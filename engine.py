"""Qwen-Image 图像引擎推理核心（CustomEngine / integrated 范式）。

（FLUX Kontext → Qwen-Image 迁移，2026-08；引擎 ID/目录/端口不变）

继承 ``CustomEngine``，实现 7 个 ``_xxx_impl`` 方法，串联三层组件：

    - pipeline_builder.py   Qwen 共享组件（VL/VAE/tokenizer/processor）+
                            GGUF Q8 transformer 加载（文件校验/单文件加载）
    - transformer_manager.py  双槽位（t2i/edit）× 单档（gguf_q8）的按需
                              加载与热切换（48GB 卡单驻 + gpu+cpu 溢出）
    - inference.py          三模式推理执行器（t2i / edit 含视角控制 / i2i 多图参考）
    - lora_manager.py       多 LoRA 叠加挂载 + transformer 切换重挂

8 端点契约（server.py 注册路由，本文件提供实现）：
    - /capabilities    variant 可用性（平台 API 优先，本地扫描 fallback）
    - /load            加载共享组件 + 默认档 transformer（to_thread 不阻塞事件循环）
    - /unload          释放 transformer 与共享组件
    - /infer           ImageInputs + images → 模式判定 → 推理 → SSE 进度帧 + 结果帧
    - /status          引擎状态（档位/槽位/显存/LoRA）
    - /lora/load       挂载单个 LoRA 到当前活跃 pipeline
    - /lora/unload     卸载单个 LoRA

variant ↔ 档位映射（与 manifest.yaml models.variants 一致，Qwen 迁移后单档）：
    Qwen/Image-GGUF-Q8 → gguf_q8（默认）
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    EngineLoadError,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
    make_error_frame,
    make_final_frame,
    make_frame,
)

from .config import EngineConfig
from .inference import InferenceRunner
from .lora_manager import LoraManager
from .pipeline_builder import (
    ModelFilesError,
    SharedComponents,
    build_edit_pipeline,
    build_t2i_pipeline,
    check_shared_files,
    check_transformer_file,
    load_shared_components,
    resolve_compute_dtype,
)
from .schemas import ImageInputs, resolve_mode
from .task_reporting import QwenImageTaskReporting, fetch_history_page
from .transformer_manager import TransformerManager

logger = logging.getLogger(__name__)

# ===== 常量 =====

# variant 名 → transformer 档位（Qwen 迁移后单档）
VARIANT_PRECISIONS: dict[str, str] = {
    "Qwen/Image-GGUF-Q8": "gguf_q8",
}

# capabilities 声明的常用输出尺寸（实际支持任意 256-2048 的 16 对齐尺寸；
# 前 5 档为 Qwen-Image 官方推荐分辨率，~1MP）
SUPPORTED_SIZES: list[str] = [
    "1328x1328", "1664x928", "928x1664", "1664x704", "704x1664",
    "1280x1280", "1024x1024", "1152x896", "896x1152",
    "1536x640", "640x1536",
]

class QwenImageEngine(CustomEngine):
    """Qwen-Image 图像引擎（文生图 / 指令编辑含视角控制 / 多图参考图生图）。

    全自定义模式（integrated 范式）：推理逻辑完全自控，不走工作流节点系统。
    （v0.2.1+ 由 FluxKontextEngine 改名而来，引擎 ID/目录/平台注册同步改为 qwen_image_5e8f3a21）
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config
        # BaseEngine.get_output_path 依赖
        self.output_root: Path = config.output_root

        # manifest 声明的默认 variant（auto-load 优先级用）
        manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant: Optional[str] = self._read_default_variant_from_manifest(manifest_path)

        # ===== 运行时组件（load() 阶段就绪）=====
        self.comps: Optional[SharedComponents] = None
        self.tm: Optional[TransformerManager] = None
        self.runner: Optional[InferenceRunner] = None
        self.loras: LoraManager = LoraManager(config.lora_dir)

        # 任务中心上报（v0.2.16 统一任务管理：gateway tasks.db 唯一真相源）
        self.task_reporting = QwenImageTaskReporting(config.engine_id)

        # ===== 状态快照字段 =====
        self._loaded_variant: Optional[str] = None
        self._device: Optional[str] = None
        self._load_time: Optional[float] = None
        self._last_used_at: Optional[float] = None

        logger.info(
            "QwenImageEngine 初始化：config=%s default_variant=%s",
            config.to_dict(), self.default_variant,
        )

    # ============================================================
    # /capabilities：variant 可用性（平台 API 优先，本地扫描 fallback）
    # ============================================================

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities。

        扫描路径遵循项目规范（日志必须记录走平台 API 还是本地 fallback 及原因）：
            1. 平台 API（AIRELIABLE_CORE_URL 已注入且可达）→ 用平台检测结果
            2. 本地 fallback：exists + size ≥ 阈值 检查共享组件与 gguf_q8 档 transformer
        """
        variants: list[VariantInfo] = []

        platform_variants = await self.get_variants()
        if platform_variants is not None:
            import os
            logger.info(
                "variant 扫描路径：平台 API（AIRELIABLE_CORE_URL=%s，%d 个 variant）",
                os.environ.get("AIRELIABLE_CORE_URL"), len(platform_variants),
            )
            for v in platform_variants:
                variants.append(VariantInfo(
                    name=v.get("name", ""),
                    base_model=v.get("base_model", "Qwen-Image-Edit-2509/Qwen-Image-2512"),
                    precision=v.get("precision", ""),
                    loader=v.get("loader", "diffusers"),
                    vram_required=v.get("vram_required", "48GB"),
                    available=bool(v.get("available", False)),
                ))
        else:
            logger.warning(
                "variant 扫描路径：本地 fallback（原因：平台 API 不可达或 AIRELIABLE_CORE_URL 未注入）"
            )
            variants = self._scan_variants_locally()

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self.loras.scan_loras(),
            supported_sizes=SUPPORTED_SIZES,
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=ImageInputs.model_json_schema(),
        )

    def _scan_variants_locally(self) -> list[VariantInfo]:
        """本地 fallback 扫描：exists + size ≥ 阈值。

        available 判定 = 共享组件齐全 且 t2i + edit 两槽位档位文件都在盘
        （variant 语义 = 一档精度的完整可运行集，与 manifest model_file_ids 一致）。
        """
        model_dir = self.config.model_dir

        shared_ok = True
        try:
            check_shared_files(model_dir)
        except ModelFilesError as e:
            shared_ok = False
            logger.warning("共享组件不完整（全部 variant 置为不可用）：%s", str(e).splitlines()[0])

        result: list[VariantInfo] = []
        for name, precision in VARIANT_PRECISIONS.items():
            t2i_ok = self._transformer_file_ok("t2i", precision)
            edit_ok = self._transformer_file_ok("edit", precision)
            available = shared_ok and t2i_ok and edit_ok
            result.append(VariantInfo(
                name=name,
                base_model="Qwen-Image-Edit-2509/Qwen-Image-2512",
                precision=precision,
                loader="diffusers",
                vram_required="48GB",
                available=available,
            ))
        logger.info(
            "本地扫描完成（model_dir=%s）：shared=%s %s",
            model_dir, shared_ok,
            {v.name: v.available for v in result},
        )
        return result

    def _transformer_file_ok(self, side: str, precision: str) -> bool:
        """检查单个槽位档位文件（exists + size ≥ 阈值）。"""
        try:
            check_transformer_file(self.config.model_dir, side, precision)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ============================================================
    # /load：共享组件 + 默认档 transformer
    # ============================================================

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load 实现。

        加载流程（全部在 to_thread 中执行，不阻塞事件循环，/health 可响应）：
            1. variant 名解析为档位（None → default_variant → 第一个在盘档位）
            2. 共享组件加载（Qwen2.5-VL-7B/VAE/tokenizer/processor/scheduler，
               常驻；VL 放置由 pipeline_builder.decide_vl_device 决策——
               剩余显存充足上 GPU，不足降 CPU 内存驻留）
            3. TransformerManager 初始化 + load_variant
               （48GB 卡 Q8 单槽 21.8GB → 单驻 + gpu+cpu 溢出策略）

        校验失败（文件缺失/残缺）抛 ModelFilesError（RuntimeError 子类），
        信息包含缺失清单与下载指引（load 路径禁止网络下载——项目规则第二部分 §2）。
        """
        variant_name, precision = self._resolve_variant(variant)
        logger.info("load() 开始：variant=%s precision=%s", variant_name, precision)

        # 保险清理（BaseEngine.load 已保证 READY 先 unload，此处防重复调用）
        if self.tm is not None:
            logger.warning("load()：检测到残留 transformer 状态，先清理再加载")
            await self._do_unload()

        t0 = time.perf_counter()

        def _blocking_load() -> tuple[str, Any, SharedComponents, TransformerManager]:
            device = self.config.resolve_device()
            dtype = resolve_compute_dtype(device)
            device_policy = self.config.device_policy
            logger.info(
                "load()：device=%s dtype=%s device_policy=%s（Turing fp16 compute 已实证"
                "NaN 黑图 → fp32 计算；VL fp16 放置由显存探测决定，Q8 单槽 ~21GB）",
                device, dtype, device_policy,
            )
            if device == "cuda":
                try:
                    import torch
                    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
                    logger.info(
                        "load() 显存探测：剩余 %.1fGB / 总 %.1fGB（gpu+cpu 策略下"
                        "不足的槽位将驻留 CPU 内存）",
                        free_mb / 1024,
                        torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    )
                except Exception:  # noqa: BLE001
                    pass

            comps = load_shared_components(self.config.model_dir, device, dtype)
            tm = TransformerManager(
                model_dir=self.config.model_dir,
                device=device,
                dtype=dtype,
                on_swap=self._on_transformer_swap,
                on_release=self._release_pipeline_refs,
                device_policy=device_policy,
            )
            tm.load_variant(precision)
            return device, dtype, comps, tm

        device, dtype, comps, tm = await asyncio.to_thread(_blocking_load)

        # 组件就绪，创建推理执行器
        self._device = device
        self.comps = comps
        self.tm = tm
        self.runner = InferenceRunner(
            transformer_manager=tm,
            lora_manager=self.loras,
            comps=comps,
            input_dir=self.config.input_dir,
            output_root=self.config.output_root,
        )
        self._loaded_variant = variant_name
        self._load_time = time.perf_counter() - t0

        logger.info(
            "load() 完成：variant=%s 槽位=%s 放置=%s 显存=%.1fMB 耗时=%.1fs",
            variant_name, tm.loaded_sides, tm.status_dict()["placement"],
            tm.vram_used_mb(), self._load_time,
        )
        return LoadResponse(
            variant=variant_name,
            state="ready",
            vram_used=self._vram_used_str(),
            load_time=round(self._load_time, 1),
        )

    def _resolve_variant(self, variant: Optional[str]) -> tuple[str, str]:
        """variant 名 → (variant 名, 档位)。

        解析优先级：显式指定 > manifest default_variant > 第一个在盘档位。
        （Qwen 迁移后仅 gguf_q8 单档）
        """
        if variant is not None:
            if variant not in VARIANT_PRECISIONS:
                raise EngineLoadError(
                    f"未知 variant：{variant}（可用：{list(VARIANT_PRECISIONS)}）"
                )
            return variant, VARIANT_PRECISIONS[variant]

        # default_variant
        if self.default_variant and self.default_variant in VARIANT_PRECISIONS:
            precision = VARIANT_PRECISIONS[self.default_variant]
            if any(
                self._transformer_file_ok(s, precision) for s in ("t2i", "edit")
            ):
                return self.default_variant, precision
            logger.warning(
                "default_variant=%s 档位文件不在盘，回退到第一个在盘档位",
                self.default_variant,
            )

        # 第一个在盘档位
        for name, precision in VARIANT_PRECISIONS.items():
            if any(self._transformer_file_ok(s, precision) for s in ("t2i", "edit")):
                return name, precision

        raise ModelFilesError(
            "transformer（gguf_q8 档 t2i/edit 槽位）均不在盘，无法加载。\n"
            f"本地路径：{self.config.model_dir}\n"
            "请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，选择镜像源后下载。"
        )

    def _on_transformer_swap(self) -> None:
        """TransformerManager 换档/换槽回调（on_swap）。

        只记日志——LoRA 重挂由 InferenceRunner.generate 统一处理
        （检测 transformer 对象身份变化后 reapply），此处重复挂载反而有害。
        """
        logger.info(
            "transformer 已切换（当前档位=%s 槽位=%s），LoRA 重挂由推理执行器按需处理",
            self.tm.loaded_precision if self.tm else None,
            self.tm.loaded_sides if self.tm else [],
        )

    def _release_pipeline_refs(self) -> None:
        """TransformerManager 换档/换出槽位前回调（on_release）。

        释放 InferenceRunner 对旧 pipeline 的引用（active_pipe 持有旧档
        transformer + PEFT adapter 层 → 显存不释放 → 新旧两档叠加溢出）。
        推理线程内被调（generate → ensure → _swap_to），引用置 None 安全。
        """
        if self.runner is not None and self.runner.active_pipe is not None:
            logger.info("换档前释放旧 pipeline 引用（旧 transformer 显存可回收）")
            self.runner.active_pipe = None

    def _sync_request_loras(self, specs: list) -> None:
        """请求级 LoRA 同步（阻塞式，供 asyncio.to_thread 调用）。

        把引擎挂载集合收敛到请求的目标集合（LoraManager.sync diff 同步）：
            - 加载缺失 / 调整权重 / 卸载多余
            - 活跃管线存在时直接挂载；无（首次推理）则记录期望状态，
              generate() 组装管线后按 transformer 身份变化 reapply 统一挂载
        """
        desired = {s.name: float(s.weight) for s in specs}
        pipe = self.runner.active_pipe if self.runner is not None else None
        self.loras.sync(pipe, desired)

    # ============================================================
    # /unload
    # ============================================================

    async def _unload_impl(self) -> UnloadResponse:
        return await self._do_unload()

    async def _do_unload(self) -> UnloadResponse:
        """实际卸载：transformer 全释放 + 共享组件释放 + LoRA 记录清空。"""
        vram_before = self._vram_used_mb()

        def _blocking_unload() -> None:
            if self.tm is not None:
                self.tm.unload()
            # 共享组件引用清空（Python GC 回收，配合 empty_cache 归还显存）
            self.comps = None
            self.runner = None
            # GC 必须在引用断开之后（accelerate hook 循环引用依赖 GC 回收），
            # 否则 empty_cache 释放不了仍存活 的 tensor（换档泄漏同根因）
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

        await asyncio.to_thread(_blocking_unload)

        # transformer 对象已释放，LoRA 运行时记录随之失效
        self.loras.clear()
        self._loaded_variant = None
        self._device = None
        self._load_time = None

        vram_freed = max(vram_before - self._vram_used_mb(), 0.0)
        logger.info("unload() 完成：释放显存约 %.1fMB", vram_freed)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed / 1024:.1f}GB" if vram_freed > 0 else None,
        )

    # ============================================================
    # /infer：三模式推理
    # ============================================================

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer 实现。

        流程：
            1. ImageInputs 校验（失败推 invalid_inputs 错误帧，不进入 ERROR 态）
            2. 输入图 staging（url 下载/b64 解码/path 直读，to_thread）
            3. 模式判定（显式 mode > images 数量推断 > 兜底 t2i）
            4. 绑定本次推理的 SSE 进度回调（diffusers 步回调跨线程 put_threadsafe）
            5. to_thread 执行 generate（含档位保证/管线组装/LoRA 重挂/采样/保存）
            6. 结果帧 + 生成历史
        """
        if self.runner is None or self.tm is None:
            raise EngineLoadError(
                "引擎未加载模型（load 未完成或已卸载），无法推理。"
                "平台 auto-load 应已触发；若直接调用请先 POST /load。"
            )

        # 1. 参数校验
        try:
            inputs = ImageInputs(**(request.inputs or {}))
        except ValidationError as e:
            logger.warning("infer 参数校验失败：%s", e)
            stream.put_threadsafe(make_error_frame(
                f"inputs 参数校验失败：{e}", "invalid_inputs",
            ))
            return

        # 1.5 gateway 档位切换：request.variant（gateway 的 model 字段解析结果，
        # 如 Qwen/Image-GGUF-Q8）→ 精度档位。inputs.precision 未显式指定时生效，
        # 使 gateway 侧通过 model 字段即可热切换精度档（与 WebUI 档位下拉等价）。
        if inputs.precision is None and request.variant in VARIANT_PRECISIONS:
            inputs.precision = VARIANT_PRECISIONS[request.variant]
            logger.info("variant 档位映射：%s → precision=%s", request.variant, inputs.precision)

        task_id = request.task_id or f"qwen_{int(time.time())}_{id(request) & 0xFFFF:x}"

        # 任务中心上报：任务创建（v0.2.16 统一任务管理）
        # 上报分工（见 task_reporting.py 模块注释）：
        #   转发链路（request.task_id 非空）只报 created（gateway 维护进度/终态）；
        #   直连链路（WebUI 等）报全生命周期（created→running→progress→终态）。
        gateway_forwarded = bool(request.task_id)
        mode_hint = inputs.mode or (
            "t2i" if not request.images
            else ("edit" if len(request.images) == 1 else "i2i")
        )
        await self.task_reporting.on_infer_created(
            task_id,
            gateway_forwarded=gateway_forwarded,
            mode=mode_hint,
            request_summary={
                "prompt": inputs.prompt[:200],
                "mode": mode_hint,
                "size": f"{inputs.width or 1024}x{inputs.height or 1024}",
                "seed": inputs.seed,
                "steps": inputs.num_inference_steps,
                "loras": [s.name for s in request.loras] or list(self.loras.loaded_loras),
            },
            variant=request.variant or self._loaded_variant or "",
        )
        self.task_reporting.on_infer_running(
            task_id, gateway_forwarded=gateway_forwarded,
        )

        # 推理执行体异常在引擎内消化（推 engine_error 帧后返回）：
        # 单次推理失败（文件缺失/网络问题/参数问题）≠ 引擎故障，模型仍在显存，
        # SDK 基类 finally 会把状态回 READY，引擎可继续服务后续请求。
        # （若异常冒泡到 SDK infer()，状态机进 ERROR 并卡死，需重新 load 才能恢复。）
        try:
            # 1.8 请求级 LoRA 同步（diff：加载缺失/调整权重/卸载多余）
            # gateway 的 extra.loras → InferRequest.loras；引擎内挂载状态随之收敛。
            # lora_explicit 语义（v0.2.16）：
            #   True  = 请求方显式指定集合（含空数组=清空全部）→ sync 收敛
            #   False = 请求方未指定 → 保持当前挂载（WebUI 会话语义，
            #           挂载状态由 /lora/load /lora/unload 独立管理）
            # 同步失败（LoRA 不在盘/键名不匹配）属请求参数问题 → 错误帧 + 保持 READY
            lora_explicit = getattr(request, "lora_explicit", False)
            if lora_explicit:
                logger.info(
                    "请求显式指定 LoRA 集合：%s（当前已挂载：%s）",
                    [(s.name, s.weight) for s in request.loras],
                    list(self.loras.loaded_loras),
                )
                stream.put_threadsafe(make_frame(
                    0.015, "preparing_loras",
                    detail=(
                        f"同步请求 LoRA：{[s.name for s in request.loras]}"
                        if request.loras else "清空全部已挂载 LoRA"
                    ),
                ))
                await asyncio.to_thread(self._sync_request_loras, request.loras)

            # 2. 输入图 staging
            images: list = []
            if request.images:
                stream.put_threadsafe(make_frame(
                    0.01, "staging_images", detail=f"加载 {len(request.images)} 张输入图",
                ))
                images = await asyncio.to_thread(self.runner.stage_images, request.images)

            # 3. 模式判定
            mode = resolve_mode(inputs, len(images))
            logger.info(
                "infer 开始：task_id=%s mode=%s precision_override=%s images=%d prompt=%s...",
                task_id, mode, inputs.precision, len(images), inputs.prompt[:50],
            )

            # 4. 绑定本次推理的进度回调（推理线程 → SSE 帧 + 任务中心进度）
            def _progress(stage: str, progress: float, detail: str, **extra: Any) -> None:
                frame = make_frame(progress, stage, detail=detail, **extra)
                stream.put_threadsafe(frame)
                # 任务中心进度（仅直连链路；推理线程 → fire 跨线程安全）
                self.task_reporting.on_infer_progress(
                    task_id,
                    gateway_forwarded=gateway_forwarded,
                    progress=progress, stage=stage,
                )

            self.runner.progress_cb = _progress
            try:
                # 5. 执行推理（阻塞式，to_thread）
                result = await asyncio.to_thread(
                    self.runner.generate, mode, inputs, images, task_id,
                )
            finally:
                self.runner.progress_cb = None

            # 6. 结果帧 + 任务中心成功上报
            self._last_used_at = time.time()
            # 注入 loras 到 result（任务中心历史视图展示用；对 SSE 消费方无害）
            result["loras"] = list(self.loras.loaded_loras)
            # 直连链路 await 先入库再返回结果帧（WebUI 刷新历史不闪空）；
            # 转发链路跳过（gateway 调度器消费 SSE 帧后自行 mark_succeeded）
            await self.task_reporting.on_infer_succeeded(
                task_id, gateway_forwarded=gateway_forwarded, result=result,
            )
            await stream.put(make_final_frame(result, {
                "total_seconds": result["duration_seconds"],
                "infer_seconds": result["infer_seconds"],
                "mode": mode,
            }))
        except Exception as e:  # noqa: BLE001
            logger.exception("推理执行失败：task_id=%s", task_id)
            # 任务中心失败上报（仅直连链路，fire-and-forget）
            self.task_reporting.on_infer_failed(
                task_id,
                gateway_forwarded=gateway_forwarded,
                error={"type": "engine_error", "message": str(e)},
            )
            stream.put_threadsafe(make_error_frame(
                f"推理失败：{e}", "engine_error",
            ))

    async def get_history(self, page: int = 1, page_size: int = 20) -> dict:
        """分页读取生成历史（server.py /history 端点调用）。

        v0.2.16 统一任务管理：内存历史已删除，本方法代理查询 gateway
        任务中心（engine_id 过滤 + status=succeeded），映射为 HistoryItem
        视图结构。gateway 不可达时降级返回空列表 + degraded 标记。
        """
        return await fetch_history_page(self.engine_id, page=page, page_size=page_size)

    # ============================================================
    # /lora/load、/lora/unload
    # ============================================================

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load 实现：挂载单个 LoRA 到当前活跃 pipeline。

        若尚无活跃 pipeline（load 后未推理过），用当前档位已加载槽位组装一个
        （优先 edit——LoRA 与槽位不匹配时 load_lora_weights 会抛错，
        由上层转为 400/500）。
        """
        pipe = await asyncio.to_thread(self._ensure_active_pipe)
        loaded = await asyncio.to_thread(self.loras.load_lora, pipe, name, weight)
        return LoraLoadResponse(
            name=name, weight=weight, loaded_loras=list(loaded.keys()),
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload 实现。"""
        pipe = self.runner.active_pipe if self.runner is not None else None
        loaded = await asyncio.to_thread(self.loras.unload_lora, pipe, name)
        return LoraUnloadResponse(name=name, loaded_loras=list(loaded.keys()))

    def _ensure_active_pipe(self):
        """获取/组装当前活跃 pipeline（LoRA 挂载点）。"""
        if self.tm is None or self.runner is None or self.comps is None:
            raise EngineLoadError("模型未加载，无法挂载 LoRA（请先 POST /load）")

        pipe = self.runner.active_pipe
        if pipe is not None and getattr(pipe, "transformer", None) is not None:
            return pipe

        # 尚无活跃 pipeline：用当前档位已加载槽位组装（优先 edit）
        side = "edit" if self.tm.is_loaded("edit") else "t2i"
        transformer = self.tm.get(side)
        if transformer is None:
            raise EngineLoadError(f"槽位 {side} 未加载，无法组装 LoRA 挂载点")
        logger.info("组装初始活跃 pipeline（side=%s）作为 LoRA 挂载点", side)
        if side == "edit":
            pipe = build_edit_pipeline(self.comps, transformer, self.tm.device)
        else:
            pipe = build_t2i_pipeline(self.comps, transformer, self.tm.device)
        self.runner.active_pipe = pipe
        return pipe

    # ============================================================
    # /status
    # ============================================================

    async def _status_impl(self) -> StatusResponse:
        """GET /status 实现。"""
        tm_status = self.tm.status_dict() if self.tm else {}
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self._loaded_variant,
            vram_used=self._vram_used_str(),
            vram_total=self._vram_total_str(),
            loaded_loras=list(self.loras.loaded_loras),
            last_used_at=self._last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                **tm_status,
                "device": self._device,
                "comps_loaded": bool(self.comps and self.comps.is_loaded),
                "load_time": round(self._load_time, 1) if self._load_time else None,
            },
        )

    # ============================================================
    # 显存统计
    # ============================================================

    def _vram_used_mb(self) -> float:
        if self.tm is not None:
            return self.tm.vram_used_mb()
        return 0.0

    def _vram_used_str(self) -> str:
        return f"{self._vram_used_mb() / 1024:.1f}GB"

    def _vram_total_str(self) -> Optional[str]:
        try:
            import torch
            if torch.cuda.is_available():
                return f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except Exception:  # noqa: BLE001
            pass
        return None
