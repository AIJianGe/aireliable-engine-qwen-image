"""三模式推理执行器：t2i（文生图）/ edit（单图指令编辑 + 视角控制）/ i2i（多图参考）。

模式与管线映射（Qwen 迁移后，分析文档 §2 决策 1/2 + §6）：
    - t2i  → QwenImagePipeline（Qwen-Image-2512 transformer）
    - edit → QwenImageEditPlusPipeline 单图输入（Qwen-Image-Edit-2509
      transformer）+ 可选视角 prompt 前缀（§4.2.1 翻译表）
    - i2i  → QwenImageEditPlusPipeline 原生多图列表（每图独立 VAE 编码 +
      latent 序列拼接 + img_shapes 元数据，官方上限 3 图；prompt 未显式
      指代时包装 Picture N 模板——官方训练分布的指代方式）

与 FLUX 时代的关键差异：
    - prompt 编码由管线内部走 Qwen2.5-VL chat template（文本 + 图像 token
      联合处理），无需手动 _encode_prompt（CLIP/T5 路线已删）
    - true_cfg_scale（默认 4.0）替代 FLUX embedded guidance；
      guidance_scale 语义变为 norm rescaling（默认 1.0）
    - negative_prompt 在 true_cfg_scale > 1 时生效（Qwen 支持）
    - 多图参考无需拼接（stitcher.py 已删）

职责：
    1. 输入图 staging：ImageInput（url/b64_json/path）→ PIL.Image
       （url 下载落盘 input_dir；b64 解码落盘；path 直接读）
    2. pipeline 组装（每次推理按 mode 重组装，组件为共享对象引用，开销可忽略）
    3. transformer 槽位保证（TransformerManager.ensure 热切换）
    4. LoRA 跨 transformer 追挂（transformer 对象变化时 reapply）
    5. diffusers callback_on_step_end → SSE 进度帧（put_threadsafe 跨线程）
    6. 输出图保存 output/image/{task_id}.png + 元信息返回（含非黑校验）

运行方式：engine._infer_impl 在 asyncio.to_thread 中调用 generate()（阻塞式）。
"""

from __future__ import annotations

import base64
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .pipeline_builder import SharedComponents, build_edit_pipeline, build_t2i_pipeline
from .schemas import ImageInputs, ViewSpec
from .transformer_manager import TransformerManager

logger = logging.getLogger(__name__)

MAX_REFERENCE_IMAGES = 5          # i2i 多图参考上限（官方训练分布 1-3 张；4-5 张
                                  # 管线可跑但指令遵循度下降；10+ 张 vision tokens
                                  # 爆炸 + 远超训练分布，不支持）
# Qwen-Image 官方推荐输出分辨率（~1MP，潜空间 16 对齐）
DEFAULT_OUTPUT_SIZE = (1328, 1328)
ALIGN = 16                        # 潜空间对齐（VAE 8x + packing 2x）

# ---- 视角控制 prompt 翻译表（分析文档 §4.2.1，官方 multiple-angles 参数化）----
# 默认值（front / level / medium）不拼任何描述 → 纯指令编辑零干扰
# 注意：必须用"相机位"语言（viewed from...），不能用"物体旋转"语言
# （rotated ... to the left 模型会理解为把主体转个角度而非移动相机，
# 2026-08-25 实测教训：用户反馈水平控制无效 → 前缀全部改为相机位描述）
_VIEW_HORIZONTAL: dict[str, str] = {
    "left45": "the same subject viewed from the left at a 45-degree angle",
    "left90": "the same subject viewed directly from its left side",
    "right45": "the same subject viewed from the right at a 45-degree angle",
    "right90": "the same subject viewed directly from its right side",
}
_VIEW_VERTICAL: dict[str, str] = {
    "up30": "low angle shot looking up at the subject",
    "down30": "elevated view of the subject",
    "down60": "high angle shot looking down at the subject",
}
_VIEW_DISTANCE: dict[str, str] = {
    "close": "close-up shot",
    "extreme": "extreme close-up shot",
    "far": "wide shot",
}

# prompt 中已含 Picture/图N 指代的检测（i2i 模板包装跳过）
_PICTURE_REF_RE = re.compile(r"(picture\s*\d|图\s*\d)", re.IGNORECASE)


def align16(v: int) -> int:
    return max(ALIGN, (int(v) // ALIGN) * ALIGN)


def build_view_prefix(view: ViewSpec) -> str:
    """视角参数 → prompt 英文前缀片段（逗号连接；全默认返回空串）。"""
    parts: list[str] = []
    if view.horizontal in _VIEW_HORIZONTAL:
        parts.append(_VIEW_HORIZONTAL[view.horizontal])
    if view.vertical in _VIEW_VERTICAL:
        parts.append(_VIEW_VERTICAL[view.vertical])
    if view.distance in _VIEW_DISTANCE:
        parts.append(_VIEW_DISTANCE[view.distance])
    return ", ".join(parts)


class InferenceRunner:
    """三模式推理执行器（阻塞式，供 asyncio.to_thread 调用）。"""

    def __init__(
        self,
        transformer_manager: TransformerManager,
        lora_manager: Any,
        comps: SharedComponents,
        input_dir: Path,
        output_root: Path,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
    ):
        """
        Args:
            comps: 已加载的共享组件（load() 阶段就绪）
            input_dir: 输入 staging 目录（url/b64 图片落盘）
            output_root: 输出根目录（output/，图片存 output/image/）
            progress_cb: SSE 进度回调（stage, progress, detail）
        """
        self.tm = transformer_manager
        self.loras = lora_manager
        self.comps = comps
        self.input_dir = input_dir
        self.output_root = output_root
        self._progress_cb = progress_cb

        # 当前活跃 pipeline 及其 transformer 身份（LoRA 跨 transformer 追挂判定）
        self._active_pipe: Any = None
        self._active_transformer_id: Optional[int] = None

    # ===== engine 层访问接口 =====

    @property
    def progress_cb(self) -> Optional[Callable[..., None]]:
        """当前 SSE 进度回调（engine._infer_impl 每次推理绑定当次 stream）。"""
        return self._progress_cb

    @progress_cb.setter
    def progress_cb(self, cb: Optional[Callable[..., None]]) -> None:
        self._progress_cb = cb

    @property
    def active_pipe(self) -> Any:
        """当前活跃 pipeline（LoRA 挂载点；None = 尚未推理过）。"""
        return self._active_pipe

    @active_pipe.setter
    def active_pipe(self, pipe: Any) -> None:
        self._active_pipe = pipe
        if pipe is not None and getattr(pipe, "transformer", None) is not None:
            self._active_transformer_id = id(pipe.transformer)

    # ============================================================
    # 主入口
    # ============================================================

    def generate(
        self,
        mode: str,
        inputs: ImageInputs,
        images: list[Image.Image],
        task_id: str,
    ) -> dict:
        """执行一次推理，返回结果元信息。

        Returns:
            {
                "image_path": 绝对路径,
                "image_url": "/output/image/{task_id}.png",
                "width": int, "height": int,
                "mode": str, "precision": str, "seed": int,
                "duration_seconds": float,
            }
        """
        t_start = time.perf_counter()

        # ---- 1. transformer 槽位保证（t2i 独槽；edit/i2i 共用 edit 槽）----
        side = "t2i" if mode == "t2i" else "edit"
        self._emit("preparing", 0.02, f"保证 {side} transformer（{inputs.precision or '当前档'}）")
        transformer = self.tm.ensure(side, inputs.precision)

        # ---- 2. 组装 pipeline（mode 决定管线类型）----
        # edit（单图指令编辑）与 i2i（2-3 图多图参考）共用 EditPlus 管线，
        # 按输入图数量区分，无槽位切换开销
        self._emit("preparing", 0.05, f"组装 {mode} 管线")
        if mode == "t2i":
            pipe = build_t2i_pipeline(self.comps, transformer, self.tm.device)
        else:
            pipe = build_edit_pipeline(self.comps, transformer, self.tm.device)

        # ---- 3. LoRA 跨 transformer 追挂 ----
        if id(pipe.transformer) != self._active_transformer_id:
            if self.loras.loaded_loras:
                self._emit("preparing", 0.08, "重挂 LoRA（transformer 已切换）")
                self.loras.reapply(pipe)
        self._active_pipe = pipe
        self._active_transformer_id = id(pipe.transformer)

        # ---- 3.5 输入图数量校验（Qwen 原生多图，无需拼接）----
        if mode == "edit" and len(images) != 1:
            raise ValueError(f"edit 模式需要恰好 1 张输入图（收到 {len(images)} 张）")
        if mode == "i2i":
            if len(images) < 2:
                raise ValueError(
                    f"i2i 多图参考模式需要 2-{MAX_REFERENCE_IMAGES} 张参考图（收到 {len(images)} 张）"
                )
            if len(images) > MAX_REFERENCE_IMAGES:
                logger.warning(
                    "参考图超上限（%d > %d），截取前 %d 张",
                    len(images), MAX_REFERENCE_IMAGES, MAX_REFERENCE_IMAGES,
                )
                images = images[:MAX_REFERENCE_IMAGES]

        # ---- 4. 推理参数 ----
        gen_params, seed = self._build_params(mode, inputs, images)

        # ---- 5. 执行（callback → SSE 进度）----
        self._emit("denoising", 0.1, f"开始采样（{gen_params['num_inference_steps']} 步）")
        t_infer = time.perf_counter()
        callback = self._make_step_callback(
            total=gen_params["num_inference_steps"],
            base=0.1, span=0.85,
        )
        result = pipe(**gen_params, callback_on_step_end=callback)
        image: Image.Image = result.images[0]
        infer_seconds = time.perf_counter() - t_infer

        # ---- 5.5 非黑校验（NaN 防御的输出侧兜底：GGUF 量化 + fp16 黑图教训）----
        extrema = image.getextrema()
        if all(lo >= 200 for lo, _hi in extrema):
            logger.warning(
                "输出图疑似全白/空白（extrema=%s）——可能存在 NaN/量化溢出，"
                "建议降低分辨率或更换 seed 重试", extrema,
            )
        elif all(hi <= 8 for _lo, hi in extrema):
            # 2026-08-25 黑图根因（fp16 compute NaN）的输出侧检测：全黑漏检
            # 让任务"succeeded"却输出 5207B 纯黑 PNG，用户无从排查
            logger.warning(
                "输出图疑似全黑（extrema=%s）——latents 可能 NaN（数值溢出），"
                "请检查引擎日志中 compute_dtype 配置是否为 fp32", extrema,
            )

        # ---- 6. 保存输出 ----
        image_dir = self.output_root / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        out_path = image_dir / f"{task_id}.png"
        image.save(out_path)
        duration = time.perf_counter() - t_start

        logger.info(
            "推理完成：mode=%s precision=%s size=%dx%d seed=%s 耗时=%.1fs（采样 %.1fs）",
            mode, self.tm.loaded_precision, image.size[0], image.size[1], seed,
            duration, infer_seconds,
        )

        return {
            "image_path": str(out_path.resolve()),
            "image_url": f"/output/image/{task_id}.png",
            # 平台契约（gateway _infer_callback 读 result.image_urls）
            "image_urls": [f"/output/image/{task_id}.png"],
            "images": [str(out_path.resolve())],
            "width": image.size[0],
            "height": image.size[1],
            "mode": mode,
            "precision": self.tm.loaded_precision,
            "seed": seed,
            "duration_seconds": round(duration, 2),
            "infer_seconds": round(infer_seconds, 2),
        }

    # ============================================================
    # 参数构造
    # ============================================================

    def _build_params(
        self,
        mode: str,
        inputs: ImageInputs,
        images: list[Image.Image],
    ) -> tuple[dict, int]:
        """构造 pipeline 调用参数（模式差异在此收敛）。

        Returns:
            (管线调用参数 dict, 生效 seed)
        """
        import torch

        # seed
        if inputs.seed is None or inputs.seed < 0:
            # 随机种子（torch.random.manual_seed() 必须传参，无参调用会 TypeError）
            import random
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(inputs.seed)
        generator = torch.Generator(device=self.tm.device if self.tm.device != "cpu" else "cpu")
        generator = generator.manual_seed(seed)

        # 输出尺寸（默认 1328×1328 官方推荐档；edit 模式默认跟随输入图）
        explicit_size = bool(inputs.width and inputs.height)
        if explicit_size:
            width, height = align16(inputs.width), align16(inputs.height)
        elif mode == "edit" and images:
            width, height = align16(images[0].size[0]), align16(images[0].size[1])
        else:
            width, height = DEFAULT_OUTPUT_SIZE

        # prompt 组装（视角前缀 + Picture 模板包装）
        prompt = inputs.prompt

        # ---- edit 模式：视角前缀拼接（§4.2.1 翻译表）----
        # 全默认值 = 不拼任何视角描述（纯指令编辑，零干扰）
        if mode == "edit" and inputs.view is not None and inputs.view.has_any:
            view_prefix = build_view_prefix(inputs.view)
            if view_prefix:
                prompt = f"{view_prefix}, {prompt}" if prompt else view_prefix
                logger.info("edit 视角控制前缀：%s", view_prefix)

        # ---- i2i 模式：Picture N 模板包装 ----
        # 「Picture N」是 EditPlus 官方训练分布的图像指代方式；用户 prompt
        # 未显式指代时补一句编号声明，帮助模型建立图 ↔ 编号锚点
        if mode == "i2i" and prompt and not _PICTURE_REF_RE.search(prompt):
            n = len(images)
            pic_list = " and ".join(f"Picture {i + 1}" for i in range(n))
            prompt = f"{pic_list} are provided as reference images. {prompt}"
            logger.info("i2i prompt 包装 Picture 模板（%d 图）", n)

        # Qwen 管线共用参数（prompt 编码由管线内部 VL chat template 处理）
        params: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": inputs.num_inference_steps,
            "true_cfg_scale": inputs.true_cfg_scale,
            "guidance_scale": inputs.guidance_scale,
            "generator": generator,
            "max_sequence_length": inputs.max_sequence_length,
        }
        # negative_prompt 在 true_cfg_scale > 1 时被管线消费
        if inputs.negative_prompt and inputs.true_cfg_scale > 1.0:
            params["negative_prompt"] = inputs.negative_prompt

        if mode == "t2i":
            return params, seed

        # ---- edit / i2i：EditPlus 管线 ----
        # Qwen 原生多图：每图独立 VAE 编码 + latent 序列拼接 + img_shapes
        # （管线内部处理，引擎侧只传 PIL 列表，无需拼接）
        params["image"] = images[0] if mode == "edit" else list(images)
        return params, seed

    # ============================================================
    # 输入图 staging（ImageInput → PIL）
    # ============================================================

    def stage_images(self, image_inputs: list[Any]) -> list[Image.Image]:
        """把 InferRequest.images（url/b64_json/path 三来源）加载为 PIL 列表。"""
        out: list[Image.Image] = []
        for i, img_input in enumerate(image_inputs):
            path = self._stage_one(img_input, index=i)
            image = Image.open(path).convert("RGB")
            out.append(image)
        logger.info("输入图 staging 完成：%d 张", len(out))
        return out

    def _stage_one(self, img_input: Any, index: int) -> Path:
        """单张输入图落盘并返回路径。"""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        stem = f"input_{int(time.time())}_{uuid.uuid4().hex[:8]}_{index}"

        # 1) 本地路径（网关已下载 / 用户本地）
        if getattr(img_input, "path", None):
            p = Path(img_input.path)
            if p.exists():
                return p
            logger.warning("ImageInput.path 不存在，尝试其他来源: %s", p)

        # 2) base64
        if getattr(img_input, "b64_json", None):
            raw = base64.b64decode(img_input.b64_json)
            p = self.input_dir / f"{stem}.png"
            p.write_bytes(raw)
            return p

        # 3) URL 下载（httpx 流式）
        if getattr(img_input, "url", None):
            import httpx
            suffix = ".png"
            if ".jpg" in img_input.url or ".jpeg" in img_input.url:
                suffix = ".jpg"
            elif ".webp" in img_input.url:
                suffix = ".webp"
            p = self.input_dir / f"{stem}{suffix}"
            with httpx.Client(timeout=httpx.Timeout(30, read=120), follow_redirects=True) as client:
                with client.stream("GET", img_input.url) as resp:
                    resp.raise_for_status()
                    with open(p, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
            logger.info("输入图已下载：%s -> %s", img_input.url[:80], p)
            return p

        raise ValueError(f"ImageInput[{index}] 缺少 url/b64_json/path 来源字段")

    # ============================================================
    # 进度回调
    # ============================================================

    def _make_step_callback(self, total: int, base: float, span: float) -> Callable:
        """构造 diffusers callback_on_step_end（采样步 → SSE 进度）。"""
        import torch  # noqa: F401（callback 签名需要）

        def callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
            progress = base + span * (step + 1) / max(total, 1)
            self._emit_threadsafe(
                "denoising", progress,
                f"采样步 {step + 1}/{total}（timestep={timestep}）",
                step=step + 1, total=total,
            )
            return callback_kwargs

        return callback

    def _emit(self, stage: str, progress: float, detail: str, **extra: Any) -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(stage, progress, detail, **extra)
            except Exception:  # noqa: BLE001
                logger.debug("进度回调失败", exc_info=True)

    def _emit_threadsafe(self, stage: str, progress: float, detail: str, **extra: Any) -> None:
        """线程安全版进度回调（diffusers callback 在推理线程触发）。"""
        if self._progress_cb is not None:
            try:
                self._progress_cb(stage, progress, detail, threadsafe=True, **extra)
            except Exception:  # noqa: BLE001
                logger.debug("进度回调失败", exc_info=True)
