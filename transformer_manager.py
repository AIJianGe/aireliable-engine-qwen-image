"""Transformer 管理器：Qwen-Image 双槽位（t2i / edit）× Q8 单档的按需加载与热切换。

Qwen 迁移后架构（分析文档 §2 决策 3/4）：
    - 双槽位：t2i（Qwen-Image-2512，文生图）/ edit（Qwen-Image-Edit-2509，
      edit 单图指令编辑 + i2i 多图参考共用，按输入图数量区分，无槽位切换）
    - 单档精度：gguf_q8（GGUF Q8_0 单文件 ~21.8GB；FLUX 时代的
      bf16 / fp8 档随 FLUX 权重一并移除，D3 决策）
    - 显存策略（Quadro RTX 8000 48GB）：
        Q8 单槽 21.8GB × 2 = 43.6GB，加 VL（16.6GB GPU 时）必然放不下双驻
        → 一律单驻热切换：
        * auto / gpu：edit 槽 GPU 常驻，t2i 槽推理时换入（旧槽先释放）
        * gpu+cpu：edit 槽 GPU 常驻，t2i 槽 CPU 内存驻留（对象保留），
          ensure 换槽时纯内存搬移（PCIe 秒级 vs 重读盘 40s+）
        * cpu：全 CPU
    - 共享组件（VL/VAE/tokenizer/processor/scheduler）由 pipeline_builder
      单独加载，本模块只管 transformer；管线组装在推理时按需进行

热切换语义：
    - ensure(side)：保证 side 槽位就绪（且在 GPU 上），返回 transformer 模块
      * 槽位未加载（auto/gpu）→ 换出另一槽位后加载
      * 槽位 CPU 驻留（gpu+cpu）→ 内存搬换入 GPU，旧 GPU 槽先搬出
    - 切换后 LoRA 丢失（新模型对象，auto/gpu 路径）→ 通过 on_swap 回调
      通知 engine 重挂；gpu+cpu 搬移路径对象身份不变，LoRA 无需重挂

设备策略（gpu+cpu 溢出，见 DEVICE_POLICY_GPU_CPU_OFFLOAD_ANALYSIS.md）：
    - load 时显存探测（torch.cuda.mem_get_info）：
        * 剩余 ≥ 双槽 → 双槽 GPU 常驻（≥80GB 卡）
        * 剩余 ≥ 单槽 → 单槽 GPU 常驻（edit 优先），t2i 槽 CPU 内存驻留
        * 剩余 < 单槽 → RuntimeError（指引仅 CPU 策略）
    - ensure 换槽：CPU 驻留槽位按需换入 GPU（旧 GPU 槽先 to("cpu") 释放），
      纯内存搬移不重读盘
    - LoRA 适配器随模块搬移（对象身份不变，无需重挂）

进度上报：load/swap 过程通过 progress_cb(stage, progress, detail) 回调，
engine 层转成 SSE 帧或日志。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from .pipeline_builder import (
    TRANSFORMER_FILES,
    TRANSFORMER_EDIT_SUBDIR,
    TRANSFORMER_T2I_SUBDIR,
    check_transformer_file,
    load_transformer,
)

logger = logging.getLogger(__name__)

SIDES = ("t2i", "edit")
PRECISIONS = ("gguf_q8",)

# Q8 槽位显存需求估算（Qwen GGUF Q8_0 单槽 ~21.8GB；探测兜底默认值）
_SLOT_NEED_GB = 22.0

# 槽位 → transformer 子目录（缺失错误信息构造用）
_SIDE_SUBDIR = {"t2i": TRANSFORMER_T2I_SUBDIR, "edit": TRANSFORMER_EDIT_SUBDIR}

# 进度回调类型：(stage: str, progress: float, detail: str) -> None
ProgressCallback = Callable[[str, float, str], None]


class TransformerManager:
    """Qwen-Image 双槽位（t2i/edit）单档（gguf_q8）管理器。"""

    def __init__(
        self,
        model_dir: Path,
        device: str,
        dtype: torch.dtype,
        progress_cb: Optional[ProgressCallback] = None,
        on_swap: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
        device_policy: str = "auto",
    ):
        """
        Args:
            model_dir: models/image/qwen-image/ 根目录
            device: 已解析的推理设备（"cuda" / "cpu"）
            dtype: 计算 dtype（fp32；Turing fp16 compute 实证 NaN 黑图，弃用）
            progress_cb: 加载进度回调（后台加载任务转发 SSE 用）
            on_swap: transformer 切换完成回调（engine 用于重挂 LoRA）
            on_release: 换档/换出槽位**前**回调（持有方释放旧 pipeline 引用，
                否则 InferenceRunner._active_pipe 持有旧档 transformer，
                显存不释放 → 新旧两档叠加溢出到系统内存，推理极慢）
            device_policy: 设备策略（auto/gpu/gpu+cpu/cpu）。
                gpu+cpu 且 device=cuda 时启用槽位显存溢出（显存不足的槽位
                驻留 CPU 内存，ensure 换槽时内存搬移换入换出）
        """
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.device_policy = device_policy
        self._progress_cb = progress_cb
        self._on_swap = on_swap
        self._on_release = on_release

        # 当前档位（None = 未加载；Qwen 迁移后恒为 "gguf_q8"）
        self._precision: Optional[str] = None
        # 槽位 → 已加载的 transformer 模块
        self._transformers: dict[str, Any] = {}
        # 槽位 → 实际放置（"gpu" / "cpu"，gpu+cpu 溢出用）
        self._placement: dict[str, str] = {}
        # 最近一次 GPU 槽位加载的实测显存用量（MB，下一槽位放置探测用）
        self._slot_usage_mb: float = 0.0
        # 各槽位加载耗时（status 展示用）
        self._load_times: dict[str, float] = {}

    # ===== 状态查询 =====

    @property
    def loaded_precision(self) -> Optional[str]:
        return self._precision

    def is_loaded(self, side: str) -> bool:
        return side in self._transformers

    @property
    def loaded_sides(self) -> list[str]:
        return list(self._transformers.keys())

    def get(self, side: str) -> Any:
        """获取已加载的 transformer（未加载返回 None）。"""
        return self._transformers.get(side)

    def vram_used_mb(self) -> float:
        if not torch.cuda.is_available() or self.device == "cpu":
            return 0.0
        try:
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**2
        except Exception:  # noqa: BLE001
            return 0.0

    # ===== 设备策略辅助（gpu+cpu 溢出）=====

    def _offload_enabled(self) -> bool:
        """gpu+cpu 策略且实际运行在 CUDA 上时启用槽位显存溢出。"""
        return self.device_policy == "gpu+cpu" and self.device != "cpu"

    def _vram_free_mb(self) -> float:
        """GPU 剩余显存（MB，torch.cuda.mem_get_info 实测，含其他进程占用）。"""
        if not torch.cuda.is_available() or self.device == "cpu":
            return 0.0
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / 1024**2
        except Exception:  # noqa: BLE001
            return 0.0

    def _cuda_alloc_mb(self) -> float:
        """本进程已分配显存（MB，memory_allocated 实测）。"""
        if not torch.cuda.is_available() or self.device == "cpu":
            return 0.0
        try:
            return torch.cuda.memory_allocated() / 1024**2
        except Exception:  # noqa: BLE001
            return 0.0

    def _fit_another_slot(self) -> bool:
        """探测剩余显存是否够再驻留一个 Q8 槽位（~21.8GB）。

        优先用已加载 GPU 槽位的实测用量（memory_allocated 增量）估算，
        留 10% 余量应对碎片与运行时 activation。
        """
        need_mb = self._slot_usage_mb * 1.1 if self._slot_usage_mb > 0 else _SLOT_NEED_GB * 1024
        free_mb = self._vram_free_mb()
        fits = free_mb >= need_mb
        logger.info(
            "gpu+cpu 显存探测：剩余 %.1fGB，下一槽位需求 ~%.1fGB → %s",
            free_mb / 1024, need_mb / 1024,
            "GPU 常驻" if fits else "CPU 内存驻留",
        )
        return fits

    def _vram_insufficient_error(self, side: str) -> RuntimeError:
        """构造显存不足错误（含换策略指引）。"""
        free_gb = self._vram_free_mb() / 1024
        return RuntimeError(
            f"设备策略 {self.device_policy}：gguf_q8 档 {side} 槽位无法载入显存"
            f"（当前剩余 {free_gb:.1f}GB，Q8 单槽需 ~21.8GB）。\n"
            f"建议：① 设备策略改为「gpu+cpu」（t2i 槽 CPU 内存驻留）；"
            f"② 改为「仅 CPU」（速度慢，仅兜底）；"
            f"③ 停止其他占用显存的引擎后重试。"
        )

    # ===== 档位文件就绪检查（capabilities/variant 扫描用，不加载）=====

    def side_available(self, side: str, precision: str = "gguf_q8") -> bool:
        """检查 side 槽位的模型文件是否在盘（不加载）。"""
        try:
            check_transformer_file(self.model_dir, side, precision)
            return True
        except Exception:  # noqa: BLE001
            return False

    def available_sides_for(self, precision: str = "gguf_q8") -> list[str]:
        """返回在盘上可用的槽位列表（Qwen 迁移后恒为 gguf_q8 单档）。"""
        return [s for s in SIDES if self.side_available(s, precision)]

    # ===== 加载 / 热切换 =====

    def load_variant(self, precision: str = "gguf_q8") -> None:
        """按档位初始化加载（engine load() 调用）。

        Qwen 迁移后仅 gguf_q8 单档。策略：
            - gpu+cpu：edit 优先加载到 GPU，t2i 按显存探测（48GB 卡必 CPU 驻留）
            - auto / gpu：单驻（edit 优先），t2i 推理时按需换入
            - cpu：全部 CPU
        """
        if precision not in PRECISIONS:
            raise ValueError(f"非法精度档位：{precision}（Qwen 迁移后仅 {PRECISIONS}）")
        self._swap_to(precision)

    def ensure(self, side: str, precision: Optional[str] = None) -> Any:
        """保证 side 槽位就绪（且在 GPU 上），返回 transformer 模块。

        Args:
            side: "t2i" / "edit"
            precision: 请求档位（None = 用当前档；Qwen 迁移后恒为 gguf_q8）

        这是推理路径的核心入口：
            - 槽位未加载（auto/gpu 单驻）→ 换出另一槽位后加载
              （新模型对象 → 触发 on_swap 回调重挂 LoRA）
            - 槽位 CPU 驻留（gpu+cpu）→ 内存搬换入 GPU，旧 GPU 槽搬出
              （对象身份不变，LoRA 随模块搬移无需重挂）
        """
        if side not in SIDES:
            raise ValueError(f"非法槽位：{side}（{SIDES}）")
        if precision is not None and precision != self._precision and precision not in PRECISIONS:
            raise ValueError(f"非法精度档位：{precision}（Qwen 迁移后仅 {PRECISIONS}）")

        if self._precision is None:
            # 未加载过（auto-load 未跑）：初始化
            self._swap_to(precision or "gguf_q8")

        if side not in self._transformers:
            if self._offload_enabled():
                # gpu+cpu：已有 GPU 槽位先降级 CPU 驻留（保留内存对象），
                # 新槽位加载到 GPU；连单槽都放不下时报错指引
                self._demote_gpu_slots(keep=side)
                try:
                    self._load_side(side, "gguf_q8", placement="gpu")
                except torch.cuda.OutOfMemoryError:
                    self._free_vram_cache()
                    raise self._vram_insufficient_error(side) from None
            else:
                # auto/gpu 单驻：换入目标槽位前换出另一槽位
                if self._transformers:
                    self._evict_others(side)
                self._load_side(side, "gguf_q8")
        elif self._offload_enabled() and self._placement.get(side) == "cpu":
            # gpu+cpu：请求槽位当前 CPU 驻留 → 换入 GPU
            self._swap_active_slot(side)

        return self._transformers[side]

    def unload(self) -> None:
        """全部卸载（释放显存与系统内存）。"""
        freed_sides = list(self._transformers.keys())
        self._release_refs()
        self._transformers.clear()
        self._placement.clear()
        self._precision = None
        self._load_times.clear()
        self._slot_usage_mb = 0.0
        self._gc_and_free_vram()
        logger.info("transformer 全部卸载（%s），显存已释放", freed_sides)

    # ===== 内部实现 =====

    def _swap_to(self, precision: str) -> None:
        """整体初始化：卸载旧状态 → 按策略加载在盘槽位。

        设备策略放置决策：
            - auto / gpu / device=cpu：单驻（edit 优先），另一槽推理时换入
            - gpu+cpu：edit 优先加载 → 显存探测（mem_get_info）：
                * 够双槽 → 双槽 GPU 常驻（≥80GB 卡）
                * 够单槽 → 单槽 GPU 常驻，t2i 槽 CPU 内存驻留
                * 连单槽都不够 → RuntimeError（指引仅 CPU 策略）
        """
        t0 = time.perf_counter()
        self._emit("switching", 0.05, f"卸载旧档（{self._precision}）")
        # 释放顺序（关键，顺序错 = 显存泄漏）：
        #   1) 持有方先释放旧 pipeline 引用（active_pipe 等）
        #   2) 断开本管理器对旧 transformer 的引用（dict.clear）
        #   3) gc.collect()——必须在 clear 之后：diffusers/accelerate 加载的模型
        #      带 hook/PEFT 循环引用，refcount 不归零，clear 前 GC 无效
        #   4) empty_cache()——只回收无引用的缓存块，活 tensor 须先被 GC
        self._release_refs()
        self._transformers.clear()
        self._placement.clear()
        self._precision = None
        self._gc_and_free_vram()

        sides = self.available_sides_for(precision)
        if not sides:
            from .pipeline_builder import _format_missing
            missing = [
                f"{_SIDE_SUBDIR[s]}/{TRANSFORMER_FILES[s][precision]}" for s in SIDES
            ]
            raise RuntimeError(_format_missing(missing, self.model_dir, f"{precision} 档 transformer"))

        offload = self._offload_enabled()
        # edit 优先（edit 槽承担 edit + i2i 两模式，t2i 仅文生图）
        sides = sorted(sides, key=lambda s: 0 if s == "edit" else 1)

        if offload:
            logger.info(
                "gpu+cpu 策略加载 %s 档（探测前剩余显存 %.1fGB）",
                precision, self._vram_free_mb() / 1024,
            )
        else:
            # auto/gpu/cpu：单驻——只加载 edit 槽（在盘时），t2i 按需换入
            sides = sides[:1]
            logger.info("单驻策略：本档仅加载 %s，另一槽位推理时按需换入", sides)

        for i, side in enumerate(sides):
            base = 0.1 + 0.8 * i / len(sides)
            if offload:
                if i == 0:
                    # 首槽位（edit）必须 GPU 驻留（OOM → 指引仅 CPU 策略）
                    try:
                        self._load_side(side, precision, progress_base=base, placement="gpu")
                    except torch.cuda.OutOfMemoryError:
                        self._free_vram_cache()
                        raise self._vram_insufficient_error(side) from None
                else:
                    # 后续槽位：显存探测，够 → GPU，不够 → CPU 内存驻留
                    # （OOM 兜底再降级一次 CPU，防探测与实际分配的偏差）
                    placement = "gpu" if self._fit_another_slot() else "cpu"
                    self._load_side(
                        side, precision, progress_base=base,
                        placement=placement, cpu_fallback_on_oom=True,
                    )
            else:
                self._load_side(side, precision, progress_base=base)

        self._precision = precision
        elapsed = time.perf_counter() - t0
        logger.info(
            "档位加载完成：%s（槽位 %s，放置 %s，耗时 %.1fs，显存 %.1fMB）",
            precision, sides, dict(self._placement), elapsed, self.vram_used_mb(),
        )
        self._emit("loaded", 1.0, f"{precision} 档就绪（{sides}）")
        self._notify_swap()

    def _load_side(
        self,
        side: str,
        precision: str,
        progress_base: float = 0.1,
        placement: str = "gpu",
        cpu_fallback_on_oom: bool = False,
    ) -> None:
        """加载单个槽位。

        Args:
            placement: 目标放置（"gpu" / "cpu"；device=cpu 时一律 CPU）
            cpu_fallback_on_oom: GPU 放置 OOM 时降级 CPU 驻留（gpu+cpu 后续
                槽位用）；False 时 OOM 原样上抛（首槽位 / auto / gpu 策略）
        """
        self._emit("loading_transformer", progress_base, f"加载 {side}（{precision}）")
        t0 = time.perf_counter()
        transformer = load_transformer(self.model_dir, side, precision, self.dtype)
        target = self.device if (placement == "gpu" and self.device != "cpu") else "cpu"
        alloc_before = self._cuda_alloc_mb() if target != "cpu" else 0.0
        try:
            transformer = transformer.to(target)
        except torch.cuda.OutOfMemoryError:
            if cpu_fallback_on_oom:
                logger.warning(
                    "gpu+cpu：%s（%s）载入 GPU 显存不足（探测偏差），降级 CPU 内存驻留",
                    side, precision,
                )
                transformer = transformer.to("cpu")
                target = "cpu"
            else:
                raise
        if target != "cpu":
            # 实测本槽位显存占用（下一槽位放置探测的估算基准）
            self._slot_usage_mb = max(self._cuda_alloc_mb() - alloc_before, 0.0)
        self._transformers[side] = transformer
        self._placement[side] = "gpu" if target != "cpu" else "cpu"
        self._load_times[side] = time.perf_counter() - t0
        logger.info(
            "%s transformer（%s）加载完成：%.1fs，放置=%s，显存 %.1fMB",
            side, precision, self._load_times[side],
            self._placement[side], self.vram_used_mb(),
        )

    def _swap_active_slot(self, side: str) -> None:
        """gpu+cpu：side 槽位从 CPU 换入 GPU，旧 GPU 槽位换出至 CPU。

        纯内存搬移（不重读盘）：Q8 单槽 ~21.8GB，PCIe 搬移十秒级，
        远快于重读盘加载（40s+）。LoRA 适配器随模块搬移（对象身份不变，
        InferenceRunner 按 id 检测不会误判为切换，无需重挂）。
        """
        t0 = time.perf_counter()
        # 持有方先释放旧 pipeline 引用（active_pipe 引用旧 GPU 槽位）
        self._release_refs()
        # 1) 旧 GPU 槽位 → CPU（换出）
        for other, tf in self._transformers.items():
            if other != side and self._placement.get(other) == "gpu":
                logger.info("gpu+cpu 换出：%s 槽位 GPU → CPU（换入 %s）", other, side)
                tf.to("cpu")
                self._placement[other] = "cpu"
        self._free_vram_cache()
        # 2) 请求槽位 CPU → GPU（换入）
        try:
            logger.info("gpu+cpu 换入：%s 槽位 CPU → %s", side, self.device)
            self._transformers[side].to(self.device)
        except torch.cuda.OutOfMemoryError:
            free_gb = self._vram_free_mb() / 1024
            raise RuntimeError(
                f"gpu+cpu 换入 {side} 槽位时显存不足（剩余 {free_gb:.1f}GB）。"
                f"可能原因：其他进程在运行中占用了显存。"
                f"建议停止占用显存的引擎后重试，或改为「仅 CPU」策略。"
            ) from None
        self._placement[side] = "gpu"
        elapsed = time.perf_counter() - t0
        logger.info(
            "gpu+cpu 槽位换入换出完成（%.1fs，显存 %.1fMB，放置 %s）",
            elapsed, self.vram_used_mb(), dict(self._placement),
        )

    def _demote_gpu_slots(self, keep: Optional[str] = None) -> None:
        """gpu+cpu：把已加载的 GPU 槽位移到 CPU 内存驻留（保留对象，不重读盘）。

        keep 指定的槽位不动（通常是即将加载到 GPU 的新槽位）。
        """
        if not self._offload_enabled():
            return
        gpu_sides = [s for s, p in self._placement.items() if p == "gpu" and s != keep]
        if not gpu_sides:
            return
        self._release_refs()
        for side in gpu_sides:
            logger.info("gpu+cpu 降级：%s 槽位 GPU → CPU（驻留内存）", side)
            self._transformers[side].to("cpu")
            self._placement[side] = "cpu"
        self._free_vram_cache()

    def _evict_others(self, keep_side: str) -> None:
        """换出除 keep_side 外的全部槽位（auto/gpu 单驻用）。"""
        evict = [s for s in self._transformers if s != keep_side]
        if not evict:
            return
        # 换出前让持有方释放旧 pipeline 引用（active_pipe 可能引用被换出的 transformer）
        self._release_refs()
        for side in evict:
            logger.info("单驻策略：换出 %s 槽位（换入 %s）", side, keep_side)
            del self._transformers[side]
            self._placement.pop(side, None)
        self._gc_and_free_vram()

    def _release_refs(self) -> None:
        """换档/换出/卸载前，让持有方释放旧 transformer 的外部引用。

        _transformers.clear() 只断开本管理器的引用；InferenceRunner 的
        active_pipe（含挂载的 PEFT adapter 层）仍引用旧档 transformer 时，
        CUDA 张量不释放 → empty_cache 无效 → 新旧两档叠加。

        注意：此处只做回调，不做 GC——GC 必须在引用断开（clear/del）之后
        由 _gc_and_free_vram 执行（accelerate hook 循环引用依赖 GC 回收）。
        """
        if self._on_release is not None:
            try:
                self._on_release()
            except Exception:  # noqa: BLE001
                logger.debug("on_release 回调失败", exc_info=True)

    def _gc_and_free_vram(self) -> None:
        """引用断开后回收显存：gc（循环引用）→ empty_cache（缓存块）。"""
        import gc
        gc.collect()
        self._free_vram_cache()

    def _free_vram_cache(self) -> None:
        if torch.cuda.is_available() and self.device != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _notify_swap(self) -> None:
        """通知 engine：transformer 已切换，需重挂 LoRA。"""
        if self._on_swap is not None:
            try:
                self._on_swap()
            except Exception:  # noqa: BLE001
                logger.exception("on_swap 回调失败（LoRA 重挂）")

    def _emit(self, stage: str, progress: float, detail: str) -> None:
        """转发进度回调。"""
        if self._progress_cb is not None:
            try:
                self._progress_cb(stage, progress, detail)
            except Exception:  # noqa: BLE001
                logger.debug("进度回调失败", exc_info=True)

    # ===== 状态快照（/status extra 用）=====

    def status_dict(self) -> dict:
        return {
            "precision": self._precision,
            "loaded_sides": self.loaded_sides,
            # 设备策略：各槽实际放置（gpu / cpu）+ 策略值
            "placement": dict(self._placement),
            "device_policy": self.device_policy,
            "vram_used_mb": round(self.vram_used_mb(), 1),
            "vram_free_mb": round(self._vram_free_mb(), 1),
            "load_times": {k: round(v, 1) for k, v in self._load_times.items()},
        }
