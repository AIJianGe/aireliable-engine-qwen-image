"""Qwen-Image 管线组件组装器（单文件权重 → diffusers 组件）。

职责（engine load() 的底层实现）：
    1. 文件完整性校验（exists + size ≥ 阈值；缺失抛 ModelFilesError 指引下载，
       禁止在 load 路径中做任何网络下载——项目规则第二部分 §2）
    2. 共享组件加载：Qwen2.5-VL-7B（文本+视觉编码器）/ QwenImage VAE /
       Qwen2Tokenizer / Qwen2VLProcessor / scheduler 配置
       （t2i 与 edit 完全共享，整个引擎生命周期只加载一份）
    3. transformer GGUF Q8 加载（QwenImageTransformer2DModel.from_single_file），
       由 transformer_manager.py 调用

迁移说明（FLUX Kontext → Qwen-Image，2026-08）：
    - t2i 槽：Qwen-Image-2512（QwenImagePipeline）
    - edit 槽：Qwen-Image-Edit-2509（QwenImageEditPlusPipeline，原生多图参考）
    - 仅 Q8 单档（D3 决策）；VL 默认 GPU、显存紧张降 CPU（D4 决策）

目录布局（与 manifest.yaml model_files.target_dir 一致）：
    models/image/qwen-image/
      ├── transformers/
      │   ├── t2i/qwen-image-2512-Q8_0.gguf          # t2i 槽（~21.8GB）
      │   └── edit/qwen-image-edit-2509-Q8_0.gguf    # edit 槽（~21.8GB）
      ├── text_encoder/                               # Qwen2.5-VL-7B bf16 4 shards
      │   ├── config.json / generation_config.json / model.safetensors.index.json
      │   └── model-0000{1..4}-of-00004.safetensors   # 共 16.6GB
      ├── vae/
      │   ├── config.json                             # AutoencoderKLQwenImage
      │   └── diffusion_pytorch_model.safetensors     # ~254MB
      ├── tokenizers/                                 # Qwen2Tokenizer（t2i 管线用）
      ├── processor/                                  # Qwen2VLProcessor（EditPlus 管线用）
      ├── scheduler/scheduler_config.json
      └── loras/*.safetensors
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# ===== 文件布局常量（相对 model_dir）=====
TRANSFORMER_T2I_SUBDIR = "transformers/t2i"
TRANSFORMER_EDIT_SUBDIR = "transformers/edit"
TEXT_ENCODER_SUBDIR = "text_encoder"
VAE_SUBDIR = "vae"
TOKENIZER_SUBDIR = "tokenizers"
PROCESSOR_SUBDIR = "processor"
SCHEDULER_SUBDIR = "scheduler"
LORA_SUBDIR = "loras"

# ===== 引擎内置 diffusers config（from_single_file 离线加载的关键）=====
# 背景：diffusers from_single_file 不传 config 时会 fetch_diffusers_config
# 推断 repo 并从 HF 拉取 config.json（离线 OSError）。内置 config 目录资产后
# from_single_file(config=<目录>) 走纯本地读取（FLUX 时代已实测离线可用）。
# 注：Edit-2509 与 2512 的 transformer config.json SHA256 完全一致 → 单份共享。
# VAE 走 from_pretrained（标准 diffusers 目录格式，无需内置 config）。
ENGINE_CONFIGS_DIR = Path(__file__).parent / "configs"
QWEN_TRANSFORMER_CONFIG_DIR = ENGINE_CONFIGS_DIR / "qwen_transformer"

# ===== transformer 文件名（t2i=Qwen-Image-2512 / edit=Qwen-Image-Edit-2509）=====
TRANSFORMER_FILES: dict[str, dict[str, str]] = {
    "t2i": {"gguf_q8": "qwen-image-2512-Q8_0.gguf"},
    "edit": {"gguf_q8": "qwen-image-edit-2509-Q8_0.gguf"},
}

# ===== 最小大小阈值（字节，防残文件；约为实际大小的 95%）=====
MIN_SIZES: dict[str, int] = {
    # 21,761,817,120 B（20.27GiB）→ 20GiB
    "qwen-image-2512-Q8_0.gguf": 20 * 1024**3,
    "qwen-image-edit-2509-Q8_0.gguf": 20 * 1024**3,
    # VL-7B bf16 shards（4.63/4.65/4.60/1.58GiB 实际）
    "model-00001-of-00004.safetensors": int(4.5 * 1024**3),
    "model-00002-of-00004.safetensors": int(4.5 * 1024**3),
    "model-00003-of-00004.safetensors": int(4.5 * 1024**3),
    "model-00004-of-00004.safetensors": int(1.5 * 1024**3),
    # VAE 253,806,966 B（242MiB）
    "diffusion_pytorch_model.safetensors": 240 * 1024**2,
}

# 共享组件文件清单（相对路径 → MIN_SIZES 键；空串 = 存在性检查即可）
SHARED_FILES: list[tuple[str, str]] = [
    # Qwen2.5-VL-7B 文本/视觉编码器（bf16 4 shards + 配置）
    (f"{TEXT_ENCODER_SUBDIR}/config.json", ""),
    (f"{TEXT_ENCODER_SUBDIR}/generation_config.json", ""),
    (f"{TEXT_ENCODER_SUBDIR}/model.safetensors.index.json", ""),
    (f"{TEXT_ENCODER_SUBDIR}/model-00001-of-00004.safetensors", "model-00001-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00002-of-00004.safetensors", "model-00002-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00003-of-00004.safetensors", "model-00003-of-00004.safetensors"),
    (f"{TEXT_ENCODER_SUBDIR}/model-00004-of-00004.safetensors", "model-00004-of-00004.safetensors"),
    # VAE（AutoencoderKLQwenImage）
    (f"{VAE_SUBDIR}/config.json", ""),
    (f"{VAE_SUBDIR}/diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors"),
    # Qwen2Tokenizer（t2i 管线）
    (f"{TOKENIZER_SUBDIR}/vocab.json", ""),
    (f"{TOKENIZER_SUBDIR}/merges.txt", ""),
    (f"{TOKENIZER_SUBDIR}/tokenizer_config.json", ""),
    (f"{TOKENIZER_SUBDIR}/special_tokens_map.json", ""),
    (f"{TOKENIZER_SUBDIR}/added_tokens.json", ""),
    (f"{TOKENIZER_SUBDIR}/chat_template.jinja", ""),
    # Qwen2VLProcessor（EditPlus 管线；model_index 声明的 processor 组件）
    (f"{PROCESSOR_SUBDIR}/preprocessor_config.json", ""),
    (f"{PROCESSOR_SUBDIR}/video_preprocessor_config.json", ""),
    (f"{PROCESSOR_SUBDIR}/tokenizer.json", ""),
    (f"{PROCESSOR_SUBDIR}/chat_template.jinja", ""),
    (f"{PROCESSOR_SUBDIR}/added_tokens.json", ""),
    (f"{PROCESSOR_SUBDIR}/merges.txt", ""),
    (f"{PROCESSOR_SUBDIR}/special_tokens_map.json", ""),
    (f"{PROCESSOR_SUBDIR}/tokenizer_config.json", ""),
    # scheduler（Qwen 专用 shift 参数：base 0.5 / max 0.9 / terminal 0.02）
    (f"{SCHEDULER_SUBDIR}/scheduler_config.json", ""),
]

# 模型仓库 ID（错误信息中指引下载用；ModelScope 主源 + hf-mirror 兜底）
HF_REPOS: dict[str, str] = {
    "t2i": "unsloth/Qwen-Image-2512-GGUF（GGUF）/ Qwen/Qwen-Image-2512（官方）",
    "edit": "unsloth/Qwen-Image-Edit-2509-GGUF（GGUF）/ Qwen/Qwen-Image-Edit-2509（官方）",
    "shared": (
        "Qwen/Qwen-Image-2512（text_encoder / vae / tokenizer / scheduler）"
        "+ Qwen/Qwen-Image-Edit-2509（processor）"
    ),
}

# VL（Qwen2.5-VL-7B）fp16 显存需求估算（bf16 16.6GB → fp16 同量级）
VL_VRAM_NEED_GB = 17.0


class ModelFilesError(RuntimeError):
    """模型文件缺失/不完整（load() 校验失败抛出，信息指引模型管理界面下载）。"""


# ============================================================
# 文件完整性校验（项目规则：exists + size ≥ 阈值，禁止 load 内下载）
# ============================================================

def check_shared_files(model_dir: Path) -> None:
    """校验共享组件文件完整性（缺失抛 ModelFilesError）。"""
    missing: list[str] = []
    for rel, min_key in SHARED_FILES:
        path = model_dir / rel
        min_size = MIN_SIZES.get(min_key, 1)
        if not path.exists():
            missing.append(rel)
        elif min_size > 1 and path.stat().st_size < min_size:
            missing.append(f"{rel}（大小不足：{path.stat().st_size / 1024**3:.2f}GB）")
    if missing:
        raise ModelFilesError(
            _format_missing(
                missing, model_dir,
                "共享组件（VL 编码器 / VAE / tokenizer / processor / scheduler）",
            )
        )


def check_transformer_file(model_dir: Path, side: str, precision: str) -> Path:
    """校验指定档位 transformer 文件并返回路径。

    Args:
        side: "t2i" / "edit"
        precision: "gguf_q8"（当前唯一档）
    """
    if side not in TRANSFORMER_FILES:
        raise ValueError(f"非法槽位：{side}（{list(TRANSFORMER_FILES)}）")
    try:
        filename = TRANSFORMER_FILES[side][precision]
    except KeyError as e:
        raise ValueError(f"非法精度档位：{precision}（Qwen 迁移后仅 gguf_q8 单档）") from e
    subdir = TRANSFORMER_T2I_SUBDIR if side == "t2i" else TRANSFORMER_EDIT_SUBDIR
    path = model_dir / subdir / filename
    min_size = MIN_SIZES.get(filename, 1)
    if not path.exists() or path.stat().st_size < min_size:
        size_note = ""
        if path.exists():
            size_note = f"（大小不足：{path.stat().st_size / 1024**3:.2f}GB < 阈值）"
        raise ModelFilesError(
            _format_missing(
                [f"{subdir}/{filename}{size_note}"],
                model_dir, f"{side} transformer（{precision} 档）", side=side,
            )
        )
    return path


def _format_missing(missing: list[str], model_dir: Path, group: str, side: Optional[str] = None) -> str:
    """构造带下载指引的缺失错误信息。"""
    repo_hint = HF_REPOS.get(side or "", "") or HF_REPOS["shared"]
    return (
        f"模型文件不完整（{group}），缺失 {len(missing)} 个文件：\n"
        f"  {missing}\n\n"
        f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，选择镜像源后下载。\n"
        f"本地路径：{model_dir}\n"
        f"模型仓库：{repo_hint}"
    )


# ============================================================
# 共享组件加载（单份常驻，t2i 与 edit 共享）
# ============================================================

@dataclass
class SharedComponents:
    """已加载的共享组件集合（整个引擎生命周期一份）。"""

    vae: Any = None                    # AutoencoderKLQwenImage（GPU）
    text_encoder: Any = None           # Qwen2_5_VLForConditionalGeneration（GPU 或 CPU，决策 D4）
    tokenizer: Any = None              # Qwen2Tokenizer（t2i 管线用）
    processor: Any = None              # Qwen2VLProcessor（EditPlus 管线用）
    scheduler_config: dict = field(default_factory=dict)
    vl_device: str = "cpu"             # VL 实际放置（"cuda" / "cpu"）

    @property
    def is_loaded(self) -> bool:
        return self.vae is not None and self.text_encoder is not None

    @property
    def vl_on_gpu(self) -> bool:
        return self.vl_device.startswith("cuda")


def resolve_compute_dtype(device: str) -> torch.dtype:
    """解析计算 dtype。

    CUDA 下用 float32（2026-08-25 黑图根因实证）：diffusers from_single_file
    的 GGUF 路径在无 kernels ggml CUDA ops 时走 GGUFLinear.forward_native
    反量化，Q8 block scale × int8 的中间积在 fp16 下溢出 → FlowMatch 第 1
    步 noise_pred 全 NaN → latents NaN → VAE decode clamp 全 0 → 黑图
    （ComfyUI 的 GGUF 动态反量化数值行为不同，社区 fp16 实证不可迁移）。
    fp32 数值范围根治；Turing fp32 原生支持（实测 13.65s/步 @1328²）。
    bf16 数值虽安全但 Turing 仿真慢 ~10×（2 步 ~5min），不可用。
    """
    if device.startswith("cuda"):
        return torch.float32
    return torch.float32


def decide_vl_device(device: str) -> str:
    """决策 VL（Qwen2.5-VL-7B）放置（决策 D4：默认 GPU，显存紧张降 CPU）。

    依据：load 时刻 torch.cuda.mem_get_info 实测剩余显存。
    单槽 Q8（~21GB，compute=fp32 不改变权重显存——GGUF 权重保持量化态）+
    VL fp16（16.6GB）+ VAE fp32（0.5GB）≈ 38GB < 48GB
    → 剩余 ≥ 19GB 时 VL 上 GPU（fp16 编码快且实证干净）；否则 VL 驻 CPU
    （fp16 内存驻留；_wrap_vl_embeds_dtype 会把编码输入重定向到 VL 所在
    设备，故 CPU 驻留时管线不再有设备不匹配问题）。
    """
    if not device.startswith("cuda"):
        return "cpu"
    try:
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
    except Exception:  # noqa: BLE001
        return "cpu"
    if free_gb >= VL_VRAM_NEED_GB + 2.0:
        logger.info("VL 放置决策：GPU（剩余显存 %.1fGB ≥ %.1fGB）", free_gb, VL_VRAM_NEED_GB + 2.0)
        return device
    logger.warning(
        "VL 放置决策：CPU 内存驻留（剩余显存 %.1fGB < %.1fGB，编码每次 +几秒）",
        free_gb, VL_VRAM_NEED_GB + 2.0,
    )
    return "cpu"


def load_shared_components(model_dir: Path, device: str, dtype: torch.dtype) -> SharedComponents:
    """加载全部共享组件（load() 时调用一次，之后常驻）。

    Args:
        device: 推理设备（"cuda" / "cpu"，transformer/VAE 的目标设备）
        dtype: 计算 dtype（fp16/fp32）
    """
    check_shared_files(model_dir)

    comps = SharedComponents()

    # ---- VAE（GPU 常驻，~0.3GB）----
    t0 = __import__("time").perf_counter()
    comps.vae = _load_vae(model_dir / VAE_SUBDIR, device, dtype)
    logger.info("QwenImage VAE 加载完成（%.1fs，%s）",
                __import__("time").perf_counter() - t0, device)

    # ---- Qwen2.5-VL-7B（决策 D4：默认 GPU，显存紧张降 CPU）----
    comps.vl_device = decide_vl_device(device)
    t0 = __import__("time").perf_counter()
    comps.text_encoder = _load_vl_encoder(model_dir / TEXT_ENCODER_SUBDIR, comps.vl_device)
    logger.info(
        "Qwen2.5-VL-7B 编码器加载完成（%.1fs，放置=%s，fp16）",
        __import__("time").perf_counter() - t0, comps.vl_device,
    )

    # ---- tokenizer（t2i 管线）/ processor（EditPlus 管线）----
    comps.tokenizer = _load_tokenizer(model_dir / TOKENIZER_SUBDIR)
    comps.processor = _load_processor(model_dir / PROCESSOR_SUBDIR)

    # ---- scheduler 配置 ----
    cfg_path = model_dir / SCHEDULER_SUBDIR / "scheduler_config.json"
    comps.scheduler_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    logger.info("tokenizer / processor / scheduler 配置加载完成")
    return comps


def make_scheduler(comps: SharedComponents) -> Any:
    """用已加载的 scheduler_config 构造 FlowMatchEulerDiscreteScheduler。"""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_config(comps.scheduler_config)


# ============================================================
# 单组件加载
# ============================================================

def _load_vae(vae_dir: Path, device: str, dtype: torch.dtype):
    """VAE → AutoencoderKLQwenImage（标准 diffusers 目录格式 from_pretrained）。

    AutoencoderKLQwenImage 不在 diffusers SINGLE_FILE_LOADABLE_CLASSES 中，
    走标准目录加载（config.json + diffusion_pytorch_model.safetensors），
    latents_mean/std 由 config 自动应用。
    """
    from diffusers import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(str(vae_dir), torch_dtype=dtype)
    return vae.to(device).eval()


def _load_vl_encoder(te_dir: Path, vl_device: str):
    """Qwen2.5-VL-7B → Qwen2_5_VLForConditionalGeneration（bf16 权重 → fp16）。

    bf16 权重 cast fp16：Qwen2.5-VL fp16 编码实证干净（_diag step1，
    nan=0 inf=0，幅度 ±139），且 fp16 显存 16.6GB 与 fp32 transformer
    （~21GB）可共存于 48GB 卡。embeds → fp32 的桥接由
    _wrap_vl_embeds_dtype 在管线组装时完成。
    """
    from transformers import Qwen2_5_VLForConditionalGeneration
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(te_dir),
        torch_dtype=torch.float16,
    )
    return te.to(vl_device).eval()


def _load_tokenizer(tok_dir: Path):
    """Qwen2Tokenizer（t2i 管线 encode_prompt 用）。"""
    from transformers import Qwen2Tokenizer
    return Qwen2Tokenizer.from_pretrained(str(tok_dir))


def _load_processor(proc_dir: Path):
    """Qwen2VLProcessor（EditPlus 管线：文本 + 视觉 token 联合处理）。"""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(proc_dir))


# ============================================================
# transformer GGUF 加载路由（transformer_manager 调用）
# ============================================================

def load_transformer(model_dir: Path, side: str, precision: str, dtype: torch.dtype):
    """按档位加载 transformer（GGUF 单文件路线，仅 gguf_q8 档）。

    Args:
        side: "t2i"（Qwen-Image-2512）/ "edit"（Qwen-Image-Edit-2509）
        precision: "gguf_q8"
        dtype: 计算 dtype（GGUF compute_dtype，fp32——fp16 实证 NaN 黑图）
    """
    path = check_transformer_file(model_dir, side, precision)
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
    logger.info("加载 GGUF Q8 transformer：%s（compute_dtype=%s）", path, dtype)
    return QwenImageTransformer2DModel.from_single_file(
        str(path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
        config=str(QWEN_TRANSFORMER_CONFIG_DIR),
    )


# ============================================================
# 管线组装（inference 按需调用，共享组件注入）
# ============================================================
# 注意：组装后【不调用 pipe.to(device)】——组件异构放置（VL 可能驻 CPU，
# transformer 由 TransformerManager 管理槽位），各组件在加载时已就位。

def _wrap_vl_embeds_dtype(pipe: Any, vl_device: str, out_dtype: torch.dtype) -> None:
    """包装管线 _get_qwen_prompt_embeds：VL 编码输出统一 cast 到 out_dtype。

    黑图修复（compute_dtype=fp32）的配套桥接，缺一不可：
    1. latents dtype 跟随 prompt_embeds.dtype（= VL dtype）：VL 是 fp16，
       不 cast 则 latents 为 fp16；而 GGUF forward_native 不做输入 dtype
       cast（本环境无 kernels ggml CUDA ops），fp16 输入 × fp32 权重直接
       RuntimeError（mat1/mat2 dtype 不一致）→ embeds cast fp32 使 latents
       与 transformer 输入全 fp32
    2. VL 驻 CPU 的降级场景：管线默认把 tokenizer 输出搬到 execution_device
       （GPU），VL 在 CPU 时设备不匹配；此处把编码输入重定向到 VL 所在
       设备，输出再搬回执行设备
    t2i 与 EditPlus 两管线的 _get_qwen_prompt_embeds 签名一致地被包装
    （prompt / image? / device / dtype，encode_prompt 内部调用）。
    """
    import inspect

    orig = pipe._get_qwen_prompt_embeds
    vl_dev = torch.device(vl_device)

    def wrapped(*args, **kwargs):
        bound = inspect.signature(orig).bind(*args, **kwargs)
        bound.arguments["device"] = vl_dev          # 编码在 VL 所在设备
        bound.arguments.pop("dtype", None)          # 用 VL 自身 dtype 编码
        embeds, mask = orig(*bound.args, **bound.kwargs)
        exec_dev = pipe._execution_device
        return embeds.to(out_dtype).to(exec_dev), mask.to(exec_dev)

    pipe._get_qwen_prompt_embeds = wrapped
    logger.info(
        "VL embeds dtype 桥接已启用（VL=%s/%s → embeds=%s，latents 跟随）",
        vl_device, torch.float16, out_dtype,
    )


def build_t2i_pipeline(comps: SharedComponents, transformer, device: str):
    """组装文生图管线（QwenImagePipeline + Qwen-Image-2512 transformer）。"""
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline(
        scheduler=make_scheduler(comps),
        vae=comps.vae,
        text_encoder=comps.text_encoder,
        tokenizer=comps.tokenizer,
        transformer=transformer,
    )
    # 禁用 tqdm 进度条：引擎以平台子进程运行（stderr 为管道），Windows 下
    # tqdm 向非控制台句柄写入+flush 抛 OSError [Errno 22]；进度已由
    # callback_on_step_end → SSE 帧 → WebUI/任务中心上报，tqdm 冗余。
    pipe.set_progress_bar_config(disable=True)
    # VL fp16 embeds → fp32 桥接（黑图修复配套，见 _wrap_vl_embeds_dtype）
    _wrap_vl_embeds_dtype(pipe, comps.vl_device, resolve_compute_dtype(device))
    return pipe


def build_edit_pipeline(comps: SharedComponents, transformer, device: str):
    """组装多图编辑管线（QwenImageEditPlusPipeline + Edit-2509 transformer）。

    edit（单图指令编辑）与 i2i（2-3 图多图参考）共用此管线（按输入图
    数量区分，无槽位切换开销）。
    """
    from diffusers import QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline(
        scheduler=make_scheduler(comps),
        vae=comps.vae,
        text_encoder=comps.text_encoder,
        tokenizer=comps.tokenizer,
        processor=comps.processor,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    # VL fp16 embeds → fp32 桥接（黑图修复配套，见 _wrap_vl_embeds_dtype）
    _wrap_vl_embeds_dtype(pipe, comps.vl_device, resolve_compute_dtype(device))
    return pipe
