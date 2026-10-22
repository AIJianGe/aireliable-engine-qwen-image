"""P0 验证脚本共享助手（阶段 2 临时脚本，不参与引擎运行时）。

职责：
1. 模型文件下载（ModelScope 优先 + hf-mirror 兜底，断点续传 + 大小校验）
2. 组件组装（ComfyUI 单文件 → diffusers 管线组件）
3. 输出布局与 manifest 计划一致（models/image/flux-kontext/...），下载不浪费

下载源策略（2026-08-22 实测）：
- BFL 官方仓库（black-forest-labs/*）在 huggingface.co/hf-mirror 均为 gated（403），
  但 ModelScope 有官方同步镜像且匿名可下载 → ae / tokenizer / scheduler / BF16 全走 ModelScope
- QuantStack / city96 / Comfy-Org / comfyanonymous 仓库 hf-mirror 公开可用，
  ModelScope 也同步了 → 统一 ModelScope 优先（国内速度快），hf-mirror 兜底

用法（各验证脚本 import 本模块）：
    from _verify_common import ensure_models, assemble_kontext_pipeline, ...
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ⚠️ HF_ENDPOINT 必须在 import huggingface_hub 之前设置（ENDPOINT 在 import 时缓存）
# 项目规则：镜像优先，避免直连 huggingface.co 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

# ---------------------------------------------------------------------------
# 路径布局（与 manifest 计划一致，见 ANALYSIS §4.1）
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"
FLUX_DIR = MODELS_ROOT / "image" / "flux-kontext"

TRANSFORMER_KONTEXT_DIR = FLUX_DIR / "transformers" / "kontext"
TRANSFORMER_T2I_DIR = FLUX_DIR / "transformers" / "t2i"
TEXT_ENCODER_DIR = FLUX_DIR / "text_encoders"
VAE_DIR = FLUX_DIR / "vae"
LORA_DIR = FLUX_DIR / "loras"

OUTPUT_DIR = Path(os.environ.get("AIRELIABLE_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "flux_kontext_image"))

# ---------------------------------------------------------------------------
# 待下载文件清单
# 条目结构：(source, repo_id, filename, 本地目标路径, 最小大小字节)
#   - source="modelscope"：ModelScope 官方/社区镜像（匿名可下载，国内快）
#   - source="hf"：hf-mirror（huggingface_hub，断点续传由其内部处理）
# 最小大小用于二次校验（防残文件），比实际略小即可
# ---------------------------------------------------------------------------

MIN_MB = 1024 * 1024

FILES = {
    # kontext GGUF Q8（T2.1 主验证对象，~12.6GB）
    # （仓库已核验 2026-08-22：QuantStack 提供 Q2~Q8 全档；ModelScope 有同步）
    "kontext_q8": (
        "modelscope",
        "QuantStack/FLUX.1-Kontext-dev-GGUF",
        "flux1-kontext-dev-Q8_0.gguf",
        TRANSFORMER_KONTEXT_DIR / "flux1-kontext-dev-Q8_0.gguf",
        12_000 * MIN_MB,
    ),
    # dev GGUF Q8（T2.2/T2.4 t2i 侧，~12.6GB）
    "dev_q8": (
        "modelscope",
        "city96/FLUX.1-dev-gguf",
        "flux1-dev-Q8_0.gguf",
        TRANSFORMER_T2I_DIR / "flux1-dev-Q8_0.gguf",
        12_000 * MIN_MB,
    ),
    # kontext FP8 scaled（T2.2，~11.9GB）
    # （文件在 split_files/diffusion_models/ 子路径下）
    "kontext_fp8": (
        "modelscope",
        "Comfy-Org/flux1-kontext-dev_ComfyUI",
        "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        TRANSFORMER_KONTEXT_DIR / "flux1-dev-kontext_fp8_scaled.safetensors",
        11_000 * MIN_MB,
    ),
    # 共享组件（~5.5GB）
    "clip_l": (
        "modelscope",
        "comfyanonymous/flux_text_encoders",
        "clip_l.safetensors",
        TEXT_ENCODER_DIR / "clip_l.safetensors",
        200 * MIN_MB,
    ),
    "t5xxl_fp8": (
        "modelscope",
        "comfyanonymous/flux_text_encoders",
        "t5xxl_fp8_e4m3fn_scaled.safetensors",
        TEXT_ENCODER_DIR / "t5xxl_fp8_e4m3fn_scaled.safetensors",
        4_800 * MIN_MB,
    ),
    # ae 直接取 Kontext 官方仓库单文件（ModelScope 镜像匿名可下；HF 侧 gated）
    "ae": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "ae.safetensors",
        VAE_DIR / "ae.safetensors",
        300 * MIN_MB,
    ),
    # tokenizer / scheduler 小文件（FLUX.1-Kontext-dev 官方 diffusers repo 的
    # ModelScope 镜像，保证与 Kontext 训练一致）
    "tok_clip_vocab": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer/vocab.json",
        FLUX_DIR / "tokenizers" / "clip" / "vocab.json",
        0,
    ),
    "tok_clip_merges": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer/merges.txt",
        FLUX_DIR / "tokenizers" / "clip" / "merges.txt",
        0,
    ),
    "tok_clip_config": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer/tokenizer_config.json",
        FLUX_DIR / "tokenizers" / "clip" / "tokenizer_config.json",
        0,
    ),
    "tok_clip_special": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer/special_tokens_map.json",
        FLUX_DIR / "tokenizers" / "clip" / "special_tokens_map.json",
        0,
    ),
    "tok_t5_spiece": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer_2/spiece.model",
        FLUX_DIR / "tokenizers" / "t5" / "spiece.model",
        0,
    ),
    "tok_t5_config": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer_2/tokenizer_config.json",
        FLUX_DIR / "tokenizers" / "t5" / "tokenizer_config.json",
        0,
    ),
    "tok_t5_special": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "tokenizer_2/special_tokens_map.json",
        FLUX_DIR / "tokenizers" / "t5" / "special_tokens_map.json",
        0,
    ),
    "scheduler_config": (
        "modelscope",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "scheduler/scheduler_config.json",
        FLUX_DIR / "tokenizers" / "scheduler_config.json",
        0,
    ),
}


# ---------------------------------------------------------------------------
# 下载与校验
# ---------------------------------------------------------------------------

def _fmt_gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f}GB"


MS_BASE = "https://modelscope.cn/models"


def _ms_download(repo_id: str, filename: str, target: Path, min_size: int) -> Path:
    """ModelScope 下载（httpx 流式 + Range 断点续传 + 大小二次校验）。

    ModelScope 匿名直链：{MS_BASE}/{repo}/resolve/master/{file}
    """
    import httpx

    url = f"{MS_BASE}/{repo_id}/resolve/master/{filename}"
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 断点续传：从 .part 现有大小继续
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}

    with httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if start > 0 and resp.status_code != 206:
                # 服务端不支持续传（返回 200 全量）→ 重下
                start = 0
            total = int(resp.headers.get("content-length", 0)) + start
            done = start
            last_log = 0.0
            mode = "ab" if start > 0 and resp.status_code == 206 else "wb"
            with open(part, mode) as fh:
                for chunk in resp.iter_bytes(chunk_size=4 * MIN_MB):
                    fh.write(chunk)
                    done += len(chunk)
                    if time.time() - last_log > 30:
                        pct = f"{done / total * 100:.1f}%" if total else "?"
                        speed = (done - start) / max(time.time() - t0, 1) / MIN_MB
                        print(f"  ... {done / MIN_MB / 1024:.2f}GB / {total / MIN_MB / 1024:.2f}GB"
                              f"（{pct}，{speed:.1f}MB/s）", flush=True)
                        last_log = time.time()

    part.replace(target)
    size = target.stat().st_size
    if size < min_size:
        raise RuntimeError(f"残文件：{target} 仅 {_fmt_gb(size)}（阈值 {_fmt_gb(min_size)}）")
    print(f"[done] {target.name} {_fmt_gb(size)}，耗时 {time.time() - t0:.0f}s")
    return target


def _hf_download(repo_id: str, filename: str, target: Path, min_size: int) -> Path:
    """hf-mirror 下载（huggingface_hub，走 HF_ENDPOINT）。"""
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target.parent))
    # local_dir 模式：filename 含子目录时落在 target.parent / <子路径>，需归位
    if not target.exists():
        alt = target.parent / Path(filename)
        if alt.exists():
            alt.replace(target)
    if not target.exists():
        raise RuntimeError(f"下载失败：{target} 不存在")
    size = target.stat().st_size
    if size < min_size:
        raise RuntimeError(f"残文件：{target} 仅 {_fmt_gb(size)}（阈值 {_fmt_gb(min_size)}）")
    return target


def ensure_file(key: str, force: bool = False) -> Path:
    """下载（或复用）单个文件并做大小二次校验。

    规则（项目规范：不能信任外部调用返回值）：
    - 已存在且 size ≥ min_size → 直接复用
    - ModelScope 优先，异常时回退 hf-mirror
    """
    source, repo_id, filename, target, min_size = FILES[key]
    if target.exists() and not force:
        size = target.stat().st_size
        if size >= min_size:
            print(f"[skip] {target.name} 已存在（{_fmt_gb(size)}）")
            return target
        print(f"[warn] {target.name} 大小不足（{_fmt_gb(size)} < 阈值），重新下载")

    print(f"[download:{source}] {repo_id}/{filename} -> {target}", flush=True)
    if source == "modelscope":
        try:
            return _ms_download(repo_id, filename, target, min_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[fallback] ModelScope 失败（{exc!r}），转 hf-mirror")
            return _hf_download(repo_id, filename, target, min_size)
    return _hf_download(repo_id, filename, target, min_size)


def ensure_models(keys: list[str]) -> dict[str, Path]:
    """批量确保文件就绪，返回 {key: path}。"""
    paths: dict[str, Path] = {}
    for key in keys:
        paths[key] = ensure_file(key)
    return paths


# Kontext 管线组装所需的共享组件键全集（各验证脚本复用）
SHARED_KEYS = [
    "clip_l", "t5xxl_fp8", "ae",
    "tok_clip_vocab", "tok_clip_merges", "tok_clip_config", "tok_clip_special",
    "tok_t5_spiece", "tok_t5_config", "tok_t5_special", "scheduler_config",
]


# ---------------------------------------------------------------------------
# 组件组装（ComfyUI 单文件 → diffusers 组件）
# ---------------------------------------------------------------------------

def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gguf_transformer(gguf_path: Path, compute_dtype=torch.bfloat16):
    """GGUF 量化 transformer → FluxTransformer2DModel。

    diffusers 官方 GGUF 路线：from_single_file + GGUFQuantizationConfig。
    compute_dtype=Turing 无 BF16 硬件时用 float16 更稳（T2.2 实测决定）。
    """
    from diffusers import FluxTransformer2DModel
    from diffusers.quantizers.gguf import GGUFQuantizationConfig

    return FluxTransformer2DModel.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype,
    )


def ensure_kontext_bf16() -> tuple[Path, str]:
    """确保 kontext BF16 transformer 就绪（ModelScope 官方镜像）。

    ModelScope 的 black-forest-labs/FLUX.1-Kontext-dev 两种形态都存在
    （2026-08-22 实测文件列表）：
    1. 根目录单文件 flux1-kontext-dev.safetensors（~24GB，ComfyUI 形态）✓ 存在
    2. diffusers 目录 transformer/（config.json + 3 分片 safetensors）✓ 存在

    单文件形态优先（与 GGUF/FP8 的单文件加载路线一致）。
    返回 (加载路径, 形态标记)：形态标记 "single_file" / "repo_dir"
    """
    repo_id = "black-forest-labs/FLUX.1-Kontext-dev"
    single = "flux1-kontext-dev.safetensors"
    target = TRANSFORMER_KONTEXT_DIR / single
    if target.exists() and target.stat().st_size > 20_000 * MIN_MB:
        print(f"[skip] {target.name} 已存在（{_fmt_gb(target.stat().st_size)}）")
        return target, "single_file"

    print(f"[download:modelscope] {repo_id}/{single} -> {target}", flush=True)
    try:
        _ms_download(repo_id, single, target, 20_000 * MIN_MB)
        return target, "single_file"
    except Exception as exc:  # noqa: BLE001
        print(f"[fallback] 单文件下载失败（{exc!r}），尝试 diffusers 目录形态")

    # diffusers 目录形态：transformer/ 下全部文件
    marker_dir = TRANSFORMER_KONTEXT_DIR / "bf16_repo"
    tf_files = [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]
    marker_dir.mkdir(parents=True, exist_ok=True)
    for f in tf_files:
        rel = Path(f).relative_to("transformer")
        dst = marker_dir / rel
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"[download:modelscope] {repo_id}/{f}")
        _ms_download(repo_id, f, dst, 0)
    return marker_dir, "repo_dir"


def load_fp8_transformer(fp8_path: Path, compute_dtype=torch.bfloat16):
    """FP8 scaled 单文件 transformer。

    两条加载路线（T2.2 实测对象）：
    1. torch_dtype=compute_dtype —— diffusers 加载时把 fp8 上抛为 bf16/fp16
       （Turing 无 FP8 计算，上抛后可跑但显存无收益，仅省磁盘）
    2. 保持 fp8 存储 + torchao weight-only（若环境支持）
    """
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_single_file(
        str(fp8_path),
        torch_dtype=compute_dtype,
    )


def load_bf16_transformer(path: Path, form: str, dtype=torch.bfloat16):
    """BF16 transformer（单文件 / diffusers 目录两形态）。"""
    from diffusers import FluxTransformer2DModel

    if form == "single_file":
        return FluxTransformer2DModel.from_single_file(str(path), torch_dtype=dtype)
    return FluxTransformer2DModel.from_pretrained(str(path), torch_dtype=dtype)


def load_vae(ae_path: Path, dtype=torch.bfloat16):
    """VAE 单文件 → AutoencoderKL。"""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_single_file(str(ae_path), torch_dtype=dtype)


def load_clip_l(clip_path: Path, dtype=torch.bfloat16):
    """CLIP-L 单文件 → CLIPTextModel。

    ComfyUI 的 clip_l.safetensors 是 openai/clip-vit-large-patch14 的完整
    CLIPModel 权重（text_model.* + vision_model.*），transformers 无单文件
    加载入口，这里手工构造 CLIP-L text 配置 + 过滤 state_dict。
    """
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel

    config = CLIPTextConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=77,
        vocab_size=49408,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        bos_token_id=49406,
        eos_token_id=49407,
    )
    model = CLIPTextModel(config)
    state = load_file(str(clip_path))
    text_state = {k: v for k, v in state.items() if k.startswith("text_model.")}
    if not text_state:
        # 兜底：打印实际键名前缀辅助诊断
        prefixes = sorted({k.split(".")[0] for k in state.keys()})
        raise RuntimeError(f"clip_l.safetensors 无 text_model.* 键，实际前缀：{prefixes}")
    missing, unexpected = model.load_state_dict(text_state, strict=False)
    # position_ids 是缓冲区，缺失属正常；其余缺失键视为失败
    real_missing = [k for k in missing if k != "text_model.embeddings.position_ids"]
    if real_missing or unexpected:
        raise RuntimeError(
            f"CLIP-L 权重不匹配：missing={real_missing[:5]}... unexpected={unexpected[:5]}..."
        )
    model = model.to(dtype=dtype).eval()
    return model


def load_t5xxl_fp8(t5_path: Path, dtype=torch.bfloat16):
    """T5-XXL fp8 单文件 → T5EncoderModel（逐张量反量化到 dtype）。

    t5xxl_fp8_e4m3fn_scaled.safetensors：权重以 float8_e4m3fn 存储，
    "_scaled" 表示权重按 per-tensor scale 放大以利用 fp8 动态范围，
    文件内含 `<key>.scale` 缩放张量，加载时 W = W_fp8.to(dtype) * scale。
    （若实测无 scale 键则按纯 cast 处理——本函数是 T2.1 的验证对象之一）
    """
    from safetensors.torch import load_file
    from transformers import T5Config, T5EncoderModel

    state = load_file(str(t5_path))
    # 诊断：键前缀与 dtype
    dtypes = {str(v.dtype) for v in state.values()}
    has_scale = any(k.endswith(".scale") for k in state)
    print(f"[t5] 键数={len(state)} dtypes={dtypes} has_scale_keys={has_scale}")

    config = T5Config(
        d_model=4096,
        d_ff=10240,
        d_kv=64,
        num_heads=64,
        num_layers=24,
        vocab_size=32128,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=False,
        is_decoder=False,
        use_cache=False,
    )
    model = T5EncoderModel(config)
    t5_state: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".scale"):
            continue
        w = tensor
        if w.dtype == torch.float8_e4m3fn:
            scale = state.get(f"{key}.scale")
            w = w.to(dtype)
            if scale is not None:
                w = w * scale.to(dtype)
        t5_state[key] = w
    # encoder-only 权重键名即 T5EncoderModel 期望键名（encoder.* / shared.*）
    missing, unexpected = model.load_state_dict(t5_state, strict=False)
    # T5EncoderModel 不需要的 decoder 键会被报告为 unexpected——过滤掉
    real_missing = [k for k in missing if not k.startswith("decoder.")]
    bad_unexpected = [k for k in unexpected if not k.startswith("decoder.")]
    if real_missing or bad_unexpected:
        raise RuntimeError(
            f"T5 权重不匹配：missing={real_missing[:5]}... unexpected={bad_unexpected[:5]}..."
        )
    model = model.to(dtype=dtype).eval()
    return model


def load_tokenizers(base_dir: Path):
    """加载 CLIP / T5 tokenizer（本地小文件目录）。"""
    from transformers import CLIPTokenizer, T5TokenizerFast

    clip_tok = CLIPTokenizer.from_pretrained(str(base_dir / "tokenizers" / "clip"))
    t5_tok = T5TokenizerFast.from_pretrained(str(base_dir / "tokenizers" / "t5"))
    return clip_tok, t5_tok


def load_scheduler(base_dir: Path):
    """加载 Kontext 官方 scheduler 配置。"""
    from diffusers import FlowMatchEulerDiscreteScheduler

    cfg = base_dir / "tokenizers" / "scheduler_config.json"
    return FlowMatchEulerDiscreteScheduler.from_config(
        __import__("json").loads(cfg.read_text(encoding="utf-8"))
    )


def assemble_kontext_pipeline(
    transformer,
    vae=None,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    scheduler=None,
    dtype=torch.bfloat16,
):
    """组装 FluxKontextPipeline（纯本地组件，无网络依赖）。"""
    from diffusers import FluxKontextPipeline

    paths = ensure_models(SHARED_KEYS)
    if vae is None:
        vae = load_vae(paths["ae"], dtype)
    if text_encoder is None:
        text_encoder = load_clip_l(paths["clip_l"], dtype)
    if text_encoder_2 is None:
        text_encoder_2 = load_t5xxl_fp8(paths["t5xxl_fp8"], dtype)
    if tokenizer is None or tokenizer_2 is None:
        tokenizer, tokenizer_2 = load_tokenizers(FLUX_DIR)
    if scheduler is None:
        scheduler = load_scheduler(FLUX_DIR)

    pipe = FluxKontextPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    )
    return pipe


# ---------------------------------------------------------------------------
# 显存/耗时度量工具
# ---------------------------------------------------------------------------

def vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 ** 2


def vram_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1024 ** 2


def free_vram_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(tag: str, t0: float) -> None:
    print(
        f"[report] {tag}: 耗时={time.time() - t0:.1f}s "
        f"allocated={vram_mb():.0f}MB reserved={vram_reserved_mb():.0f}MB"
    )


def banner(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


if __name__ == "__main__":
    # 独立运行：只做下载（不组装），供预先批量拉取
    keys = sys.argv[1:] or [
        "kontext_q8", "clip_l", "t5xxl_fp8", "ae",
        "tok_clip_vocab", "tok_clip_merges", "tok_clip_config", "tok_clip_special",
        "tok_t5_spiece", "tok_t5_config", "tok_t5_special", "scheduler_config",
    ]
    banner(f"FLUX Kontext P0 下载：{keys}")
    ensure_models(keys)
    print("全部就绪")
