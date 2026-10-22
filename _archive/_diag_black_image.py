"""黑图根因诊断：T5-XXL / CLIP-L fp16 编码输出检查。

背景：8/22 起 FLUX 引擎所有输出均为纯黑图（1024×1024 全 0 像素，PNG 仅 3.1KB）。
采样流程无异常抛出、任务状态 succeeded → 数值问题（NaN/Inf → clamp 全 0）而非流程问题。

嫌疑链（按社区已知问题概率排序）：
    1. T5-XXL fp16 激活溢出（transformers 官方明确警告 T5 不支持 fp16，推荐 bf16）
    2. GGUF Q8 + fp16 transformer 计算溢出
    3. VAE fp16 解码问题

本脚本独立于引擎进程（CPU 推导，NaN 语义 CPU/GPU 一致）：
    - Step 1: 加载 T5 fp8 → 反量化 fp16（与 pipeline_builder._load_t5xxl_fp8 相同逻辑）
    - Step 2: fp16 encode 用户真实 prompt，逐 block hook 检查首个 Inf/NaN 出现层
    - Step 3: 对照 fp32 encode 同一 prompt（fp32 为金标准）
    - Step 4: CLIP-L fp16 encode 对照（排除法）

用法：
    cd engines/qwen_image_5e8f3a21
    ..\..\envs\image-qwen-image\.venv\Scripts\python.exe _diag_black_image.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("_diag")

MODEL_DIR = Path(r"D:\myProInstallDir\AIReliable\models\image\flux-kontext")
T5_PATH = MODEL_DIR / "text_encoders" / "t5xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_PATH = MODEL_DIR / "text_encoders" / "clip_l.safetensors"
T5_TOK_DIR = MODEL_DIR / "tokenizers" / "t5"
CLIP_TOK_DIR = MODEL_DIR / "tokenizers" / "clip"

# 用户真实黑图请求的 prompt（2026-08-23 11:14:48 引擎日志）
PROMPT = "a red lighthouse on a rocky coast at sunset, dramatic sky"


def load_t5_fp8(dtype: torch.dtype):
    """复刻 pipeline_builder._load_t5xxl_fp8 的加载逻辑（独立复制，避免相对导入）。"""
    from safetensors.torch import load_file
    from transformers import T5Config, T5EncoderModel

    state = load_file(str(T5_PATH))
    config = T5Config(
        d_model=4096, d_ff=10240, d_kv=64, num_heads=64, num_layers=24,
        vocab_size=32128, feed_forward_proj="gated-gelu",
        is_encoder_decoder=False, is_decoder=False, use_cache=False,
    )
    model = T5EncoderModel(config)
    t5_state: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".scale_weight") or key == "scaled_fp8":
            continue
        w = tensor
        if w.dtype == torch.float8_e4m3fn:
            scale = state.get(f"{key}.scale_weight")
            w = w.to(dtype)
            if scale is not None:
                w = w * scale.to(dtype)
        t5_state[key] = w
    missing, unexpected = model.load_state_dict(t5_state, strict=False)
    real_missing = [k for k in missing if not k.startswith("decoder.")]
    bad_unexpected = [k for k in unexpected if not k.startswith("decoder.")]
    if real_missing or bad_unexpected:
        raise RuntimeError(f"T5 权重不匹配: missing={real_missing[:3]} unexpected={bad_unexpected[:3]}")
    return model.to(dtype=dtype).eval()


def stat(t: torch.Tensor) -> str:
    """张量数值统计（nan/inf/范围）。"""
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())} "
            f"min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else 'N/A'} "
            f"max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'N/A'} "
            f"mean={t[torch.isfinite(t)].mean().item() if torch.isfinite(t).any() else 'N/A'}"
        )
    return f"shape={tuple(t.shape)} dtype={t.dtype}"


def diag_t5(dtype: torch.dtype, label: str) -> None:
    """加载 T5 并 encode，逐 block 定位首个 Inf/NaN。"""
    from transformers import T5TokenizerFast

    log.info("=== T5 %s 加载（fp8 反量化 → %s）===", label, dtype)
    t0 = time.perf_counter()
    model = load_t5_fp8(dtype)
    log.info("T5 加载完成 %.1fs", time.perf_counter() - t0)

    # 逐 block hook：记录每个 block 输出的 nan/inf 计数
    block_reports: list[str] = []

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                n, i = int(torch.isnan(t).sum()), int(torch.isinf(t).sum())
                if n or i:
                    block_reports.append(f"block[{idx}]: nan={n} inf={i}")
        return hook

    for i, block in enumerate(model.encoder.block):
        block.register_forward_hook(make_hook(i))

    tok = T5TokenizerFast.from_pretrained(str(T5_TOK_DIR))
    inputs = tok(PROMPT, return_tensors="pt", padding=True, truncation=True, max_length=512)

    log.info("T5 %s encode 开始（prompt=%r）", label, PROMPT[:60])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    log.info("T5 %s encode 完成 %.1fs", label, time.perf_counter() - t0)

    log.info("T5 %s prompt_embeds: %s", label, stat(out.last_hidden_state))
    if block_reports:
        log.info("T5 %s 首个异常层及全部异常层: %s", label, block_reports)
    else:
        log.info("T5 %s 所有 block 输出无 nan/inf", label)


def diag_clip(dtype: torch.dtype) -> None:
    """CLIP-L fp16 encode 对照（排除法）。"""
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    config = CLIPTextConfig(
        hidden_size=768, intermediate_size=3072, num_attention_heads=12,
        num_hidden_layers=12, max_position_embeddings=77, vocab_size=49408,
        hidden_act="quick_gelu", layer_norm_eps=1e-05,
        bos_token_id=49406, eos_token_id=49407,
    )
    model = CLIPTextModel(config)
    state = load_file(str(CLIP_PATH))
    text_state = {k: v for k, v in state.items() if k.startswith("text_model.")}
    model_keys = set(model.state_dict().keys())
    if not any(k.startswith("text_model.") for k in model_keys):
        text_state = {k.removeprefix("text_model."): v for k, v in text_state.items()}
    model.load_state_dict(text_state, strict=False)
    model = model.to(dtype=dtype).eval()

    tok = CLIPTokenizer.from_pretrained(str(CLIP_TOK_DIR))
    inputs = tok(PROMPT, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    log.info("CLIP-L fp16 prompt_embeds: %s", stat(out.last_hidden_state))


def main() -> None:
    if not T5_PATH.exists():
        log.error("T5 权重不存在: %s", T5_PATH)
        sys.exit(1)

    # Step 1: T5 fp16（引擎当前实际计算 dtype）
    diag_t5(torch.float16, "fp16")

    # Step 2: T5 fp32 金标准对照
    diag_t5(torch.float32, "fp32")

    # Step 3: CLIP-L fp16 排除法
    diag_clip(torch.float16)

    log.info("=== 诊断结论判读 ===")
    log.info("若 fp16 输出含 nan/inf 而 fp32 干净 → 根因实锤：T5 fp16 激活溢出")
    log.info("若两者都干净 → 根因在 transformer（GGUF Q8 + fp16）或 VAE，需进一步诊断")


if __name__ == "__main__":
    main()
