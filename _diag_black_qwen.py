"""QwenImage 黑图根因诊断（GPU 实证，复刻引擎加载/编码路径）。

背景：Qwen 迁移后所有 t2i 输出均为纯黑图（1328×1328 全 0 像素，PNG 仅 5207B）。
采样流程无异常、任务 succeeded → 数值问题（NaN/Inf → clamp 全 0）。

嫌疑链（Qwen 官方文档：Qwen2.5-VL 明确要求 bf16，fp16 激活溢出是已知问题）：
    1. Qwen2.5-VL-7B fp16 编码溢出（引擎当前 torch_dtype=torch.float16）→ NaN embeds
    2. GGUF Q8 transformer + fp16 compute_dtype 溢出
    3. VAE fp16 解码问题

诊断步骤：
    Step 0: 确认黑图像素（排除保存链路问题）
    Step 1: VL fp16 GPU encode 真实 prompt（复刻 QwenImagePipeline.encode_prompt
            的 chat template 路径）→ NaN/Inf 统计 + 逐 block 定位
    Step 2: VL bf16 对照（Turing 仿真 bf16，数值正确仅慢）
    Step 3: （VL 干净时）transformer GGUF fp16 单步 forward → NaN 检查

用法：
    cd engines/qwen_image_5e8f3a21
    ..\..\envs\image-qwen-image\.venv\Scripts\python.exe _diag_black_qwen.py [step1|step2|step3|black]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("_diag")

ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "models" / "image" / "qwen-image"
VL_DIR = MODEL_DIR / "text_encoder"
TOK_DIR = MODEL_DIR / "tokenizers"
OUTPUT_DIR = ROOT / "output" / "image"

# 引擎日志中的真实黑图 prompt（2026-08-25 16:01 / 17:08）
PROMPTS = [
    "a cute orange cat wearing sunglasses, beach background",
    "生成一个美女",
]


def stat(t: torch.Tensor) -> str:
    if t.is_floating_point():
        fin = torch.isfinite(t)
        if fin.any():
            return (f"shape={tuple(t.shape)} dtype={t.dtype} "
                    f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())} "
                    f"min={t[fin].min().item():.4g} max={t[fin].max().item():.4g}")
        return f"shape={tuple(t.shape)} dtype={t.dtype} nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())} 全非有限"
    return f"shape={tuple(t.shape)} dtype={t.dtype}"


def step0_black() -> None:
    """Step 0: 黑图像素确认。"""
    log.info("=== Step 0: 黑图像素确认 ===")
    for p in sorted(OUTPUT_DIR.glob("qwen_*.png"))[-3:]:
        img = Image.open(p)
        log.info("%s: size=%s mode=%s extrema=%s → %s",
                 p.name, img.size, img.mode, img.getextrema(),
                 "纯黑" if all(hi <= 8 for _lo, hi in img.getextrema()) else "非黑")


def load_vl(dtype: torch.dtype, device: str = "cuda"):
    """复刻 pipeline_builder._load_vl_encoder 的加载方式。"""
    from transformers import Qwen2_5_VLForConditionalGeneration
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(VL_DIR), torch_dtype=dtype)
    return te.to(device).eval()


def encode(te, tokenizer, prompt: str, max_len: int = 1024):
    """复刻 diffusers QwenImagePipeline._get_qwen_prompt_embeds 的模板路径。"""
    template = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects "
        "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    drop_idx = 34
    txt = template.format(prompt)
    tokens = tokenizer(
        txt, max_length=max_len + drop_idx, padding=True,
        truncation=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = te(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
    return out.hidden_states[-1]


def diag_vl(dtype: torch.dtype, label: str) -> None:
    """VL 编码器指定 dtype 诊断（逐 block hook 定位首个 NaN 层）。"""
    from transformers import AutoTokenizer
    log.info("=== Step: VL %s（GPU）===", label)
    t0 = time.perf_counter()
    te = load_vl(dtype)
    log.info("VL 加载 %.1fs", time.perf_counter() - t0)
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))

    reports: list[str] = []

    def make_hook(name: str):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                n, i = int(torch.isnan(t).sum()), int(torch.isinf(t).sum())
                if n or i:
                    reports.append(f"{name}: nan={n} inf={i}")
        return hook

    # LLM decoder 层 hook
    for i, layer in enumerate(te.model.language_model.layers):
        layer.register_forward_hook(make_hook(f"llm.layer[{i}]"))

    for prompt in PROMPTS:
        t0 = time.perf_counter()
        embeds = encode(te, tokenizer, prompt)
        log.info("prompt=%r → %s（%.1fs）", prompt[:40], stat(embeds), time.perf_counter() - t0)
    if reports:
        log.info("异常层（前 5）: %s", reports[:5])
    else:
        log.info("所有 layer 输出无 nan/inf")
    del te
    torch.cuda.empty_cache()


def step6_locate() -> None:
    """Step 6: fp16 transformer 逐层定位（权重 inf 检查 + 60 block 首个 NaN 层）。

    目的：若 NaN 仅出现在固定少数层 → 混合精度局部修复（只把溢出层 fp32 化）；
    若第一步第一个 block 就 NaN → 权重反量化问题。
    """
    from diffusers import (
        AutoencoderKLQwenImage, GGUFQuantizationConfig,
        QwenImagePipeline, QwenImageTransformer2DModel,
    )
    from transformers import AutoTokenizer
    import json

    cfg_dir = Path(__file__).parent / "configs" / "qwen_transformer"
    gguf = MODEL_DIR / "transformers" / "t2i" / "qwen-image-2512-Q8_0.gguf"
    log.info("=== Step 6: fp16 transformer 逐层定位 ===")

    t0 = time.perf_counter()
    tr = QwenImageTransformer2DModel.from_single_file(
        str(gguf), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        torch_dtype=torch.float16, config=str(cfg_dir),
    ).to("cuda").eval()
    log.info("transformer 加载 %.1fs", time.perf_counter() - t0)

    # 6.1 权重 inf/nan 检查（GGUF 反量化是否产生 inf 权重）
    bad_params: list[str] = []
    for name, p in tr.named_parameters():
        if p.is_floating_point():
            n, i = int(torch.isnan(p).sum()), int(torch.isinf(p).sum())
            if n or i:
                bad_params.append(f"{name}: nan={n} inf={i}")
    if bad_params:
        log.info("权重异常（前 10）: %s", bad_params[:10])
    else:
        log.info("所有权重无 nan/inf（反量化干净）")

    # 6.2 逐 block hook
    blocks = tr.transformer_blocks
    log.info("transformer_blocks 数量=%d", len(blocks))
    first_nan: list[str] = []

    def make_hook(idx: int):
        def hook(_m, _inp, out):
            if first_nan:
                return  # 已定位，跳过
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                n, i = int(torch.isnan(t).sum()), int(torch.isinf(t).sum())
                if n or i:
                    first_nan.append(f"block[{idx}]: nan={n} inf={i} max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'N/A'}")
        return hook

    for i, b in enumerate(blocks):
        b.register_forward_hook(make_hook(i))

    # img_in / norm 输入侧也 hook
    def hook_img_in(_m, inp, out):
        if isinstance(out, torch.Tensor):
            log.info("img_in 输出: %s", stat(out.detach()))
    tr.img_in.register_forward_hook(hook_img_in)

    vae = AutoencoderKLQwenImage.from_pretrained(
        str(MODEL_DIR / "vae"), torch_dtype=torch.float16).to("cuda").eval()
    te = load_vl(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))
    sched_cfg = json.loads((MODEL_DIR / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    pipe = QwenImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=te,
        tokenizer=tokenizer, transformer=tr,
    )
    pipe.set_progress_bar_config(disable=True)

    result = pipe(
        prompt=PROMPTS[0], width=1328, height=1328,
        num_inference_steps=2, true_cfg_scale=1.0, guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(1207524730),
        max_sequence_length=512,
    )
    img = result.images[0]
    log.info("2 步完成，最终图 extrema=%s", img.getextrema())
    # 管线实际使用的 timesteps/sigmas（推理后仍保留在 scheduler 对象上）
    log.info("pipe timesteps=%s", [float(x) for x in pipe.scheduler.timesteps.tolist()])
    log.info("pipe sigmas=%s", [float(x) for x in pipe.scheduler.sigmas.tolist()])
    if first_nan:
        log.info("首个 NaN 层: %s", first_nan[0])
    else:
        log.info("所有 block 无 NaN（1 步）")


def step7_fp32_bench() -> None:
    """Step 7: fp32 transformer 显存占用 + 速度实测（prompt_embeds 注入绕过 VL）。

    决策数据：fp32 档 transformer 实际显存；Turing fp32 原生路径单步耗时
    （对照 fp16 引擎实测 1.7s/步）。embeds 用真实 prompt 形状（48 tokens，
    Step1 实测），随机值即可（只测速度不测数值——fp32 数值必然干净）。
    """
    from diffusers import (
        AutoencoderKLQwenImage, GGUFQuantizationConfig,
        QwenImagePipeline, QwenImageTransformer2DModel,
    )
    import json

    cfg_dir = Path(__file__).parent / "configs" / "qwen_transformer"
    gguf = MODEL_DIR / "transformers" / "t2i" / "qwen-image-2512-Q8_0.gguf"

    log.info("=== Step 7: fp32 transformer bench（prompt_embeds 注入）===")
    t0 = time.perf_counter()
    tr = QwenImageTransformer2DModel.from_single_file(
        str(gguf), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
        torch_dtype=torch.float32, config=str(cfg_dir),
    ).to("cuda").eval()
    log.info("transformer 加载 %.1fs，显存 %.1fGB",
             time.perf_counter() - t0, torch.cuda.memory_allocated() / 1024**3)

    vae = AutoencoderKLQwenImage.from_pretrained(
        str(MODEL_DIR / "vae"), torch_dtype=torch.float16).to("cuda").eval()
    sched_cfg = json.loads((MODEL_DIR / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    pipe = QwenImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None,
                             tokenizer=None, transformer=tr)
    pipe.set_progress_bar_config(disable=True)

    # 真实 prompt 编码形状（Step1 实测 48 tokens），随机值注入
    torch.manual_seed(0)
    embeds = torch.randn(1, 48, 3584, dtype=torch.float32, device="cuda") * 10
    mask = torch.ones(1, 48, dtype=torch.long, device="cuda")

    t0 = time.perf_counter()
    result = pipe(
        prompt=None, prompt_embeds=embeds, prompt_embeds_mask=mask,
        width=1328, height=1328, num_inference_steps=3, true_cfg_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(1207524730),
        max_sequence_length=512,
    )
    total = time.perf_counter() - t0
    img = result.images[0]
    log.info("3 步完成 %.1fs（≈%.2fs/步，50 步 ≈ %.1f 分钟），显存峰值 %.1fGB",
             total, total / 3, total / 3 * 50 / 60,
             torch.cuda.max_memory_allocated() / 1024**3)
    log.info("输出图 extrema=%s（随机 embeds 不代表真实画面，仅速度参考）", img.getextrema())


def step3_pipeline(compute_dtype: torch.dtype = torch.float16, vae_dtype: torch.dtype = torch.float16,
                   steps: int = 2, vl_device: str = "cuda", vl_dtype: torch.dtype | None = None,
                   embeds_cast: torch.dtype | None = None) -> None:
    """Step 3: 完整 t2i 管线 2 步复现（hook 每步 noise_pred + VAE 解码数值）。

    覆盖嫌疑链：transformer GGUF compute_dtype / VAE dtype / 保存链路。
    VL dtype 默认与 compute_dtype 一致；embeds_cast 非 None 时模拟生产修复
    （VL fp16 GPU 编码 + _get_qwen_prompt_embeds 输出 cast fp32 → latents fp32，
    桥接 GGUF forward_native 不做输入 dtype cast 的限制，见 step8）。
    """
    from diffusers import (
        AutoencoderKLQwenImage, GGUFQuantizationConfig,
        QwenImagePipeline, QwenImageTransformer2DModel,
    )
    from transformers import AutoTokenizer
    import json

    cfg_dir = Path(__file__).parent / "configs" / "qwen_transformer"
    gguf = MODEL_DIR / "transformers" / "t2i" / "qwen-image-2512-Q8_0.gguf"
    log.info("=== Step 3: 完整 t2i 管线 %d 步（transformer=%s vae=%s）===",
             steps, compute_dtype, vae_dtype)

    t0 = time.perf_counter()
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(MODEL_DIR / "vae"), torch_dtype=vae_dtype).to("cuda").eval()
    log.info("VAE 加载 %.1fs（%s）", time.perf_counter() - t0, vae_dtype)

    t0 = time.perf_counter()
    tr = QwenImageTransformer2DModel.from_single_file(
        str(gguf), quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype, config=str(cfg_dir),
    ).to("cuda").eval()
    log.info("transformer 加载 %.1fs（compute=%s）", time.perf_counter() - t0, compute_dtype)

    te = load_vl(vl_dtype or compute_dtype, device=vl_device)  # VL dtype 决定 latents dtype（prompt_embeds.dtype）
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))
    sched_cfg = json.loads((MODEL_DIR / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    pipe = QwenImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=te,
        tokenizer=tokenizer, transformer=tr,
    )
    pipe.set_progress_bar_config(disable=True)

    # 生产修复模拟：包装 _get_qwen_prompt_embeds，VL 编码输出统一 cast 到
    # embeds_cast（latents dtype 跟随 prompt_embeds.dtype）
    if embeds_cast is not None:
        _orig_get_embeds = pipe._get_qwen_prompt_embeds
        def _wrapped_get_embeds(*args, **kwargs):
            embeds, mask = _orig_get_embeds(*args, **kwargs)
            return embeds.to(embeds_cast), mask
        pipe._get_qwen_prompt_embeds = _wrapped_get_embeds
        log.info("已包装 _get_qwen_prompt_embeds：输出 cast → %s", embeds_cast)

    # hook：每步 noise_pred 数值统计
    step_reports: list[str] = []

    orig_forward = tr.forward

    def wrapped_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        t = out[0] if isinstance(out, (tuple, list)) else out
        t = t.sample if hasattr(t, "sample") else t
        step_reports.append(stat(t.detach()))
        return out

    tr.forward = wrapped_forward

    # hook：VAE decode 输入/输出统计
    orig_decode = vae.decode

    def wrapped_decode(*args, **kwargs):
        lat = args[0] if args else kwargs.get("z")
        log.info("VAE decode 输入: %s", stat(lat.detach()))
        out = orig_decode(*args, **kwargs)
        t = out[0] if isinstance(out, (tuple, list)) else out
        t = t.sample if hasattr(t, "sample") else t
        log.info("VAE decode 输出: %s", stat(t.detach()))
        return out

    vae.decode = wrapped_decode

    torch.manual_seed(1207524730)  # 用真实黑图请求的 seed
    t0 = time.perf_counter()
    result = pipe(
        prompt=PROMPTS[0], width=1328, height=1328,
        num_inference_steps=steps, true_cfg_scale=1.0, guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(1207524730),
        max_sequence_length=512,
    )
    img = result.images[0]
    log.info("管线完成 %.1fs，最终图 extrema=%s → %s",
             time.perf_counter() - t0, img.getextrema(),
             "纯黑" if all(hi <= 8 for _lo, hi in img.getextrema()) else "非黑")
    log.info("各步 noise_pred:")
    for i, r in enumerate(step_reports):
        log.info("  step[%d]: %s", i, r)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("black", "all"):
        step0_black()
    if mode in ("step1", "all"):
        diag_vl(torch.float16, "fp16（引擎当前档）")
    if mode in ("step2", "all"):
        diag_vl(torch.bfloat16, "bf16 对照")
    if mode in ("step3", "all"):
        step3_pipeline()
    if mode == "step4":
        # bf16 compute（Turing 仿真 = fp32 数值范围，验证 NaN 修复）
        step3_pipeline(compute_dtype=torch.bfloat16)
    if mode == "step5":
        # fp32 compute + VL fp32 驻 CPU（速度验证：Turing fp32 原生支持）
        step3_pipeline(compute_dtype=torch.float32, vl_device="cpu")
    if mode == "step6":
        # 逐层定位：fp16 transformer 权重 inf 检查 + 60 block 逐层 hook（第一个 NaN 层）
        step6_locate()
    if mode == "step7":
        # fp32 transformer 显存 + 单步速度实测（决策关键数据）
        step7_fp32_bench()
    if mode == "step8":
        # 生产修复候选：transformer compute=fp32（Turing 原生，无 NaN）+ VL fp16 GPU
        # + embeds cast fp32（桥接 GGUF forward_native 不做输入 dtype cast 的限制：
        # 本环境无 kernels ggml ops → forward_native → mat1/mat2 dtype 必须一致）
        step3_pipeline(compute_dtype=torch.float32, vl_dtype=torch.float16, vl_device="cuda",
                       embeds_cast=torch.float32)
    log.info("=== 诊断完成 ===")
