"""T2.1 组件组装验证：ComfyUI 单文件 → diffusers FluxKontextPipeline 出图。

验证目标（分析 §6 风险 1）：
  GGUF transformer + fp8 T5 + clip_l + ae 四类 ComfyUI 单文件能否在
  diffusers 侧组装出可出图的 FluxKontextPipeline。

失败回退：官方 diffusers repo 整仓路线（影响 manifest，需重写 T3.1）。

运行：
  uv run --no-sync --project envs/image-qwen-image \
      python engines/qwen_image_5e8f3a21/_verify_assemble.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from _verify_common import (
    FLUX_DIR,
    OUTPUT_DIR,
    SHARED_KEYS,
    assemble_kontext_pipeline,
    banner,
    ensure_models,
    load_gguf_transformer,
    report,
    resolve_device,
)


def main() -> int:
    banner("T2.1 组件组装验证（ComfyUI 单文件 → diffusers 管线）")
    device = resolve_device()
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} compute_dtype={compute_dtype}")

    # 1) 下载最小验证集（kontext Q8 + 共享组件）
    paths = ensure_models(["kontext_q8", *SHARED_KEYS])

    # 2) GGUF transformer 加载
    t0 = time.time()
    transformer = load_gguf_transformer(paths["kontext_q8"], compute_dtype)
    report("transformer(GGUF Q8) 加载", t0)

    # 3) 组装完整管线（内部逐组件加载：VAE / CLIP-L / T5 fp8 / tokenizer / scheduler）
    t0 = time.time()
    pipe = assemble_kontext_pipeline(transformer, dtype=compute_dtype)
    report("管线组装（含 VAE/CLIP/T5/tokenizer/scheduler）", t0)
    pipe = pipe.to(device)

    # 4) 单图编辑推理出图（PIL 输入）
    from PIL import Image

    test_img = Image.new("RGB", (512, 512), color=(30, 60, 120))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = pipe(
        image=test_img,
        prompt="change the background to a sunny beach, keep the composition",
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=2.5,
    )
    images = result.images
    out_path = OUTPUT_DIR / "_verify_t2_1_assemble.png"
    images[0].save(out_path)
    report(f"推理出图（8 步 512x512）-> {out_path}", t0)

    # 5) NaN 检查
    import numpy as np

    arr = np.asarray(images[0], dtype=np.float32)
    has_nan = bool(np.isnan(arr).any())
    print(f"[check] 输出 NaN={has_nan} 均值={arr.mean():.1f} 尺寸={images[0].size}")

    if has_nan:
        print("❌ T2.1 失败：输出含 NaN（fp16 数值溢出？下一步 T2.2 换 dtype 对比）")
        return 1

    print("✅ T2.1 通过：单文件组件可在 diffusers 组装出图")
    return 0


if __name__ == "__main__":
    sys.exit(main())
