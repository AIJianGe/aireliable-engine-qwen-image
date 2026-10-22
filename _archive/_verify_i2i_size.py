"""T2.3 Kontext I2I 多图参考与输出尺寸独立性验证。

验证目标（分析 §6 风险 3）：多图参考模式下，输出尺寸能否独立于参考图
尺寸指定（如 2×512 参考图 + 输出 1024×1024）。

2026-08-22 重大发现（源码核实，diffusers 0.40.0）：
- `FluxKontextImage2ImagePipeline` 不存在（分析文档的类名假设错误）
- `FluxKontextPipeline.__call__(image=[img1, img2, ...])` 原生支持多图：
  多图在 prepare_latents 中按序列维拼接（in-context 机制，latent_ids
  与 image_ids concat），无需手工拼接参考图
- `height/width` 参数独立控制输出尺寸；`max_area=1024²` + `_auto_resize`
  自动把参考图缩放到 PREFERRED_KONTEXT_RESOLUTIONS 最近档

验证内容：
1. 多图列表输入（2 张 512×512）是否可跑通
2. width/height=1024×1024 时输出是否等于 1024×1024
3. （对照）手工拼接 stitcher 路线仍可用作回退

运行：
  uv run --no-sync --project envs/image-qwen-image \
      python engines/qwen_image_5e8f3a21/_verify_i2i_size.py
"""

from __future__ import annotations

import sys
import time

import torch
from PIL import Image

import stitcher
from _verify_common import (
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
    banner("T2.3 Kontext I2I 多图参考 + 输出尺寸独立性验证")
    device = resolve_device()
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} compute_dtype={compute_dtype}")

    paths = ensure_models(["kontext_q8", *SHARED_KEYS])
    transformer = load_gguf_transformer(paths["kontext_q8"], compute_dtype)
    pipe = assemble_kontext_pipeline(transformer, dtype=compute_dtype).to(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[str] = []

    # ---- A) 原生多图列表输入 + 输出尺寸独立指定 ----
    refs = [
        Image.new("RGB", (512, 512), color=(200, 80, 80)),
        Image.new("RGB", (512, 512), color=(80, 80, 200)),
    ]
    try:
        t0 = time.time()
        result = pipe(
            image=refs,
            prompt="a person wearing a red hat and a person wearing a blue hat, standing together",
            width=1024,
            height=1024,
            num_inference_steps=8,
            guidance_scale=2.5,
        )
        out_size = result.images[0].size
        out_path = OUTPUT_DIR / "_verify_t2_3_native_multi.png"
        result.images[0].save(out_path)
        report("原生多图推理（2×512 → 1024x1024）", t0)
        ok = out_size == (1024, 1024)
        results.append(f"原生多图列表：输入 2×512x512，请求 1024x1024，输出 {out_size}"
                       f" -> {'✅ 尺寸独立' if ok else '❌ 尺寸不符'}")
    except Exception as exc:  # noqa: BLE001
        results.append(f"原生多图列表：❌ 失败 {exc!r}")

    # ---- B) 3~4 张参考图上限验证（多图参考能力边界）----
    try:
        refs4 = [Image.new("RGB", (512, 512), color=(60, 60, 60)) for _ in range(4)]
        t0 = time.time()
        result = pipe(
            image=refs4,
            prompt="four abstract gray squares arranged in a row",
            width=1024,
            height=1024,
            num_inference_steps=4,
            guidance_scale=2.5,
        )
        out_path = OUTPUT_DIR / "_verify_t2_3_multi4.png"
        result.images[0].save(out_path)
        report("4 图参考推理", t0)
        results.append(f"4 图参考：✅ 输出 {result.images[0].size}")
    except Exception as exc:  # noqa: BLE001
        results.append(f"4 图参考：❌ 失败 {exc!r}")

    # ---- C) 对照：手工拼接 stitcher 路线（回退方案验证）----
    try:
        st = stitcher.stitch_references(refs, notes=[])
        t0 = time.time()
        result = pipe(
            image=st.image,
            prompt="two people standing together",
            width=1024,
            height=1024,
            num_inference_steps=8,
            guidance_scale=2.5,
        )
        out_path = OUTPUT_DIR / "_verify_t2_3_stitched.png"
        result.images[0].save(out_path)
        report("拼接图推理（stitcher 对照）", t0)
        results.append(f"拼接对照：拼接图 {st.image.size}，输出 {result.images[0].size}")
    except Exception as exc:  # noqa: BLE001
        results.append(f"拼接对照：❌ 失败 {exc!r}")

    banner("T2.3 结论")
    for line in results:
        print(line)
    native_ok = any("尺寸独立" in line for line in results)
    if native_ok:
        print("\n✅ 通过：FluxKontextPipeline 原生多图 + 输出尺寸独立可控")
        print("   → i2i 多参考模式采用原生 image 列表（无需拼接器）")
        return 0
    print("\n⚠️ 原生多图路线未通过，引擎侧回退 stitcher 拼接路线")
    return 1


if __name__ == "__main__":
    sys.exit(main())
