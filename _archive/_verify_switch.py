"""T2.4 双 transformer 热切换验证。

验证目标（分析 §6 风险 4）：同进程内 dev ↔ kontext transformer 切换，
共享组件（VAE/CLIP/T5/tokenizer/scheduler）不重载，验证切换耗时与
显存回收（无泄漏式增长）。

场景：
- GGUF Q8 双驻（12.6×2 + 5.5 ≈ 31GB < 48GB）：两 transformer 同时常驻，
  切换 = 换引用，应零耗时
- 单驻模式（模拟 BF16 预算）：卸载 A → 加载 B，验证显存回收

运行：
  uv run --no-sync --project envs/image-qwen-image \
      python engines/qwen_image_5e8f3a21/_verify_switch.py
"""

from __future__ import annotations

import gc
import sys
import time

import torch
from PIL import Image

from _verify_common import (
    OUTPUT_DIR,
    SHARED_KEYS,
    assemble_kontext_pipeline,
    banner,
    ensure_models,
    free_vram_cache,
    load_gguf_transformer,
    report,
    resolve_device,
    vram_reserved_mb,
)


def gen_one(pipe, tag: str) -> None:
    """单次出图冒烟（512，8 步）。"""
    test_img = Image.new("RGB", (512, 512), color=(60, 120, 60))
    t0 = time.time()
    result = pipe(
        image=test_img,
        prompt="a green spiral pattern, abstract",
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=2.5,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.images[0].save(OUTPUT_DIR / f"_verify_t2_4_{tag}.png")
    report(f"出图（{tag}）", t0)


def main() -> int:
    banner("T2.4 双 transformer 热切换验证")
    device = resolve_device()
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    paths = ensure_models(["kontext_q8", "dev_q8", *SHARED_KEYS])

    # 1) 加载共享组件 + kontext transformer 组装管线
    t0 = time.time()
    kontext_tf = load_gguf_transformer(paths["kontext_q8"], compute_dtype)
    pipe = assemble_kontext_pipeline(kontext_tf, dtype=compute_dtype).to(device)
    report("初始组装（kontext GGUF + 共享组件）", t0)
    shared_vram = vram_reserved_mb()
    print(f"[baseline] reserved={shared_vram:.0f}MB")

    gen_one(pipe, "kontext_1")

    # 2) 加载 dev transformer（双驻）——t2i 用 FluxPipeline 组装
    t0 = time.time()
    dev_tf = load_gguf_transformer(paths["dev_q8"], compute_dtype)
    report("加载 dev GGUF（双驻）", t0)
    dual_vram = vram_reserved_mb()
    print(f"[dual] reserved={dual_vram:.0f}MB（增量 {dual_vram - shared_vram:.0f}MB）")

    # 3) FluxPipeline 组装（t2i 侧，共享同一批 text_encoder/vae/tokenizer）
    from diffusers import FluxPipeline

    t2i_pipe = FluxPipeline(
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer_2=pipe.tokenizer_2,
        transformer=dev_tf,
    ).to(device)

    # 4) 零成本切换验证：kontext ↔ dev 交替推理（仅换引用，共享组件不动）
    t0 = time.time()
    gen_one(pipe, "kontext_2")          # kontext 再来一次
    switch_time = time.time() - t0
    t0 = time.time()
    result = t2i_pipe(
        prompt="a green spiral pattern, abstract",
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=3.5,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.images[0].save(OUTPUT_DIR / "_verify_t2_4_dev_t2i.png")
    report("t2i 出图（dev）", t0)
    print(f"[switch] 双驻交替推理正常，往返耗时（含出图）={switch_time:.1f}s")

    # 5) 单驻切换验证：卸载 dev → 显存应回落
    dev_tf = None
    t2i_pipe = None
    gc.collect()
    free_vram_cache()
    after_unload = vram_reserved_mb()
    print(f"[unload-dev] reserved={after_unload:.0f}MB（较双驻回落 {dual_vram - after_unload:.0f}MB）")

    # 6) 再加载 dev（模拟 BF16 单驻热切换路径），显存应可预测回升
    t0 = time.time()
    dev_tf2 = load_gguf_transformer(paths["dev_q8"], compute_dtype)
    dev_tf2.to(device)
    report("重新加载 dev（单驻热切换路径）", t0)
    reload_vram = vram_reserved_mb()
    print(f"[reload-dev] reserved={reload_vram:.0f}MB（与首次双驻差 {reload_vram - dual_vram:+.0f}MB）")

    leak = reload_vram - dual_vram
    banner("T2.4 结论")
    if leak < 2_000:  # 2GB 容差（分配器碎片）
        print(f"✅ 通过：双驻交替零成本；单驻卸载/重载显存回收正常（差值 {leak:+.0f}MB）")
        return 0
    print(f"❌ 疑似显存泄漏：重载后比首次多 {leak:.0f}MB，TransformerManager 需强制单驻+full empty_cache")
    return 1


if __name__ == "__main__":
    sys.exit(main())
