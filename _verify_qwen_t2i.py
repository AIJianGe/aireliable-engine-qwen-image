"""M2 冒烟：QwenImagePipeline + Qwen-Image-2512 Q8（任务表 T4.2）。

验证目标：
    1. t2i 槽独立验证（2512 transformer + VL 文本编码）
    2. 中文字符渲染（Qwen-Image 强项，验收核心）

用法：
    cd envs\image-qwen-image
    .venv\Scripts\python.exe ..\..\engines\qwen_image_5e8f3a21\_verify_qwen_t2i.py
    可选：--model-dir / --steps（默认 10）/ --size 1328（默认 832 冒烟档）

通过标准：输出非黑/无 NaN；中文海报场景文字正确渲染（目视）
输出保存：output/_verify_qwen/t2i_*.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

import pipeline_builder as pb  # noqa: E402

PROJECT_ROOT = ENGINE_DIR.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "image" / "qwen-image"
OUT_DIR = PROJECT_ROOT / "output" / "_verify_qwen"


def _check_output(img, scene: str) -> bool:
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    ok = mn < 200.0 and mx > 10.0 and not math.isnan(mn) and not math.isnan(mx)
    print(f"    [{scene}] extrema: min={mn:.1f} max={mx:.1f} → {'✅ 非黑' if ok else '❌ 疑似黑图/NaN'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen t2i M2 冒烟")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--size", type=int, default=832, help="冒烟边长（默认 832，正式 1328）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"
    dtype = pb.resolve_compute_dtype(device)
    props = torch.cuda.get_device_properties(0)
    print(f"[0] GPU: {props.name}（sm_{props.major}{props.minor}）；compute_dtype={dtype}")

    # ---- 共享组件（与 edit 冒烟同一套；VL 决策 D4）----
    t0 = time.perf_counter()
    comps = pb.load_shared_components(args.model_dir, device, dtype)
    print(f"[1] 共享组件加载完成（{time.perf_counter()-t0:.1f}s；VL={comps.vl_device}）")

    # ---- t2i 槽 transformer（2512 Q8）----
    t0 = time.perf_counter()
    transformer = pb.load_transformer(args.model_dir, "t2i", "gguf_q8", dtype)
    print(f"[2] Qwen-Image-2512 Q8 transformer 加载完成（{time.perf_counter()-t0:.1f}s）")

    # ---- 组装 t2i 管线 ----
    pipe = pb.build_t2i_pipeline(comps, transformer, device)
    print("[3] QwenImagePipeline 组装完成")

    all_ok = True

    def run_scene(name: str, prompt: str, negative: str | None = None, seed: int = 2026):
        nonlocal all_ok
        print(f"\n[{name}] prompt: {prompt!r}（{args.steps} 步，{args.size}×{args.size}）")
        t0 = time.perf_counter()
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=args.steps,
            height=args.size,
            width=args.size,
            generator=torch.Generator(device).manual_seed(seed),
        )
        if negative:
            kwargs["negative_prompt"] = negative
        try:
            out = pipe(**kwargs).images[0]
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ 推理异常：{type(e).__name__}: {e}")
            all_ok = False
            return
        elapsed = time.perf_counter() - t0
        out_path = OUT_DIR / f"t2i_{name}.png"
        out.save(out_path)
        print(f"    耗时 {elapsed:.1f}s → {out_path}")
        if not _check_output(out, name):
            all_ok = False

    # 场景 1：常规文生图
    run_scene(
        "scene1_basic",
        "A serene mountain lake at sunrise, mist over the water, "
        "snow-capped peaks reflected in the still water, photorealistic.",
    )

    # 场景 2：中文字符渲染（Qwen 强项验收）
    run_scene(
        "scene2_chinese",
        "一张现代简约风格的咖啡店海报，海报上写着大号中文标题「山水咖啡」，"
        "下方小字「每一杯，都是山间的清晨」，暖色调，木质纹理背景。",
        negative="blurry, low quality, distorted text",
    )

    if torch.cuda.is_available():
        print(f"\n[显存] 峰值分配：{torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

    print("\n" + ("✅ M2 t2i 冒烟通过（中文渲染效果请目视 t2i_scene2_chinese.png）" if all_ok else "❌ 存在失败场景"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
