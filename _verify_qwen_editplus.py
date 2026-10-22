"""M1 离线冒烟：QwenImageEditPlusPipeline + Edit-2509 Q8（任务表 T3.1/T3.2/T4.1）。

验证目标（Turing fp16 + GGUF Q8 计算稳定性，风险 R1）：
    场景 1（T3.1）：单图指令编辑 —— 管线贯通 + 输出非黑/非 NaN
    场景 2（T3.2）：跨图交互 —— 2 张输入图 + "Picture 1 的女孩抱着 Picture 2 的猫"
    场景 3（T4.1）：视角控制 —— 单图 × {右 90°, 俯视 30°, 近景}（--view 场景开启）

用法（在 image-qwen-image venv 中运行）：
    cd envs\image-qwen-image
    .venv\Scripts\python.exe ..\..\engines\qwen_image_5e8f3a21\_verify_qwen_editplus.py
    可选参数：
        --model-dir D:\myProInstallDir\AIReliable\models\image\qwen-image（默认自动推导）
        --image1/--image2 <路径>   指定真实测试图（推荐；缺省用 PIL 合成占位图）
        --view                     追加 T4.1 视角控制三组场景
        --steps N                  采样步数（默认 10，冒烟用短步程）
        --skip-scene1/--skip-scene2 跳过指定场景

通过标准：
    - 所有场景输出图 extrema 非黑（min < 200）且无 NaN
    - 场景 2 输出图目视包含两个输入图的主体（女孩 + 猫）
    - 全程无 NaN warning / CUDA error
输出保存：output/_verify_qwen/editplus_scene*.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# 引擎目录加入 sys.path（直接复用引擎 pipeline_builder，同时验证引擎代码）
ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

import pipeline_builder as pb  # noqa: E402

# 项目根（D:\myProInstallDir\AIReliable）
PROJECT_ROOT = ENGINE_DIR.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "image" / "qwen-image"
OUT_DIR = PROJECT_ROOT / "output" / "_verify_qwen"

# 视角前缀翻译表（与 inference.py build_view_prefix 一致，T4.1 用）
_VIEW_HORIZONTAL = {
    "left45": "turned 45 degrees to the left",
    "left90": "turned 90 degrees to the left, left side view",
    "right45": "turned 45 degrees to the right",
    "right90": "turned 90 degrees to the right, right side view",
}
_VIEW_VERTICAL = {
    "up30": "viewed from 30 degrees above, slight high angle",
    "down30": "viewed from 30 degrees below, slight low angle",
    "down60": "viewed from 60 degrees below, dramatic low angle",
}
_VIEW_DISTANCE = {
    "close": "close-up shot",
    "far": "wide shot from a distance",
}


def _make_placeholder(kind: str, size: int = 768) -> Image.Image:
    """PIL 合成占位图（未提供真实图片时验证管线贯通；语义效果验证需 --image1/2）。"""
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    if kind == "girl":
        # 简笔人物：圆头 + 三角裙
        d.ellipse((size*0.35, size*0.10, size*0.65, size*0.40), fill=(245, 200, 170))
        d.polygon([(size*0.5, size*0.38), (size*0.25, size*0.85), (size*0.75, size*0.85)],
                  fill=(200, 80, 120))
        d.rectangle((size*0.44, size*0.85, size*0.48, size*0.98), fill=(245, 200, 170))
        d.rectangle((size*0.52, size*0.85, size*0.56, size*0.98), fill=(245, 200, 170))
    else:  # cat
        # 简笔猫：圆身 + 双耳
        d.ellipse((size*0.25, size*0.35, size*0.75, size*0.90), fill=(120, 110, 100))
        d.polygon([(size*0.28, size*0.28), (size*0.30, size*0.10), (size*0.42, size*0.24)], fill=(120, 110, 100))
        d.polygon([(size*0.72, size*0.28), (size*0.70, size*0.10), (size*0.58, size*0.24)], fill=(120, 110, 100))
        d.ellipse((size*0.32, size*0.12, size*0.68, size*0.44), fill=(140, 130, 118))
    return img


def _check_output(img: Image.Image, scene: str) -> bool:
    """输出图体检：非黑（min < 200）+ 像素值合法（无 NaN 产生的灰带）。"""
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    ok = mn < 200.0 and mx > 10.0 and not math.isnan(mn) and not math.isnan(mx)
    print(f"    [{scene}] extrema: min={mn:.1f} max={mx:.1f} → {'✅ 非黑' if ok else '❌ 疑似黑图/NaN'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen EditPlus M1 冒烟")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--image1", type=Path, default=None, help="真实测试图 1（人物）")
    ap.add_argument("--image2", type=Path, default=None, help="真实测试图 2（猫）")
    ap.add_argument("--view", action="store_true", help="追加 T4.1 视角控制场景")
    ap.add_argument("--steps", type=int, default=10, help="冒烟步数（默认 10）")
    ap.add_argument("--skip-scene1", action="store_true")
    ap.add_argument("--skip-scene2", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 0. 环境信息 ----
    assert torch.cuda.is_available(), "需要 CUDA（Turing fp16 验证）"
    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    dtype = pb.resolve_compute_dtype(device)
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"[0] GPU: {props.name}（sm_{props.major}{props.minor}，{props.total_memory/1024**3:.0f}GB，"
          f"剩余 {free_gb:.1f}GB）；compute_dtype={dtype}")
    print(f"[0] model_dir: {args.model_dir}")

    # ---- 1. 共享组件（VL 放置走引擎 D4 决策）----
    t0 = time.perf_counter()
    comps = pb.load_shared_components(args.model_dir, device, dtype)
    print(f"[1] 共享组件加载完成（{time.perf_counter()-t0:.1f}s；VL 放置={comps.vl_device}）")

    # ---- 2. edit 槽 transformer（Q8 单档）----
    t0 = time.perf_counter()
    transformer = pb.load_transformer(args.model_dir, "edit", "gguf_q8", dtype)
    print(f"[2] Edit-2509 Q8 transformer 加载完成（{time.perf_counter()-t0:.1f}s）")

    # ---- 3. 组装 EditPlus 管线 ----
    pipe = pb.build_edit_pipeline(comps, transformer, device)
    print("[3] QwenImageEditPlusPipeline 组装完成")

    all_ok = True

    def run_scene(name: str, images: list[Image.Image], prompt: str, seed: int = 2026) -> Image.Image | None:
        nonlocal all_ok
        print(f"\n[{name}] prompt: {prompt!r}（{len(images)} 图，{args.steps} 步）")
        t0 = time.perf_counter()
        try:
            out = pipe(
                prompt=prompt,
                image=images,
                num_inference_steps=args.steps,
                true_cfg_scale=4.0,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ 推理异常：{type(e).__name__}: {e}")
            all_ok = False
            return None
        elapsed = time.perf_counter() - t0
        out_path = OUT_DIR / f"editplus_{name}.png"
        out.save(out_path)
        print(f"    耗时 {elapsed:.1f}s → {out_path}")
        if not _check_output(out, name):
            all_ok = False
        return out

    girl = Image.open(args.image1).convert("RGB") if args.image1 else _make_placeholder("girl")
    cat = Image.open(args.image2).convert("RGB") if args.image2 else _make_placeholder("cat")

    # ---- 场景 1（T3.1）：单图指令编辑 ----
    if not args.skip_scene1:
        run_scene(
            "scene1_edit",
            [girl],
            "Change the background of this image to a sunny beach with palm trees.",
        )

    # ---- 场景 2（T3.2）：跨图交互（核心验收：女孩抱猫）----
    if not args.skip_scene2:
        run_scene(
            "scene2_cross",
            [girl, cat],
            "The girl in Picture 1 holding the cat in Picture 2.",
        )

    # ---- 场景 3（T4.1）：视角控制 ----
    if args.view:
        views = [
            ("view_right90", f"{_VIEW_HORIZONTAL['right90']}"),
            ("view_up30", f"{_VIEW_VERTICAL['up30']}"),
            ("view_close", f"{_VIEW_DISTANCE['close']}"),
        ]
        for name, prefix in views:
            run_scene(
                name,
                [girl],
                f"Show the person in the image, {prefix}.",
            )

    # ---- 显存快照 ----
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[显存] 峰值分配：{alloc:.1f}GB")

    print("\n" + ("✅ M1 冒烟全部通过" if all_ok else "❌ 存在失败场景（见上）"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
