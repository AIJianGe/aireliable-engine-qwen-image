"""FLUX Kontext 三模式功能测试（HTTP SSE 直连 7007）。

流程：
1. POST /load（Q8 默认档）
2. POST /infer × 3 模式（t2i / edit / i2i）
3. 每模式：保存 result 帧图片 + 灰阶 extrema 校验 + size>10KB

输出目录：output/image/_functionaltest/
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ENGINE_URL = "http://127.0.0.1:7007"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "image" / "_functionaltest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY: list[dict] = []


def parse_sse_frames(text_iter) -> list[dict]:
    frames = []
    for line in text_iter:
        if not line:
            continue
        line = line.strip()
        if line.startswith("data:"):
            try:
                frames.append(json.loads(line[5:]))
            except json.JSONDecodeError:
                pass
    return frames


def save_check(case: str, b64img: str, fname: str) -> dict:
    buf = base64.b64decode(b64img)
    pil = Image.open(io.BytesIO(buf)).convert("RGB")
    out = OUT_DIR / fname
    pil.save(out)
    extrema = pil.convert("L").getextrema()
    size_kb = out.stat().st_size / 1024
    status = "PASS" if (extrema[1] > 200 and size_kb > 10) else "FAIL"
    info = {
        "case": case, "output": str(out),
        "size_wh": [pil.width, pil.height],
        "extrema_L": list(extrema), "size_kb": round(size_kb, 1),
        "status": status,
    }
    print(f"\n=== [{status}] {case} ===")
    for k, v in info.items():
        if k != "case":
            print(f"  {k}: {v}")
    SUMMARY.append(info)
    return out


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    # 1. health
    try:
        r = httpx.get(f"{ENGINE_URL}/health", timeout=10)
        print(f"[health] {r.status_code}: {r.json()}")
    except Exception as e:
        print(f"[health] FAIL: {e}")
        sys.exit(1)

    # 2. load（若已 ready 则会秒回）
    t0 = time.time()
    with httpx.Client(timeout=900) as c:
        r = c.post(f"{ENGINE_URL}/load", json={"variant": "FLUX/Kontext-GGUF-Q8"})
        print(f"[load] {r.status_code} ({time.time()-t0:.1f}s): {r.text[:300]}")
        if r.status_code != 200:
            sys.exit(2)

    # ====== CASE 1: t2i 文生图 ======
    print("\n>>> CASE 1: t2i 文生图（金毛犬）")
    payload = {
        "variant": "FLUX/Kontext-GGUF-Q8",
        "inputs": {
            "prompt": "A golden retriever dog running on a green park lawn, sunny day, photorealistic, ultra detailed, 8k",
            "mode": "t2i", "width": 1024, "height": 1024,
            "num_inference_steps": 28, "guidance_scale": 3.5, "seed": 42,
        },
        "images": [],
        "task_id": f"t2i_{int(time.time())}",
    }
    with httpx.Client(timeout=1800) as c:
        with c.stream("POST", f"{ENGINE_URL}/infer", json=payload) as r:
            r.raise_for_status()
            frames = parse_sse_frames(r.iter_lines())
    print(f"  frames: {len(frames)}")
    result = next((f for f in reversed(frames) if f.get("type") == "result"), None)
    if not result or "image" not in result:
        print(f"  FAIL: no result frame. last frame keys={frames[-1].keys() if frames else None}")
        print(f"  last frame preview: {str(frames[-1])[:300] if frames else None}")
        sys.exit(3)
    dog_path = save_check("T2I 文生图-金毛犬", result["image"], "01_t2i_golden.png")

    # ====== CASE 2: edit 单图编辑（戴红帽）======
    print("\n>>> CASE 2: edit 单图指令编辑（戴红棒球帽）")
    dog_b64 = _img_b64(dog_path)
    payload = {
        "variant": "FLUX/Kontext-GGUF-Q8",
        "inputs": {
            "prompt": "Put a red baseball cap on the dog's head. Keep everything else unchanged. Photorealistic, seamless edit.",
            "mode": "edit", "strength": 0.85,
            "num_inference_steps": 28, "guidance_scale": 3.5, "seed": 43,
        },
        "images": [{"b64_json": dog_b64}],
        "task_id": f"edit_{int(time.time())}",
    }
    with httpx.Client(timeout=1800) as c:
        with c.stream("POST", f"{ENGINE_URL}/infer", json=payload) as r:
            r.raise_for_status()
            frames = parse_sse_frames(r.iter_lines())
    print(f"  frames: {len(frames)}")
    result = next((f for f in reversed(frames) if f.get("type") == "result"), None)
    if not result or "image" not in result:
        print("  FAIL: no result frame")
        sys.exit(4)
    save_check("EDIT 戴红棒球帽", result["image"], "02_edit_cap.png")

    # ====== CASE 3: i2i 多图参考（女孩抱猫）======
    print("\n>>> CASE 3: i2i 多图参考（女孩抱猫组合）")
    ref1 = PROJECT_ROOT / "data" / "inputs" / "ref1.png"
    ref2 = PROJECT_ROOT / "data" / "inputs" / "ref2.png"
    if ref1.exists() and ref2.exists():
        girl_path, cat_path = ref1, ref2
        print(f"  复用 data/inputs 参考图: ref1={ref1.name} ref2={ref2.name}")
    else:
        # 无参考图：先 t2i 造两张
        print("  data/inputs/ref1/ref2 不存在，先 t2i 造两张参考图...")
        # 女孩
        p = {
            "variant": "FLUX/Kontext-GGUF-Q8",
            "inputs": {"prompt": "Portrait of a young asian girl, upper body, cozy room background, studio lighting, photorealistic",
                       "mode": "t2i", "width": 896, "height": 1152,
                       "num_inference_steps": 24, "guidance_scale": 3.5, "seed": 100},
            "images": [], "task_id": f"ref_girl_{int(time.time())}",
        }
        with httpx.Client(timeout=1800) as c:
            with c.stream("POST", f"{ENGINE_URL}/infer", json=p) as r:
                frames = parse_sse_frames(r.iter_lines())
        res = next((f for f in reversed(frames) if f.get("type") == "result"), None)
        girl_path = save_check("I2I 参考-女孩图", res["image"], "03a_ref_girl.png")
        # 猫
        p["inputs"] = {"prompt": "A cute orange tabby cat sitting on a pillow, close-up, natural light, photorealistic",
                       "mode": "t2i", "width": 896, "height": 896,
                       "num_inference_steps": 24, "guidance_scale": 3.5, "seed": 101}
        p["task_id"] = f"ref_cat_{int(time.time())}"
        with httpx.Client(timeout=1800) as c:
            with c.stream("POST", f"{ENGINE_URL}/infer", json=p) as r:
                frames = parse_sse_frames(r.iter_lines())
        res = next((f for f in reversed(frames) if f.get("type") == "result"), None)
        cat_path = save_check("I2I 参考-猫图", res["image"], "03b_ref_cat.png")

    girl_b64 = _img_b64(girl_path)
    cat_b64 = _img_b64(cat_path)
    payload = {
        "variant": "FLUX/Kontext-GGUF-Q8",
        "inputs": {
            "prompt": (
                "The girl in Picture 1 holding the cat from Picture 2 in her arms. "
                "Girl gently embraces the cat. Cozy indoor background, natural lighting, "
                "photorealistic, 8k quality."
            ),
            "mode": "i2i", "width": 1024, "height": 1024,
            "num_inference_steps": 28, "guidance_scale": 3.5, "seed": 50,
            "strength": 1.0,
        },
        "images": [{"b64_json": girl_b64}, {"b64_json": cat_b64}],
        "task_id": f"i2i_{int(time.time())}",
    }
    with httpx.Client(timeout=1800) as c:
        with c.stream("POST", f"{ENGINE_URL}/infer", json=payload) as r:
            r.raise_for_status()
            frames = parse_sse_frames(r.iter_lines())
    print(f"  frames: {len(frames)}")
    result = next((f for f in reversed(frames) if f.get("type") == "result"), None)
    if not result or "image" not in result:
        # 诊断：打印最后几帧
        for f in frames[-5:]:
            print(f"  frame debug: type={f.get('type')} keys={sorted(f.keys())[:10]} preview={str(f)[:200]}")
        sys.exit(5)
    save_check("I2I 多图参考-女孩抱猫", result["image"], "03_i2i_girl_hold_cat.png")

    # ====== SUMMARY ======
    print("\n" + "=" * 70)
    print("FLUX KONTEXT 三模式功能测试汇总")
    print("=" * 70)
    print(f"{'Case':<32} {'Status':<6} {'Size(KB)':<10} {'Extrema L':<16} {'WH':<14}")
    print("-" * 70)
    for r in SUMMARY:
        wh = f"{r['size_wh'][0]}x{r['size_wh'][1]}"
        print(f"{r['case']:<32} {r['status']:<6} {r['size_kb']:<10} "
              f"{str(r['extrema_L']):<16} {wh:<14}")
    n_pass = sum(1 for r in SUMMARY if r["status"] == "PASS")
    n_total = len(SUMMARY)
    print("-" * 70)
    print(f"PASS: {n_pass}/{n_total}")
    (OUT_DIR / "_summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整输出目录: {OUT_DIR}")
    print(f"汇总 JSON: {OUT_DIR / '_summary.json'}")
    sys.exit(0 if n_pass == n_total else 10)


if __name__ == "__main__":
    main()
