"""T5.x 推理冒烟：三模式 SSE 流（对运行中的 7007 服务）。

用法：
    uv run --no-sync --project envs/image-qwen-image \
        python engines/qwen_image_5e8f3a21/_test_infer.py [t2i|edit|i2i]

默认 edit。i2i 需准备 2 张输入图（data/inputs/ref1.png、ref2.png）。
"""
import json
import sys

import httpx

MODE = sys.argv[1] if len(sys.argv) > 1 else "edit"

payload = {
    "inputs": {
        "prompt": (
            "a cat holding a sign that says \"hello world\", detailed, high quality"
            if MODE == "t2i"
            else "change the background to a sunny beach, keep the composition"
        ),
        "mode": MODE,
        "width": 512,
        "height": 512,
        "num_inference_steps": 8,
        "guidance_scale": 2.5,
        "seed": 42,
    },
    "images": [{"path": "data/inputs/test_edit_512.png"}],
    "task_id": f"test_{MODE}_001",
}
if MODE == "i2i":
    payload["images"] = [
        {"path": "data/inputs/ref1.png"},
        {"path": "data/inputs/ref2.png"},
    ]

frames = []
with httpx.Client(timeout=1800) as c:
    with c.stream("POST", "http://127.0.0.1:7007/infer", json=payload) as r:
        print(f"[infer:{MODE}] HTTP", r.status_code)
        for line in r.iter_lines():
            if line.startswith("data:"):
                f = json.loads(line[5:])
                frames.append(f)
                stage = f.get("stage") or f.get("type") or f.get("status")
                print(f"  frame: type={f.get('type')} stage={stage} "
                      f"progress={f.get('progress')} detail={str(f.get('detail'))[:60]}")

print("总帧数:", len(frames))
final = frames[-1] if frames else {}
print("最终帧 keys:", sorted(final.keys()))
for k in ("image_urls", "image_url", "images", "error", "message"):
    if k in final:
        print(f"  {k}: {str(final[k])[:150]}")
if isinstance(final.get("result"), dict):
    print("  result:", json.dumps(final["result"], ensure_ascii=False)[:300])
