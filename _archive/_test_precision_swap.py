"""T2.4 验证：精度档热切换（对运行中的 7007 服务）。

序列：
    1. edit 推理（默认当前档 gguf_q8）→ 基准
    2. edit 推理（precision=fp8）→ 触发热切换 Q8→FP8
    3. edit 推理（precision=gguf_q8）→ 切回 Q8
    4. /status 确认最终档位与显存

判定：三步推理全部出图，切换无崩溃、显存无泄漏式增长。
"""
import json

import httpx

BASE = "http://127.0.0.1:7007"


def infer(precision: str | None, task_id: str) -> dict:
    inputs = {
        "prompt": "change the background to a sunny beach, keep the composition",
        "mode": "edit",
        "width": 512,
        "height": 512,
        "num_inference_steps": 6,
        "guidance_scale": 2.5,
        "seed": 55,
    }
    if precision:
        inputs["precision"] = precision
    payload = {
        "inputs": inputs,
        "images": [{"path": "data/inputs/test_edit_512.png"}],
        "task_id": task_id,
    }
    final = {}
    with httpx.Client(timeout=1800) as c:
        with c.stream("POST", f"{BASE}/infer", json=payload) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    f = json.loads(line[5:])
                    if f.get("stage") in ("completed", "error"):
                        final = f
    return final


with httpx.Client(timeout=30) as c:
    s = c.get(f"{BASE}/status").json()
    print(f"[0] 初始：precision={s['extra']['precision']} vram={s['vram_used']}")

for tag, prec in [("Q8-基准", None), ("FP8-热切换", "fp8"), ("Q8-切回", "gguf_q8")]:
    f = infer(prec, f"test_swap_{tag}")
    res = f.get("result") or {}
    if f.get("stage") == "completed":
        print(f"[{tag}] ✅ {res.get('duration_seconds')}s precision={res.get('precision')}")
    else:
        print(f"[{tag}] ❌ {json.dumps(f, ensure_ascii=False)[:200]}")
    with httpx.Client(timeout=30) as c:
        s = c.get(f"{BASE}/status").json()
        print(f"    status: precision={s['extra']['precision']} vram={s['vram_used']} sides={s['extra']['loaded_sides']}")
