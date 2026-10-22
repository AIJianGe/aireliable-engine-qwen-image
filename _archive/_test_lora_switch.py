"""T6.2 验证：LoRA 挂载下 transformer 槽位往返切换（对运行中的 7007 服务）。

序列：
    1. /lora/load flux-koda 0.8（挂到当前活跃 pipe——上次 t2i 推理 → dev 侧）
    2. /infer edit → 切换到 kontext transformer → reapply
    3. /infer t2i  → 切回 dev transformer → reapply（关键边界：adapter 已在 dev 上）
    4. /status → loaded_loras 应保持 ['flux-koda']

判定：
    - 步骤 3 后 /status 的 loaded_loras 仍含 flux-koda → 通过
    - 若日志出现"重挂失败（…already exists…）"且 loaded_loras 丢失 → 边界 bug 确认
"""
import json

import httpx

BASE = "http://127.0.0.1:7007"


def infer(mode: str, task_id: str, images: list | None = None) -> dict:
    payload = {
        "inputs": {
            "prompt": "a red apple on a wooden table, studio lighting",
            "mode": mode,
            "width": 512,
            "height": 512,
            "num_inference_steps": 6,
            "guidance_scale": 2.5,
            "seed": 33,
        },
        "task_id": task_id,
    }
    if images:
        payload["images"] = images
    final = {}
    with httpx.Client(timeout=600) as c:
        with c.stream("POST", f"{BASE}/infer", json=payload) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    f = json.loads(line[5:])
                    if f.get("stage") in ("completed", None) and f.get("type") != "progress":
                        final = f
            # 兜底：最后一帧
    return final


with httpx.Client(timeout=60) as c:
    # 0. 初始状态
    s = c.get(f"{BASE}/status").json()
    print(f"[0] state={s['state']} loaded_loras={s['loaded_loras']} "
          f"extra={s.get('extra', {}).get('loaded_sides')}")

    # 1. 挂 LoRA（挂到当前活跃 pipe）
    r = c.post(f"{BASE}/lora/load", json={"name": "flux-koda", "weight": 0.8})
    print(f"[1] lora/load: HTTP {r.status_code} -> {r.json()}")
    assert r.status_code == 200

    # 2. edit 推理（切到 kontext → reapply）
    f = infer("edit", "test_lorasw_edit", [{"path": "data/inputs/test_edit_512.png"}])
    res = f.get("result") or {}
    print(f"[2] infer edit: {res.get('duration_seconds')}s precision={res.get('precision')}")
    s = c.get(f"{BASE}/status").json()
    print(f"    status: loaded_loras={s['loaded_loras']}")

    # 3. t2i 推理（切回 dev → reapply，关键边界）
    f = infer("t2i", "test_lorasw_t2i")
    res = f.get("result") or {}
    print(f"[3] infer t2i: {res.get('duration_seconds')}s precision={res.get('precision')}")
    s = c.get(f"{BASE}/status").json()
    print(f"    status: loaded_loras={s['loaded_loras']}")

    ok = "flux-koda" in s["loaded_loras"]
    print(f"\n[判定] LoRA 跨槽位往返保持：{'✅ 通过' if ok else '❌ 丢失（边界 bug 确认）'}")

    # 4. 清理：卸载 LoRA
    r = c.post(f"{BASE}/lora/unload", json={"name": "flux-koda"})
    print(f"[4] lora/unload: HTTP {r.status_code} -> {r.json().get('loaded_loras')}")
