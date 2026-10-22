"""T6.x 验证：LoRA 挂载状态下推理（对运行中的 7007 服务）。"""
import json

import httpx

payload = {
    "inputs": {
        "prompt": "a vintage movie poster of a lighthouse at sunset",
        "mode": "edit",
        "width": 512,
        "height": 512,
        "num_inference_steps": 6,
        "guidance_scale": 2.5,
        "seed": 21,
    },
    "images": [{"path": "data/inputs/test_edit_512.png"}],
    "task_id": "test_lora_001",
}
frames = []
with httpx.Client(timeout=600) as c:
    with c.stream("POST", "http://127.0.0.1:7007/infer", json=payload) as r:
        print("[lora-infer] HTTP", r.status_code)
        for line in r.iter_lines():
            if line.startswith("data:"):
                f = json.loads(line[5:])
                frames.append(f)
                if f.get("stage") != "denoising" or f.get("progress", 0) in (0.1, 1.0):
                    print(f"  {f.get('stage')} p={f.get('progress')} {str(f.get('detail'))[:50]}")
final = frames[-1] if frames else {}
res = final.get("result") or final.get("error")
print("final:", json.dumps(res, ensure_ascii=False)[:250] if res else final)
