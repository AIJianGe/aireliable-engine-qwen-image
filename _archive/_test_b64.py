"""T5.3 验证：b64_json 输入来源（对运行中的 7007 服务）。"""
import base64
import json

import httpx

b64 = base64.b64encode(open(r"data\inputs\test_edit_512.png", "rb").read()).decode()
payload = {
    "inputs": {
        "prompt": "turn it into a pencil sketch",
        "mode": "edit",
        "width": 512,
        "height": 512,
        "num_inference_steps": 6,
        "guidance_scale": 2.5,
        "seed": 7,
    },
    "images": [{"b64_json": b64}],
    "task_id": "test_b64_001",
}
frames = []
with httpx.Client(timeout=600) as c:
    with c.stream("POST", "http://127.0.0.1:7007/infer", json=payload) as r:
        print("[b64] HTTP", r.status_code)
        for line in r.iter_lines():
            if line.startswith("data:"):
                f = json.loads(line[5:])
                frames.append(f)
                print(f"  {f.get('stage')} p={f.get('progress')}")
final = frames[-1] if frames else {}
res = final.get("result") or final.get("error")
print("final:", json.dumps(res, ensure_ascii=False)[:220] if res else final)
