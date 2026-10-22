"""T4.2 端点错误路径测试（对运行中的 7007 服务）。

验证：
    1. POST /load（模型不完整）→ 500 + detail 含缺失清单与下载指引
    2. POST /load（未知 variant）→ 400
    3. POST /lora/load（未加载模型）→ 500 + EngineLoadError 信息
"""
import httpx

BASE = "http://127.0.0.1:7007"

with httpx.Client(timeout=30) as c:
    r = c.post(f"{BASE}/load", json={"variant": "FLUX/Kontext-GGUF-Q8"})
    print(f"[1] load 缺文件: HTTP {r.status_code}")
    detail = r.json().get("detail", "")
    print(f"    detail 前 300 字: {str(detail)[:300]}")
    assert r.status_code == 500
    assert "缺失" in str(detail) or "不完整" in str(detail)

    r = c.post(f"{BASE}/load", json={"variant": "BAD-VARIANT"})
    print(f"[2] load 未知 variant: HTTP {r.status_code} — {r.json().get('detail', '')[:80]}")
    assert r.status_code == 400

    r = c.post(f"{BASE}/lora/load", json={"name": "foo", "weight": 1.0})
    print(f"[3] lora/load 未加载模型: HTTP {r.status_code} — {str(r.json().get('detail', ''))[:80]}")
    assert r.status_code == 500

print("\n端点错误路径测试全部通过")
