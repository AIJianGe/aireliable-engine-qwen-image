"""列出 ModelScope 上 Kontext 仓库文件结构并测试匿名下载（临时脚本）。"""

import httpx

MS = "https://modelscope.cn"


def list_files(repo: str) -> list[str]:
    """列出仓库文件（递归）。"""
    r = httpx.get(f"{MS}/api/v1/models/{repo}/repo/files", params={"Recursive": "true"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    files = []
    for f in data.get("Data", {}).get("Files", []):
        if f.get("Type") == "blob":
            files.append(f["Path"])
    return files


for repo in ["black-forest-labs/FLUX.1-Kontext-dev", "QuantStack/FLUX.1-Kontext-dev-GGUF", "city96/FLUX.1-dev-gguf", "comfyanonymous/flux_text_encoders"]:
    try:
        files = list_files(repo)
        print(f"== {repo}（{len(files)} 个文件）==")
        for f in files[:40]:
            print("   ", f)
        if len(files) > 40:
            print(f"    ...（还有 {len(files) - 40} 个）")
    except Exception as e:
        print(f"== {repo} == ERROR: {e!r}")

# 匿名下载小文件测试
try:
    url = f"{MS}/models/black-forest-labs/FLUX.1-Kontext-dev/resolve/master/scheduler/scheduler_config.json"
    r = httpx.get(url, follow_redirects=True, timeout=20)
    print("\n[scheduler_config 下载测试]", r.status_code)
    print(r.text[:400])
except Exception as e:
    print("\n[scheduler_config 下载测试] ERROR:", e)
