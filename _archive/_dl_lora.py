"""T6.1 准备：下载测试用 FLUX LoRA（hf-mirror，diffusers 键格式）。

下载 2 个公开 FLUX dev LoRA 到 lora_dir：
  1. shakker-logo   ← Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design lora.safetensors
  2. shakker-poster ← Shakker-Labs/FLUX.1-dev-LoRA-Movie-Poster lora.safetensors
（Shakker-Labs 的 LoRA 均为 diffusers transformer.* 键格式，README 官方 diffusers 演示）
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download

LORA_DIR = Path(r"D:\myProInstallDir\AIReliable\models\image\flux-kontext\loras")
LORA_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    ("Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design", "FLUX-dev-lora-Logo-Design.safetensors", "shakker-logo.safetensors"),
    ("alvdansen/flux-koda", "araminta_k_flux_koda.safetensors", "flux-koda.safetensors"),
]

for repo, filename, local_name in FILES:
    dest = LORA_DIR / local_name
    if dest.exists() and dest.stat().st_size > 10 * 1024 * 1024:
        print(f"[skip] {local_name} 已存在（{dest.stat().st_size / 1e6:.0f}MB）")
        continue
    print(f"[download] {repo}/{filename} -> {dest}")
    p = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(LORA_DIR),
    )
    src = Path(p)
    if src != dest:
        src.rename(dest)
    print(f"[ok] {local_name}（{dest.stat().st_size / 1e6:.0f}MB）")

print("完成")
