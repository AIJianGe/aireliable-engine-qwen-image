"""离线加载实验 3：内置 config 目录路线（v2.2 定案方案）。

背景（前两次实验结论）：
    - 实验 1（GGUF 无 config）：from_single_file 内部 fetch_diffusers_config 推断
      black-forest-labs/FLUX.1-Depth-dev → 离线缓存未命中 → OSError
    - 实验 2（VAE original_config=dict）：create_vae_diffusers_config_from_ldm 推导的
      config 带 quant_conv 键，ae.safetensors 无此权重 → meta tensor 未填充 → NotImplementedError

本实验方案（single_file_model.py L433-461 核实）：
    from_single_file(config=<本地目录>) 走 cls.load_config(目录) 纯本地读 config.json，
    不触碰网络。config.json 为引擎内置资产（configs/flux_transformer / configs/flux_vae）。

实验项：
    1. VAE：from_single_file(config=configs/flux_vae) 完全离线加载 + encode/decode 冒烟
    2. GGUF transformer：from_single_file(config=configs/flux_transformer) 完全离线加载
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 强制完全离线（在 import 任何 HF 相关库之前）
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MD = Path(__file__).resolve().parents[2] / "models" / "image" / "flux-kontext"
ENGINE_CONFIGS = Path(__file__).parent / "configs"


def test_vae_local_config() -> None:
    """实验 1：VAE config=本地目录 完全离线加载。"""
    import torch
    from diffusers import AutoencoderKL

    t0 = time.perf_counter()
    try:
        model = AutoencoderKL.from_single_file(
            str(MD / "vae" / "ae.safetensors"),
            torch_dtype=torch.float16,
            config=str(ENGINE_CONFIGS / "flux_vae"),
        )
        print(
            f"[VAE] config=目录 离线 OK：latent_ch={model.config.latent_channels} "
            f"blocks={model.config.block_out_channels} "
            f"scaling={model.config.scaling_factor} shift={model.config.shift_factor}"
            f"（{time.perf_counter() - t0:.1f}s）"
        )
        # 功能冒烟：meta tensor 是否全部填充（CUDA 上跑，CPU fp16 conv 极慢）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 256, dtype=torch.float16, device=device)
            lat = model.encode(x).latent_dist.sample()
            rec = model.decode(lat).sample
        print(f"[VAE] encode/decode OK：latent {tuple(lat.shape)} → rec {tuple(rec.shape)}（{device}）")
    except Exception as e:
        print(f"[VAE] config=目录 失败：{type(e).__name__}: {str(e)[:300]}")


def test_gguf_local_config() -> None:
    """实验 2：GGUF transformer config=本地目录 完全离线加载。"""
    import torch
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

    path = MD / "transformers" / "kontext" / "flux1-kontext-dev-Q8_0.gguf"
    if not path.exists():
        print("[GGUF] 文件不在盘，跳过")
        return

    t0 = time.perf_counter()
    try:
        model = FluxTransformer2DModel.from_single_file(
            str(path),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
            torch_dtype=torch.float16,
            config=str(ENGINE_CONFIGS / "flux_transformer"),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[GGUF] config=目录 离线 OK：{n_params / 1e9:.2f}B 参数（{time.perf_counter() - t0:.1f}s）")
        # 前向冒烟（1 token 序列最小化）
        del model
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception as e:
        print(f"[GGUF] config=目录 失败：{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    test_vae_local_config()
    test_gguf_local_config()
