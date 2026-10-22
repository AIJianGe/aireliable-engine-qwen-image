"""诊断：ae.safetensors 键格式 vs convert_ldm_vae_checkpoint 转换 vs 模型期望键。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MD = Path(__file__).resolve().parents[2] / "models" / "image" / "flux-kontext"
ENGINE_CONFIGS = Path(__file__).parent / "configs"


def main() -> None:
    from safetensors.torch import load_file
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    # 1. 原始 checkpoint 键
    state = load_file(str(MD / "vae" / "ae.safetensors"))
    ckpt_keys = set(state.keys())
    print(f"[1] ae.safetensors 键数={len(ckpt_keys)}")
    sample = sorted(ckpt_keys)[:8]
    for k in sample:
        print(f"    {k}")
    prefixes = sorted({k.split(".")[0] for k in ckpt_keys})
    print(f"    顶层前缀：{prefixes}")

    # 2. 模型期望键（config 构建）
    model = AutoencoderKL.from_config(str(ENGINE_CONFIGS / "flux_vae" / "config.json"))
    model_keys = set(model.state_dict().keys())
    print(f"[2] 模型期望键数={len(model_keys)}")
    for k in sorted(model_keys)[:8]:
        print(f"    {k}")

    # 3. convert 转换
    import json
    cfg = json.loads((ENGINE_CONFIGS / "flux_vae" / "config.json").read_text(encoding="utf-8"))
    converted = convert_ldm_vae_checkpoint(config=cfg, checkpoint=state)
    conv_keys = set(converted.keys())
    print(f"[3] 转换后键数={len(conv_keys)}")
    for k in sorted(conv_keys)[:8]:
        print(f"    {k}")

    # 4. 匹配分析
    missing_in_conv = model_keys - conv_keys
    unexpected_conv = conv_keys - model_keys
    print(f"[4] 模型期望但转换结果缺失（前 10，共 {len(missing_in_conv)}）：")
    for k in sorted(missing_in_conv)[:10]:
        print(f"    {k}")
    print(f"    转换结果多余（前 10，共 {len(unexpected_conv)}）：")
    for k in sorted(unexpected_conv)[:10]:
        print(f"    {k}")

    # 5. 原始键与模型期望键直接匹配度（是否已是 diffusers 格式？）
    direct_match = model_keys & ckpt_keys
    print(f"[5] 原始键与模型期望键交集：{len(direct_match)}/{len(model_keys)}")


if __name__ == "__main__":
    main()
