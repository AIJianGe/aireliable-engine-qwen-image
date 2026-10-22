"""T2.2 Turing 三档精度稳定性与性能实测。

验证目标（分析 §6 风险 2）：Quadro RTX 8000（sm_75）无原生 BF16/FP8 硬件，
三档精度分别出图（512 与 1024），记录耗时 / 显存峰值 / NaN。

用法（precisions 可选子集，默认全部）：
  uv run --no-sync --project envs/image-qwen-image \
      python engines/qwen_image_5e8f3a21/_verify_precision.py [gguf] [fp8] [bf16]

注意：
- GGUF 档同时实测 compute_dtype ∈ {bf16, fp16}（Turing bf16 软件回退风险
  可用 GGUF 权重验证，无需先下 24GB BF16 文件）
- BF16 档需下载 ~24GB，默认放最后
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np
import torch
from PIL import Image

from _verify_common import (
    OUTPUT_DIR,
    SHARED_KEYS,
    assemble_kontext_pipeline,
    banner,
    ensure_kontext_bf16,
    ensure_models,
    free_vram_cache,
    load_bf16_transformer,
    load_fp8_transformer,
    load_gguf_transformer,
    report,
    resolve_device,
    vram_mb,
    vram_reserved_mb,
)

RESULTS: list[dict] = []


def run_case(tag: str, pipe, *, size: int, steps: int = 10) -> None:
    """单次出图 + 记录。"""
    device = next(pipe.transformer.parameters()).device
    test_img = Image.new("RGB", (size, size), color=(90, 60, 30))
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t0 = time.time()
    try:
        result = pipe(
            image=test_img,
            prompt="a red apple on a wooden table, studio lighting",
            height=size,
            width=size,
            num_inference_steps=steps,
            guidance_scale=2.5,
        )
        elapsed = time.time() - t0
        arr = np.asarray(result.images[0], dtype=np.float32)
        out = OUTPUT_DIR / f"_verify_t2_2_{tag}_{size}.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0
        row = {
            "档位": tag,
            "尺寸": f"{size}x{size}",
            "步数": steps,
            "耗时s": round(elapsed, 1),
            "峰值MB": round(peak),
            "NaN": bool(np.isnan(arr).any()),
            "均值": round(float(arr.mean()), 1),
        }
        RESULTS.append(row)
        print(f"[case] {row} -> {out.name}")
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"档位": tag, "尺寸": f"{size}x{size}", "错误": repr(exc)[:200]})
        print(f"[case] {tag} {size}px 失败：{exc!r}")


def cleanup(pipe=None) -> None:
    if pipe is not None:
        for attr in ("transformer", "vae", "text_encoder", "text_encoder_2"):
            setattr(pipe, attr, None)
    gc.collect()
    free_vram_cache()


def main() -> int:
    banner("T2.2 Turing 三档精度稳定性实测")
    device = resolve_device()
    if device != "cuda":
        print("⚠️ 无 CUDA，仅 CPU 路径参考（性能数据无意义）")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"bf16_supported={torch.cuda.is_bf16_supported() if device == 'cuda' else 'N/A'}")

    wanted = set(sys.argv[1:]) or {"gguf", "fp8", "bf16"}
    paths = ensure_models(["kontext_q8", *SHARED_KEYS])

    # ---- GGUF Q8（compute_dtype 双测：bf16 vs fp16）----
    if "gguf" in wanted:
        for cd_name, cd in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            try:
                t0 = time.time()
                transformer = load_gguf_transformer(paths["kontext_q8"], cd)
                pipe = assemble_kontext_pipeline(transformer, dtype=cd).to(device)
                report(f"GGUF Q8({cd_name}) 加载+组装", t0)
                run_case(f"gguf-q8-cd{cd_name}", pipe, size=512, steps=10)
                run_case(f"gguf-q8-cd{cd_name}", pipe, size=1024, steps=10)
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] GGUF({cd_name})：{exc!r}")
                RESULTS.append({"档位": f"gguf-q8-cd{cd_name}", "错误": repr(exc)[:200]})
            finally:
                cleanup(pipe if 'pipe' in dir() else None)
                pipe = None

    # ---- FP8 scaled（上抛 compute_dtype 路线）----
    if "fp8" in wanted:
        pipe = None
        try:
            paths = ensure_models(["kontext_fp8"])
            cd = torch.float16
            t0 = time.time()
            transformer = load_fp8_transformer(paths["kontext_fp8"], cd)
            pipe = assemble_kontext_pipeline(transformer, dtype=cd).to(device)
            report("FP8(上抛fp16) 加载+组装", t0)
            run_case("fp8-upcast-fp16", pipe, size=512, steps=10)
            run_case("fp8-upcast-fp16", pipe, size=1024, steps=10)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] FP8：{exc!r}")
            RESULTS.append({"档位": "fp8", "错误": repr(exc)[:200]})
        finally:
            cleanup(pipe)
            pipe = None

    # ---- BF16 ----
    if "bf16" in wanted:
        pipe = None
        try:
            bf16_path, form = ensure_kontext_bf16()
            t0 = time.time()
            transformer = load_bf16_transformer(bf16_path, form, torch.bfloat16)
            pipe = assemble_kontext_pipeline(transformer, dtype=torch.bfloat16).to(device)
            report(f"BF16({form}) 加载+组装", t0)
            run_case("bf16", pipe, size=512, steps=10)
            run_case("bf16", pipe, size=1024, steps=10)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] BF16：{exc!r}")
            RESULTS.append({"档位": "bf16", "错误": repr(exc)[:200]})
        finally:
            cleanup(pipe)
            pipe = None

    banner("T2.2 结果汇总")
    for row in RESULTS:
        print(row)
    print(f"（记录完毕，共 {len(RESULTS)} 条；显存当前 allocated={vram_mb():.0f}MB "
          f"reserved={vram_reserved_mb():.0f}MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
