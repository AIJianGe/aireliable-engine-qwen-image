"""TransformerManager 设备策略（gpu+cpu 溢出）逻辑单测。

不加载真实模型：mock load_transformer / torch.cuda，验证放置决策与
ensure 换槽的内存搬移路径。运行：
    cd envs\image-qwen-image
    uv run --no-sync python -m engines.qwen_image_5e8f3a21._test_device_policy
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

from .transformer_manager import TransformerManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 显存模拟参数（MB）
SIM_TOTAL_MB = 48 * 1024
SIM_INITIAL_FREE_MB = 46 * 1024  # 共享组件等占用后的剩余
SLOT_USAGE_MB = 24 * 1024  # 模拟 BF16 单槽显存占用

state = {"free": SIM_INITIAL_FREE_MB}


class FakeTransformer:
    """记录 to() 搬移轨迹的假 transformer。"""

    def __init__(self, side: str):
        self.side = side
        self.device = "cpu"
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(f"{self.device}->{device}")
        self.device = device
        # 模拟显存增减（cuda 占用 / cpu 释放）
        if device != "cpu":
            state["free"] -= SLOT_USAGE_MB
        else:
            state["free"] += SLOT_USAGE_MB
        return self


def fake_load_transformer(model_dir, side, precision, dtype):
    return FakeTransformer(side)


def fake_mem_get_info():
    return state["free"] * 1024**2, SIM_TOTAL_MB * 1024**2


def fake_memory_allocated():
    return (SIM_TOTAL_MB - SIM_INITIAL_FREE_MB + (SIM_TOTAL_MB - state["free"])) * 1024**2 // 2


def make_tm(policy: str) -> TransformerManager:
    tm = TransformerManager(
        model_dir=PROJECT_ROOT / "models" / "image" / "flux-kontext",
        device="cuda",
        dtype=torch.float16,
        device_policy=policy,
    )
    return tm


def fake_available_sides(self, precision):
    """文件在盘模拟：三档两槽全在盘。"""
    return ["t2i", "kontext"]


def run_test(name: str, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e!r}")
        sys.exit(1)


def _patches():
    return [
        patch("engines.qwen_image_5e8f3a21.transformer_manager.load_transformer", fake_load_transformer),
        patch.object(torch.cuda, "mem_get_info", staticmethod(fake_mem_get_info)),
        patch.object(torch.cuda, "memory_allocated", staticmethod(fake_memory_allocated)),
        patch.object(TransformerManager, "available_sides_for", fake_available_sides),
    ]


def test_dual_fit():
    """显存够双槽 → 双槽 GPU 常驻。"""
    global state
    state = {"free": 46 * 1024}
    with _patches()[0], _patches()[1], _patches()[2], _patches()[3]:
        tm = make_tm("gpu+cpu")
        tm.load_variant("gguf_q8")
        assert tm._placement == {"kontext": "gpu", "t2i": "gpu"}, tm._placement
        assert tm.loaded_sides == ["kontext", "t2i"]


def test_single_fit():
    """显存仅够单槽 → kontext GPU 常驻 + t2i CPU 驻留；ensure 换槽内存搬移。"""
    global state, SLOT_USAGE_MB
    state = {"free": 13 * 1024}  # 13GB：够一个 Q8 槽
    SLOT_USAGE_MB = 11 * 1024  # Q8 单槽 11GB
    try:
        with _patches()[0], _patches()[1], _patches()[2], _patches()[3]:
            tm = make_tm("gpu+cpu")
            tm.load_variant("gguf_q8")
            # 13GB 剩余，加载 kontext 11GB 后剩 2GB < 11*1.1 → t2i 应 CPU 驻留
            assert tm._placement == {"kontext": "gpu", "t2i": "cpu"}, tm._placement

            # ensure kontext（已在 GPU）→ 直接返回，不搬移
            tf = tm.ensure("kontext", None)
            assert tf.device == "cuda"
            assert tm._placement == {"kontext": "gpu", "t2i": "cpu"}

            # ensure t2i（CPU 驻留）→ 换入 GPU，kontext 换出 CPU（内存搬移）
            tf2 = tm.ensure("t2i", None)
            assert tf2.device == "cuda", tf2.moves
            assert tm._placement == {"kontext": "cpu", "t2i": "gpu"}, tm._placement
            assert tf2.moves == ["cpu->cpu", "cpu->cuda"], tf2.moves  # 加载后 to(cpu)，换入 to(cuda)
            assert tm._transformers["kontext"].moves == ["cpu->cuda", "cuda->cpu"]
    finally:
        SLOT_USAGE_MB = 24 * 1024


def test_auto_policy_unchanged():
    """auto 策略 → 现状行为（全 GPU；BF16 单驻）。"""
    global state
    state = {"free": 46 * 1024}
    with _patches()[0], _patches()[1], _patches()[2], _patches()[3]:
        tm = make_tm("auto")
        tm.load_variant("gguf_q8")
        assert tm._placement == {"t2i": "gpu", "kontext": "gpu"}, tm._placement
        # BF16：auto 策略下单驻（只加载第一个在盘槽位）
        tm.load_variant("bf16")
        assert len(tm.loaded_sides) == 1, tm.loaded_sides
        # 档位切换后 LoRA 重挂通知（on_swap）触发过
        tm.load_variant("gguf_q8")


def test_insufficient():
    """连单槽都不够 → RuntimeError 带指引。"""
    global state, SLOT_USAGE_MB
    state = {"free": 2 * 1024}  # 2GB：连 Q8 单槽都放不下
    SLOT_USAGE_MB = 11 * 1024
    try:
        class OOMTransformer(FakeTransformer):
            def to(self, device):
                if device != "cpu":
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                return self

        def oom_loader(model_dir, side, precision, dtype):
            return OOMTransformer(side)

        with _patches()[1], _patches()[2], _patches()[3], \
             patch("engines.qwen_image_5e8f3a21.transformer_manager.load_transformer", oom_loader):
            tm = make_tm("gpu+cpu")
            try:
                tm.load_variant("gguf_q8")
                raise AssertionError("应抛 RuntimeError")
            except RuntimeError as e:
                assert "Q8" in str(e) and "gpu+cpu" in str(e), str(e)
    finally:
        SLOT_USAGE_MB = 24 * 1024


def test_status_dict():
    """status_dict 暴露 placement / device_policy / vram_free_mb。"""
    global state, SLOT_USAGE_MB
    state = {"free": 46 * 1024}
    SLOT_USAGE_MB = 11 * 1024  # 与 memory_allocated 模拟的实测值（~12GB）一致
    try:
        with _patches()[0], _patches()[1], _patches()[2], _patches()[3]:
            tm = make_tm("gpu+cpu")
            tm.load_variant("gguf_q8")
            d = tm.status_dict()
            assert d["placement"] == {"kontext": "gpu", "t2i": "gpu"}, d
            assert d["device_policy"] == "gpu+cpu"
            assert d["vram_free_mb"] > 0
    finally:
        SLOT_USAGE_MB = 24 * 1024


if __name__ == "__main__":
    run_test("双槽显存充足 → 双 GPU 常驻", test_dual_fit)
    run_test("显存仅够单槽 → 单 GPU + CPU 驻留 + ensure 换槽搬移", test_single_fit)
    run_test("auto 策略行为不变（全 GPU / BF16 单驻）", test_auto_policy_unchanged)
    run_test("连单槽都不够 → RuntimeError 指引", test_insufficient)
    run_test("status_dict 暴露放置与策略", test_status_dict)
    print("\n全部通过 ✓")
