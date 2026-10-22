"""T4.3 冒烟测试：QwenImageEngine 骨架（不加载模型）。

验证点：
    1. 构造成功（CustomEngine 继承链 + manifest default_variant 读取）
    2. _capabilities_impl：本地 fallback 扫描（无 AIRELIABLE_CORE_URL）→ 三档 variant 结构
    3. _status_impl：idle 态 + extra 字段
    4. _load_lora_impl 未加载时抛 EngineLoadError
    5. get_history 分页结构
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.qwen_image_5e8f3a21.config import EngineConfig
from engines.qwen_image_5e8f3a21.engine import QwenImageEngine


async def main() -> None:
    # --models-root 参数：用真实模型目录验证 capabilities 扫描（真实下载状态）
    if len(sys.argv) > 2 and sys.argv[1] == "--models-root":
        root = Path(sys.argv[2]).resolve()
        config = EngineConfig(
            models_root=root / "models",
            project_root=root,
            data_root=root / "data",
            output_root=root / "output",
        )
    else:
        config = EngineConfig()
    engine = QwenImageEngine(config)
    print(f"[1] 构造 OK：engine_id={engine.engine_id} default_variant={engine.default_variant}")
    assert engine.default_variant == "FLUX/Kontext-GGUF-Q8"

    caps = await engine._capabilities_impl()
    print(f"[2] capabilities OK：{len(caps.variants)} 个 variant，loras={caps.loras}")
    for v in caps.variants:
        print(f"    - {v.name}: precision={v.precision} available={v.available}")
    assert len(caps.variants) == 3
    assert caps.default_variant == "FLUX/Kontext-GGUF-Q8"

    status = await engine._status_impl()
    print(f"[3] status OK：state={status.state} vram={status.vram_used} extra_keys={sorted(status.extra)}")
    assert status.state == "idle"
    assert status.extra["precision"] is None

    try:
        await engine._load_lora_impl("test_lora", 1.0)
        raise AssertionError("[4] 应当抛 EngineLoadError")
    except Exception as e:
        assert type(e).__name__ == "EngineLoadError", f"异常类型错误: {type(e).__name__}"
        print(f"[4] 未加载时挂 LoRA 正确抛错：{e}")

    hist = engine.get_history(page=1, page_size=20)
    print(f"[5] history OK：total={hist['total']} page={hist['page']} items={len(hist['items'])}")
    assert hist["total"] == 0

    print("\n全部冒烟测试通过")


if __name__ == "__main__":
    asyncio.run(main())
