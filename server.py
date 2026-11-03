"""Qwen-Image 图像引擎 HTTP 服务（引擎 ID qwen_image_5e8f3a21）。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件只保留引擎特有路由。

    标准端点（骨架）：
        GET  /health  /capabilities  /status
        POST /load  /unload  /infer(SSE)  /lora/load  /lora/unload

    扩展端点：
        GET  /history        生成历史分页（WebUI 结果区）
        GET|POST /log_level  日志级别动态调整（骨架）
        GET  /output/*       输出图片静态服务
        GET  /ui/*           WebUI（四区布局）
        GET  /               307 → /ui/

启动方式（平台 EngineManager）：
    python -m engines.qwen_image_5e8f3a21.server --port 7007
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException

from aireliable_engine import build_engine_app, run_engine_server

from .config import EngineConfig, parse_args
from .engine import QwenImageEngine

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).parent / "webui"


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[QwenImageEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> QwenImageEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: QwenImageEngine) -> None:
    """启动钩子：绑定全局实例。"""
    global _engine

    _engine = engine
    logger.info(
        "引擎实例化完成: state=%s default_variant=%s",
        engine.state_machine.state.value, engine.default_variant,
    )


async def _on_shutdown(engine: QwenImageEngine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== Qwen-Image 图像引擎关闭 =====")
    try:
        if engine.tm is not None:
            try:
                await engine._unload_impl()
            except Exception as e:  # noqa: BLE001
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


@router.get("/history")
async def history(page: int = 1, page_size: int = 20) -> dict:
    """生成历史（最新在前，gateway 任务中心代理视图）。"""
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="page ≥ 1 且 1 ≤ page_size ≤ 100")
    return await get_engine().get_history(page=page, page_size=page_size)


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        QwenImageEngine,
        config=config,
        title="Qwen-Image Engine",
        description=(
            "AIReliable 平台的 Qwen-Image 图像引擎"
            "（文生图 / 指令编辑含视角控制 / 多图参考图生图，GGUF Q8 单档）"
        ),
        version="0.2.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global app

    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
