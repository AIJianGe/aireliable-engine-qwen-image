"""Qwen-Image 图像引擎 HTTP 服务（FastAPI，引擎 ID qwen_image_5e8f3a21）。

8 个标准端点（SDK 契约）+ 引擎扩展端点：

    标准端点：
        - GET  /health         健康检查
        - GET  /capabilities   variant 能力声明（Q8 单档）+ LoRA 列表 + 输出尺寸
        - POST /load           加载共享组件 + 默认档 transformer
        - POST /unload         释放显存
        - POST /infer          三模式推理（SSE 流式：进度帧 + 结果帧）
        - GET  /status         引擎状态（档位/槽位/显存/LoRA）
        - POST /lora/load      挂载单个 LoRA
        - POST /lora/unload    卸载单个 LoRA

    扩展端点（WebUI / 平台对接用）：
        - GET  /history        生成历史分页（WebUI 结果区）
        - GET/POST /log_level  日志级别动态调整（SDK 提供）
        - GET  /output/*       输出图片静态服务（结果图 URL 直访）

启动方式（平台 EngineManager）：
    python -m engines.qwen_image_5e8f3a21.server --port 7007
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import QwenImageEngine

logger = logging.getLogger(__name__)

# ===== 全局引擎实例（FastAPI 启动时初始化，所有端点共享）=====

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


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时初始化引擎，关闭时释放模型。"""
    global _engine, _config

    if _config is None:
        _config = parse_args()

    # 日志（v0.2.15+ 统一 logs/engines/{engine_id}.log；stdout JSON 供 core 采集）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== Qwen-Image 图像引擎启动（qwen_image_5e8f3a21）=====")
    logger.info("配置: %s", _config.to_dict())

    _engine = QwenImageEngine(_config)
    logger.info(
        "引擎实例化完成: state=%s default_variant=%s",
        _engine.state_machine.state.value, _engine.default_variant,
    )

    yield

    logger.info("===== Qwen-Image 图像引擎关闭 =====")
    if _engine is not None and _engine.tm is not None:
        try:
            await _engine._unload_impl()
        except Exception as e:  # noqa: BLE001
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="Qwen-Image Engine",
        description=(
            "AIReliable 平台的 Qwen-Image 图像引擎"
            "（文生图 / 指令编辑含视角控制 / 多图参考图生图，GGUF Q8 单档）"
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS（平台 iframe 嵌入 WebUI 需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_endpoints(app)

    # 根路径重定向到 WebUI（平台「浏览器打开」打开基础 URL，无此路由会 404）
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    # 输出目录静态服务（/output/image/{task_id}.png → output_root/image/...）
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    # 自定义 WebUI（阶段 7 实现；webui/ 目录存在时挂载到 /ui）
    _mount_webui(app)

    return app


def _mount_webui(app: FastAPI) -> None:
    """挂载模块化 WebUI（四区布局：设置/参数/LoRA/结果）。

    阶段 7（T7.1）实现 webui/index.html 后自动生效；
    目录不存在时跳过（仅提供 API 端点）。
    """
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
        logger.info("WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.info("WebUI 未挂载（webui/index.html 不存在，阶段 7 实现）")


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准端点 + 扩展端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（三档 variant / LoRA / 输出尺寸）。"""
        resp = await get_engine().capabilities()
        # 手动序列化 variants，保留子类扩展字段（Pydantic 按声明类型序列化会丢弃）
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        return result

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """加载模型（共享组件 + 默认档 transformer）。

        文件缺失/残缺抛 500（ModelFilesError，信息含缺失清单与下载指引）；
        档位名非法抛 400。
        """
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败: variant=%s", request.variant)
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """卸载模型（释放显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """三模式推理（SSE 流式返回进度帧 + 结果帧）。

        请求体：
            - variant: variant 名（可选，auto-load 用）
            - inputs: ImageInputs 字段
              （prompt / mode: t2i|i2i|edit / width / height / steps /
               guidance / seed / strength / precision: bf16|fp8|gguf_q8）
            - images: 输入图列表（url/b64_json/path 三来源；
              0 图→t2i、1 图→edit、≥2 图→i2i 多图参考，上限 5 图）
            - loras: LoRA 规格（可选，阶段 6 支持请求级 diff）
            - task_id: 任务 ID（可选）
        """
        engine = get_engine()
        sse_stream = engine.infer(request)
        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态（档位/槽位/显存/LoRA/历史数）。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """挂载单个 LoRA（叠加，不卸载已挂载的）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("LoRA 加载失败: %s", request.name)
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载单个 LoRA。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败: %s", request.name)
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 扩展端点 =====

    # 9. /history（WebUI 结果区分页；v0.2.16 起代理查询 gateway 任务中心）
    @app.get("/history")
    async def history(page: int = 1, page_size: int = 20) -> dict:
        """生成历史（最新在前，gateway 任务中心代理视图）。"""
        if page < 1 or page_size < 1 or page_size > 100:
            raise HTTPException(status_code=400, detail="page ≥ 1 且 1 ≤ page_size ≤ 100")
        return await get_engine().get_history(page=page, page_size=page_size)

    # 10. /log_level（SDK 提供：GET 查询 + POST 调整）
    register_log_level_endpoint(app)


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    _config = parse_args()

    # 启动阶段日志先行（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 Qwen-Image 图像引擎: host=%s, port=%d",
        _config.host, _config.port,
    )

    app = create_app(_config)

    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
