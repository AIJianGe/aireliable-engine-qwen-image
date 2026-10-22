"""任务中心对接（QwenImage 引擎，v0.2.16+）。

设计依据：TASK_CENTER_ENGINE_HISTORY_UNIFY_ANALYSIS.md（方案 A）——
gateway ``data/tasks.db`` 为任务元数据**唯一真相源**；引擎删除内存历史后，
WebUI 历史改为 gateway 查询视图（本模块 ``fetch_history_page``）。

上报分工规则（避免双写终态）：
    - **gateway 转发链路**（``request.task_id`` 非空）：gateway 已建任务并维护
      进度/终态（调度器消费 SSE 帧后 mark_succeeded/mark_failed），
      引擎只报 created（upsert 更新状态，不带 request/task_type——
      gateway UPDATE 分支不覆盖二者，此为双保险）
    - **直连链路**（``request.task_id`` 为空，如 WebUI 会话）：引擎报全生命周期
      created（INSERT，带 task_type + request 摘要）→ running → progress →
      succeeded/failed。succeeded 用 await（先入库再返回结果帧，
      WebUI 刷新历史不闪空；SDK 3s 超时容错）

task_type 映射（与 gateway 任务中心筛选语义对齐，见 openai_endpoints.py
create_image/edit_image 的取值）：
    t2i → image_generation；edit / i2i → image_editing

容错原则（与 SDK PlatformTaskReporter 一致）：
    gateway 不可达时所有上报降级（warning 日志），**不阻断推理主流程**。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

from aireliable_engine import PlatformTaskReporter

logger = logging.getLogger(__name__)

# 模式 → 任务中心 task_type（与 gateway /v1/images/* 端点取值一致）
TASK_TYPE_BY_MODE: dict[str, str] = {
    "t2i": "image_generation",
    "edit": "image_editing",
    "i2i": "image_editing",
}

# task_type → 模式（历史视图回退映射：转发链路无精确 mode，用 task_type 近似）
MODE_BY_TASK_TYPE: dict[str, str] = {
    "image_generation": "t2i",
    "image_editing": "edit",
}

# gateway 默认地址（平台通过 engine_manager 注入 AIRELIABLE_GATEWAY_URL；
# 引擎独立调试时回退本机默认）
_DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"


def _gateway_url() -> str:
    """读取 gateway 地址（AIRELIABLE_GATEWAY_URL 优先）。"""
    return (os.environ.get("AIRELIABLE_GATEWAY_URL", "") or _DEFAULT_GATEWAY_URL).rstrip("/")


class QwenImageTaskReporting:
    """QwenImage 引擎任务上报薄封装（惰性初始化，容错优先）。"""

    def __init__(self, engine_id: str):
        self._engine_id = engine_id
        # 自建实例（显式 engine_id，独立调试脱离平台时 env 未注入也能正确归属）
        self._reporter: Optional[PlatformTaskReporter] = None

    async def _ensure_reporter(self) -> Optional[PlatformTaskReporter]:
        """惰性初始化上报器（首次 awaited 上报时创建，捕获主 loop）。"""
        if self._reporter is None:
            try:
                self._reporter = PlatformTaskReporter(engine_id=self._engine_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("任务上报器初始化失败（不阻断推理）: %s", e)
        return self._reporter

    # ===== 推理生命周期埋点（engine.py _infer_impl 调用）=====

    async def on_infer_created(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        mode: str,
        request_summary: dict[str, Any],
        variant: str,
    ) -> None:
        """推理入口上报任务创建。

        - 转发链路：仅 upsert 状态（gateway 侧 request/task_type 为权威，
          且 UPDATE 分支本就不覆盖，双保险）
        - 直连链路：INSERT 新任务，携带 task_type + request 摘要
          （摘要含 prompt/mode/size/seed 等历史视图所需字段）
        """
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            if gateway_forwarded:
                await reporter.report_created(task_id=task_id, variant=variant)
            else:
                await reporter.report_created(
                    task_id=task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=TASK_TYPE_BY_MODE.get(mode, "image_generation"),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("report_created 失败（不阻断推理）: %s", e)

    def on_infer_running(self, task_id: str, *, gateway_forwarded: bool) -> None:
        """推理开始上报 running（仅直连链路；转发链路由 gateway 调度器维护）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_running(task_id)

    def on_infer_progress(
        self, task_id: str, *, gateway_forwarded: bool, progress: float, stage: str,
    ) -> None:
        """推理进度上报（仅直连链路，fire-and-forget，推理线程跨线程安全）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_progress(task_id, progress=progress, stage=stage)

    async def on_infer_succeeded(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        result: dict[str, Any],
    ) -> None:
        """推理成功上报（仅直连链路；转发链路 gateway 消费 SSE 帧后自行 mark_succeeded）。

        使用 await（而非 fire）：先入库再返回结果帧，保证 WebUI 收到结果后
        立即刷新历史能查到本条（避免闪空）。SDK 3s 超时容错。
        """
        if gateway_forwarded:
            return
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            await reporter.report_succeeded(task_id=task_id, result=result)
        except Exception as e:  # noqa: BLE001
            logger.warning("report_succeeded 失败（不阻断推理）: %s", e)

    def on_infer_failed(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        error: dict[str, Any],
    ) -> None:
        """推理失败上报（仅直连链路，fire-and-forget）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_failed(task_id, error=error)


# ============================================================
# 历史视图代理（/history 端点 → gateway /v1/tasks 查询）
# ============================================================

async def fetch_history_page(engine_id: str, page: int, page_size: int) -> dict:
    """代理查询 gateway 任务中心，映射为 WebUI HistoryItem 视图结构。

    查询链路：
        GET {gateway}/v1/tasks?engine_id=<self>&status=succeeded&limit&offset
        （列表摘要：image_url/prompt/task_type/created_at/duration_seconds）
        + 并发补全 GET /v1/tasks/{task_id}（完整 request/result，
          提供 width/height/seed/precision/loras）

    降级：gateway 不可达（3s 超时）→ {"items": [], "degraded": true}
    （WebUI 历史区显示降级提示，生成功能不受影响）

    Returns:
        dict: {items, total, page, page_size[, degraded]}，
        items 为 HistoryItem 形状的 plain dict（width/height/seed 可能为 None，
        WebUI 端做兜底显示）
    """
    offset = (page - 1) * page_size
    try:
        async with httpx.AsyncClient(
            base_url=_gateway_url(),
            timeout=httpx.Timeout(3.0, connect=1.5),
            trust_env=False,  # 平台内部回环，不走系统代理
        ) as client:
            resp = await client.get("/v1/tasks", params={
                "engine_id": engine_id,
                "status": "succeeded",
                "limit": page_size,
                "offset": offset,
            })
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks") or []

            # 并发补全详情（列表摘要不含 width/height/seed/precision/loras）
            details = await asyncio.gather(
                *(_fetch_task_detail(client, t.get("task_id", "")) for t in tasks),
                return_exceptions=True,
            )
            items = [
                item
                for item in (
                    _map_task_to_history(
                        t,
                        None if isinstance(d, Exception) else d,
                    )
                    for t, d in zip(tasks, details)
                )
                # 无图任务（如推理中取消的 succeeded 记录）不进历史视图
                if item.get("image_url")
            ]
            return {
                "items": items,
                "total": data.get("total", 0),
                "page": page,
                "page_size": page_size,
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "历史视图查询 gateway 失败（降级为空列表，原因：%s）", e,
        )
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "degraded": True,
        }


async def _fetch_task_detail(client: httpx.AsyncClient, task_id: str) -> Optional[dict]:
    """查询单任务详情（完整 request/result）。失败返回 None（不阻断整页）。"""
    if not task_id:
        return None
    try:
        resp = await client.get(f"/v1/tasks/{task_id}")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def _map_task_to_history(summary: dict, detail: Optional[dict]) -> dict:
    """任务记录 → HistoryItem 视图形状。

    字段优先级（直连链路数据完整；转发链路部分字段缺失时逐级回退）：
        - mode:     result.mode（直连）> request_summary.mode > task_type 近似
        - 图片:      顶层 image_url（gateway v0.3.9+ 便利字段，两条链路都有）
        - width/height: result.width/height（直连）> result/request.size 解析（转发）
        - seed:     result.seed（直连）> request.extra.seed（转发）
        - precision: result.precision（直连）> variant_id 档位映射
        - loras:    result.loras（直连上报时注入；转发链路无 → []）
    """
    req: dict = (detail or {}).get("request") or {}
    result: dict = (detail or {}).get("result") or {}
    req_summary: dict = summary.get("request_summary") or {}

    # mode
    mode = result.get("mode") or req_summary.get("mode") or req.get("mode")
    if not mode:
        mode = MODE_BY_TASK_TYPE.get(summary.get("task_type", ""), "t2i")

    # 图片 URL（列表摘要的顶层便利字段，兼容两条链路）
    image_url = summary.get("image_url") or result.get("image_url") or ""

    # width / height
    width = result.get("width")
    height = result.get("height")
    if not width or not height:
        size = str(result.get("size") or req.get("size") or "").lower()
        if "x" in size:
            try:
                w_str, h_str = size.split("x", 1)
                width = width or int(w_str)
                height = height or int(h_str)
            except ValueError:
                pass

    # seed（转发链路：gateway 存的 request 是 OpenAI 格式，seed 在 extra 里）
    seed = result.get("seed")
    if seed is None:
        extra = req.get("extra")
        if isinstance(extra, dict):
            seed = extra.get("seed")

    # precision（直连 result 自带；转发链路用 variant_id 档位映射）
    precision = result.get("precision")
    if precision is None:
        precision = _precision_from_variant(summary.get("variant_id", ""))

    # loras（直连上报时注入 result；转发链路无记录 → 空）
    loras = result.get("loras")
    if not isinstance(loras, list):
        loras = []

    duration = summary.get("duration_seconds")
    if duration is None:
        duration = result.get("duration_seconds", 0.0)

    return {
        "task_id": summary.get("task_id", ""),
        "mode": mode,
        "prompt": req_summary.get("prompt") or req.get("prompt") or "",
        "image_url": image_url,
        "thumbnail_url": None,
        "width": width,
        "height": height,
        "seed": seed,
        "precision": precision,
        "loras": loras,
        "created_at": summary.get("created_at", 0.0),
        "duration_seconds": duration or 0.0,
    }


def _precision_from_variant(variant_id: str) -> Optional[str]:
    """variant 名 → 精度档位（延迟导入避免与 engine.py 循环依赖）。"""
    from .engine import VARIANT_PRECISIONS
    return VARIANT_PRECISIONS.get(variant_id)
