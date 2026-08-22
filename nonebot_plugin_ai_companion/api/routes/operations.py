"""运维总览路由：首页控制台聚合数据。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..helpers import (
    maybe_await,
    runtime_summary,
    training_config,
    training_operation,
)


def build_router(
    *,
    config: Any,
    auth,
    store: Any | None,
    trainer: Any | None,
    mlx_runtime: Any | None,
    qwen_client: Any | None,
) -> APIRouter:
    router = APIRouter(prefix="/api/operations", dependencies=[Depends(auth)])

    @router.get("/overview")
    async def operations_overview() -> dict[str, Any]:
        runtime_status = (
            await maybe_await(qwen_client.status_snapshot())
            if qwen_client and hasattr(qwen_client, "status_snapshot")
            else (
                await maybe_await(mlx_runtime.status_snapshot())
                if mlx_runtime and hasattr(mlx_runtime, "status_snapshot")
                else {"state": "mlx_runtime_unavailable", "healthy": False}
            )
        )
        dashboard = (
            await maybe_await(store.dashboard_stats())
            if store and hasattr(store, "dashboard_stats")
            else {"totals": {}, "training": {}, "charts": {}}
        )
        events = (
            await maybe_await(store.list_inference_events(limit=300, offset=0))
            if store and hasattr(store, "list_inference_events")
            else {"items": [], "total": 0, "limit": 300, "offset": 0}
        )
        client_metrics = (
            qwen_client.metrics_snapshot()
            if qwen_client and hasattr(qwen_client, "metrics_snapshot")
            else {"last": None, "history": []}
        )
        return {
            "runtime": runtime_summary(dict(runtime_status)),
            "learning": dict(dashboard.get("totals") or {}),
            "training": {
                "summary": dashboard.get("training") or {},
                "operation": training_operation(trainer),
                "config": training_config(config),
            },
            "charts": dashboard.get("charts") or {},
            "inference": {"client": client_metrics},
            "inference_events": events,
            "errors": [runtime_status.get("last_error")] if runtime_status.get("last_error") else [],
        }

    return router
