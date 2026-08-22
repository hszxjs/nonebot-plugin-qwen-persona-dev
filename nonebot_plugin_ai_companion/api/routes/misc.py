"""杂项路由：推理事件分页、插件日志、诊断、模型列表。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..helpers import file_log_entries, maybe_await, plugin_log_path, runtime_summary

logger = logging.getLogger(__name__)


def build_router(
    *,
    config: Any,
    auth,
    store: Any | None,
    mlx_runtime: Any | None,
    static_root: Path,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[auth])

    @router.get("/inference-events")
    async def inference_events(limit: int = 300, offset: int = 0) -> dict[str, Any]:
        if not store or not hasattr(store, "list_inference_events"):
            return {"items": [], "total": 0, "limit": min(limit, 300), "offset": max(0, offset)}
        return await maybe_await(store.list_inference_events(limit=limit, offset=offset))

    @router.get("/plugin/logs")
    async def plugin_logs(tail: int = 300) -> dict[str, Any]:
        path = plugin_log_path(config)
        return {
            "logs": file_log_entries(path, tail=tail, stream="plugin"),
            "path": str(path),
            "exists": path.is_file(),
        }

    @router.get("/diagnostics")
    async def diagnostics() -> dict[str, Any]:
        runtime_status = (
            await maybe_await(mlx_runtime.status_snapshot())
            if mlx_runtime and hasattr(mlx_runtime, "status_snapshot")
            else {"state": "mlx_runtime_unavailable", "healthy": False}
        )
        runtime_logs = (
            mlx_runtime.logs(tail=300) if mlx_runtime and hasattr(mlx_runtime, "logs") else []
        )
        from ...runtime_config import admin_config_path, config_to_dict

        return {
            "runtime": {
                "status": runtime_summary(dict(runtime_status)),
                "logs": runtime_logs,
            },
            "settings": {
                "values": config_to_dict(config),
                "override_file": str(admin_config_path(config)),
            },
            "static": {
                "root": str(static_root),
                "built": (static_root / "index.html").is_file(),
                "index": str(static_root / "index.html"),
            },
            "build": {
                "admin_server": "fastapi-vite",
            },
        }

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        # 从在线推理配置派生 models 端点；未配置则回退到辅助训练配置。
        chat_url = (config.ai_companion_online_api_url or "").strip()
        api_key = (config.ai_companion_online_api_key or "").strip()
        if not chat_url:
            chat_url = (config.ai_companion_lora_helper_api_url or "").strip()
            api_key = (config.ai_companion_lora_helper_api_key or "").strip()
        try:
            from ...providers.glm_backend import fetch_online_models

            models, source = await fetch_online_models(chat_url, api_key, timeout=8.0)
        except Exception as exc:  # noqa: BLE001 — 任何异常都走兜底，不阻塞前端
            models, source = ["glm-5.2", "glm-4.5-flash", "glm-4.6", "glm-4.5"], "fallback"
            logger.warning("list_models fallback after error: %r", exc)
        return {"models": models, "source": source}

    return router
