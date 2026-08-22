"""运行时路由：状态/日志/加载/重载/卸载。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..helpers import maybe_await, runtime_summary


def build_router(*, config: Any, auth, mlx_runtime: Any | None, qwen_client: Any | None) -> APIRouter:
    router = APIRouter(prefix="/api/runtime", dependencies=[auth])

    async def current_status() -> dict[str, Any]:
        if qwen_client and hasattr(qwen_client, "status_snapshot"):
            return dict(await maybe_await(qwen_client.status_snapshot()))
        if mlx_runtime and hasattr(mlx_runtime, "status_snapshot"):
            return dict(await maybe_await(mlx_runtime.status_snapshot()))
        return {"state": "mlx_runtime_unavailable", "healthy": False}

    @router.get("/status")
    async def runtime_status() -> dict[str, Any]:
        return runtime_summary(await current_status())

    @router.get("/logs")
    async def runtime_logs(tail: int = 300) -> dict[str, Any]:
        logs = (
            mlx_runtime.logs(tail=max(0, min(int(tail), 1000)))
            if mlx_runtime and hasattr(mlx_runtime, "logs")
            else []
        )
        return {"logs": logs}

    async def run_action(action: str, confirmed: bool = False):
        if action in {"reload", "unload"} and not confirmed:
            return JSONResponse(
                {
                    "error": "confirmation required",
                    "confirmation_required": True,
                    "action": action,
                },
                status_code=409,
            )
        if not mlx_runtime or not hasattr(mlx_runtime, action):
            raise HTTPException(status_code=503, detail="mlx runtime unavailable")
        model = str(getattr(config, "qwen_persona_mlx_model", "") or "").strip() or "unknown"
        result = await maybe_await(getattr(mlx_runtime, action)())
        runtime = runtime_summary(dict(result or {}))
        return {
            "action": action,
            "message": f"{action} model '{model}'",
            "runtime": runtime,
        }

    @router.post("/load")
    async def runtime_load():
        return await run_action("load", confirmed=True)

    @router.post("/reload")
    async def runtime_reload(request: Request):
        payload = await request.json()
        return await run_action("reload", confirmed=bool(payload.get("confirmed")))

    @router.post("/unload")
    async def runtime_unload(request: Request):
        payload = await request.json()
        return await run_action("unload", confirmed=bool(payload.get("confirmed")))

    return router
