"""模型管理器路由（文档十二.2）：本地模型列表、后台下载、SSE 进度、取消、删除。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...model_manager.manager import ModelManager, ModelOperationError

logger = logging.getLogger(__name__)


class DownloadPayload(BaseModel):
    model_ref: str


def build_router(*, auth, model_manager: ModelManager | None) -> APIRouter:
    router = APIRouter(prefix="/api/models/manage", dependencies=[Depends(auth)])

    def require() -> ModelManager:
        if model_manager is None:
            raise HTTPException(status_code=503, detail="model manager unavailable")
        return model_manager

    @router.get("")
    async def manage_overview() -> dict[str, Any]:
        manager = require()
        local = manager.list_local_models()
        total = sum(item["size_bytes"] for item in local)
        return {
            "local": local,
            "presets": manager.presets(),
            "total_size_bytes": total,
            "operation": manager.operation_snapshot(),
        }

    @router.post("/download", response_model=None)
    async def start_download(payload: DownloadPayload) -> dict[str, Any]:
        manager = require()
        try:
            snapshot = await manager.start_download(payload.model_ref)
        except ModelOperationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"operation": snapshot}

    @router.post("/download/cancel")
    async def cancel_download() -> dict[str, Any]:
        manager = require()
        try:
            return manager.cancel_download()
        except ModelOperationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/download/events")
    async def download_events() -> StreamingResponse:
        """SSE：每秒推送下载操作快照；操作结束（含失败/取消）后发终帧并关闭。"""
        manager = require()

        async def event_stream():
            idle_sent = False
            while True:
                snapshot = manager.operation_snapshot()
                if snapshot is None:
                    if idle_sent:
                        break
                    idle_sent = True
                    yield f"data: {json.dumps({'state': 'idle'})}\n\n"
                    break
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                if snapshot.get("state") != "running":
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.delete("")
    async def delete_model(model_ref: str) -> dict[str, Any]:
        manager = require()
        try:
            return await manager.delete_model(model_ref)
        except ModelOperationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
