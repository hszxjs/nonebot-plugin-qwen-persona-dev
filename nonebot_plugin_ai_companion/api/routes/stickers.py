"""表情仓库路由：列表/事件/预览/摘要更新/批量删除。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..helpers import maybe_await


def build_router(*, config: Any, auth, store: Any | None) -> APIRouter:
    router = APIRouter(prefix="/api/stickers", dependencies=[Depends(auth)])

    @router.get("")
    async def stickers(limit: int = 80, offset: int = 0, q: str = "", type: str = "") -> dict[str, Any]:
        if not store or not hasattr(store, "list_stickers"):
            return {
                "items": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "query": q,
                "type": type,
                "type_counts": {},
            }
        return await maybe_await(store.list_stickers(limit=limit, offset=offset, query=q, segment_type=type))

    @router.get("/events")
    async def sticker_events(limit: int = 100, offset: int = 0) -> dict[str, Any]:
        if not store or not hasattr(store, "list_sticker_events"):
            return {"items": [], "total": 0, "limit": min(limit, 300), "offset": max(0, offset)}
        return await maybe_await(store.list_sticker_events(limit=limit, offset=offset))

    @router.get("/{sticker_key:path}/preview")
    async def sticker_preview(sticker_key: str) -> FileResponse:
        if not store or not hasattr(store, "get_sticker_preview_path"):
            raise HTTPException(status_code=404, detail="sticker preview not found")
        repo_dir = getattr(config, "ai_companion_sticker_repo_dir", "data/qwen_persona/sticker_repo")
        path = await maybe_await(store.get_sticker_preview_path(sticker_key, repo_dir=repo_dir))
        if path is None or not Path(path).is_file():
            raise HTTPException(status_code=404, detail="sticker preview not found")
        return FileResponse(Path(path))

    @router.patch("/{sticker_key:path}")
    async def update_sticker(sticker_key: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict) or "summary" not in payload:
            raise HTTPException(status_code=400, detail="summary is required")
        if not store or not hasattr(store, "update_sticker_summary"):
            raise HTTPException(status_code=503, detail="sticker store unavailable")
        return await maybe_await(store.update_sticker_summary(sticker_key, str(payload["summary"])))

    @router.delete("")
    async def delete_stickers(request: Request) -> dict[str, int]:
        payload = await request.json()
        raw_keys = payload.get("keys") if isinstance(payload, dict) else None
        keys = [raw_keys] if isinstance(raw_keys, str) else raw_keys
        if not isinstance(keys, list):
            raise HTTPException(status_code=400, detail="keys is required")
        if not store or not hasattr(store, "delete_stickers"):
            raise HTTPException(status_code=503, detail="sticker store unavailable")
        try:
            return await maybe_await(store.delete_stickers([str(key) for key in keys]))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
