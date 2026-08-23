"""可观测性路由（文档十三节）：统一日志查询（HTTP 历史）+ SSE 实时时间线 + 黑名单测试。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...observability.bus import EventBus

logger = logging.getLogger(__name__)


class BlacklistTestPayload(BaseModel):
    text: str


def build_router(
    *,
    auth,
    store: Any | None,
    event_bus: EventBus | None,
    env: Any | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    @router.get("/logs")
    async def unified_logs(kinds: str = "all", group_id: str = "", limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """跨 message_logs / inference_events / sticker_events 的合并时间线（HTTP 历史）。"""
        if store is None:
            return {"items": [], "total": 0}
        wanted = {part.strip() for part in str(kinds or "all").split(",") if part.strip()}
        per_source_limit = max(200, int(limit) + int(offset))
        entries: list[dict[str, Any]] = []

        async def collect(inc: list[dict[str, Any]], source: str) -> None:
            for item in inc:
                entry = dict(item)
                entry["source"] = source
                entry.setdefault("group_id", None)
                entries.append(entry)

        if not wanted or "all" in wanted or "message" in wanted:
            payload = await store.message_logs.list_logs(group_id=group_id, limit=per_source_limit, offset=0)
            await collect(payload.get("items") or [], "message")
        if not wanted or "all" in wanted or "inference" in wanted:
            payload = await store.list_inference_events(limit=300, offset=0)
            await collect(payload.get("items") or [], "inference")
        if not wanted or "all" in wanted or "sticker" in wanted:
            payload = await store.list_sticker_events(limit=300, offset=0)
            await collect(payload.get("items") or [], "sticker")

        if group_id:
            entries = [entry for entry in entries if str(entry.get("group_id") or "") == group_id]
        entries.sort(key=lambda entry: float(entry.get("created_at") or 0), reverse=True)
        start = max(0, int(offset or 0))
        end = start + max(1, min(int(limit or 100), 300))
        return {"items": entries[start:end], "total": len(entries)}

    @router.get("/logs/stream")
    async def logs_stream(max_seconds: float = 0) -> StreamingResponse:
        """SSE 实时时间线：仅推送增量（历史由 GET /api/logs 拉取，前端合并）。

        max_seconds > 0 时到时自动收流（EventSource 会自动重连；测试与弱网客户端用）。
        """
        if event_bus is None:
            raise HTTPException(status_code=503, detail="event bus unavailable")
        deadline = asyncio.get_event_loop().time() + float(max_seconds) if max_seconds and max_seconds > 0 else None

        async def event_stream():
            heartbeat = 0
            loop = asyncio.get_event_loop()
            async with event_bus.subscribe() as queue:
                while True:
                    timeout = 15.0
                    if deadline is not None:
                        timeout = min(timeout, max(0.05, deadline - loop.time()))
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=timeout)
                        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    except TimeoutError:
                        if deadline is not None and loop.time() >= deadline:
                            yield "event: closed\ndata: {}\n\n"
                            return
                        heartbeat += 1
                        yield f": heartbeat {heartbeat}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/blacklist/test", response_model=None)
    async def blacklist_test(payload: BlacklistTestPayload) -> dict[str, Any]:
        """测试一段文本会命中哪些黑名单规则（词库管理页用）。"""
        if env is None:
            raise HTTPException(status_code=503, detail="environment unavailable")
        matches = env.blacklist_matches(payload.text or "")
        return {
            "text": payload.text,
            "matched": bool(matches),
            "matches": matches,
            "words": list(env.config.ai_companion_blacklist_words),
            "regexes": list(env.config.ai_companion_blacklist_regexes),
        }

    return router
