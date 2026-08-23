"""KPI Dashboard 与会话监控路由（文档十四节）。

- GET /api/dashboard/kpi：消息量、接话率（按触发原因拆分）、Provider 延迟/tps、
  后端健康、工具调用成功率、活跃会话数
- GET /api/sessions：按会话（群）聚合的最近回复记录
- POST /api/sessions/poke：手动触发一次主动接话（需有在线 bot）
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ProactiveSender = Callable[[str], Awaitable[str]]


class PokePayload(BaseModel):
    group_id: str


def build_router(
    *,
    auth,
    store: Any | None,
    qwen_client: Any | None,
    proactive_sender: ProactiveSender | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    @router.get("/dashboard/kpi")
    async def dashboard_kpi(days: int = 7) -> dict[str, Any]:
        if store is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        since = time.time() - max(1, min(int(days), 30)) * 86400

        def _sync() -> dict[str, Any]:
            with store.db.connection() as conn:
                total_replies = int(
                    conn.execute("SELECT COUNT(*) FROM message_logs WHERE created_at >= ?", (since,)).fetchone()[0]
                )
                reason_rows = conn.execute(
                    """
                    SELECT trigger_reason AS reason, COUNT(*) AS count
                    FROM message_logs
                    WHERE created_at >= ? AND trigger_reason != ''
                    GROUP BY trigger_reason ORDER BY count DESC
                    """,
                    (since,),
                ).fetchall()
                provider_rows = conn.execute(
                    """
                    SELECT model,
                           COUNT(*) AS total,
                           AVG(latency_seconds) AS avg_latency,
                           AVG(tps) AS avg_tps
                    FROM inference_events
                    WHERE created_at >= ? AND model != ''
                    GROUP BY model ORDER BY total DESC
                    """,
                    (since,),
                ).fetchall()
                active_sessions = int(
                    conn.execute(
                        """
                        SELECT COUNT(DISTINCT group_id) FROM message_logs
                        WHERE created_at >= ? AND group_id IS NOT NULL
                        """,
                        (since - 86400,),
                    ).fetchone()[0]
                )
                learned = int(
                    conn.execute("SELECT COUNT(*) FROM chat_samples WHERE created_at >= ?", (since,)).fetchone()[0]
                )
            return {
                "total_replies": total_replies,
                "reply_reasons": [{"reason": row["reason"], "count": int(row["count"])} for row in reason_rows],
                "providers": [
                    {
                        "model": row["model"],
                        "total": int(row["total"]),
                        "avg_latency_seconds": round(float(row["avg_latency"] or 0), 3),
                        "avg_tps": round(float(row["avg_tps"] or 0), 2),
                    }
                    for row in provider_rows
                ],
                "active_sessions": active_sessions,
                "learned_messages": learned,
            }

        aggregate = await asyncio.to_thread(_sync)
        tool_stats = await store.tool_call_logs.success_stats(days=int(days))
        backend_health: dict[str, Any] = {"healthy": None, "model": "", "backend": ""}
        if qwen_client is not None and hasattr(qwen_client, "status_snapshot"):
            snapshot = qwen_client.status_snapshot()
            backend_health = {
                "healthy": bool(snapshot.get("healthy")),
                "model": str(snapshot.get("model") or ""),
                "backend": str(qwen_client.active_backend()),
            }
        return {"window_days": max(1, min(int(days), 30)), **aggregate, "tools": tool_stats, "backend": backend_health}

    @router.get("/sessions")
    async def sessions(limit: int = 50) -> dict[str, Any]:
        if store is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        payload = await store.message_logs.list_logs(limit=max(1, min(int(limit), 200)))
        # 按会话聚合：group_id → 最近一条 + 计数（私聊归 private:user）
        sessions: dict[str, dict[str, Any]] = {}
        for item in payload.get("items") or []:
            key = str(item.get("group_id") or f"private:{item.get('user_id')}")
            entry = sessions.setdefault(key, {"session_key": key, "count": 0, "latest": item})
            entry["count"] += 1
            entry["latest"] = item
        items = sorted(sessions.values(), key=lambda e: float(e["latest"].get("created_at") or 0), reverse=True)
        return {"items": items, "total": len(items)}

    @router.post("/sessions/poke", response_model=None)
    async def session_poke(payload: PokePayload) -> dict[str, Any]:
        if proactive_sender is None:
            raise HTTPException(status_code=503, detail="proactive sender unavailable (no bot online?)")
        try:
            message_id = await proactive_sender(payload.group_id.strip())
        except Exception as exc:  # noqa: BLE001 — 主动发送失败给出可读错误
            raise HTTPException(status_code=502, detail=f"proactive reply failed: {exc}") from exc
        return {"group_id": payload.group_id, "sent": True, "reply_message_id": message_id}

    return router
