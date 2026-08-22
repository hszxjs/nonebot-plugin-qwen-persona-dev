"""可观测性域：推理调用事件与表情事件的环形缓冲持久化（各保留最新 1000 条）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from typing import Any

from ..memory.db import PersonaDatabase


class InferenceEventRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def record_inference_event(
        self,
        *,
        event_id: str,
        status: str,
        trigger_reason: str,
        model: str,
        adapter: str = "",
        latency_seconds: float | None = None,
        tps: float | None = None,
        media_count: int = 0,
        response_chars: int = 0,
        error_summary: str = "",
        input_text: str = "",
        output_text: str = "",
        group_id: str | None = None,
        user_id: str | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._record_inference_event_sync,
            event_id=event_id,
            status=status,
            trigger_reason=trigger_reason,
            model=model,
            adapter=adapter,
            latency_seconds=latency_seconds,
            tps=tps,
            media_count=media_count,
            response_chars=response_chars,
            error_summary=error_summary,
            input_text=input_text,
            output_text=output_text,
            group_id=group_id,
            user_id=user_id,
            created_at=created_at,
        )

    def _record_inference_event_sync(
        self,
        *,
        event_id: str,
        status: str,
        trigger_reason: str,
        model: str,
        adapter: str = "",
        latency_seconds: float | None = None,
        tps: float | None = None,
        media_count: int = 0,
        response_chars: int = 0,
        error_summary: str = "",
        input_text: str = "",
        output_text: str = "",
        group_id: str | None = None,
        user_id: str | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        self.db.ensure_schema()
        timestamp = float(created_at if created_at is not None else time.time())
        item = {
            "event_id": str(event_id or f"inference-{timestamp:.6f}"),
            "created_at": timestamp,
            "status": str(status or "unknown"),
            "trigger_reason": str(trigger_reason or "unknown"),
            "model": str(model or ""),
            "adapter": str(adapter or ""),
            "latency_seconds": None if latency_seconds is None else float(latency_seconds),
            "tps": None if tps is None else float(tps),
            "media_count": max(0, int(media_count or 0)),
            "response_chars": max(0, int(response_chars or 0)),
            "error_summary": self.db.preview_text(error_summary, limit=240),
            "input_preview": self.db.preview_text(input_text, limit=240),
            "output_preview": self.db.preview_text(output_text, limit=240),
            "group_id": None if group_id is None else str(group_id),
            "user_id": None if user_id is None else str(user_id),
        }
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO inference_events (
                    event_id, created_at, status, trigger_reason, model, adapter,
                    latency_seconds, tps, media_count, response_chars, error_summary,
                    input_preview, output_preview, group_id, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["event_id"],
                    item["created_at"],
                    item["status"],
                    item["trigger_reason"],
                    item["model"],
                    item["adapter"],
                    item["latency_seconds"],
                    item["tps"],
                    item["media_count"],
                    item["response_chars"],
                    item["error_summary"],
                    item["input_preview"],
                    item["output_preview"],
                    item["group_id"],
                    item["user_id"],
                ),
            )
            conn.execute(
                """
                DELETE FROM inference_events
                WHERE event_id NOT IN (
                    SELECT event_id
                    FROM inference_events
                    ORDER BY created_at DESC
                    LIMIT 1000
                )
                """
            )
        return item

    async def list_inference_events(
        self,
        *,
        limit: int = 300,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._list_inference_events_sync,
            limit=limit,
            offset=offset,
        )

    def _list_inference_events_sync(
        self,
        *,
        limit: int = 300,
        offset: int = 0,
    ) -> dict[str, Any]:
        self.db.ensure_schema()
        limit = max(1, min(int(limit or 300), 300))
        offset = max(0, int(offset or 0))
        with self.db.connection() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM inference_events").fetchone()[0])
            rows = conn.execute(
                """
                SELECT event_id, created_at, status, trigger_reason, model, adapter,
                       latency_seconds, tps, media_count, response_chars, error_summary,
                       input_preview, output_preview, group_id, user_id
                FROM inference_events
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return {
            "items": [self._inference_event_row_payload(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _preview_text(value: str, *, limit: int = 240) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _inference_event_row_payload(row: sqlite3.Row) -> dict[str, Any]:
        created_at = float(row["created_at"])
        return {
            "event_id": row["event_id"],
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at)),
            "created_at": created_at,
            "status": row["status"],
            "trigger_reason": row["trigger_reason"],
            "model": row["model"],
            "adapter": row["adapter"],
            "latency_seconds": row["latency_seconds"],
            "tps": row["tps"],
            "media_count": int(row["media_count"] or 0),
            "response_chars": int(row["response_chars"] or 0),
            "error_summary": row["error_summary"],
            "input_preview": row["input_preview"],
            "output_preview": row["output_preview"],
            "group_id": row["group_id"],
            "user_id": row["user_id"],
        }


class StickerEventRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def record_sticker_event(
        self,
        *,
        event_type: str,
        status: str,
        key: str | None = None,
        group_id: str | None = None,
        user_id: str | None = None,
        message: str = "",
        details: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._record_sticker_event_sync,
            event_type=event_type,
            status=status,
            key=key,
            group_id=group_id,
            user_id=user_id,
            message=message,
            details=details or {},
            created_at=created_at,
        )

    def _record_sticker_event_sync(
        self,
        *,
        event_type: str,
        status: str,
        key: str | None = None,
        group_id: str | None = None,
        user_id: str | None = None,
        message: str = "",
        details: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        self.db.ensure_schema()
        timestamp = float(created_at if created_at is not None else time.time())
        details_payload = details if isinstance(details, dict) else {}
        basis = json.dumps(
            {
                "created_at": timestamp,
                "event_type": event_type,
                "status": status,
                "key": key,
                "message": message,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        event_id = "sticker-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:20]
        item = {
            "event_id": event_id,
            "created_at": timestamp,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "event_type": str(event_type or "unknown"),
            "status": str(status or "unknown"),
            "key": None if key is None else str(key),
            "group_id": None if group_id is None else str(group_id),
            "user_id": None if user_id is None else str(user_id),
            "message": self.db.preview_text(message, limit=240),
            "details": details_payload,
        }
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sticker_events (
                    event_id, created_at, event_type, status, key, group_id,
                    user_id, message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["event_id"],
                    item["created_at"],
                    item["event_type"],
                    item["status"],
                    item["key"],
                    item["group_id"],
                    item["user_id"],
                    item["message"],
                    json.dumps(item["details"], ensure_ascii=False, default=str),
                ),
            )
            conn.execute(
                """
                DELETE FROM sticker_events
                WHERE event_id NOT IN (
                    SELECT event_id
                    FROM sticker_events
                    ORDER BY created_at DESC
                    LIMIT 1000
                )
                """
            )
        return item

    async def list_sticker_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._list_sticker_events_sync,
            limit=limit,
            offset=offset,
        )

    def _list_sticker_events_sync(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self.db.ensure_schema()
        limit = max(1, min(int(limit or 100), 300))
        offset = max(0, int(offset or 0))
        with self.db.connection() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM sticker_events").fetchone()[0])
            rows = conn.execute(
                """
                SELECT event_id, created_at, event_type, status, key, group_id,
                       user_id, message, details_json
                FROM sticker_events
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return {
            "items": [self._sticker_event_row_payload(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def _sticker_event_row_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        created_at = float(row["created_at"])
        return {
            "event_id": row["event_id"],
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at)),
            "created_at": created_at,
            "event_type": row["event_type"],
            "status": row["status"],
            "key": row["key"],
            "group_id": row["group_id"],
            "user_id": row["user_id"],
            "message": row["message"],
            "details": self.db.json_object(row["details_json"]),
        }


async def safe_record_sticker_event(
    store: Any,
    *,
    logger: Any,
    event_type: str,
    status: str,
    key: str | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
    message: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    """记录表情事件的容错包装：store 缺能力或写入失败时只告警不抛出。"""
    if not hasattr(store, "record_sticker_event"):
        return
    try:
        await store.record_sticker_event(
            event_type=event_type,
            status=status,
            key=key,
            group_id=group_id,
            user_id=user_id,
            message=message,
            details=details or {},
        )
    except Exception as exc:
        logger.warning(f"ai companion sticker event record failed: {exc!r}")
