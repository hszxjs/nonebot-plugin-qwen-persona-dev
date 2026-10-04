"""离线信箱仓库：被 @/引用的记录持久化，目标上线后投递。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..memory.db import PersonaDatabase


@dataclass
class PingInboxEntry:
    id: int
    session_key: str
    group_id: str
    target_user_id: str
    finder_user_id: str
    reason: str
    source_message_id: str
    text: str
    attachments: list[str] = field(default_factory=list)
    created_at: float = 0.0


def _entry_from_row(row: Any) -> PingInboxEntry:
    try:
        attachments = [str(item) for item in json.loads(row["attachments"] or "[]")]
    except (TypeError, ValueError):
        attachments = []
    return PingInboxEntry(
        id=int(row["id"]),
        session_key=str(row["session_key"] or ""),
        group_id=str(row["group_id"] or ""),
        target_user_id=str(row["target_user_id"] or ""),
        finder_user_id=str(row["finder_user_id"] or ""),
        reason=str(row["reason"] or "at"),
        source_message_id=str(row["source_message_id"] or ""),
        text=str(row["text"] or ""),
        attachments=attachments,
        created_at=float(row["created_at"] or 0.0),
    )


class PingInboxRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def add(
        self,
        *,
        session_key: str,
        group_id: str,
        target_user_id: str,
        finder_user_id: str = "",
        reason: str = "at",
        source_message_id: str = "",
        text: str = "",
        attachments: list[str] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._add_sync,
            session_key,
            group_id,
            target_user_id,
            finder_user_id,
            reason,
            source_message_id,
            text,
            list(attachments or []),
        )

    def _add_sync(
        self,
        session_key: str,
        group_id: str,
        target_user_id: str,
        finder_user_id: str,
        reason: str,
        source_message_id: str,
        text: str,
        attachments: list[str],
    ) -> int:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ping_inbox (
                    session_key, group_id, target_user_id, finder_user_id, reason,
                    source_message_id, text, attachments, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_key,
                    group_id,
                    target_user_id,
                    finder_user_id,
                    reason,
                    source_message_id,
                    text,
                    json.dumps(attachments, ensure_ascii=False),
                    time.time(),
                ),
            )
            return int(cursor.lastrowid or 0)

    async def list_for_target(self, *, session_key: str, target_user_id: str) -> list[PingInboxEntry]:
        return await asyncio.to_thread(self._list_sync, session_key, target_user_id)

    def _list_sync(self, session_key: str, target_user_id: str) -> list[PingInboxEntry]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ping_inbox
                WHERE session_key = ? AND target_user_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_key, target_user_id),
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    async def delete_entries(self, entry_ids: list[int]) -> int:
        return await asyncio.to_thread(self._delete_sync, list(entry_ids))

    def _delete_sync(self, entry_ids: list[int]) -> int:
        if not entry_ids:
            return 0
        self.db.ensure_schema()
        placeholders = ",".join("?" for _ in entry_ids)
        with self.db.connection() as conn:
            cursor = conn.execute(f"DELETE FROM ping_inbox WHERE id IN ({placeholders})", entry_ids)
            return int(cursor.rowcount or 0)

    async def purge_expired(self, *, ttl_seconds: float) -> list[PingInboxEntry]:
        """清掉超过 TTL 的信箱条目，返回被清的条目（调用方负责删附件文件）。"""
        return await asyncio.to_thread(self._purge_sync, float(ttl_seconds))

    def _purge_sync(self, ttl_seconds: float) -> list[PingInboxEntry]:
        cutoff = time.time() - max(60.0, ttl_seconds)
        self.db.ensure_schema()
        expired: list[PingInboxEntry] = []
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM ping_inbox WHERE created_at < ?", (cutoff,)).fetchall()
            for row in rows:
                expired.append(_entry_from_row(row))
            if expired:
                conn.execute("DELETE FROM ping_inbox WHERE created_at < ?", (cutoff,))
        return expired


__all__ = ["PingInboxEntry", "PingInboxRepo"]
