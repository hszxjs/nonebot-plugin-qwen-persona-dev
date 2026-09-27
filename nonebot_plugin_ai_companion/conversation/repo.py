"""会话时间线仓库：写入、窗口读取、昵称表、历史回填。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..memory.db import PersonaDatabase
from .model import SPEAKER_BOT, SPEAKER_MEMBER, ConversationTurn, session_key_for, turn_from_row

logger = logging.getLogger(__name__)

# 窗口读取上限：即使配置写得很大也不无限膨胀（在线端点与本地模型共用的硬上限）。
MAX_WINDOW_TURNS = 200


class ConversationRepo:
    """conversation_turns 的读写。全部方法线程池执行，调用方直接 await。"""

    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    # ---- 写入 ----
    async def record_turn(
        self,
        *,
        session_key: str,
        speaker_kind: str,
        text: str,
        group_id: str | None = None,
        user_id: str | None = None,
        speaker_name: str = "",
        sticker_keys: list[str] | None = None,
        trigger_reason: str = "",
        reply_to_message_id: str = "",
        source_message_id: str = "",
        created_at: float | None = None,
    ) -> None:
        turn = ConversationTurn(
            session_key=session_key,
            speaker_kind=speaker_kind,
            text=str(text or ""),
            group_id=group_id,
            user_id=user_id,
            speaker_name=str(speaker_name or ""),
            sticker_keys=[str(key) for key in (sticker_keys or []) if str(key).strip()],
            trigger_reason=str(trigger_reason or ""),
            reply_to_message_id=str(reply_to_message_id or ""),
            source_message_id=str(source_message_id or ""),
            created_at=float(created_at if created_at is not None else time.time()),
        )
        await asyncio.to_thread(self._insert_sync, turn)

    def _insert_sync(self, turn: ConversationTurn) -> None:
        self.db.ensure_schema()
        row = turn.to_row()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns (
                    session_key, group_id, user_id, speaker_kind, speaker_name, text,
                    sticker_keys, trigger_reason, reply_to_message_id, source_message_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["session_key"],
                    row["group_id"],
                    row["user_id"],
                    row["speaker_kind"],
                    row["speaker_name"],
                    row["text"],
                    json.dumps(row["sticker_keys"], ensure_ascii=False),
                    row["trigger_reason"],
                    row["reply_to_message_id"],
                    row["source_message_id"],
                    row["created_at"],
                ),
            )

    # ---- 读取 ----
    async def recent_turns(
        self,
        *,
        session_key: str,
        limit: int,
        since: float | None = None,
    ) -> list[ConversationTurn]:
        """取该会话最近 limit 条，按时间正序返回（旧 → 新）。

        since（如角色卡绑定时刻）之后的时间线才算数：换绑人格后不从旧上下文续。
        """
        limit = max(0, min(int(limit), MAX_WINDOW_TURNS))
        if limit <= 0:
            return []
        return await asyncio.to_thread(self._recent_turns_sync, session_key, limit, since)

    def _recent_turns_sync(self, session_key: str, limit: int, since: float | None) -> list[ConversationTurn]:
        self.db.ensure_schema()
        sql = "SELECT * FROM conversation_turns WHERE session_key = ?"
        params: list[Any] = [session_key]
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(float(since))
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [turn_from_row(row) for row in reversed(rows)]

    async def count_turns(self, *, session_key: str | None = None) -> int:
        return await asyncio.to_thread(self._count_turns_sync, session_key)

    async def list_session_groups(self) -> list[str]:
        """时间线里出现过的群号（去重）——昵称回填与统计用。"""
        return await asyncio.to_thread(self._list_session_groups_sync)

    def _list_session_groups_sync(self) -> list[str]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT group_id FROM conversation_turns WHERE group_id IS NOT NULL AND group_id != ''"
            ).fetchall()
        return [str(row["group_id"]) for row in rows if str(row["group_id"] or "").strip()]

    def _count_turns_sync(self, session_key: str | None) -> int:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            if session_key:
                row = conn.execute(
                    "SELECT COUNT(*) FROM conversation_turns WHERE session_key = ?", (session_key,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()
        return int(row[0] or 0)

    # ---- 历史回填 ----
    async def backfill_from_samples(self, *, limit: int = 0) -> dict[str, int]:
        """把 chat_samples（学习语料）回填成会话时间线，供冷启动的上下文与记忆使用。

        幂等：(session_key, source_message_id, created_at) 已存在的样本跳过。
        昵称无法离线解析，留空由渲染层兜底显示「群友xxxx」。
        """
        return await asyncio.to_thread(self._backfill_sync, limit)

    def _backfill_sync(self, limit: int) -> dict[str, int]:
        self.db.ensure_schema()
        scanned = inserted = skipped = 0
        with self.db.connection() as conn:
            sql = """
                SELECT created_at, group_id, user_id, message_id, plain_text, sticker_keys
                FROM chat_samples
                WHERE plain_text != '' OR sticker_keys != '[]'
                ORDER BY created_at ASC
            """
            params: tuple[Any, ...] = ()
            if int(limit) > 0:
                sql += " LIMIT ?"
                params = (int(limit),)
            rows = conn.execute(sql, params).fetchall()
            for row in rows:
                scanned += 1
                group_id = row["group_id"]
                user_id = row["user_id"]
                session_key = session_key_for(group_id=group_id, user_id=user_id)
                message_id = str(row["message_id"] or "")
                created_at = float(row["created_at"] or 0.0)
                if message_id:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM conversation_turns
                        WHERE session_key = ? AND source_message_id = ?
                        LIMIT 1
                        """,
                        (session_key, message_id),
                    ).fetchone()
                else:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM conversation_turns
                        WHERE session_key = ? AND created_at = ? AND speaker_kind = 'member'
                        LIMIT 1
                        """,
                        (session_key, created_at),
                    ).fetchone()
                if existing:
                    skipped += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO conversation_turns (
                        session_key, group_id, user_id, speaker_kind, speaker_name, text,
                        sticker_keys, trigger_reason, reply_to_message_id, source_message_id, created_at
                    ) VALUES (?, ?, ?, ?, '', ?, ?, '', '', ?, ?)
                    """,
                    (
                        session_key,
                        group_id,
                        user_id,
                        SPEAKER_MEMBER,
                        str(row["plain_text"] or ""),
                        str(row["sticker_keys"] or "[]"),
                        message_id,
                        created_at,
                    ),
                )
                inserted += 1
        logger.info(f"conversation backfill: scanned={scanned} inserted={inserted} skipped={skipped}")
        return {"scanned": scanned, "inserted": inserted, "skipped": skipped}


class MemberNameRepo:
    """昵称表：写时快照，历史渲染零额外 API。"""

    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def upsert_name(
        self,
        *,
        group_id: str | None,
        user_id: str,
        name: str,
        updated_at: float | None = None,
    ) -> None:
        clean_name = str(name or "").strip()
        clean_user = str(user_id or "").strip()
        if not clean_user or not clean_name:
            return
        await asyncio.to_thread(self._upsert_sync, group_id or "", clean_user, clean_name, updated_at)

    def _upsert_sync(self, group_id: str, user_id: str, name: str, updated_at: float | None) -> None:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO member_names (group_id, user_id, name, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    name = excluded.name,
                    updated_at = excluded.updated_at
                """,
                (group_id, user_id, name, float(updated_at if updated_at is not None else time.time())),
            )

    async def get_names(self, *, group_id: str | None, user_ids: list[str]) -> dict[str, str]:
        clean_ids = [str(item).strip() for item in user_ids if str(item).strip()]
        if not clean_ids:
            return {}
        return await asyncio.to_thread(self._get_names_sync, group_id or "", clean_ids)

    def _get_names_sync(self, group_id: str, user_ids: list[str]) -> dict[str, str]:
        self.db.ensure_schema()
        placeholders = ",".join("?" for _ in user_ids)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT user_id, name FROM member_names WHERE group_id = ? AND user_id IN ({placeholders})",
                (group_id, *user_ids),
            ).fetchall()
        return {str(row["user_id"]): str(row["name"] or "") for row in rows if str(row["name"] or "").strip()}

    async def list_names(self, *, group_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_names_sync, (group_id or "") if group_id is not None else None, limit)

    def _list_names_sync(self, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            if group_id is None:
                rows = conn.execute(
                    "SELECT group_id, user_id, name, updated_at FROM member_names ORDER BY updated_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT group_id, user_id, name, updated_at FROM member_names
                    WHERE group_id = ? ORDER BY updated_at DESC LIMIT ?
                    """,
                    (group_id, max(1, int(limit))),
                ).fetchall()
        return [
            {
                "group_id": str(row["group_id"]),
                "user_id": str(row["user_id"]),
                "name": str(row["name"]),
                "updated_at": float(row["updated_at"] or 0.0),
            }
            for row in rows
        ]


__all__ = ["MAX_WINDOW_TURNS", "SPEAKER_BOT", "SPEAKER_MEMBER", "ConversationRepo", "MemberNameRepo"]
