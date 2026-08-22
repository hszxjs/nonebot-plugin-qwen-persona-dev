"""记忆域：聊天样本（短期上下文来源）仓库。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Iterable

from ..prompt.media_text import context_text_from_sample
from ..prompt.text import is_training_noise, normalize_text
from ..sticker.assets import insert_media_asset_conn
from .db import PersonaDatabase, context_search_terms


class ChatSampleRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def record_sample(
        self,
        *,
        platform: str | None,
        user_id: str | None,
        group_id: str | None,
        message_id: str | None,
        plain_text: str,
        training_text: str,
        raw_segments: list[dict[str, Any]],
        stickers: list[dict[str, Any]],
        learn_reason: str,
    ) -> None:
        await asyncio.to_thread(
            self._record_sample_sync,
            platform,
            user_id,
            group_id,
            message_id,
            plain_text,
            training_text,
            raw_segments,
            stickers,
            learn_reason,
        )
    def _record_sample_sync(
        self,
        platform: str | None,
        user_id: str | None,
        group_id: str | None,
        message_id: str | None,
        plain_text: str,
        training_text: str,
        raw_segments: list[dict[str, Any]],
        stickers: list[dict[str, Any]],
        learn_reason: str,
    ) -> None:
        self.db.ensure_schema()
        now = time.time()
        sticker_keys = [item["key"] for item in stickers]
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_samples (
                    created_at, platform, user_id, group_id, message_id,
                    plain_text, training_text, raw_segments, sticker_keys, learn_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    platform,
                    user_id,
                    group_id,
                    message_id,
                    plain_text,
                    training_text,
                    json.dumps(raw_segments, ensure_ascii=False, default=str),
                    json.dumps(sticker_keys, ensure_ascii=False),
                    learn_reason,
                ),
            )
            for sticker in stickers:
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key) DO UPDATE SET
                        summary=excluded.summary,
                        uri=CASE
                            WHEN COALESCE(stickers.uri, '') LIKE 'file://%' THEN stickers.uri
                            WHEN COALESCE(excluded.uri, '') != '' THEN excluded.uri
                            ELSE stickers.uri
                        END,
                        data_json=excluded.data_json,
                        group_id=excluded.group_id,
                        user_id=excluded.user_id,
                        last_seen=excluded.last_seen,
                        count=stickers.count + 1
                    """,
                    (
                        sticker["key"],
                        sticker["type"],
                        sticker.get("summary"),
                        sticker.get("uri"),
                        json.dumps(sticker.get("data") or {}, ensure_ascii=False, default=str),
                        group_id,
                        user_id,
                        now,
                        now,
                    ),
                )
                uri = str(sticker.get("uri") or "").strip()
                summary = str(sticker.get("summary") or "").strip()
                if uri and summary:
                    insert_media_asset_conn(
                        conn,
                        source="sticker",
                        source_key=str(sticker["key"]),
                        uri=uri,
                        prompt="请用一句中文描述这个群聊表情的含义。",
                        answer=summary,
                        metadata={
                            "group_id": group_id,
                            "user_id": user_id,
                            "sticker_type": sticker.get("type"),
                        },
                        created_at=now,
                    )
    async def recent_samples(self, *, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._recent_samples_sync, group_id, limit)

    def _recent_samples_sync(self, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        self.db.ensure_schema()
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, limit) if group_id else (limit,)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                FROM chat_samples
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]
    async def search_context_samples(
        self,
        *,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_context_samples_sync,
            group_id=group_id,
            query=query,
            limit=limit,
        )

    def _search_context_samples_sync(
        self,
        *,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.db.ensure_schema()
        terms = context_search_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit or 5), 30))
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, max(80, limit * 12)) if group_id else (max(80, limit * 12),)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                FROM chat_samples
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        scored: list[tuple[int, float, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            context_text = context_text_from_sample(item)
            if not context_text or is_training_noise(context_text):
                continue
            haystack = self._context_search_haystack(item, context_text)
            score = sum(len(term) for term in terms if term in haystack)
            if score <= 0:
                continue
            scored.append((score, float(item.get("created_at") or 0), item))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _score, _created_at, item in scored[:limit]]

    @staticmethod
    def _context_search_haystack(item: dict[str, Any], context_text: str) -> str:
        return " ".join(
            str(value or "").lower()
            for value in (
                context_text,
                item.get("plain_text"),
                item.get("training_text"),
                item.get("user_id"),
            )
        )

    async def cleanup_empty_samples(self) -> int:
        def _sync() -> int:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM chat_samples
                    WHERE trim(coalesce(plain_text, '')) = ''
                      AND trim(coalesce(training_text, '')) = ''
                      AND trim(coalesce(sticker_keys, '[]')) IN ('', '[]')
                    """
                )
            return max(0, cursor.rowcount)

        return await asyncio.to_thread(_sync)

    async def cleanup_noise_samples(
        self,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> int:
        def _sync() -> int:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, training_text FROM chat_samples WHERE training_text != ''"
                ).fetchall()
                ids = [
                    int(row["id"])
                    for row in rows
                    if self.db.is_noise_text(
                        str(row["training_text"] or ""),
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    )
                ]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(f"DELETE FROM chat_samples WHERE id IN ({placeholders})", ids)
            return len(ids)

        return await asyncio.to_thread(_sync)
