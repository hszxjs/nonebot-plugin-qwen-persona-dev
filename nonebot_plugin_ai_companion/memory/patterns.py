"""记忆域：群聊对话模式（trigger/response 对）仓库。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

from .db import PersonaDatabase


class ConversationPatternRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def record_conversation_pattern(
        self,
        *,
        group_id: str,
        trigger_signature: str,
        trigger_text: str,
        response_signature: str,
        response_text: str,
        response_sticker_keys: list[str],
    ) -> None:
        await asyncio.to_thread(
            self._record_conversation_pattern_sync,
            group_id=group_id,
            trigger_signature=trigger_signature,
            trigger_text=trigger_text,
            response_signature=response_signature,
            response_text=response_text,
            response_sticker_keys=response_sticker_keys,
        )

    def _record_conversation_pattern_sync(
        self,
        *,
        group_id: str,
        trigger_signature: str,
        trigger_text: str,
        response_signature: str,
        response_text: str,
        response_sticker_keys: list[str],
    ) -> None:
        self.db.ensure_schema()
        now = time.time()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_patterns (
                    group_id, trigger_signature, trigger_text,
                    response_signature, response_text, response_sticker_keys,
                    first_seen, last_seen, count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(group_id, trigger_signature, response_signature)
                DO UPDATE SET
                    trigger_text=excluded.trigger_text,
                    response_text=excluded.response_text,
                    response_sticker_keys=excluded.response_sticker_keys,
                    last_seen=excluded.last_seen,
                    count=conversation_patterns.count + 1
                """,
                (
                    group_id,
                    trigger_signature,
                    trigger_text,
                    response_signature,
                    response_text,
                    json.dumps(response_sticker_keys, ensure_ascii=False),
                    now,
                    now,
                ),
            )
    async def get_conversation_pattern(
        self,
        *,
        group_id: str,
        trigger_signature: str,
        min_count: int,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._get_conversation_pattern_sync,
            group_id=group_id,
            trigger_signature=trigger_signature,
            min_count=min_count,
        )

    def _get_conversation_pattern_sync(
        self,
        *,
        group_id: str,
        trigger_signature: str,
        min_count: int,
    ) -> dict[str, Any] | None:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT trigger_text, response_signature, response_text,
                       response_sticker_keys, count, last_seen
                FROM conversation_patterns
                WHERE group_id = ?
                  AND trigger_signature = ?
                  AND count >= ?
                ORDER BY count DESC, last_seen DESC
                LIMIT 1
                """,
                (group_id, trigger_signature, max(1, int(min_count))),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["response_sticker_keys"] = self.db.json_list(
            item.get("response_sticker_keys")
        )
        return item

    async def cleanup_empty_patterns(self) -> int:
        def _sync() -> int:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM conversation_patterns
                    WHERE trim(coalesce(response_text, '')) = ''
                      AND trim(coalesce(response_sticker_keys, '[]')) IN ('', '[]')
                    """
                )
            return max(0, cursor.rowcount)

        return await asyncio.to_thread(_sync)

    async def cleanup_noise_patterns(
        self,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> int:
        def _sync() -> int:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT rowid, trigger_text, response_text FROM conversation_patterns"
                ).fetchall()
                ids = [
                    int(row["rowid"])
                    for row in rows
                    if self.db.is_noise_text(
                        str(row["trigger_text"] or ""),
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    )
                    or self.db.is_noise_text(
                        str(row["response_text"] or ""),
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    )
                ]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(
                        f"DELETE FROM conversation_patterns WHERE rowid IN ({placeholders})",
                        ids,
                    )
            return len(ids)

        return await asyncio.to_thread(_sync)
