"""记忆域：外部语料样本仓库。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Iterable

from ..prompt.text import normalize_text
from .db import PersonaDatabase


def external_corpus_rows(db: PersonaDatabase, *, limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        return []
    with db.connection() as conn:
        if limit == 0:
            rows = conn.execute(
                """
                SELECT id, source, source_id, text, metadata_json, created_at
                FROM external_corpus_samples
                ORDER BY created_at DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, source, source_id, text, metadata_json, created_at
                FROM external_corpus_samples
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    return [dict(row) for row in rows]


class ExternalCorpusRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def import_external_corpus_samples(
        self,
        *,
        source: str,
        records: Iterable[dict[str, Any]],
        min_chars: int,
        max_chars: int,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._import_external_corpus_samples_sync,
            source=source,
            records=records,
            min_chars=min_chars,
            max_chars=max_chars,
            command_prefixes=command_prefixes,
            command_regexes=command_regexes,
        )

    def _import_external_corpus_samples_sync(
        self,
        *,
        source: str,
        records: Iterable[dict[str, Any]],
        min_chars: int,
        max_chars: int,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        self.db.ensure_schema()
        source = normalize_text(source) or "external"
        inserted = 0
        skipped = 0
        duplicates = 0
        with self.db.connection() as conn:
            for record in records:
                text = normalize_text(str(record.get("text") or ""))
                if (
                    len(text) < max(1, int(min_chars))
                    or len(text) > max(1, int(max_chars))
                    or self.db.is_noise_text(
                        text,
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    )
                ):
                    skipped += 1
                    continue

                source_id = normalize_text(str(record.get("source_id") or ""))
                if not source_id:
                    source_id = hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()
                metadata = record.get("metadata")
                metadata_json = json.dumps(
                    metadata if isinstance(metadata, dict) else {},
                    ensure_ascii=False,
                    default=str,
                )
                created_at = record.get("created_at")
                try:
                    created_at_value = float(created_at)
                except (TypeError, ValueError):
                    created_at_value = time.time()
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO external_corpus_samples (
                        source, source_id, text, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source, source_id, text, metadata_json, created_at_value),
                )
                if cursor.rowcount:
                    inserted += 1
                else:
                    duplicates += 1
        return {"inserted": inserted, "skipped": skipped, "duplicates": duplicates}
