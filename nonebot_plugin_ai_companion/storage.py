from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote, unquote, urlparse

from .filters import is_command_like, is_training_noise, normalize_text
from .generation_safety import is_low_quality_generated_text
from .media import context_text_from_sample
from .sticker import (
    build_training_text,
    is_transient_image_sticker_data,
    meaningful_protocol_summary,
    parse_sticker_tokens,
    sticker_to_record,
)
from .sticker_classifier import classify_sticker_record, is_sendable_sticker_record

MEDIA_ASSET_SOURCES = {"sticker", "admin_approved", "explicit_command"}
CHAT_LOG_PREFIX_RE = re.compile(r"^\s*\d{5,12}\s*[:：]\s*")


class PersonaStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "persona.sqlite3"

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def _init_sync(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    platform TEXT,
                    user_id TEXT,
                    group_id TEXT,
                    message_id TEXT,
                    plain_text TEXT NOT NULL,
                    training_text TEXT NOT NULL,
                    raw_segments TEXT NOT NULL,
                    sticker_keys TEXT NOT NULL,
                    learn_reason TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stickers (
                    key TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    summary TEXT,
                    uri TEXT,
                    data_json TEXT NOT NULL,
                    group_id TEXT,
                    user_id TEXT,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    dataset_path TEXT,
                    output_dir TEXT,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    log TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_patterns (
                    group_id TEXT NOT NULL,
                    trigger_signature TEXT NOT NULL,
                    trigger_text TEXT NOT NULL,
                    response_signature TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    response_sticker_keys TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (group_id, trigger_signature, response_signature)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_corpus_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(source, source_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(source, source_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_events (
                    event_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    model TEXT NOT NULL,
                    adapter TEXT NOT NULL,
                    latency_seconds REAL,
                    tps REAL,
                    media_count INTEGER NOT NULL DEFAULT 0,
                    response_chars INTEGER NOT NULL DEFAULT 0,
                    error_summary TEXT NOT NULL,
                    input_preview TEXT NOT NULL,
                    output_preview TEXT NOT NULL,
                    group_id TEXT,
                    user_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sticker_events (
                    event_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    key TEXT,
                    group_id TEXT,
                    user_id TEXT,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_samples_group_time "
                "ON chat_samples(group_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stickers_group_count "
                "ON stickers(group_id, count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_patterns_lookup "
                "ON conversation_patterns(group_id, trigger_signature, count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_external_corpus_source_time "
                "ON external_corpus_samples(source, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_assets_source_time "
                "ON media_assets(source, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_inference_events_time "
                "ON inference_events(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sticker_events_time "
                "ON sticker_events(created_at DESC)"
            )

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
        self._init_sync()
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
            "error_summary": self._preview_text(error_summary, limit=240),
            "input_preview": self._preview_text(input_text, limit=240),
            "output_preview": self._preview_text(output_text, limit=240),
            "group_id": None if group_id is None else str(group_id),
            "user_id": None if user_id is None else str(user_id),
        }
        with self._connection() as conn:
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
        self._init_sync()
        limit = max(1, min(int(limit or 300), 300))
        offset = max(0, int(offset or 0))
        with self._connection() as conn:
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
        self._init_sync()
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
            "message": self._preview_text(message, limit=240),
            "details": details_payload,
        }
        with self._connection() as conn:
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
        self._init_sync()
        limit = max(1, min(int(limit or 100), 300))
        offset = max(0, int(offset or 0))
        with self._connection() as conn:
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
            "details": self._json_object(row["details_json"]),
        }

    async def cleanup_empty_chat_artifacts(self) -> dict[str, int]:
        return await asyncio.to_thread(self._cleanup_empty_chat_artifacts_sync)

    def _cleanup_empty_chat_artifacts_sync(self) -> dict[str, int]:
        self._init_sync()
        with self._connection() as conn:
            sample_cursor = conn.execute(
                """
                DELETE FROM chat_samples
                WHERE trim(coalesce(plain_text, '')) = ''
                  AND trim(coalesce(training_text, '')) = ''
                  AND trim(coalesce(sticker_keys, '[]')) IN ('', '[]')
                """
            )
            pattern_cursor = conn.execute(
                """
                DELETE FROM conversation_patterns
                WHERE trim(coalesce(response_text, '')) = ''
                  AND trim(coalesce(response_sticker_keys, '[]')) IN ('', '[]')
                """
            )
        return {
            "samples_deleted": max(0, sample_cursor.rowcount),
            "patterns_deleted": max(0, pattern_cursor.rowcount),
        }

    async def cleanup_training_noise(
        self,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._cleanup_training_noise_sync,
            command_prefixes=command_prefixes,
            command_regexes=command_regexes,
        )

    def _cleanup_training_noise_sync(
        self,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        self._init_sync()
        with self._connection() as conn:
            sample_rows = conn.execute(
                "SELECT id, training_text FROM chat_samples WHERE training_text != ''"
            ).fetchall()
            sample_ids = [
                int(row["id"])
                for row in sample_rows
                if self._is_training_noise_text(
                    str(row["training_text"] or ""),
                    command_prefixes=command_prefixes,
                    command_regexes=command_regexes,
                )
            ]
            if sample_ids:
                placeholders = ",".join("?" for _ in sample_ids)
                conn.execute(
                    f"DELETE FROM chat_samples WHERE id IN ({placeholders})",
                    sample_ids,
                )

            pattern_rows = conn.execute(
                """
                SELECT rowid, trigger_text, response_text
                FROM conversation_patterns
                """
            ).fetchall()
            pattern_ids = [
                int(row["rowid"])
                for row in pattern_rows
                if self._is_training_noise_text(
                    str(row["trigger_text"] or ""),
                    command_prefixes=command_prefixes,
                    command_regexes=command_regexes,
                )
                or self._is_training_noise_text(
                    str(row["response_text"] or ""),
                    command_prefixes=command_prefixes,
                    command_regexes=command_regexes,
                )
            ]
            if pattern_ids:
                placeholders = ",".join("?" for _ in pattern_ids)
                conn.execute(
                    f"DELETE FROM conversation_patterns WHERE rowid IN ({placeholders})",
                    pattern_ids,
                )
        return {
            "samples_deleted": len(sample_ids),
            "patterns_deleted": len(pattern_ids),
        }

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
        self._init_sync()
        source = normalize_text(source) or "external"
        inserted = 0
        skipped = 0
        duplicates = 0
        with self._connection() as conn:
            for record in records:
                text = normalize_text(str(record.get("text") or ""))
                if (
                    len(text) < max(1, int(min_chars))
                    or len(text) > max(1, int(max_chars))
                    or self._is_training_noise_text(
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
        self._init_sync()
        now = time.time()
        sticker_keys = [item["key"] for item in stickers]
        with self._connection() as conn:
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
                    self._record_media_asset_conn(
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

    async def record_media_asset(
        self,
        *,
        source: str,
        source_key: str,
        uri: str,
        prompt: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._record_media_asset_sync,
            source=source,
            source_key=source_key,
            uri=uri,
            prompt=prompt,
            answer=answer,
            metadata=metadata or {},
        )

    def _record_media_asset_sync(
        self,
        *,
        source: str,
        source_key: str,
        uri: str,
        prompt: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        self._init_sync()
        source = str(source or "").strip()
        source_key = str(source_key or "").strip()
        uri = str(uri or "").strip()
        prompt = normalize_text(prompt)
        answer = normalize_text(answer)
        if source not in MEDIA_ASSET_SOURCES or not source_key or not uri or not answer:
            return {"inserted": 0, "duplicates": 0, "skipped": 1}
        with self._connection() as conn:
            cursor = self._record_media_asset_conn(
                conn,
                source=source,
                source_key=source_key,
                uri=uri,
                prompt=prompt or "请用一句中文描述这张图。",
                answer=answer,
                metadata=metadata or {},
                created_at=time.time(),
            )
        inserted = int(cursor.rowcount or 0)
        return {
            "inserted": inserted,
            "duplicates": 0 if inserted else 1,
            "skipped": 0,
        }

    async def record_sticker_asset(
        self,
        *,
        sticker: dict[str, Any],
        group_id: str | None,
        user_id: str | None,
        summary: str = "",
        increment_count: bool = True,
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._record_sticker_asset_sync,
            sticker=sticker,
            group_id=group_id,
            user_id=user_id,
            summary=summary,
            increment_count=increment_count,
        )

    def _record_sticker_asset_sync(
        self,
        *,
        sticker: dict[str, Any],
        group_id: str | None,
        user_id: str | None,
        summary: str = "",
        increment_count: bool = True,
    ) -> dict[str, int]:
        self._init_sync()
        key = str(sticker.get("key") or "").strip()
        segment_type = str(sticker.get("type") or "").strip()
        data = dict(sticker.get("data") or {})
        uri = str(
            sticker.get("uri")
            or data.get("url")
            or data.get("file")
            or data.get("file_id")
            or data.get("id")
            or ""
        ).strip()
        clean_summary = normalize_text(summary or str(sticker.get("summary") or ""))
        if not key or not segment_type:
            return {"stickers_upserted": 0, "media_inserted": 0, "skipped": 1}
        if clean_summary:
            data["summary"] = clean_summary
        now = time.time()
        count_delta = 1 if increment_count else 0
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO stickers (
                    key, type, summary, uri, data_json, group_id, user_id,
                    first_seen, last_seen, count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    summary=COALESCE(NULLIF(excluded.summary, ''), stickers.summary),
                    uri=CASE
                        WHEN COALESCE(stickers.uri, '') LIKE 'file://%' THEN stickers.uri
                        WHEN COALESCE(excluded.uri, '') != '' THEN excluded.uri
                        ELSE stickers.uri
                    END,
                    data_json=excluded.data_json,
                    group_id=COALESCE(excluded.group_id, stickers.group_id),
                    user_id=COALESCE(excluded.user_id, stickers.user_id),
                    last_seen=excluded.last_seen,
                    count=stickers.count + excluded.count
                """,
                (
                    key,
                    segment_type,
                    clean_summary,
                    uri,
                    json.dumps(data, ensure_ascii=False, default=str),
                    group_id,
                    user_id,
                    now,
                    now,
                    count_delta,
                ),
            )
            media_inserted = 0
            if uri and clean_summary:
                media_cursor = self._record_media_asset_conn(
                    conn,
                    source="sticker",
                    source_key=key,
                    uri=uri,
                    prompt="请用一句中文描述这个群聊表情的含义。",
                    answer=clean_summary,
                    metadata={
                        "group_id": group_id,
                        "user_id": user_id,
                        "sticker_type": segment_type,
                        "background_analysis": True,
                    },
                    created_at=now,
                )
                media_inserted = int(media_cursor.rowcount or 0)
        return {
            "stickers_upserted": int(cursor.rowcount or 0),
            "media_inserted": media_inserted,
            "skipped": 0,
        }

    def _record_media_asset_conn(
        self,
        conn: sqlite3.Connection,
        *,
        source: str,
        source_key: str,
        uri: str,
        prompt: str,
        answer: str,
        metadata: dict[str, Any],
        created_at: float,
    ) -> sqlite3.Cursor:
        return conn.execute(
            """
            INSERT OR IGNORE INTO media_assets (
                source, source_key, uri, prompt, answer, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                source_key,
                uri,
                prompt,
                answer,
                json.dumps(metadata, ensure_ascii=False, default=str),
                created_at,
            ),
        )

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
        self._init_sync()
        now = time.time()
        with self._connection() as conn:
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
        self._init_sync()
        with self._connection() as conn:
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
        item["response_sticker_keys"] = self._json_list(
            item.get("response_sticker_keys")
        )
        return item

    async def recent_samples(self, *, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._recent_samples_sync, group_id, limit)

    def _recent_samples_sync(self, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        self._init_sync()
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, limit) if group_id else (limit,)
        with self._connection() as conn:
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
        self._init_sync()
        terms = self._context_search_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit or 5), 30))
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, max(80, limit * 12)) if group_id else (max(80, limit * 12),)
        with self._connection() as conn:
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
    def _context_search_terms(query: str) -> list[str]:
        raw_terms = [
            normalize_text(term)
            for term in re.split(r"\s+", str(query or ""))
            if normalize_text(term)
        ]
        terms: list[str] = []
        seen: set[str] = set()
        for term in raw_terms:
            candidates = [term]
            if len(term) >= 8 and re.search(r"[\u4e00-\u9fff]", term):
                candidates.extend(term[index : index + 2] for index in range(0, len(term) - 1))
            for candidate in candidates:
                candidate = candidate.lower()
                if len(candidate) < 1 or candidate in seen:
                    continue
                seen.add(candidate)
                terms.append(candidate)
        return terms

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

    async def search_media_assets(
        self,
        *,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_media_assets_sync,
            group_id=group_id,
            query=query,
            limit=limit,
        )

    def _search_media_assets_sync(
        self,
        *,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self._init_sync()
        terms = self._context_search_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit or 5), 30))
        rows = self._media_asset_rows(limit=max(80, limit * 12))
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for item in rows:
            metadata = self._json_object(item.get("metadata_json"))
            if group_id and str(metadata.get("group_id") or "") != str(group_id):
                continue
            haystack = self._media_asset_search_haystack(item, metadata)
            score = sum(len(term) for term in terms if term in haystack)
            if score <= 0:
                continue
            item = dict(item)
            item["metadata"] = metadata
            scored.append((score, float(item.get("created_at") or 0), item))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _score, _created_at, item in scored[:limit]]

    @staticmethod
    def _media_asset_search_haystack(
        item: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        return " ".join(
            str(value or "").lower()
            for value in (
                item.get("prompt"),
                item.get("answer"),
                item.get("uri"),
                item.get("source_key"),
                json.dumps(metadata, ensure_ascii=False, default=str),
            )
        )

    async def top_stickers(self, *, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._top_stickers_sync, group_id, limit)

    def _top_stickers_sync(self, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        self._init_sync()
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, limit) if group_id else (limit,)
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT key, type, summary, uri, data_json, count, last_seen
                FROM stickers
                {where}
                ORDER BY count DESC, last_seen DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._sticker_row_payload(row) for row in rows]

    async def get_sticker(self, key: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_sticker_sync, key)

    def _get_sticker_sync(self, key: str) -> dict[str, Any] | None:
        self._init_sync()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT key, type, summary, uri, data_json, count, last_seen
                FROM stickers
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if not row:
            return None
        return self._sticker_row_payload(row)

    async def list_stickers(
        self,
        *,
        limit: int = 80,
        offset: int = 0,
        query: str = "",
        segment_type: str = "",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._list_stickers_sync,
            limit,
            offset,
            query,
            segment_type,
        )

    def _list_stickers_sync(
        self,
        limit: int = 80,
        offset: int = 0,
        query: str = "",
        segment_type: str = "",
    ) -> dict[str, Any]:
        self._init_sync()
        limit = max(1, min(int(limit or 80), 200))
        offset = max(0, int(offset or 0))
        query = str(query or "").strip()
        segment_type = str(segment_type or "").strip()

        where_parts: list[str] = []
        params: list[Any] = []
        if segment_type:
            where_parts.append("type = ?")
            params.append(segment_type)
        if query:
            like = f"%{query}%"
            where_parts.append(
                """
                (
                    key LIKE ?
                    OR type LIKE ?
                    OR COALESCE(summary, '') LIKE ?
                    OR COALESCE(uri, '') LIKE ?
                    OR COALESCE(group_id, '') LIKE ?
                    OR COALESCE(user_id, '') LIKE ?
                    OR data_json LIKE ?
                )
                """
            )
            params.extend([like, like, like, like, like, like, like])
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        with self._connection() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM stickers {where}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT key, type, summary, uri, data_json, group_id, user_id,
                       first_seen, last_seen, count
                FROM stickers
                {where}
                ORDER BY last_seen DESC, count DESC, key ASC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
            type_rows = conn.execute(
                """
                SELECT type, COUNT(*) AS count
                FROM stickers
                GROUP BY type
                ORDER BY count DESC, type ASC
                """
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(self._sticker_row_payload(row))

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "type": segment_type,
            "type_counts": {str(row["type"]): int(row["count"] or 0) for row in type_rows},
        }

    def _sticker_row_payload(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["data"] = self._json_object(item.pop("data_json", "{}"))
        item["count"] = int(item.get("count") or 0)
        if item.get("first_seen") is not None:
            item["first_seen"] = float(item["first_seen"])
        if item.get("last_seen") is not None:
            item["last_seen"] = float(item["last_seen"])
        classification = classify_sticker_record(item)
        item["classification"] = classification.to_dict()
        item["sendable"] = bool(classification.sendable)
        item["preview_url"] = self._sticker_preview_url(item)
        return item

    def _sticker_preview_url(self, item: dict[str, Any]) -> str | None:
        key = str(item.get("key") or "").strip()
        if not key:
            return None
        if self._local_sticker_media_path(item, require_exists=True) is None:
            return None
        return f"/api/stickers/{quote(key, safe='')}/preview"

    def _local_sticker_media_path(
        self,
        item: dict[str, Any],
        *,
        require_exists: bool,
    ) -> Path | None:
        data = dict(item.get("data") or {})
        for value in (
            item.get("uri"),
            data.get("url"),
            data.get("file"),
            data.get("file_id"),
            data.get("uri"),
        ):
            path = self._path_from_file_uri(value)
            if path is None:
                continue
            if require_exists and not path.is_file():
                continue
            return path
        return None

    @staticmethod
    def _path_from_file_uri(value: Any) -> Path | None:
        parsed = urlparse(str(value or ""))
        if parsed.scheme != "file":
            return None
        path = Path(unquote(parsed.path)).expanduser()
        try:
            return path.resolve()
        except OSError:
            return None

    async def get_sticker_preview_path(
        self,
        key: str,
        *,
        repo_dir: str | Path,
    ) -> Path | None:
        return await asyncio.to_thread(self._get_sticker_preview_path_sync, key, repo_dir)

    def _get_sticker_preview_path_sync(
        self,
        key: str,
        repo_dir: str | Path,
    ) -> Path | None:
        item = self._get_sticker_sync(key)
        if not item:
            return None
        path = self._local_sticker_media_path(item, require_exists=True)
        if path is None:
            return None
        media_dir = (Path(repo_dir) / "media").resolve()
        try:
            path.relative_to(media_dir)
        except ValueError:
            return None
        return path

    async def delete_stickers(self, keys: list[str]) -> dict[str, int]:
        return await asyncio.to_thread(self._delete_stickers_sync, keys)

    def _delete_stickers_sync(self, keys: list[str]) -> dict[str, int]:
        self._init_sync()
        unique_keys: list[str] = []
        seen: set[str] = set()
        for key in keys:
            value = str(key or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            unique_keys.append(value)
        if not unique_keys:
            return {"requested": 0, "deleted": 0}
        if len(unique_keys) > 500:
            raise ValueError("too many stickers selected")

        placeholders = ",".join("?" for _ in unique_keys)
        with self._connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM stickers WHERE key IN ({placeholders})",
                unique_keys,
            )
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
        return {"requested": len(unique_keys), "deleted": int(deleted)}

    async def update_sticker_summary(self, key: str, summary: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_sticker_summary_sync, key, summary)

    def _update_sticker_summary_sync(self, key: str, summary: str) -> dict[str, Any]:
        self._init_sync()
        key = str(key or "").strip()
        if not key:
            raise ValueError("key is required")
        summary = " ".join(str(summary or "").split())[:200]

        with self._connection() as conn:
            row = conn.execute(
                "SELECT data_json FROM stickers WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return {"updated": 0, "item": None}
            data = self._json_object(row["data_json"])
            data["summary"] = summary
            cursor = conn.execute(
                """
                UPDATE stickers
                SET summary = ?, data_json = ?
                WHERE key = ?
                """,
                (summary, json.dumps(data, ensure_ascii=False), key),
            )
            updated = int(cursor.rowcount or 0)

        return {"updated": updated, "item": self._get_sticker_sync(key)}

    async def normalize_sticker_summaries_from_protocol(self) -> dict[str, int]:
        return await asyncio.to_thread(self._normalize_sticker_summaries_from_protocol_sync)

    def _normalize_sticker_summaries_from_protocol_sync(self) -> dict[str, int]:
        self._init_sync()
        updated = 0
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT key, summary, data_json FROM stickers"
            ).fetchall()
            for row in rows:
                data = self._json_object(row["data_json"])
                protocol_summary = meaningful_protocol_summary(data)
                if not protocol_summary:
                    continue
                if str(row["summary"] or "") == protocol_summary:
                    continue
                data["summary"] = protocol_summary
                cursor = conn.execute(
                    """
                    UPDATE stickers
                    SET summary = ?, data_json = ?
                    WHERE key = ?
                    """,
                    (
                        protocol_summary,
                        json.dumps(data, ensure_ascii=False),
                        row["key"],
                    ),
                )
                updated += int(cursor.rowcount or 0)
        return {"updated": updated}

    async def deduplicate_stickers_by_canonical_key(
        self,
        sticker_segment_types: Iterable[str],
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._deduplicate_stickers_by_canonical_key_sync,
            sticker_segment_types,
        )

    def _deduplicate_stickers_by_canonical_key_sync(
        self,
        sticker_segment_types: Iterable[str],
    ) -> dict[str, int]:
        self._init_sync()
        enabled_types = {str(item) for item in sticker_segment_types}
        merged_stickers = 0
        rekeyed_stickers = 0
        rewritten_samples = 0
        with self._connection() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT key, type, summary, uri, data_json, group_id, user_id,
                           first_seen, last_seen, count
                    FROM stickers
                    ORDER BY first_seen ASC
                    """
                ).fetchall()
            ]
            groups: dict[str, list[dict[str, Any]]] = {}
            key_map: dict[str, str] = {}
            for row in rows:
                data = self._json_object(row.get("data_json"))
                canonical = sticker_to_record(
                    {"type": row.get("type"), "data": data},
                    enabled_types,
                )
                if canonical is None:
                    continue
                canonical_key = str(canonical["key"])
                groups.setdefault(canonical_key, []).append(row)
                old_key = str(row["key"])
                if old_key != canonical_key:
                    key_map[old_key] = canonical_key

            for canonical_key, grouped_rows in groups.items():
                old_keys = [str(row["key"]) for row in grouped_rows]
                if len(grouped_rows) == 1 and old_keys[0] == canonical_key:
                    continue
                preferred = max(
                    grouped_rows,
                    key=lambda row: (
                        str(row.get("uri") or "").startswith("file://"),
                        bool(str(row.get("summary") or "").strip()),
                        float(row.get("last_seen") or 0),
                    ),
                )
                summary = str(preferred.get("summary") or "").strip()
                if not summary:
                    summary = next(
                        (
                            str(row.get("summary") or "").strip()
                            for row in grouped_rows
                            if str(row.get("summary") or "").strip()
                        ),
                        "",
                    )
                uri = str(preferred.get("uri") or "").strip() or None
                if not uri:
                    uri = next(
                        (
                            str(row.get("uri") or "").strip()
                            for row in grouped_rows
                            if str(row.get("uri") or "").strip()
                        ),
                        None,
                    )
                placeholders = ",".join("?" for _ in old_keys)
                conn.execute(
                    f"DELETE FROM stickers WHERE key IN ({placeholders})",
                    old_keys,
                )
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_key,
                        preferred["type"],
                        summary,
                        uri,
                        preferred["data_json"],
                        preferred.get("group_id"),
                        preferred.get("user_id"),
                        min(float(row["first_seen"]) for row in grouped_rows),
                        max(float(row["last_seen"]) for row in grouped_rows),
                        sum(int(row["count"]) for row in grouped_rows),
                    ),
                )
                merged_stickers += max(0, len(grouped_rows) - 1)
                rekeyed_stickers += sum(
                    1 for old_key in old_keys if old_key != canonical_key
                )

            if key_map:
                samples = conn.execute(
                    "SELECT id, training_text, sticker_keys FROM chat_samples"
                ).fetchall()
                for row in samples:
                    current_keys = [
                        str(key)
                        for key in self._json_list(row["sticker_keys"])
                        if str(key).strip()
                    ]
                    mapped_keys = list(
                        dict.fromkeys(key_map.get(key, key) for key in current_keys)
                    )
                    training_text = str(row["training_text"] or "")
                    base_text, embedded_keys = parse_sticker_tokens(training_text)
                    mapped_embedded_keys = list(
                        dict.fromkeys(key_map.get(key, key) for key in embedded_keys)
                    )
                    next_training_text = training_text
                    if mapped_embedded_keys != embedded_keys:
                        next_training_text = build_training_text(
                            base_text,
                            mapped_embedded_keys,
                        )
                    next_sticker_keys = json.dumps(mapped_keys, ensure_ascii=False)
                    if (
                        next_training_text == training_text
                        and next_sticker_keys == str(row["sticker_keys"] or "[]")
                    ):
                        continue
                    conn.execute(
                        """
                        UPDATE chat_samples
                        SET training_text = ?, sticker_keys = ?
                        WHERE id = ?
                        """,
                        (next_training_text, next_sticker_keys, row["id"]),
                    )
                    rewritten_samples += 1

        return {
            "merged_stickers": merged_stickers,
            "rekeyed_stickers": rekeyed_stickers,
            "rewritten_samples": rewritten_samples,
        }

    async def cleanup_legacy_media_artifacts(self) -> dict[str, int]:
        return await asyncio.to_thread(self._cleanup_legacy_media_artifacts_sync)

    def _cleanup_legacy_media_artifacts_sync(self) -> dict[str, int]:
        self._init_sync()
        deleted_image_stickers = 0
        updated_samples = 0
        with self._connection() as conn:
            invalid_image_keys = {
                str(row["key"])
                for row in conn.execute(
                    """
                    SELECT key, type, summary, uri, data_json
                    FROM stickers
                    WHERE type = 'image'
                    """
                ).fetchall()
                if not is_sendable_sticker_record(
                    {
                        "key": row["key"],
                        "type": row["type"],
                        "summary": row["summary"],
                        "uri": row["uri"],
                        "data": self._json_object(row["data_json"]),
                    }
                )
            }
            if invalid_image_keys:
                placeholders = ",".join("?" for _ in invalid_image_keys)
                cursor = conn.execute(
                    f"DELETE FROM stickers WHERE key IN ({placeholders})",
                    list(invalid_image_keys),
                )
                deleted_image_stickers = int(cursor.rowcount or 0)

            rows = conn.execute(
                """
                SELECT id, plain_text, training_text, sticker_keys
                FROM chat_samples
                """
            ).fetchall()
            for row in rows:
                sample = dict(row)
                training_text = str(sample.get("training_text") or "")
                has_legacy_media_description = (
                    "图片：" in training_text
                    or "表情：" in training_text
                    or "图：" in training_text
                )
                sticker_keys = [
                    key
                    for key in self._json_list(sample.get("sticker_keys"))
                    if str(key).strip() and str(key) not in invalid_image_keys
                ]
                if has_legacy_media_description:
                    sample["sticker_keys"] = json.dumps(sticker_keys, ensure_ascii=False)
                    safe_text = context_text_from_sample(sample)
                elif invalid_image_keys:
                    base_text, _removed_keys = parse_sticker_tokens(training_text)
                    safe_text = build_training_text(base_text, sticker_keys)
                else:
                    continue
                next_sticker_keys = json.dumps(sticker_keys, ensure_ascii=False)
                if safe_text == training_text and next_sticker_keys == str(
                    sample.get("sticker_keys") or "[]"
                ):
                    continue
                conn.execute(
                    """
                    UPDATE chat_samples
                    SET training_text = ?, sticker_keys = ?
                    WHERE id = ?
                    """,
                    (
                        safe_text,
                        next_sticker_keys,
                        sample["id"],
                    ),
                )
                updated_samples += 1
        return {
            "deleted_image_stickers": deleted_image_stickers,
            "updated_samples": updated_samples,
        }

    async def backfill_stickers_from_raw_segments(
        self,
        sticker_segment_types: Iterable[str],
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self._backfill_stickers_from_raw_segments_sync,
            sticker_segment_types,
        )

    def _backfill_stickers_from_raw_segments_sync(
        self,
        sticker_segment_types: Iterable[str],
    ) -> dict[str, int]:
        self._init_sync()
        enabled_types = {str(item) for item in sticker_segment_types}
        samples_updated = 0
        stickers_upserted = 0
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, user_id, group_id, plain_text, training_text,
                       raw_segments, sticker_keys
                FROM chat_samples
                WHERE raw_segments LIKE '%"type": "image"%'
                   OR raw_segments LIKE '%"type": "face"%'
                   OR raw_segments LIKE '%"type": "mface"%'
                   OR raw_segments LIKE '%"type": "emoji"%'
                   OR raw_segments LIKE '%"type": "sticker"%'
                """
            ).fetchall()
            for row in rows:
                sample = dict(row)
                try:
                    raw_segments = json.loads(sample.get("raw_segments") or "[]")
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw_segments, list):
                    continue

                recovered: list[dict[str, Any]] = []
                seen_recovered: set[str] = set()
                for segment in raw_segments:
                    if not isinstance(segment, dict):
                        continue
                    if (
                        str(segment.get("type") or "") == "image"
                        and is_transient_image_sticker_data(dict(segment.get("data") or {}))
                    ):
                        continue
                    record = sticker_to_record(segment, enabled_types)
                    if record is None or record["key"] in seen_recovered:
                        continue
                    seen_recovered.add(record["key"])
                    recovered.append(record)
                if not recovered:
                    continue

                for sticker in recovered:
                    conn.execute(
                        """
                        INSERT INTO stickers (
                            key, type, summary, uri, data_json, group_id, user_id,
                            first_seen, last_seen, count
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(key) DO UPDATE SET
                            summary=COALESCE(NULLIF(stickers.summary, ''), excluded.summary),
                            uri=CASE
                                WHEN COALESCE(stickers.uri, '') LIKE 'file://%' THEN stickers.uri
                                WHEN COALESCE(excluded.uri, '') != '' THEN excluded.uri
                                ELSE stickers.uri
                            END,
                            data_json=excluded.data_json,
                            group_id=COALESCE(stickers.group_id, excluded.group_id),
                            user_id=COALESCE(stickers.user_id, excluded.user_id),
                            last_seen=MAX(stickers.last_seen, excluded.last_seen),
                            count=MAX(stickers.count, 1)
                        """,
                        (
                            sticker["key"],
                            sticker["type"],
                            sticker.get("summary"),
                            sticker.get("uri"),
                            json.dumps(sticker.get("data") or {}, ensure_ascii=False, default=str),
                            sample.get("group_id"),
                            sample.get("user_id"),
                            float(sample.get("created_at") or time.time()),
                            float(sample.get("created_at") or time.time()),
                        ),
                    )
                    stickers_upserted += 1

                current_keys = [str(key) for key in self._json_list(sample.get("sticker_keys")) if str(key).strip()]
                merged_keys: list[str] = []
                seen_keys: set[str] = set()
                for key in [*current_keys, *(item["key"] for item in recovered)]:
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged_keys.append(key)

                training_text = str(sample.get("training_text") or "")
                base_text, embedded_keys = parse_sticker_tokens(training_text)
                if not base_text:
                    base_text = str(sample.get("plain_text") or "").strip()
                if not base_text:
                    summaries = [
                        f"表情：{item.get('summary')}"
                        for item in recovered
                        if str(item.get("summary") or "").strip()
                    ]
                    base_text = " ".join(summaries)
                for key in embedded_keys:
                    if key not in seen_keys:
                        seen_keys.add(key)
                        merged_keys.append(key)

                next_training_text = build_training_text(base_text, merged_keys)
                next_sticker_keys = json.dumps(merged_keys, ensure_ascii=False)
                if (
                    next_training_text != training_text
                    or next_sticker_keys != str(sample.get("sticker_keys") or "[]")
                ):
                    conn.execute(
                        """
                        UPDATE chat_samples
                        SET training_text = ?, sticker_keys = ?
                        WHERE id = ?
                        """,
                        (next_training_text, next_sticker_keys, sample["id"]),
                    )
                    samples_updated += 1
        return {"samples_updated": samples_updated, "stickers_upserted": stickers_upserted}

    @staticmethod
    def _json_object(raw: Any) -> dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _json_list(raw: Any) -> list[Any]:
        try:
            data = json.loads(raw or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    async def export_sft_dataset(
        self,
        *,
        output_path: Path,
        limit: int,
        history_window: int,
        system_prompt: str,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
        external_limit: int = 0,
        include_multimodal_assets: bool = False,
        excluded_user_ids: Iterable[str] = (),
        excluded_message_ids: Iterable[str] = (),
    ) -> int:
        return await asyncio.to_thread(
            self._export_sft_dataset_sync,
            output_path,
            limit,
            history_window,
            system_prompt,
            command_prefixes,
            command_regexes,
            external_limit,
            include_multimodal_assets,
            excluded_user_ids,
            excluded_message_ids,
        )

    def _export_sft_dataset_sync(
        self,
        output_path: Path,
        limit: int,
        history_window: int,
        system_prompt: str,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
        external_limit: int = 0,
        include_multimodal_assets: bool = False,
        excluded_user_ids: Iterable[str] = (),
        excluded_message_ids: Iterable[str] = (),
    ) -> int:
        self._init_sync()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_limit = max(int(limit or 0), 0)
        unlimited_dataset = dataset_limit <= 0
        excluded_users = {str(item) for item in excluded_user_ids if str(item).strip()}
        excluded_messages = {str(item) for item in excluded_message_ids if str(item).strip()}
        with self._connection() as conn:
            if unlimited_dataset:
                rows = conn.execute(
                    """
                    SELECT id, user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                    FROM chat_samples
                    WHERE training_text != ''
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            else:
                fetch_limit = max(dataset_limit + history_window + 32, dataset_limit)
                rows = conn.execute(
                    """
                    SELECT id, user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                    FROM chat_samples
                    WHERE training_text != ''
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (fetch_limit,),
                ).fetchall()

        ordered = [dict(row) for row in reversed(rows)]
        exported = 0
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for idx, row in enumerate(ordered):
                if not unlimited_dataset and exported >= dataset_limit:
                    break
                history = ordered[max(0, idx - history_window) : idx]
                history_lines = []
                for item in history:
                    if self._is_excluded_dataset_row(
                        item,
                        excluded_user_ids=excluded_users,
                        excluded_message_ids=excluded_messages,
                    ):
                        continue
                    context_text = context_text_from_sample(item)
                    if self._is_excluded_dataset_row(
                        item,
                        excluded_user_ids=excluded_users,
                        excluded_message_ids=excluded_messages,
                        assistant_text=context_text,
                    ):
                        continue
                    context_text = self._strip_chat_log_prefix(context_text)
                    if context_text and not self._is_training_noise_text(
                        context_text,
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    ):
                        history_lines.append(f"上一句：{context_text}")
                prompt = "\n".join(history_lines) or "群聊刚开始，请直接接住当前话题。"
                assistant_text = context_text_from_sample(row)
                if self._is_excluded_dataset_row(
                    row,
                    excluded_user_ids=excluded_users,
                    excluded_message_ids=excluded_messages,
                    assistant_text=assistant_text,
                ):
                    continue
                assistant_text = self._strip_chat_log_prefix(assistant_text)
                if not assistant_text or self._is_training_noise_text(
                    assistant_text,
                    command_prefixes=command_prefixes,
                    command_regexes=command_regexes,
                ) or is_low_quality_generated_text(assistant_text):
                    continue
                record = {
                    "images": [],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "下面是群聊上下文。请优先理解当前语境，按自然群聊方式回复：\n" + prompt,
                        },
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "metadata": {
                        "source": "chat_samples",
                        "sample_id": row["id"],
                        "group_id": row.get("group_id"),
                        "user_id": row.get("user_id"),
                        "created_at": row.get("created_at"),
                    },
                }
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                exported += 1
            external_limit = int(external_limit or 0)
            external_enabled = external_limit >= 0
            has_dataset_room = unlimited_dataset or exported < dataset_limit
            if external_enabled and has_dataset_room:
                if external_limit > 0:
                    row_limit = external_limit if unlimited_dataset else min(external_limit, dataset_limit - exported)
                else:
                    row_limit = 0 if unlimited_dataset else max(dataset_limit - exported, 0)
                for row in self._external_corpus_rows(limit=row_limit):
                    raw_text = normalize_text(str(row.get("text") or ""))
                    if (
                        not raw_text
                        or self._has_chat_log_prefix(raw_text)
                        or self._is_training_noise_text(
                            raw_text,
                            command_prefixes=command_prefixes,
                            command_regexes=command_regexes,
                        )
                        or is_low_quality_generated_text(raw_text)
                    ):
                        continue
                    text = self._strip_chat_log_prefix(raw_text)
                    if not text:
                        text,
                        continue
                    metadata = self._sanitize_metadata_chat_log_prefixes(
                        self._json_object(row.get("metadata_json"))
                    )
                    user_prompt = normalize_text(str(metadata.get("user_prompt") or ""))
                    user_content = (
                        "下面是外部公开社区语境，请按这种口吻自然接话：\n" + user_prompt
                        if user_prompt
                        else "下面是一条公开社区语料，请学习其自然口吻，不要输出来源信息："
                    )
                    record = {
                        "images": [],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": text},
                        ],
                        "metadata": {
                            "source": row.get("source"),
                            "source_id": row.get("source_id"),
                            "external_sample_id": row.get("id"),
                            "created_at": row.get("created_at"),
                            "external": True,
                            "extra": metadata,
                        },
                    }
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    exported += 1
                    if not unlimited_dataset and exported >= dataset_limit:
                        break
            has_dataset_room = unlimited_dataset or exported < dataset_limit
            if include_multimodal_assets and has_dataset_room:
                if unlimited_dataset:
                    media_limit = 0
                else:
                    media_limit = max(dataset_limit - exported, 0)
                for row in self._media_asset_rows(limit=media_limit):
                    uri = str(row.get("uri") or "").strip()
                    raw_answer = normalize_text(str(row.get("answer") or ""))
                    if (
                        not uri
                        or not raw_answer
                        or self._has_chat_log_prefix(raw_answer)
                        or is_low_quality_generated_text(raw_answer)
                    ):
                        continue
                    answer = self._strip_chat_log_prefix(raw_answer)
                    prompt = self._strip_chat_log_prefix(
                        normalize_text(str(row.get("prompt") or ""))
                    )
                    metadata = self._sanitize_metadata_chat_log_prefixes(
                        self._json_object(row.get("metadata_json"))
                    )
                    record = {
                        "images": [uri],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": uri},
                                    {
                                        "type": "text",
                                        "text": prompt or "请按群聊语境理解这张图片。",
                                    },
                                ],
                            },
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": {
                            "source": row.get("source"),
                            "source_key": row.get("source_key"),
                            "media_asset_id": row.get("id"),
                            "created_at": row.get("created_at"),
                            "multimodal": True,
                            "extra": metadata,
                        },
                    }
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    exported += 1
                    if not unlimited_dataset and exported >= dataset_limit:
                        break
        return exported

    @staticmethod
    def _is_excluded_dataset_row(
        row: dict[str, Any],
        *,
        excluded_user_ids: set[str],
        excluded_message_ids: set[str],
        assistant_text: str | None = None,
    ) -> bool:
        user_id = str(row.get("user_id") or "")
        message_id = str(row.get("message_id") or "")
        if user_id and user_id in excluded_user_ids:
            return True
        if message_id and message_id in excluded_message_ids:
            return True
        if assistant_text is not None and CHAT_LOG_PREFIX_RE.match(str(assistant_text or "")):
            return True
        return False

    @staticmethod
    def _strip_chat_log_prefix(text: str) -> str:
        return CHAT_LOG_PREFIX_RE.sub("", str(text or "")).strip()

    @staticmethod
    def _has_chat_log_prefix(text: str) -> bool:
        return bool(CHAT_LOG_PREFIX_RE.match(str(text or "")))

    @classmethod
    def _sanitize_metadata_chat_log_prefixes(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_metadata_chat_log_prefixes(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize_metadata_chat_log_prefixes(item) for item in value]
        if isinstance(value, str):
            return cls._strip_chat_log_prefix(value)
        return value

    def _external_corpus_rows(self, *, limit: int) -> list[dict[str, Any]]:
        if limit < 0:
            return []
        with self._connection() as conn:
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

    def _media_asset_rows(self, *, limit: int) -> list[dict[str, Any]]:
        if limit < 0:
            return []
        with self._connection() as conn:
            if limit == 0:
                rows = conn.execute(
                    """
                    SELECT id, source, source_key, uri, prompt, answer, metadata_json, created_at
                    FROM media_assets
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, source, source_key, uri, prompt, answer, metadata_json, created_at
                    FROM media_assets
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _is_training_noise_text(
        text: str,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> bool:
        return is_training_noise(text) or is_command_like(
            text,
            prefixes=command_prefixes,
            regexes=command_regexes,
        )

    async def begin_training_run(
        self,
        *,
        run_id: str,
        reason: str,
        dataset_path: str,
        output_dir: str,
        sample_count: int,
    ) -> None:
        await asyncio.to_thread(
            self._begin_training_run_sync,
            run_id,
            reason,
            dataset_path,
            output_dir,
            sample_count,
        )

    def _begin_training_run_sync(
        self,
        run_id: str,
        reason: str,
        dataset_path: str,
        output_dir: str,
        sample_count: int,
    ) -> None:
        self._init_sync()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_runs (
                    id, status, reason, started_at, dataset_path, output_dir, sample_count, log
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "running",
                    reason,
                    time.time(),
                    dataset_path,
                    output_dir,
                    sample_count,
                    "",
                ),
            )

    async def finish_training_run(self, *, run_id: str, status: str, log: str) -> None:
        await asyncio.to_thread(self._finish_training_run_sync, run_id, status, log)

    def _finish_training_run_sync(self, run_id: str, status: str, log: str) -> None:
        self._init_sync()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, finished_at = ?, log = ?
                WHERE id = ?
                """,
                (status, time.time(), log[-12000:], run_id),
            )

    async def stats(self) -> dict[str, int]:
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, int]:
        self._init_sync()
        with self._connection() as conn:
            samples = conn.execute("SELECT COUNT(*) FROM chat_samples").fetchone()[0]
            stickers = conn.execute("SELECT COUNT(*) FROM stickers").fetchone()[0]
            training_runs = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            media_assets = conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0]
        return {
            "samples": int(samples),
            "stickers": int(stickers),
            "training_runs": int(training_runs),
            "media_assets": int(media_assets),
        }

    async def dashboard_stats(self, *, days: int = 14, group_limit: int = 8) -> dict[str, Any]:
        return await asyncio.to_thread(self._dashboard_stats_sync, days, group_limit)

    def _dashboard_stats_sync(self, days: int = 14, group_limit: int = 8) -> dict[str, Any]:
        self._init_sync()
        days = max(1, min(int(days), 60))
        group_limit = max(1, min(int(group_limit), 20))
        now = time.time()
        start_day = time.localtime(now - (days - 1) * 86400)
        start_epoch = time.mktime(
            (
                start_day.tm_year,
                start_day.tm_mon,
                start_day.tm_mday,
                0,
                0,
                0,
                start_day.tm_wday,
                start_day.tm_yday,
                start_day.tm_isdst,
            )
        )

        with self._connection() as conn:
            total_samples = int(conn.execute("SELECT COUNT(*) FROM chat_samples").fetchone()[0])
            trainable_samples = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_samples WHERE training_text != ''"
                ).fetchone()[0]
            )
            stickers = int(conn.execute("SELECT COUNT(*) FROM stickers").fetchone()[0])
            external_corpus_samples = int(
                conn.execute("SELECT COUNT(*) FROM external_corpus_samples").fetchone()[0]
            )
            media_assets = int(
                conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0]
            )
            training_runs = int(conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0])
            groups = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT group_id) FROM chat_samples WHERE group_id IS NOT NULL"
                ).fetchone()[0]
            )
            users = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM chat_samples WHERE user_id IS NOT NULL"
                ).fetchone()[0]
            )
            samples_with_stickers = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_samples WHERE sticker_keys != '[]'"
                ).fetchone()[0]
            )
            today_start = time.mktime(
                (
                    time.localtime(now).tm_year,
                    time.localtime(now).tm_mon,
                    time.localtime(now).tm_mday,
                    0,
                    0,
                    0,
                    time.localtime(now).tm_wday,
                    time.localtime(now).tm_yday,
                    time.localtime(now).tm_isdst,
                )
            )
            today_samples = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_samples WHERE created_at >= ?",
                    (today_start,),
                ).fetchone()[0]
            )
            last_24h_samples = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_samples WHERE created_at >= ?",
                    (now - 86400,),
                ).fetchone()[0]
            )
            minmax = conn.execute(
                """
                SELECT MIN(created_at), MAX(created_at), AVG(LENGTH(training_text))
                FROM chat_samples
                """
            ).fetchone()
            last_success = conn.execute(
                """
                SELECT id, status, reason, started_at, finished_at, sample_count, dataset_path, output_dir
                FROM training_runs
                WHERE status = 'success'
                ORDER BY COALESCE(finished_at, started_at) DESC
                LIMIT 1
                """
            ).fetchone()
            last_training = conn.execute(
                """
                SELECT id, status, reason, started_at, finished_at, sample_count, dataset_path, output_dir
                FROM training_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            pending_since = (
                float(last_success["finished_at"] or last_success["started_at"])
                if last_success
                else 0.0
            )
            pending_samples = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM chat_samples
                    WHERE training_text != '' AND created_at > ?
                    """,
                    (pending_since,),
                ).fetchone()[0]
            )
            sample_times = [
                float(row[0])
                for row in conn.execute(
                    """
                    SELECT created_at
                    FROM chat_samples
                    WHERE training_text != '' AND created_at >= ?
                    """,
                    (start_epoch,),
                ).fetchall()
            ]
            group_rows = conn.execute(
                """
                SELECT group_id, COUNT(*) AS count
                FROM chat_samples
                WHERE training_text != ''
                GROUP BY group_id
                ORDER BY count DESC
                LIMIT ?
                """,
                (group_limit,),
            ).fetchall()

        buckets: dict[str, int] = {}
        for offset in range(days):
            bucket_time = start_epoch + offset * 86400
            buckets[time.strftime("%m-%d", time.localtime(bucket_time))] = 0
        for created_at in sample_times:
            label = time.strftime("%m-%d", time.localtime(created_at))
            if label in buckets:
                buckets[label] += 1

        return {
            "totals": {
                "learned_messages": total_samples,
                "trainable_samples": trainable_samples,
                "pending_training_samples": pending_samples,
                "stickers": stickers,
                "external_corpus_samples": external_corpus_samples,
                "media_assets": media_assets,
                "training_runs": training_runs,
                "groups": groups,
                "users": users,
                "samples_with_stickers": samples_with_stickers,
                "today_samples": today_samples,
                "last_24h_samples": last_24h_samples,
                "first_sample_at": float(minmax[0]) if minmax and minmax[0] is not None else None,
                "last_sample_at": float(minmax[1]) if minmax and minmax[1] is not None else None,
                "avg_training_chars": round(float(minmax[2]), 1) if minmax and minmax[2] is not None else None,
            },
            "training": {
                "last_success": self._training_row_payload(last_success),
                "last_run": self._training_row_payload(last_training),
                "pending_since": pending_since or None,
            },
            "charts": {
                "learning_trend": [
                    {"label": label, "count": count} for label, count in buckets.items()
                ],
                "group_contribution": [
                    {
                        "label": self._mask_group_id(row["group_id"]),
                        "count": int(row["count"]),
                    }
                    for row in group_rows
                ],
            },
        }

    @staticmethod
    def _training_row_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": row["id"],
            "status": row["status"],
            "reason": row["reason"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "sample_count": int(row["sample_count"] or 0),
            "dataset_path": row["dataset_path"],
            "output_dir": row["output_dir"],
        }

    @staticmethod
    def _mask_group_id(group_id: Any) -> str:
        value = str(group_id or "unknown")
        if value == "unknown":
            return "unknown"
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}...{value[-2:]}"



