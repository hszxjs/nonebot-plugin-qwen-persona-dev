"""记忆域共享数据库层：SQLite 连接、DDL、JSON/文本工具。

所有领域仓库共享同一个 PersonaDatabase 实例（同一 persona.sqlite3），
每个同步方法仍先 ensure_schema()（幂等 IF NOT EXISTS），与旧实现一致。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..prompt.text import is_command_like, is_training_noise, normalize_text


def context_search_terms(query: str) -> list[str]:
    raw_terms = [normalize_text(term) for term in re.split(r"\s+", str(query or "")) if normalize_text(term)]
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


class PersonaDatabase:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "persona.sqlite3"

    async def init(self) -> None:
        await asyncio.to_thread(self.ensure_schema)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
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

    @staticmethod
    def preview_text(value: str, *, limit: int = 240) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def json_object(raw: Any) -> dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def json_list(raw: Any) -> list[Any]:
        try:
            data = json.loads(raw or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def ensure_schema(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_samples_group_time ON chat_samples(group_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stickers_group_count ON stickers(group_id, count)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_patterns_lookup "
                "ON conversation_patterns(group_id, trigger_signature, count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_external_corpus_source_time "
                "ON external_corpus_samples(source, created_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_assets_source_time ON media_assets(source, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_events_time ON inference_events(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sticker_events_time ON sticker_events(created_at DESC)")

    @staticmethod
    def is_noise_text(
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
