"""表情包域持久化：stickers 与 media_assets 两个实体的仓库。

去重/回填/遗留清理等维护例程会顺带重写 chat_samples 中的表情 token
（跨表但同属表情域语义，见各方法 docstring 与架构文档）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from ..memory.db import PersonaDatabase, context_search_terms
from ..prompt.media_text import context_text_from_sample
from ..prompt.text import normalize_text
from .classifier import classify_sticker_record, is_sendable_sticker_record
from .model import (
    build_training_text,
    is_transient_image_sticker_data,
    meaningful_protocol_summary,
    parse_sticker_tokens,
    sticker_to_record,
)

MEDIA_ASSET_SOURCES = {"sticker", "admin_approved", "explicit_command"}


def insert_media_asset_conn(
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


def media_asset_rows(db: PersonaDatabase, *, limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        return []
    with db.connection() as conn:
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


class StickerAssetRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

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
        self.db.ensure_schema()
        source = str(source or "").strip()
        source_key = str(source_key or "").strip()
        uri = str(uri or "").strip()
        prompt = normalize_text(prompt)
        answer = normalize_text(answer)
        if source not in MEDIA_ASSET_SOURCES or not source_key or not uri or not answer:
            return {"inserted": 0, "duplicates": 0, "skipped": 1}
        with self.db.connection() as conn:
            cursor = insert_media_asset_conn(
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
        self.db.ensure_schema()
        key = str(sticker.get("key") or "").strip()
        segment_type = str(sticker.get("type") or "").strip()
        data = dict(sticker.get("data") or {})
        uri = str(
            sticker.get("uri") or data.get("url") or data.get("file") or data.get("file_id") or data.get("id") or ""
        ).strip()
        clean_summary = normalize_text(summary or str(sticker.get("summary") or ""))
        if not key or not segment_type:
            return {"stickers_upserted": 0, "media_inserted": 0, "skipped": 1}
        if clean_summary:
            data["summary"] = clean_summary
        now = time.time()
        count_delta = 1 if increment_count else 0
        with self.db.connection() as conn:
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
                media_cursor = insert_media_asset_conn(
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
        self.db.ensure_schema()
        terms = context_search_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit or 5), 30))
        rows = media_asset_rows(self.db, limit=max(80, limit * 12))
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for item in rows:
            metadata = self.db.json_object(item.get("metadata_json"))
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
        self.db.ensure_schema()
        where = "WHERE group_id = ?" if group_id else ""
        params: tuple[Any, ...] = (group_id, limit) if group_id else (limit,)
        with self.db.connection() as conn:
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
        self.db.ensure_schema()
        with self.db.connection() as conn:
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
        self.db.ensure_schema()
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

        with self.db.connection() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM stickers {where}", params).fetchone()[0])
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
        item["data"] = self.db.json_object(item.pop("data_json", "{}"))
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
        self.db.ensure_schema()
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
        with self.db.connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM stickers WHERE key IN ({placeholders})",
                unique_keys,
            )
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
        return {"requested": len(unique_keys), "deleted": int(deleted)}

    async def update_sticker_summary(self, key: str, summary: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_sticker_summary_sync, key, summary)

    def _update_sticker_summary_sync(self, key: str, summary: str) -> dict[str, Any]:
        self.db.ensure_schema()
        key = str(key or "").strip()
        if not key:
            raise ValueError("key is required")
        summary = " ".join(str(summary or "").split())[:200]

        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT data_json FROM stickers WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return {"updated": 0, "item": None}
            data = self.db.json_object(row["data_json"])
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
        self.db.ensure_schema()
        updated = 0
        with self.db.connection() as conn:
            rows = conn.execute("SELECT key, summary, data_json FROM stickers").fetchall()
            for row in rows:
                data = self.db.json_object(row["data_json"])
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
        self.db.ensure_schema()
        enabled_types = {str(item) for item in sticker_segment_types}
        merged_stickers = 0
        rekeyed_stickers = 0
        rewritten_samples = 0
        with self.db.connection() as conn:
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
                data = self.db.json_object(row.get("data_json"))
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
                rekeyed_stickers += sum(1 for old_key in old_keys if old_key != canonical_key)

            if key_map:
                samples = conn.execute("SELECT id, training_text, sticker_keys FROM chat_samples").fetchall()
                for row in samples:
                    current_keys = [str(key) for key in self.db.json_list(row["sticker_keys"]) if str(key).strip()]
                    mapped_keys = list(dict.fromkeys(key_map.get(key, key) for key in current_keys))
                    training_text = str(row["training_text"] or "")
                    base_text, embedded_keys = parse_sticker_tokens(training_text)
                    mapped_embedded_keys = list(dict.fromkeys(key_map.get(key, key) for key in embedded_keys))
                    next_training_text = training_text
                    if mapped_embedded_keys != embedded_keys:
                        next_training_text = build_training_text(
                            base_text,
                            mapped_embedded_keys,
                        )
                    next_sticker_keys = json.dumps(mapped_keys, ensure_ascii=False)
                    if next_training_text == training_text and next_sticker_keys == str(row["sticker_keys"] or "[]"):
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
        self.db.ensure_schema()
        deleted_image_stickers = 0
        updated_samples = 0
        with self.db.connection() as conn:
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
                        "data": self.db.json_object(row["data_json"]),
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
                    "图片：" in training_text or "表情：" in training_text or "图：" in training_text
                )
                sticker_keys = [
                    key
                    for key in self.db.json_list(sample.get("sticker_keys"))
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
                if safe_text == training_text and next_sticker_keys == str(sample.get("sticker_keys") or "[]"):
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
        self.db.ensure_schema()
        enabled_types = {str(item) for item in sticker_segment_types}
        samples_updated = 0
        stickers_upserted = 0
        with self.db.connection() as conn:
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
                    if str(segment.get("type") or "") == "image" and is_transient_image_sticker_data(
                        dict(segment.get("data") or {})
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

                current_keys = [str(key) for key in self.db.json_list(sample.get("sticker_keys")) if str(key).strip()]
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
                        f"表情：{item.get('summary')}" for item in recovered if str(item.get("summary") or "").strip()
                    ]
                    base_text = " ".join(summaries)
                for key in embedded_keys:
                    if key not in seen_keys:
                        seen_keys.add(key)
                        merged_keys.append(key)

                next_training_text = build_training_text(base_text, merged_keys)
                next_sticker_keys = json.dumps(merged_keys, ensure_ascii=False)
                if next_training_text != training_text or next_sticker_keys != str(sample.get("sticker_keys") or "[]"):
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
