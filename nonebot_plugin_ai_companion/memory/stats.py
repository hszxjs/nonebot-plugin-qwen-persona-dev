"""统计聚合：管理后台 dashboard 所需的跨表只读统计。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..finetune.runs import training_row_payload
from .db import PersonaDatabase


def _mask_group_id(group_id: Any) -> str:
    value = str(group_id or "unknown")
    if value == "unknown":
        return "unknown"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}...{value[-2:]}"

class StatsRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def stats(self) -> dict[str, int]:
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, int]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
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
        self.db.ensure_schema()
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

        with self.db.connection() as conn:
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
                "last_success": training_row_payload(last_success),
                "last_run": training_row_payload(last_training),
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
