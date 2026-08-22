"""微调域：训练运行记录仓库。"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any

from ..memory.db import PersonaDatabase


def training_row_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
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


class TrainingRunRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

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
        self.db.ensure_schema()
        with self.db.connection() as conn:
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
        self.db.ensure_schema()
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, finished_at = ?, log = ?
                WHERE id = ?
                """,
                (status, time.time(), log[-12000:], run_id),
            )
