"""工具调用日志（文档十三节 ToolCallLog）：KPI 工具成功率与耗时来源。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..memory.db import PersonaDatabase


class ToolCallLogRepo:
    def __init__(self, db: PersonaDatabase, *, max_records: int = 2000) -> None:
        self.db = db
        self.max_records = max(50, int(max_records))

    async def record(
        self,
        *,
        name: str,
        ok: bool,
        latency_seconds: float,
        group_id: str | None = None,
        arguments: Any = None,
        result: Any = None,
        created_at: float | None = None,
    ) -> None:
        import json

        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_call_logs (
                        created_at, name, ok, latency_seconds, group_id,
                        arguments_preview, result_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        float(created_at if created_at is not None else time.time()),
                        str(name or ""),
                        1 if ok else 0,
                        round(max(0.0, float(latency_seconds)), 3),
                        group_id,
                        _preview(json.dumps(arguments, ensure_ascii=False, default=str))
                        if arguments is not None
                        else "",
                        _preview(
                            result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                        )
                        if result is not None
                        else "",
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM tool_call_logs
                    WHERE id NOT IN (
                        SELECT id FROM tool_call_logs ORDER BY created_at DESC LIMIT ?
                    )
                    """,
                    (self.max_records,),
                )

        await asyncio.to_thread(_sync)

    async def success_stats(self, *, days: int = 7) -> dict[str, Any]:
        def _sync() -> dict[str, Any]:
            self.db.ensure_schema()
            since = time.time() - max(1, int(days)) * 86400
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT name,
                           COUNT(*) AS total,
                           SUM(ok) AS ok_count,
                           AVG(latency_seconds) AS avg_latency
                    FROM tool_call_logs
                    WHERE created_at >= ?
                    GROUP BY name
                    ORDER BY total DESC
                    """,
                    (since,),
                ).fetchall()
            items = [
                {
                    "name": row["name"],
                    "total": int(row["total"] or 0),
                    "ok": int(row["ok_count"] or 0),
                    "success_rate": round(int(row["ok_count"] or 0) / int(row["total"] or 1), 3),
                    "avg_latency_seconds": round(float(row["avg_latency"] or 0), 3),
                }
                for row in rows
            ]
            total = sum(item["total"] for item in items)
            ok = sum(item["ok"] for item in items)
            return {
                "items": items,
                "total_calls": total,
                "overall_success_rate": round(ok / total, 3) if total else None,
            }

        return await asyncio.to_thread(_sync)


def _preview(text: str, *, limit: int = 200) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:limit]
