"""成就计数仓库：每群每人独立计数，记录全部消息（含命令/黑名单消息）。

与 conversation_turns 的分工：时间线只收「干净」消息（命令、黑名单被学习管线
拦掉），成就计数是群活跃度的统计口径，在里程碑 matcher 里独立自增，与学习管线
完全解耦——学习开关、命令、黑名单都不影响数字。

老群不归零、也不补发：某群第一次出现计数行时，用时间线做一次性基线回填
（排除正在处理的这条消息，由随后的 +1 统一计入），并把「已播报阈值」设为
已达成的最高档，因此部署那一刻不会给历史发言补一摞成就卡。

计数、基线建立、跨线认领在同一事务里完成（BEGIN IMMEDIATE），并发消息不会
重复认领同一个阈值。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass

from ..memory.db import PersonaDatabase
from .milestone import crossed_milestone, milestone_baseline


@dataclass
class MilestoneBump:
    group_id: str
    user_id: str
    total: int
    claimed: int | None  # 本次跨过且尚未播报的阈值（取最大的一个）；无则 None
    announced: int  # 更新后的已播报阈值
    previous: int  # 本次认领前的已播报阈值（发送失败回滚用）


class MilestoneCounterRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def bump(
        self,
        *,
        group_id: str,
        user_id: str,
        configured: Iterable[int],
        source_message_id: str = "",
    ) -> MilestoneBump:
        thresholds = [int(item) for item in configured]
        return await asyncio.to_thread(
            self._bump_sync,
            str(group_id),
            str(user_id),
            thresholds,
            str(source_message_id or ""),
        )

    def _bump_sync(
        self,
        group_id: str,
        user_id: str,
        thresholds: list[int],
        source_message_id: str,
    ) -> MilestoneBump:
        self.db.ensure_schema()
        now = time.time()
        # 当前消息可能已经被学习管线写进时间线（学习 matcher 优先级在前）：
        # 回填时排除它，否则「回填 + 本次 +1」会把这条消息算两遍。
        exclude_message_id = source_message_id or "\x00"
        with self.db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            seeded = conn.execute(
                "SELECT 1 FROM member_message_counts WHERE group_id = ? LIMIT 1",
                (group_id,),
            ).fetchone()
            if seeded is None:
                conn.execute(
                    """
                    INSERT INTO member_message_counts
                        (group_id, user_id, message_count, announced_threshold, last_message_at)
                    SELECT group_id, user_id, COUNT(*), NULL, 0.0
                    FROM conversation_turns
                    WHERE group_id = ? AND speaker_kind = 'member'
                      AND user_id IS NOT NULL AND user_id != ''
                      AND source_message_id != ?
                    GROUP BY user_id
                    ON CONFLICT(group_id, user_id) DO NOTHING
                    """,
                    (group_id, exclude_message_id),
                )
            conn.execute(
                """
                INSERT INTO member_message_counts
                    (group_id, user_id, message_count, announced_threshold, last_message_at)
                VALUES (?, ?, 1, NULL, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    message_count = message_count + 1,
                    last_message_at = excluded.last_message_at
                """,
                (group_id, user_id, now),
            )
            row = conn.execute(
                "SELECT message_count, announced_threshold FROM member_message_counts "
                "WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            ).fetchone()
            total = int(row["message_count"])
            announced_raw = row["announced_threshold"]
            if announced_raw is None:
                # 基线取「本条之前」的档位（total-1）：若正是本条跨线，照常播报
                announced = milestone_baseline(max(0, total - 1), thresholds)
            else:
                announced = int(announced_raw)
            previous = announced
            claimed = crossed_milestone(total, announced, thresholds)
            if claimed is not None:
                announced = claimed
            conn.execute(
                "UPDATE member_message_counts SET announced_threshold = ? WHERE group_id = ? AND user_id = ?",
                (announced, group_id, user_id),
            )
        return MilestoneBump(
            group_id=group_id,
            user_id=user_id,
            total=total,
            claimed=claimed,
            announced=announced,
            previous=previous,
        )

    async def rollback_claim(self, *, group_id: str, user_id: str, claimed: int, previous: int) -> bool:
        """发送彻底失败时撤销本次认领，让该成员下条消息自然重试。

        只在 announced 仍等于本次认领值时回滚（已被后续消息推进或回滚过则不动）。
        """
        return await asyncio.to_thread(
            self._rollback_sync, str(group_id), str(user_id), int(claimed), max(0, int(previous))
        )

    def _rollback_sync(self, group_id: str, user_id: str, claimed: int, previous: int) -> bool:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE member_message_counts SET announced_threshold = ? "
                "WHERE group_id = ? AND user_id = ? AND announced_threshold = ?",
                (previous, group_id, user_id, claimed),
            )
            return bool(cursor.rowcount)


__all__ = ["MilestoneBump", "MilestoneCounterRepo"]
