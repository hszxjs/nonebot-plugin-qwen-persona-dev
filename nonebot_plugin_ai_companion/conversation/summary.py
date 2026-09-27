"""滚动话题摘要：每群一份，把旧对话压成一段「最近在聊什么」。

为什么需要：时间线窗口只有几十条，跨小时/跨天的话题靠窗口接不上；
摘要由辅助模型增量更新（旧摘要 + 新增对话 → 新摘要），注入提示词的 scene 段。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..memory.db import PersonaDatabase
from .model import ConversationTurn
from .repo import ConversationRepo, MemberNameRepo

logger = logging.getLogger(__name__)

SummaryChat = Callable[[str], Awaitable[str]]

SUMMARY_PROMPT = """你在维护一个 QQ 群的「最近话题摘要」，供机器人在群里接话时参考。

把「已有摘要」和「新增对话」合并成一段新的摘要，要求：
1. 只写这个群最近在聊什么、进行到哪一步、有哪些未结的事与约定；不要流水账、不要复述每句话。
2. 保留仍然有效的信息，丢掉已经过时的；总长不超过 150 字。
3. 不写机器人自身的运行规则，不写隐私性内容（真实姓名/住址/联系方式/经济状况）。
4. 只输出摘要正文，不要标题、不要解释、不要引号。

【已有摘要】
{previous}

【新增对话】
{dialogue}
"""

MAX_SUMMARY_CHARS = 300


@dataclass
class GroupSummary:
    group_id: str
    summary: str = ""
    covered_turn_id: int = 0
    turns_covered: int = 0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "summary": self.summary,
            "covered_turn_id": self.covered_turn_id,
            "turns_covered": self.turns_covered,
            "updated_at": self.updated_at,
        }


class GroupSummaryRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def get(self, *, group_id: str) -> GroupSummary | None:
        return await asyncio.to_thread(self._get_sync, str(group_id or ""))

    def _get_sync(self, group_id: str) -> GroupSummary | None:
        if not group_id:
            return None
        self.db.ensure_schema()
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM group_summaries WHERE group_id = ?", (group_id,)).fetchone()
        if row is None:
            return None
        return GroupSummary(
            group_id=str(row["group_id"]),
            summary=str(row["summary"] or ""),
            covered_turn_id=int(row["covered_turn_id"] or 0),
            turns_covered=int(row["turns_covered"] or 0),
            updated_at=float(row["updated_at"] or 0.0),
        )

    async def list_all(self) -> list[GroupSummary]:
        return await asyncio.to_thread(self._list_all_sync)

    def _list_all_sync(self) -> list[GroupSummary]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM group_summaries ORDER BY updated_at DESC").fetchall()
        return [
            GroupSummary(
                group_id=str(row["group_id"]),
                summary=str(row["summary"] or ""),
                covered_turn_id=int(row["covered_turn_id"] or 0),
                turns_covered=int(row["turns_covered"] or 0),
                updated_at=float(row["updated_at"] or 0.0),
            )
            for row in rows
        ]

    async def upsert(
        self,
        *,
        group_id: str,
        summary: str,
        covered_turn_id: int,
        turns_covered: int,
        updated_at: float | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_sync, str(group_id or ""), str(summary or ""), int(covered_turn_id), int(turns_covered), updated_at
        )

    def _upsert_sync(
        self,
        group_id: str,
        summary: str,
        covered_turn_id: int,
        turns_covered: int,
        updated_at: float | None,
    ) -> None:
        if not group_id:
            return
        now = float(updated_at if updated_at is not None else time.time())
        self.db.ensure_schema()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO group_summaries (group_id, summary, covered_turn_id, turns_covered, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    summary = excluded.summary,
                    covered_turn_id = excluded.covered_turn_id,
                    turns_covered = excluded.turns_covered,
                    updated_at = excluded.updated_at
                """,
                (group_id, summary[:MAX_SUMMARY_CHARS], covered_turn_id, turns_covered, now),
            )

    async def delete(self, *, group_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, str(group_id or ""))

    def _delete_sync(self, group_id: str) -> bool:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            cursor = conn.execute("DELETE FROM group_summaries WHERE group_id = ?", (group_id,))
            return bool(cursor.rowcount)


def clean_summary_text(raw: str) -> str:
    """清洗模型输出：去引号/标题/换行，压成一段。"""
    text = str(raw or "").strip()
    for prefix in ("摘要：", "摘要:", "【摘要】", "- "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = " ".join(text.split()).strip('“”"\' ')
    return text[:MAX_SUMMARY_CHARS]


async def update_group_summary(
    *,
    group_id: str,
    turns: list[ConversationTurn],
    names: dict[str, str],
    repo: GroupSummaryRepo,
    chat: SummaryChat,
) -> str:
    """增量更新单个群的摘要：旧摘要 + 新对话 → 新摘要。"""
    from .window import render_history_lines

    dialogue = "\n".join(render_history_lines(turns, names=names))[-2000:]
    if not dialogue.strip():
        return ""
    current = await repo.get(group_id=group_id)
    prompt = SUMMARY_PROMPT.format(previous=(current.summary if current else "") or "（暂无）", dialogue=dialogue)
    raw = await chat(prompt)
    summary = clean_summary_text(raw)
    if not summary:
        return ""
    last_id = int(turns[-1].id or 0) if turns else 0
    await repo.upsert(
        group_id=group_id,
        summary=summary,
        covered_turn_id=last_id,
        turns_covered=int(current.turns_covered if current else 0) + len(turns),
    )
    return summary


async def run_summary_round(
    *,
    conversation: ConversationRepo,
    names_repo: MemberNameRepo,
    repo: GroupSummaryRepo,
    chat: SummaryChat,
    min_new_turns: int = 40,
    interval_seconds: float = 3600.0,
    max_turns: int = 200,
    now: float | None = None,
    disabled_groups: set[str] | None = None,
) -> list[dict[str, Any]]:
    """扫描所有群：新对话够多、或距上次摘要够久且有新内容时才更新。"""
    current = float(now if now is not None else time.time())
    skip = {str(item) for item in (disabled_groups or set())}
    results: list[dict[str, Any]] = []
    for group_id in await conversation.list_session_groups():
        if str(group_id) in skip:
            continue
        session_key = f"group:{group_id}"
        existing = await repo.get(group_id=str(group_id))
        after_id = existing.covered_turn_id if existing else 0
        pending = await conversation.turns_after(session_key=session_key, after_id=after_id, limit=max_turns)
        if not pending:
            continue
        waited = current - (existing.updated_at if existing else 0.0)
        if len(pending) < max(1, int(min_new_turns)) and waited < max(60.0, float(interval_seconds)):
            continue
        names = await names_repo.get_names(
            group_id=str(group_id),
            user_ids=[str(turn.user_id or "") for turn in pending if str(turn.user_id or "").strip()],
        )
        try:
            summary = await update_group_summary(
                group_id=str(group_id), turns=pending, names=names, repo=repo, chat=chat
            )
        except Exception as exc:  # noqa: BLE001 — 摘要失败不影响其它群，下次再试
            logger.warning(f"group summary failed for {group_id}: {exc!r}")
            results.append({"group_id": str(group_id), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if summary:
            results.append({"group_id": str(group_id), "turns": len(pending), "summary": summary})
            logger.info(f"group summary updated for {group_id}: turns={len(pending)}")
    return results


__all__ = [
    "GroupSummary",
    "GroupSummaryRepo",
    "MAX_SUMMARY_CHARS",
    "SUMMARY_PROMPT",
    "clean_summary_text",
    "run_summary_round",
    "update_group_summary",
]
