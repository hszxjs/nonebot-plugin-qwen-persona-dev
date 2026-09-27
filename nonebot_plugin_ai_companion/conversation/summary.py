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

SUMMARY_PROMPT = """你在维护一个 QQ 群的「当前在聊的话题」，供机器人在群里接话时参考。

把「已有摘要」和「新增对话」合并成最新的并行话题列表，要求：
1. 群里可能同时在聊好几件事：最多输出 3 条，每条一行，各自独立、互不混淆。
2. 每条只写「在聊什么 + 到哪一步了（未结的事/约定）」，一行不超过 40 字；不要流水账、不要复述每句话。
3. 保留仍然有效的话题，丢掉已经结束或过时的；只有一件事就只输出一行。
4. 不写机器人自身的运行规则或表现，不写隐私性内容（真实姓名/住址/联系方式/经济状况）。
5. 只输出话题本身，每行以「- 」开头，不要标题、不要解释、不要引号。

【已有摘要】
{previous}

【新增对话】
{dialogue}
"""

MAX_SUMMARY_CHARS = 300
MAX_SUMMARY_LINES = 3


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
            self._upsert_sync,
            str(group_id or ""),
            str(summary or ""),
            int(covered_turn_id),
            int(turns_covered),
            updated_at,
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
    """清洗模型输出：每条一行、最多 3 条并行话题，去掉标题、引号与 bot 元信息。"""
    from ..memory.bot_meta import looks_like_bot_meta

    lines: list[str] = []
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for prefix in ("摘要：", "摘要:", "【摘要】", "- ", "•", "*", "1.", "2.", "3."):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        line = " ".join(line.split()).strip("“”\"' ")
        if line and looks_like_bot_meta(line):
            continue  # 摘要不该写"机器人在调试/复读"这类自身机制
        if line and line not in lines:
            lines.append(line)
    return "\n".join(lines[:MAX_SUMMARY_LINES])[:MAX_SUMMARY_CHARS]


def summary_lines(summary: str) -> list[str]:
    """摘要 → 列表（兼容旧的一段式摘要）。"""
    return [line.strip() for line in str(summary or "").splitlines() if line.strip()]


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
    "MAX_SUMMARY_LINES",
    "SUMMARY_PROMPT",
    "clean_summary_text",
    "run_summary_round",
    "summary_lines",
    "update_group_summary",
]
