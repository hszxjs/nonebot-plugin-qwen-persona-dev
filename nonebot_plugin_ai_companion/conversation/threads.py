"""话题线程聚类：把群里的消息串成「谁在接谁的话」。

为什么需要：群里大多数消息既没有 @ 也没有引用（实测两天 58 条里只有 6 条带 @），
但人读聊天记录时能自然分辨出"A 在回 B"。这里用可解释的启发式把这件事显式化：

1. 显式优先：引用/回复指向的那条消息 → 它所在的线程；
2. 否则看窗口内的上一条消息：
   - 同一个人的又一次发言 → 继续他自己的那条线（自言自语式补充）；
   - 别人刚说完（在窗口内）→ 视为接话，加入对方那条线；
3. 窗口内没有前文 → 开一条新线。

线程 id 用"该线程第一条消息的 turn id"表示，天然稳定、无需额外表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_THREAD_WINDOW_SECONDS = 300.0
# 推断只看最近这些 turn，避免在长会话里翻全表
LOOKBACK_TURNS = 60


@dataclass(frozen=True)
class ThreadAssignment:
    thread_id: int
    reply_to_turn_id: int
    reason: str  # explicit | same_speaker | reply_to_previous | new_thread


def infer_thread(
    previous_turns: list[dict[str, Any]],
    *,
    new_turn_id: int,
    user_id: str | None,
    created_at: float,
    reply_to_message_id: str = "",
    window_seconds: float = DEFAULT_THREAD_WINDOW_SECONDS,
) -> ThreadAssignment:
    """为一条新 turn 推断 (thread_id, reply_to_turn_id)。

    previous_turns 需按时间正序，元素含 id/user_id/source_message_id/thread_id/created_at。
    """
    window = max(1.0, float(window_seconds))
    clean_user = str(user_id or "")
    recent = [item for item in previous_turns if str(item.get("user_id") or "") or item.get("speaker_kind") == "bot"]
    recent = recent[-LOOKBACK_TURNS:]

    # 1) 显式引用/回复：按消息 id 找到被引用的那条
    if reply_to_message_id:
        for item in reversed(recent):
            if str(item.get("source_message_id") or "") == str(reply_to_message_id):
                thread_id = int(item.get("thread_id") or 0) or int(item.get("id") or 0)
                return ThreadAssignment(thread_id, int(item.get("id") or 0), "explicit")

    # 2) 窗口内的上一条消息
    antecedent = None
    for item in reversed(recent):
        created = float(item.get("created_at") or 0.0)
        if created_at - created > window:
            break
        antecedent = item
        break

    if antecedent is None:
        return ThreadAssignment(int(new_turn_id), 0, "new_thread")

    antecedent_user = str(antecedent.get("user_id") or "")
    antecedent_id = int(antecedent.get("id") or 0)
    thread_id = int(antecedent.get("thread_id") or 0) or antecedent_id
    if clean_user and antecedent_user == clean_user:
        # 同一个人接着说：不算"回复谁"，但仍在同一条线上
        return ThreadAssignment(thread_id, antecedent_id, "same_speaker")
    return ThreadAssignment(thread_id, antecedent_id, "reply_to_previous")


@dataclass
class ThreadDigest:
    """当前话题的摘要信息：参与者 + 最近几句（供提示词聚焦一条线）。"""

    thread_id: int
    participants: list[str]
    lines: list[str]


def current_thread_digest(
    turns: list[dict[str, Any]],
    *,
    thread_id: int,
    names: dict[str, str],
    max_lines: int = 8,
) -> ThreadDigest:
    """取某条线程最近的若干句 + 参与者，用于"只回这一条线"。"""
    from .window import render_history_lines, safe_speaker_name

    thread_turns = [item for item in turns if int(item.get("thread_id") or 0) == int(thread_id)]
    thread_turns = thread_turns[-max(1, int(max_lines)) :]
    participants: list[str] = []
    lines = render_history_lines(
        [item["turn"] for item in thread_turns if item.get("turn") is not None],
        names=names,
    )
    for item in thread_turns:
        turn = item.get("turn")
        if turn is None:
            continue
        if turn.is_bot:
            label = "你"
        else:
            label = safe_speaker_name(turn.speaker_name or names.get(str(turn.user_id or ""), ""), user_id=turn.user_id)
        if label not in participants:
            participants.append(label)
    return ThreadDigest(thread_id=int(thread_id), participants=participants, lines=lines)


__all__ = [
    "DEFAULT_THREAD_WINDOW_SECONDS",
    "LOOKBACK_TURNS",
    "ThreadAssignment",
    "ThreadDigest",
    "current_thread_digest",
    "infer_thread",
]
