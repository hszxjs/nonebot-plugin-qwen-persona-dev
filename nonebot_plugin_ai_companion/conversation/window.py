"""会话窗口渲染：把时间线转成进入提示词的「昵称：内容」多行文本。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from ..sticker.model import build_training_text
from .model import ConversationTurn

_NAME_UNSAFE_RE = re.compile(r"[\s:：]+")
FALLBACK_NAME_PREFIX = "群友"


def safe_speaker_name(name: str, *, user_id: str | None = None) -> str:
    """昵称清洗：去掉换行/冒号（避免破坏「昵称：内容」结构），超长截断。

    名字永远不回落成 QQ 号：解析不到就用「群友xxxx」，避免数字 id 进提示词后被复述。
    """
    cleaned = _NAME_UNSAFE_RE.sub("", str(name or "").strip())
    if cleaned:
        return cleaned[:16]
    tail = str(user_id or "").strip()[-4:]
    return f"{FALLBACK_NAME_PREFIX}{tail}" if tail else "群友"


def render_turn_line(
    turn: ConversationTurn,
    *,
    names: Mapping[str, str] | None = None,
    bot_label: str = "你",
) -> str:
    """单条时间线 → 提示词行。bot 自己的话用「你：」开头。"""
    body = build_training_text(turn.text, turn.sticker_keys)
    if not body.strip():
        return ""
    if turn.is_bot:
        return f"{bot_label}：{body}"
    name_map = names or {}
    name = turn.speaker_name or name_map.get(str(turn.user_id or ""), "")
    return f"{safe_speaker_name(name, user_id=turn.user_id)}：{body}"


def render_history_lines(
    turns: Iterable[ConversationTurn],
    *,
    names: Mapping[str, str] | None = None,
    bot_label: str = "你",
    max_lines: int = 0,
    exclude_message_ids: Iterable[str] | None = None,
) -> list[str]:
    """时间线（旧 → 新）→ 提示词行。空行丢弃；可按 source_message_id 排除当前消息。

    max_lines > 0 时只保留最后 max_lines 行（贴近"最近几轮"语义）。
    """
    excluded = {str(item) for item in (exclude_message_ids or []) if str(item).strip()}
    lines: list[str] = []
    for turn in turns:
        if excluded and str(turn.source_message_id or "") in excluded:
            continue
        line = render_turn_line(turn, names=names, bot_label=bot_label)
        if line:
            lines.append(line)
    if max_lines > 0:
        lines = lines[-max_lines:]
    return lines


def collect_user_ids(turns: Iterable[ConversationTurn]) -> list[str]:
    """时间线里出现过的成员 QQ（用于批量补齐昵称），去重且保持顺序。"""
    seen: set[str] = set()
    ids: list[str] = []
    for turn in turns:
        if turn.is_bot:
            continue
        user_id = str(turn.user_id or "").strip()
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        ids.append(user_id)
    return ids


__all__ = ["collect_user_ids", "render_history_lines", "render_turn_line", "safe_speaker_name"]
