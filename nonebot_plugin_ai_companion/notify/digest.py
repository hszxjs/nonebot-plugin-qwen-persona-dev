"""错过的上下文：从时间线提取每条触发的 ±N 窗口，相连的合并成连续段。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DigestBlock:
    """一段连续的时间线摘录（由一个或多个相连的触发窗口合并而成）。"""

    turns: list[dict[str, Any]] = field(default_factory=list)
    trigger_message_ids: set[str] = field(default_factory=set)


def _turn_position_index(turns: list[dict[str, Any]]) -> dict[str, int]:
    """source_message_id → 在时间线里的下标。"""
    index: dict[str, int] = {}
    for position, turn in enumerate(turns):
        message_id = str(turn.get("source_message_id") or "")
        if message_id and message_id not in index:
            index[message_id] = position
    return index


def _anchor_position(
    turns: list[dict[str, Any]],
    position_index: dict[str, int],
    *,
    source_message_id: str,
    created_at: float,
) -> int | None:
    """触发消息在时间线里的锚点下标：优先按消息 id，退化按"最近一条不晚于触发时间的"。"""
    direct = position_index.get(str(source_message_id or ""))
    if direct is not None:
        return direct
    best: int | None = None
    for position, turn in enumerate(turns):
        if float(turn.get("created_at") or 0.0) <= float(created_at) + 0.001:
            best = position
        else:
            break
    return best


def merge_context_blocks(
    turns: list[dict[str, Any]],
    anchors: list[tuple[str, float]],
    *,
    context_lines: int = 3,
    max_lines: int = 40,
) -> tuple[list[DigestBlock], int]:
    """按锚点取 ±context_lines 窗口；相连（重叠或紧邻）的合并成一个块。

    turns 需按时间正序；anchors 为 (source_message_id, created_at) 列表（按时间正序）。
    超过 max_lines 时从最老的行开始截断，返回值第二项是被截掉的行数。
    """
    position_index = _turn_position_index(turns)
    windows: list[tuple[int, int, str]] = []  # (start, end, trigger_message_id)
    for source_message_id, created_at in anchors:
        anchor = _anchor_position(turns, position_index, source_message_id=source_message_id, created_at=created_at)
        if anchor is None:
            continue
        start = max(0, anchor - max(0, int(context_lines)))
        end = min(len(turns) - 1, anchor + max(0, int(context_lines)))
        windows.append((start, end, str(source_message_id or "")))
    if not windows:
        return [], 0
    windows.sort(key=lambda item: item[0])

    blocks: list[DigestBlock] = []
    current: DigestBlock | None = None
    current_end = -1
    for start, end, trigger_id in windows:
        if current is not None and start <= current_end + 1:
            # 重叠或紧邻：并入当前块
            for position in range(current_end + 1, end + 1):
                current.turns.append(turns[position])
            if trigger_id:
                current.trigger_message_ids.add(trigger_id)
            current_end = max(current_end, end)
        else:
            current = DigestBlock(
                turns=[turns[position] for position in range(start, end + 1)],
                trigger_message_ids={trigger_id} if trigger_id else set(),
            )
            blocks.append(current)
            current_end = end

    total = sum(len(block.turns) for block in blocks)
    dropped = 0
    while total > max(1, int(max_lines)) and blocks:
        # 从最老的块开始丢行
        oldest = blocks[0]
        if len(oldest.turns) <= total - int(max_lines) or len(blocks) == 1:
            take = min(len(oldest.turns), total - int(max_lines)) if len(blocks) > 1 else len(oldest.turns) - 1
            if take <= 0:
                break
            del oldest.turns[:take]
            total -= take
            dropped += take
            if not oldest.turns:
                blocks.pop(0)
        else:
            take = total - int(max_lines)
            del oldest.turns[:take]
            total -= take
            dropped += take
    return [block for block in blocks if block.turns], dropped


__all__ = ["DigestBlock", "merge_context_blocks"]
