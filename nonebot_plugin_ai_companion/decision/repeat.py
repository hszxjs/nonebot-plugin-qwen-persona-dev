from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..prompt.text import is_question_like
from ..sticker.model import extract_plain_text, extract_segments, sticker_to_record


@dataclass(frozen=True)
class RepeatCandidate:
    signature: str
    plain_text: str
    sticker_keys: list[str]
    stickers: list[dict[str, Any]]
    count: int


@dataclass(frozen=True)
class ConversationPatternMessage:
    signature: str
    user_id: str
    plain_text: str
    sticker_keys: list[str]
    observed_at: float


@dataclass(frozen=True)
class ConversationPatternCandidate:
    trigger: ConversationPatternMessage
    response: ConversationPatternMessage


class RepeatTracker:
    def __init__(self) -> None:
        self._last_signature_by_group: dict[str, str] = {}
        self._count_by_group: dict[str, int] = {}
        self._last_repeat_at_by_group: dict[str, float] = {}

    def observe(
        self,
        *,
        group_id: str | None,
        message: Any,
        sticker_segment_types: set[str],
        min_count: int,
        cooldown_seconds: int,
    ) -> RepeatCandidate | None:
        if not group_id:
            return None
        plain_text = extract_plain_text(message).strip()
        segments = extract_segments(message)
        stickers = [
            sticker
            for sticker in (sticker_to_record(segment, sticker_segment_types) for segment in segments)
            if sticker is not None
        ]
        sticker_keys = [item["key"] for item in stickers]
        signature = build_repeat_signature(plain_text, sticker_keys)
        if not signature:
            return None

        last_signature = self._last_signature_by_group.get(group_id)
        if signature == last_signature:
            count = self._count_by_group.get(group_id, 1) + 1
        else:
            count = 1
        self._last_signature_by_group[group_id] = signature
        self._count_by_group[group_id] = count
        if is_question_like(plain_text):
            return None

        if count < min_count:
            return None
        now = time.time()
        last_repeat_at = self._last_repeat_at_by_group.get(group_id, 0)
        if cooldown_seconds > 0 and now - last_repeat_at < cooldown_seconds:
            return None
        self._last_repeat_at_by_group[group_id] = now
        return RepeatCandidate(
            signature=signature,
            plain_text=plain_text,
            sticker_keys=sticker_keys,
            stickers=stickers,
            count=count,
        )


class ConversationPatternTracker:
    def __init__(self, *, max_gap_seconds: int = 180) -> None:
        self.max_gap_seconds = max(0, int(max_gap_seconds))
        self._last_message_by_group: dict[str, ConversationPatternMessage] = {}

    def observe(
        self,
        *,
        group_id: str | None,
        user_id: str | None,
        message: Any | None = None,
        sticker_segment_types: set[str] | None = None,
        plain_text: str | None = None,
        sticker_keys: list[str] | None = None,
        observed_at: float | None = None,
    ) -> ConversationPatternCandidate | None:
        if not group_id or not user_id:
            return None
        current = build_conversation_pattern_message(
            user_id=user_id,
            message=message,
            sticker_segment_types=sticker_segment_types or set(),
            plain_text=plain_text,
            sticker_keys=sticker_keys,
            observed_at=observed_at,
        )
        if current is None:
            self.reset(group_id)
            return None

        previous = self._last_message_by_group.get(group_id)
        self._last_message_by_group[group_id] = current
        if previous is None or previous.user_id == current.user_id:
            return None
        if self.max_gap_seconds > 0 and current.observed_at - previous.observed_at > self.max_gap_seconds:
            return None
        return ConversationPatternCandidate(trigger=previous, response=current)

    def reset(self, group_id: str | None) -> None:
        if group_id:
            self._last_message_by_group.pop(group_id, None)


def build_conversation_pattern_message(
    *,
    user_id: str,
    message: Any | None = None,
    sticker_segment_types: set[str],
    plain_text: str | None = None,
    sticker_keys: list[str] | None = None,
    observed_at: float | None = None,
) -> ConversationPatternMessage | None:
    text = extract_plain_text(message).strip() if plain_text is None else plain_text.strip()
    keys = sticker_keys
    if keys is None:
        segments = extract_segments(message)
        keys = [
            sticker["key"]
            for sticker in (sticker_to_record(segment, sticker_segment_types) for segment in segments)
            if sticker is not None
        ]
    signature = build_repeat_signature(text, keys)
    if not signature:
        return None
    return ConversationPatternMessage(
        signature=signature,
        user_id=str(user_id),
        plain_text=text,
        sticker_keys=keys,
        observed_at=time.time() if observed_at is None else float(observed_at),
    )


def build_repeat_signature(plain_text: str, sticker_keys: list[str]) -> str:
    parts: list[str] = []
    if plain_text.strip():
        parts.append("text:" + " ".join(plain_text.split()))
    parts.extend(f"sticker:{key}" for key in sticker_keys)
    return "|".join(parts)
