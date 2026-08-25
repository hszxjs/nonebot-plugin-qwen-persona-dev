"""接话/学习决策门：是否回应、是否学习、是否携带表情。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..prompt.text import is_command_like, is_training_noise, normalize_text


@dataclass(frozen=True)
class FilterDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ReplyDecision:
    allowed: bool
    reason: str


def _clamp_probability(value: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def is_private_chat(event: Any, ids: dict[str, str | None]) -> bool:
    message_type = str(_get_field(event, "message_type", "") or "").lower()
    if message_type:
        return message_type == "private"

    detail_type = str(_get_field(event, "detail_type", "") or "").lower()
    if detail_type in {"private", "direct", "direct_message"}:
        return True
    if detail_type in {"group", "channel", "guild"}:
        return False

    return not ids.get("group_id")


def message_mentions_bot(message: Any, bot_self_id: str | None) -> bool:
    if not bot_self_id:
        return False
    wanted = str(bot_self_id)
    try:
        segments = list(message)
    except TypeError:
        return False

    for segment in segments:
        segment_type = _get_field(segment, "type", "")
        if str(segment_type).lower() != "at":
            continue
        data = _get_field(segment, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        for key in ("qq", "id", "user_id", "uid"):
            value = data.get(key)
            if value is not None and str(value) == wanted:
                return True
    return False


def should_force_reply(
    event: Any,
    ids: dict[str, str | None],
    message: Any,
    *,
    bot_self_id: str | None,
    to_me: bool,
) -> bool:
    return bool(to_me) or is_private_chat(event, ids) or message_mentions_bot(message, bot_self_id)


def should_use_sticker_reply(
    *,
    incoming_has_sticker: bool,
    text_sticker_probability: float,
    random_value: float,
) -> bool:
    if incoming_has_sticker:
        return True
    try:
        probability = float(text_sticker_probability)
    except (TypeError, ValueError):
        return False
    probability = min(1.0, max(0.0, probability))
    return probability > 0 and float(random_value) < probability


def should_random_reply(text: str, *, probability: float, random_value: float) -> bool:
    if not normalize_text(text):
        return False
    try:
        probability_value = float(probability)
    except (TypeError, ValueError):
        return False
    probability_value = min(1.0, max(0.0, probability_value))
    return probability_value > 0 and float(random_value) < probability_value


def should_learn_text(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    prefixes: Iterable[str] = (),
    regexes: Iterable[str] = (),
) -> FilterDecision:
    normalized = normalize_text(text)
    if not normalized:
        return FilterDecision(False, "empty")
    if is_command_like(normalized, prefixes=prefixes, regexes=regexes):
        return FilterDecision(False, "command")
    if is_training_noise(normalized):
        return FilterDecision(False, "noise")
    if len(normalized) < min_chars:
        return FilterDecision(False, "too_short")
    if len(normalized) > max_chars:
        return FilterDecision(False, "too_long")
    return FilterDecision(True, "accepted")


def should_attempt_reply(
    *,
    enable_reply: bool,
    group_enabled: bool,
    self_event: bool,
    blacklisted: bool,
    command_like: bool,
    force_reply: bool,
    has_text: bool,
    reply_probability: float,
    follow_reply_probability: float,
    random_value: float,
    has_media: bool = False,
) -> ReplyDecision:
    if not enable_reply:
        return ReplyDecision(False, "reply_disabled")
    if not group_enabled:
        return ReplyDecision(False, "group_disabled")
    if self_event:
        return ReplyDecision(False, "self_event")
    if blacklisted:
        return ReplyDecision(False, "blacklisted")
    if force_reply:
        return ReplyDecision(True, "force_reply")
    if command_like:
        return ReplyDecision(False, "command_like")
    if not has_text and not has_media:
        return ReplyDecision(False, "empty")

    probability = _clamp_probability(reply_probability)
    reason = "random_reply"
    if probability <= 0:
        probability = _clamp_probability(follow_reply_probability)
        reason = "follow_reply"
    if probability > 0 and float(random_value) < probability:
        return ReplyDecision(True, reason)
    return ReplyDecision(False, "probability_miss")
