from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FilterDecision:
    allowed: bool
    reason: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_text(text)).lower()


def is_question_like(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if "?" in normalized or "？" in normalized:
        return True

    compact = _compact_text(normalized)
    if not compact:
        return False
    question_prefixes = (
        "为什么",
        "为啥",
        "怎么",
        "咋",
        "如何",
        "什么",
        "啥",
        "哪",
        "哪个",
        "哪里",
        "谁",
        "多少",
        "几",
        "能不能",
        "可不可以",
        "是不是",
        "有没有",
        "是否",
    )
    if compact.startswith(question_prefixes):
        return True
    if compact.endswith(("吗", "嘛", "么", "呢", "行不行", "好不好", "可以不", "能不能")):
        return True
    return any(marker in compact for marker in ("怎么", "为什么", "为啥", "什么", "有没有"))


def is_echo_reply(reply: str, current_text: str) -> bool:
    reply_key = _compact_text(reply)
    current_key = _compact_text(current_text)
    if not reply_key or not current_key:
        return False
    if reply_key == current_key:
        return True
    if min(len(reply_key), len(current_key)) < 6:
        return False
    if reply_key in current_key or current_key in reply_key:
        return min(len(reply_key), len(current_key)) / max(len(reply_key), len(current_key)) >= 0.55
    return False


def is_question_deflection_reply(reply: str, current_text: str) -> bool:
    if not is_question_like(current_text):
        return False
    reply_key = _compact_text(reply)
    if not reply_key:
        return False
    if "你问的问题" in reply_key or "你问我" in reply_key or "问你" in reply_key:
        return True
    if re.search(r"你.*问.*我.*问.*你", normalize_text(reply)):
        return True
    return False


def is_training_noise(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    noise_patterns = (
        r"该消息类型暂不支持查看",
        r"\b(?:TypeError|ReferenceError|SyntaxError|AttributeError|RuntimeError|HTTPStatusError)\b",
        r"\bTraceback\b",
        r"undefined is not an object",
        r"object has no attribute",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in noise_patterns)


def is_command_like(
    text: str,
    *,
    prefixes: Iterable[str] = (),
    regexes: Iterable[str] = (),
) -> bool:
    stripped = (text or "").lstrip()
    if not stripped:
        return False

    for prefix in sorted({str(item) for item in prefixes if str(item)}, key=len, reverse=True):
        if stripped.startswith(prefix):
            return True

    for pattern in regexes:
        try:
            if re.search(pattern, text or ""):
                return True
        except re.error:
            continue
    return False


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


def sanitize_reply(
    text: str,
    *,
    prefixes: Iterable[str] = (),
    regexes: Iterable[str] = (),
) -> str:
    kept_lines: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if is_command_like(line, prefixes=prefixes, regexes=regexes):
            continue
        if is_training_noise(line):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    if is_command_like(cleaned, prefixes=prefixes, regexes=regexes):
        return ""
    return cleaned
