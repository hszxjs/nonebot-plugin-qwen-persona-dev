from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .filters import normalize_text


@dataclass(frozen=True)
class MessageContextEnvelope:
    current_text: str
    quoted_texts: list[str]
    forwarded_texts: list[str]
    media_parts: list[dict[str, Any]]
    sticker_keys: list[str]
    forced_by_at: bool
    context_sources: list[str]


def build_message_context(
    *,
    message: Any,
    event: Any,
    current_text: str,
    media_parts: list[dict[str, Any]],
    sticker_keys: list[str],
    forced_by_at: bool,
) -> MessageContextEnvelope:
    quoted_texts = _texts_from_reply(event)
    forwarded_texts = _texts_from_forwarded_nodes(message)
    sources: list[str] = []
    if normalize_text(current_text):
        sources.append("current")
    if quoted_texts:
        sources.append("quoted")
    if forwarded_texts:
        sources.append("forwarded")
    if media_parts:
        sources.append("media")
    if sticker_keys:
        sources.append("sticker")
    if forced_by_at:
        sources.append("force_reply")

    return MessageContextEnvelope(
        current_text=normalize_text(current_text),
        quoted_texts=quoted_texts,
        forwarded_texts=forwarded_texts,
        media_parts=list(media_parts),
        sticker_keys=[str(key) for key in sticker_keys if str(key).strip()],
        forced_by_at=bool(forced_by_at),
        context_sources=sources,
    )


def _texts_from_reply(event: Any) -> list[str]:
    reply = _get_field(event, "reply", None)
    if reply is None:
        return []
    message = _get_field(reply, "message", None)
    return _texts_from_message(message)


def _texts_from_forwarded_nodes(message: Any) -> list[str]:
    texts: list[str] = []
    for segment in _iter_segments(message):
        if str(_get_field(segment, "type", "")).lower() != "node":
            continue
        data = _get_field(segment, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        for field in ("content", "message"):
            texts.extend(_texts_from_message(data.get(field)))
    return _dedupe_texts(texts)


def _texts_from_message(message: Any) -> list[str]:
    texts: list[str] = []
    if message is None:
        return texts
    if isinstance(message, str):
        cleaned = normalize_text(message)
        return [cleaned] if cleaned else []
    for segment in _iter_segments(message):
        segment_type = str(_get_field(segment, "type", "")).lower()
        data = _get_field(segment, "data", {}) or {}
        if segment_type == "text":
            text = data.get("text") if isinstance(data, dict) else None
            cleaned = normalize_text(str(text or ""))
            if cleaned:
                texts.append(cleaned)
            continue
        if segment_type == "node" and isinstance(data, dict):
            texts.extend(_texts_from_message(data.get("content") or data.get("message")))
    return _dedupe_texts(texts)


def _iter_segments(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, dict):
        return [message]
    if isinstance(message, str):
        return [{"type": "text", "data": {"text": message}}]
    try:
        return list(message)
    except TypeError:
        return [{"type": "text", "data": {"text": str(message or "")}}]


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _dedupe_texts(texts: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = normalize_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
