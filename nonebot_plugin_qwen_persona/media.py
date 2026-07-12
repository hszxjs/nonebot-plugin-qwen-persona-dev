from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Mapping, Protocol

from .sticker import (
    STICKER_TOKEN_RE,
    build_training_text,
    is_generic_media_placeholder,
    sticker_to_record,
)


CQ_CODE_RE = re.compile(r"\[CQ:(?P<type>[A-Za-z0-9_]+)(?P<params>(?:,[^\]]*)?)\]")
MEDIA_TOKEN_RE = re.compile(r"\[\[(?!(?:sticker):)[A-Za-z0-9_.:-]+:[^\]]+\]\]")
CONTEXT_MEDIA_DESCRIPTION_RE = re.compile(
    r"(?:^|\s)(?:图|图片|表情|表情包)[：:]\s*.*?"
    r"(?=(?:\s+(?:图|图片|表情|表情包)[：:])|\s+\[\[sticker:|$)"
)
REPLY_MEDIA_ARTIFACT_RE = re.compile(
    r"(?:图|图片|表情|表情包)[：:]\s*"
    r"(?:图片|表情包|表情|贴图|动图|image|sticker)"
    r"(?:[（(][^）)]{0,24}[）)])?",
    re.IGNORECASE,
)
GENERIC_MEDIA_PLACEHOLDER_TEXT_RE = re.compile(r"\[(?:动画消息|动画表情|表情消息)\]")
MEDIA_SEGMENT_TYPES = {"image", "face", "mface", "emoji", "sticker"}
logger = logging.getLogger(__name__)


class MediaCaptioner(Protocol):
    async def caption_media(self, part: dict[str, Any]) -> str:
        ...


def message_to_parts(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, str):
        return _cq_text_to_parts(message)
    try:
        segments = list(message)
    except TypeError:
        return _cq_text_to_parts(str(message or ""))

    parts: list[dict[str, Any]] = []
    for segment in segments:
        item = _segment_to_dict(segment)
        segment_type = item["type"]
        data = item["data"]
        if segment_type == "text":
            text = str(data.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text, "data": {"text": text}})
            continue
        parts.append({"type": segment_type, "data": data})
    return parts


async def prepare_message_learning(
    message: Any,
    *,
    sticker_segment_types: Iterable[str],
    captioner: MediaCaptioner | None,
    image_fallback_text: str,
    sticker_fallback_text: str,
) -> dict[str, Any]:
    parts = message_to_parts(message)
    plain_chunks: list[str] = []
    training_chunks: list[str] = []
    media_context_chunks: list[str] = []
    sticker_keys: list[str] = []
    stickers: list[dict[str, Any]] = []

    for part in parts:
        part_type = str(part.get("type") or "")
        if part_type == "text":
            text = _clean_text(str(part.get("text") or ""))
            if text:
                plain_chunks.append(text)
                training_chunks.append(text)
            continue

        if part_type == "image":
            record = sticker_to_record(part, sticker_segment_types)
            if record is not None:
                caption = _record_prompt_summary(record)
                if not caption:
                    caption = await _caption_or_fallback(captioner, part, "")
                if not caption:
                    caption = _safe_caption(sticker_fallback_text)
                if caption:
                    training_chunks.append(f"表情：{caption}")
                    record["summary"] = caption
                stickers.append(record)
                sticker_keys.append(record["key"])
            else:
                caption = await _caption_or_fallback(captioner, part, image_fallback_text)
                if caption:
                    media_context_chunks.append(f"图片：{caption}")
            continue

        if part_type in MEDIA_SEGMENT_TYPES:
            record = sticker_to_record(part, sticker_segment_types)
            caption = _record_prompt_summary(record) if record is not None else ""
            if not caption:
                caption = await _caption_or_fallback(captioner, part, sticker_fallback_text)
            if caption:
                training_chunks.append(f"表情：{caption}")
            if record is not None:
                stickers.append(record)
                sticker_keys.append(record["key"])
            continue

    plain_text = _clean_text(" ".join(plain_chunks))
    training_text = build_training_text(
        _clean_text(" ".join(training_chunks)),
        sticker_keys,
    )
    return {
        "plain_text": plain_text,
        "training_text": training_text,
        "media_context_text": _clean_text(" ".join(media_context_chunks)),
        "raw_segments": parts,
        "stickers": stickers,
        "media_parts": [part for part in parts if part.get("type") != "text"],
    }


def sanitize_model_reply_media_text(text: str) -> str:
    without_cq = CQ_CODE_RE.sub("", text or "")
    without_media_tokens = MEDIA_TOKEN_RE.sub("", without_cq)
    without_artifacts = REPLY_MEDIA_ARTIFACT_RE.sub("", without_media_tokens)
    return _clean_text(without_artifacts)


def context_text_from_sample(sample: Mapping[str, Any]) -> str:
    plain_text = _clean_text(str(sample.get("plain_text") or ""))
    sticker_keys = _safe_sticker_keys(sample.get("sticker_keys"))
    if plain_text:
        return build_training_text(plain_text, sticker_keys)
    fallback_text = context_text_from_training_text(str(sample.get("training_text") or ""))
    if fallback_text:
        return fallback_text
    return build_training_text("", sticker_keys)


def context_text_from_training_text(text: str) -> str:
    without_descriptions = CONTEXT_MEDIA_DESCRIPTION_RE.sub(" ", text or "")
    return _clean_text(_strip_prompt_unsafe_sticker_tokens(without_descriptions))


def reply_context_text_from_learning(learning: Mapping[str, Any]) -> str:
    plain_text = _clean_text(str(learning.get("plain_text") or ""))
    training_text = str(learning.get("training_text") or "")
    without_sticker_tokens = STICKER_TOKEN_RE.sub("", training_text)
    media_context = _clean_text(str(learning.get("media_context_text") or ""))
    if (
        not plain_text
        and not media_context
        and not context_text_from_training_text(without_sticker_tokens)
    ):
        return ""
    chunks = []
    base = _clean_text(without_sticker_tokens)
    if base:
        chunks.append(base)
    if media_context:
        chunks.append(media_context)
    return _clean_text(" ".join(chunks))


def filter_training_text_sticker_tokens(
    training_text: str,
    stickers: Iterable[Mapping[str, Any]],
) -> str:
    allowed_keys = {
        str(sticker.get("key") or "").strip()
        for sticker in stickers
        if str(sticker.get("key") or "").strip()
    }
    base_text, sticker_keys = _parse_sticker_tokens_unique(training_text)
    kept_keys = [key for key in sticker_keys if key in allowed_keys]
    return build_training_text(base_text, kept_keys)


async def _caption_or_fallback(
    captioner: MediaCaptioner | None,
    part: dict[str, Any],
    fallback: str,
) -> str:
    if captioner is not None:
        try:
            caption = await captioner.caption_media(part)
            caption = _safe_caption(caption)
            if caption:
                return caption
        except Exception as exc:
            logger.warning("qwen persona media caption fallback: %r", exc)
    return _safe_caption(fallback)


def _record_prompt_summary(record: dict[str, Any]) -> str:
    data = record.get("data")
    if not isinstance(data, Mapping):
        return ""
    for field in ("summary", "alt", "name", "text"):
        caption = _safe_caption(str(data.get(field) or ""))
        if caption:
            return caption
    return ""


def _cq_text_to_parts(text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    cursor = 0
    for match in CQ_CODE_RE.finditer(text or ""):
        if match.start() > cursor:
            chunk = text[cursor : match.start()]
            if chunk:
                parts.append({"type": "text", "text": chunk, "data": {"text": chunk}})
        parts.append(
            {
                "type": match.group("type"),
                "data": _parse_cq_params(match.group("params") or ""),
            }
        )
        cursor = match.end()
    if cursor < len(text or ""):
        chunk = text[cursor:]
        if chunk:
            parts.append({"type": "text", "text": chunk, "data": {"text": chunk}})
    return parts


def _parse_cq_params(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in str(raw or "").lstrip(",").split(","):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        params[_cq_unescape(key)] = _cq_unescape(value)
    return params


def _cq_unescape(value: str) -> str:
    return (
        str(value)
        .replace("&#44;", ",")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
    )


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    segment_type = getattr(segment, "type", None)
    data = getattr(segment, "data", None)
    if isinstance(segment, dict):
        segment_type = segment.get("type", segment_type)
        data = segment.get("data", data)
    return {"type": str(segment_type or "unknown"), "data": dict(data or {})}


def _safe_caption(text: str) -> str:
    cleaned = sanitize_model_reply_media_text(str(text or ""))
    cleaned = re.sub(r"(?:https?://|file://|base64://)\S+", "", cleaned, flags=re.IGNORECASE)
    if is_generic_media_placeholder(cleaned):
        return ""
    return _clean_text(cleaned)[:120]


def _safe_sticker_keys(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "[]")
        except json.JSONDecodeError:
            parsed = []
    else:
        parsed = raw
    if not isinstance(parsed, list):
        return []

    keys: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        key = str(item or "").strip()
        if not key or key in seen or not _is_prompt_safe_sticker_key(key):
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _is_prompt_safe_sticker_key(key: str) -> bool:
    return not str(key or "").startswith("image:")


def _strip_prompt_unsafe_sticker_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = str(match.group(1) or "")
        return match.group(0) if _is_prompt_safe_sticker_key(key) else " "

    return STICKER_TOKEN_RE.sub(replace, text or "")


def _parse_sticker_tokens_unique(text: str) -> tuple[str, list[str]]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in STICKER_TOKEN_RE.findall(text or ""):
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    clean_text = STICKER_TOKEN_RE.sub("", text or "").strip()
    return _clean_text(clean_text), keys


def _clean_text(text: str) -> str:
    without_placeholders = GENERIC_MEDIA_PLACEHOLDER_TEXT_RE.sub(" ", str(text or ""))
    return re.sub(r"\s+", " ", without_placeholders).strip()
