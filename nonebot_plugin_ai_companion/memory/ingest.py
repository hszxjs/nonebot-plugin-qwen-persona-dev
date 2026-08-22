"""学习摄入管线：把一条原始消息解析为可学习的纯文本/训练文本/表情记录。"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from ..context.media_parts import message_to_parts
from ..contracts import MediaCaptioner
from ..prompt.media_text import filter_training_text_sticker_tokens, sanitize_media_caption
from ..prompt.text import clean_media_text
from ..sticker.classifier import StickerClassification, classify_sticker_segment
from ..sticker.model import build_training_text, sticker_to_record

logger = logging.getLogger(__name__)

MEDIA_SEGMENT_TYPES = {"image", "face", "mface", "emoji", "sticker"}

__all__ = [
    "MEDIA_SEGMENT_TYPES",
    "prepare_message_learning",
    "filter_training_text_sticker_tokens",
]


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
    sticker_events: list[dict[str, Any]] = []
    enabled_sticker_types = {str(item) for item in sticker_segment_types}

    for part in parts:
        part_type = str(part.get("type") or "")
        if part_type == "text":
            text = clean_media_text(str(part.get("text") or ""))
            if text:
                plain_chunks.append(text)
                training_chunks.append(text)
            continue

        if part_type == "image":
            caption = ""
            classification = classify_sticker_segment(part)
            if not classification.sendable:
                caption = await _caption_or_fallback(captioner, part, "")
                if caption:
                    classification = classify_sticker_segment(part, caption=caption)
            record = (
                sticker_to_record(part, enabled_sticker_types, classification=classification)
                if classification.sendable
                else None
            )
            if record is not None:
                caption = _record_prompt_summary(record) or caption
                if not caption:
                    caption = await _caption_or_fallback(captioner, part, "")
                if not caption:
                    caption = sanitize_media_caption(sticker_fallback_text)
                if caption:
                    training_chunks.append(f"表情：{caption}")
                    record["summary"] = caption
                    # 表情/图片也进回复上下文，让 bot 这次回复就能"看到"图的内容。
                    media_context_chunks.append(f"图片/表情：{caption}")
                stickers.append(record)
                sticker_keys.append(record["key"])
                sticker_events.append(
                    _classifier_event(
                        part=part,
                        classification=classification,
                        status="accepted",
                        key=str(record.get("key") or ""),
                        message=f"accepted image sticker: {caption or classification.reason}",
                    )
                )
            else:
                if not caption:
                    caption = await _caption_or_fallback(captioner, part, image_fallback_text)
                if caption:
                    media_context_chunks.append(f"图片：{caption}")
                sticker_events.append(
                    _classifier_event(
                        part=part,
                        classification=classification,
                        status="rejected",
                        key="",
                        message=f"rejected image: {caption or classification.reason}",
                    )
                )
            continue

        if part_type in MEDIA_SEGMENT_TYPES:
            classification = classify_sticker_segment(part)
            record = sticker_to_record(part, enabled_sticker_types, classification=classification)
            caption = _record_prompt_summary(record) if record is not None else ""
            if not caption:
                caption = await _caption_or_fallback(captioner, part, sticker_fallback_text)
            if caption:
                training_chunks.append(f"表情：{caption}")
            if record is not None:
                stickers.append(record)
                sticker_keys.append(record["key"])
            continue

    plain_text = clean_media_text(" ".join(plain_chunks))
    training_text = build_training_text(
        clean_media_text(" ".join(training_chunks)),
        sticker_keys,
    )
    return {
        "plain_text": plain_text,
        "training_text": training_text,
        "media_context_text": clean_media_text(" ".join(media_context_chunks)),
        "raw_segments": parts,
        "stickers": stickers,
        "sticker_events": sticker_events,
        "media_parts": [part for part in parts if part.get("type") != "text"],
    }


async def _caption_or_fallback(
    captioner: MediaCaptioner | None,
    part: dict[str, Any],
    fallback: str,
) -> str:
    if captioner is not None:
        try:
            caption = await captioner.caption_media(part)
            caption = sanitize_media_caption(caption)
            if caption:
                return caption
        except Exception as exc:
            # 摄图失败降级为兜底文案，属预期内的可吞异常，但仍记录日志以便排查。
            logger.warning("ai companion media caption fallback: %r", exc)
    return sanitize_media_caption(fallback)


def _record_prompt_summary(record: dict[str, Any]) -> str:
    data = record.get("data")
    if not isinstance(data, Mapping):
        return ""
    for field in ("summary", "alt", "name", "text"):
        caption = sanitize_media_caption(str(data.get(field) or ""))
        if caption:
            return caption
    return ""


def _classifier_event(
    *,
    part: Mapping[str, Any],
    classification: StickerClassification,
    status: str,
    key: str,
    message: str,
) -> dict[str, Any]:
    data = part.get("data") if isinstance(part.get("data"), Mapping) else {}
    return {
        "event_type": "classifier",
        "status": status,
        "key": key,
        "message": clean_media_text(message)[:240],
        "details": {
            **classification.to_dict(),
            "segment_type": str(part.get("type") or ""),
            "summary": str((data or {}).get("summary") or "")[:120],
            "uri": str((data or {}).get("url") or (data or {}).get("file") or "")[:240],
        },
    }
