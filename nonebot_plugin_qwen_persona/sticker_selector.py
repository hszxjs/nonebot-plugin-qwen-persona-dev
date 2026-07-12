from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .reply_planner import ReplyIntent
from .sticker_classifier import is_sendable_sticker_record

logger = logging.getLogger(__name__)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
IMAGE_CAPABLE_TYPES = {"image", "mface", "emoji", "sticker"}


class StickerSelectionRuntime(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class StickerSelection:
    key: str | None
    reason: str = ""
    raw_text: str = ""
    candidate_count: int = 0
    usable_count: int = 0


async def select_reply_sticker(
    runtime: StickerSelectionRuntime,
    *,
    candidates: list[dict[str, Any]],
    trigger_text: str,
    reply_text: str,
    intent: ReplyIntent | str,
    max_candidate_images: int = 4,
) -> StickerSelection:
    candidate_count = len(candidates)
    usable_candidates = [
        item
        for item in candidates
        if str(item.get("key") or "").strip() and is_sendable_sticker_record(item)
    ]
    if not usable_candidates:
        return StickerSelection(key=None, candidate_count=candidate_count, usable_count=0)

    messages = _selector_messages(
        candidates=usable_candidates,
        trigger_text=trigger_text,
        reply_text=reply_text,
        intent=str(getattr(intent, "value", intent)),
    )
    media_parts = _candidate_media_parts(usable_candidates, max_candidate_images=max_candidate_images)
    try:
        raw_text = await runtime.chat(messages, media_parts=media_parts)
    except Exception as exc:
        logger.warning("sticker selection failed: %r", exc)
        return StickerSelection(
            key=None,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    payload = _parse_json_object(raw_text)
    if payload is None:
        logger.warning("invalid sticker selection: non-json output=%r", raw_text)
        return StickerSelection(
            key=None,
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    key = str(payload.get("key") or "").strip()
    if not key or key.upper() == "NONE":
        return StickerSelection(
            key=None,
            reason=str(payload.get("reason") or ""),
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    allowed_keys = {str(item.get("key") or "").strip() for item in usable_candidates}
    if key not in allowed_keys:
        logger.warning(
            "invalid sticker selection: key=%r is not an exact candidate key",
            key,
        )
        return StickerSelection(
            key=None,
            reason=str(payload.get("reason") or ""),
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    return StickerSelection(
        key=key,
        reason=str(payload.get("reason") or ""),
        raw_text=raw_text,
        candidate_count=candidate_count,
        usable_count=len(usable_candidates),
    )


def _selector_messages(
    *,
    candidates: list[dict[str, Any]],
    trigger_text: str,
    reply_text: str,
    intent: str,
) -> list[dict[str, Any]]:
    candidate_lines = []
    for index, item in enumerate(candidates, start=1):
        key = str(item.get("key") or "").strip()
        summary = str(item.get("summary") or item.get("type") or "").strip()
        segment_type = str(item.get("type") or "").strip()
        count = str(item.get("count") or 0)
        candidate_lines.append(f"{index}. key={key} type={segment_type} summary={summary} count={count}")

    return [
        {
            "role": "system",
            "content": (
                "你是群聊表情选择器。只能选择候选 key，不能发明、补全或改写 key。"
                "如果没有合适表情，返回 NONE。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"intent={intent}\n"
                f"trigger={str(trigger_text or '').strip()}\n"
                f"reply={str(reply_text or '').strip()}\n\n"
                "只能选择候选 key，不能输出不存在的 key。\n"
                "候选表情：\n"
                + "\n".join(candidate_lines)
                + '\n\n只返回 JSON：{"key":"候选 key 或 NONE","reason":"简短原因"}'
            ),
        },
    ]


def _candidate_media_parts(
    candidates: list[dict[str, Any]],
    *,
    max_candidate_images: int,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    limit = max(0, int(max_candidate_images))
    for item in candidates:
        if len(parts) >= limit:
            break
        segment_type = str(item.get("type") or "").strip()
        if not is_sendable_sticker_record(item):
            continue
        if segment_type not in IMAGE_CAPABLE_TYPES:
            continue
        data = dict(item.get("data") or {})
        uri = str(item.get("uri") or data.get("url") or data.get("file") or data.get("file_id") or "").strip()
        if not uri or uri.startswith("base64://"):
            continue
        if not any(str(data.get(field) or "").strip() for field in ("url", "file", "file_id", "uri")):
            data["url"] = uri
        parts.append({"type": segment_type, "data": data})
    return parts


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    match = JSON_OBJECT_RE.search(text)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
