from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .generation_profiles import GenerationPurpose
from .reply_planner import ReplyIntent
from .sticker_classifier import is_sendable_sticker_record

logger = logging.getLogger(__name__)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
IMAGE_CAPABLE_TYPES = {"image", "mface", "emoji", "sticker"}
MIN_REPLY_STICKER_COUNT = 5


class StickerSelectionRuntime(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: GenerationPurpose | str = GenerationPurpose.PERSONA_REPLY,
    ) -> str:
        ...


@dataclass(frozen=True)
class StickerSelection:
    key: str | None
    reason: str = ""
    raw_text: str = ""
    candidate_count: int = 0
    usable_count: int = 0
    fallback: bool = False


@dataclass(frozen=True)
class _CandidateRef:
    id: str
    key: str
    item: dict[str, Any]


async def select_reply_sticker(
    runtime: StickerSelectionRuntime,
    *,
    candidates: list[dict[str, Any]],
    trigger_text: str,
    reply_text: str,
    intent: ReplyIntent | str,
    max_candidate_images: int = 4,
    min_count: int = MIN_REPLY_STICKER_COUNT,
) -> StickerSelection:
    candidate_count = len(candidates)
    min_count_value = max(1, int(min_count))
    usable_candidates = [
        item
        for item in candidates
        if str(item.get("key") or "").strip() and is_sendable_sticker_record(item)
        and _sticker_count(item) >= min_count_value
    ]
    if not usable_candidates:
        return StickerSelection(key=None, candidate_count=candidate_count, usable_count=0)

    candidate_refs = [
        _CandidateRef(
            id=f"S{index}",
            key=str(item.get("key") or "").strip(),
            item=item,
        )
        for index, item in enumerate(usable_candidates, start=1)
    ]
    messages = _selector_messages(
        candidates=candidate_refs,
        trigger_text=trigger_text,
        reply_text=reply_text,
        intent=str(getattr(intent, "value", intent)),
    )
    media_parts = _candidate_media_parts(usable_candidates, max_candidate_images=max_candidate_images)
    try:
        raw_text = await runtime.chat(
            messages,
            media_parts=media_parts,
            purpose=GenerationPurpose.STICKER_SELECTOR,
        )
    except Exception as exc:
        logger.warning("sticker selection failed: %r", exc)
        fallback = _fallback_sticker_selection(
            candidate_refs,
            raw_text="",
            candidate_count=candidate_count,
            reason=f"fallback after selector failure: {type(exc).__name__}",
        )
        if fallback is not None:
            return fallback
        return StickerSelection(
            key=None,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    payload = _parse_json_object(raw_text)
    if payload is None:
        logger.warning("invalid sticker selection: non-json output=%r", raw_text)
        fallback = _fallback_sticker_selection(
            candidate_refs,
            raw_text=raw_text,
            candidate_count=candidate_count,
            reason=(
                "fallback after empty sticker selector output"
                if not str(raw_text or "").strip()
                else "fallback after invalid sticker selector output"
            ),
        )
        if fallback is not None:
            return fallback
        return StickerSelection(
            key=None,
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    selected = str(payload.get("id") or payload.get("key") or "").strip()
    if not selected or selected.upper() == "NONE":
        return StickerSelection(
            key=None,
            reason=str(payload.get("reason") or ""),
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    ref_by_id = {item.id: item for item in candidate_refs}
    allowed_keys = {item.key for item in candidate_refs}
    if selected in ref_by_id:
        key = ref_by_id[selected].key
    elif selected in allowed_keys:
        key = selected
    else:
        logger.warning(
            "invalid sticker selection: id_or_key=%r is not an exact candidate",
            selected,
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
    candidates: list[_CandidateRef],
    trigger_text: str,
    reply_text: str,
    intent: str,
) -> list[dict[str, Any]]:
    candidate_lines = []
    for ref in candidates:
        item = ref.item
        summary = str(item.get("summary") or item.get("type") or "").strip()
        segment_type = str(item.get("type") or "").strip()
        count = str(item.get("count") or 0)
        candidate_lines.append(f"{ref.id}. type={segment_type} summary={summary} count={count}")

    return [
        {
            "role": "system",
            "content": (
                "你是群聊表情选择器。只能选择候选 id，不能发明、补全或改写 id。"
                "如果没有合适表情，返回 NONE。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"intent={intent}\n"
                f"trigger={str(trigger_text or '').strip()}\n"
                f"reply={str(reply_text or '').strip()}\n\n"
                "只能选择候选 id，不能输出不存在的 id。\n"
                "候选表情：\n"
                + "\n".join(candidate_lines)
                + '\n\n只返回候选 id 的 JSON：{"id":"候选 id 或 NONE","reason":"简短原因"}'
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


def _sticker_count(candidate: dict[str, Any]) -> int:
    try:
        return int(candidate.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _fallback_sticker_selection(
    candidates: list[_CandidateRef],
    *,
    raw_text: str,
    candidate_count: int,
    reason: str = "",
) -> StickerSelection | None:
    if not candidates:
        return None
    return StickerSelection(
        key=candidates[0].key,
        reason=reason or "fallback to first eligible sticker",
        raw_text=raw_text,
        candidate_count=candidate_count,
        usable_count=len(candidates),
        fallback=True,
    )
