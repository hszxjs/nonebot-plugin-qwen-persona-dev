from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .filters import normalize_text
from .sticker import sticker_to_record
from .sticker_classifier import classify_sticker_segment

logger = logging.getLogger(__name__)


class MediaCaptioner(Protocol):
    async def caption_media(self, part: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class MediaAnalysisResult:
    accepted: int = 0
    rejected: int = 0
    skipped: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BackgroundMediaAnalysisDecision:
    allowed: bool
    reason: str


def should_analyze_background_media(
    *,
    enable_learning: bool,
    group_enabled: bool,
    self_event: bool,
    blacklisted: bool,
    command_like: bool,
    force_reply: bool,
    has_media: bool,
) -> BackgroundMediaAnalysisDecision:
    if not enable_learning:
        return BackgroundMediaAnalysisDecision(False, "learning_disabled")
    if not has_media:
        return BackgroundMediaAnalysisDecision(False, "no_media")
    if not group_enabled:
        return BackgroundMediaAnalysisDecision(False, "group_disabled")
    if self_event:
        return BackgroundMediaAnalysisDecision(False, "self_event")
    if blacklisted:
        return BackgroundMediaAnalysisDecision(False, "blacklisted")
    if force_reply:
        return BackgroundMediaAnalysisDecision(True, "force_reply")
    if command_like:
        return BackgroundMediaAnalysisDecision(False, "command_like")
    return BackgroundMediaAnalysisDecision(True, "accepted")


async def analyze_media_parts(
    *,
    store: Any,
    captioner: MediaCaptioner | None,
    parts: list[dict[str, Any]],
    group_id: str | None,
    user_id: str | None,
    sticker_segment_types: set[str] | None = None,
) -> MediaAnalysisResult:
    enabled_types = sticker_segment_types or {"image", "face", "mface", "emoji", "sticker"}
    accepted = 0
    rejected = 0
    skipped = 0
    events: list[dict[str, Any]] = []

    for part in parts:
        part_type = str(part.get("type") or "")
        if part_type != "image":
            skipped += 1
            continue
        initial_classification = classify_sticker_segment(part)
        caption = await _caption_or_empty(captioner, part)
        classification = classify_sticker_segment(part, caption=caption)
        data = dict(part.get("data") or {})
        uri = str(data.get("url") or data.get("file") or data.get("file_id") or "").strip()
        event = {
            "event_type": "media_analysis",
            "status": "accepted" if classification.sendable else "rejected",
            "key": "",
            "message": caption or classification.reason,
            "details": {
                **classification.to_dict(),
                "uri": uri[:240],
                "caption": caption[:240],
            },
        }

        if not classification.sendable:
            rejected += 1
            events.append(event)
            continue

        record = sticker_to_record(part, enabled_types, classification=classification)
        if record is None:
            skipped += 1
            event["status"] = "skipped"
            events.append(event)
            continue
        if caption:
            record["summary"] = caption
            record.setdefault("data", {})["summary"] = caption
        event["key"] = str(record.get("key") or "")

        if hasattr(store, "record_sticker_asset"):
            kwargs = {
                "sticker": record,
                "group_id": group_id,
                "user_id": user_id,
                "summary": caption,
                "increment_count": not initial_classification.sendable,
            }
            try:
                await store.record_sticker_asset(**kwargs)
            except TypeError:
                kwargs.pop("increment_count", None)
                await store.record_sticker_asset(**kwargs)
        else:
            await store.record_media_asset(
                source="sticker",
                source_key=str(record.get("key") or _media_source_key(part, caption)),
                uri=uri,
                prompt="请用一句中文描述这个群聊表情的含义。",
                answer=caption or str(record.get("summary") or ""),
                metadata={
                    "group_id": group_id,
                    "user_id": user_id,
                    "classification": classification.to_dict(),
                },
            )
        accepted += 1
        events.append(event)

    return MediaAnalysisResult(
        accepted=accepted,
        rejected=rejected,
        skipped=skipped,
        events=events,
    )


async def _caption_or_empty(captioner: MediaCaptioner | None, part: dict[str, Any]) -> str:
    if captioner is None:
        return ""
    try:
        return normalize_text(await captioner.caption_media(part))[:200]
    except Exception as exc:
        logger.warning("qwen persona media analysis caption failed: %r", exc)
        return ""


def _media_source_key(part: dict[str, Any], caption: str) -> str:
    payload = repr({"part": part, "caption": caption}).encode("utf-8", errors="replace")
    return "media:" + hashlib.sha1(payload).hexdigest()[:20]
