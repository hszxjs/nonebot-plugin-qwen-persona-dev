from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from ..contracts import StickerSelectionRuntime
from ..prompt.planner import ReplyIntent
from ..providers.generation_profiles import GenerationPurpose
from .classifier import is_sendable_sticker_record
from .model import _sticker_local_file_uri

logger = logging.getLogger(__name__)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
IMAGE_CAPABLE_TYPES = {"image", "mface", "emoji", "sticker"}
MIN_REPLY_STICKER_COUNT = 5


@dataclass(frozen=True)
class StickerSelection:
    key: str | None
    reason: str = ""
    raw_text: str = ""
    candidate_count: int = 0
    usable_count: int = 0
    fallback: bool = False
    keys: tuple[str, ...] = ()

    @property
    def selected_keys(self) -> list[str]:
        """去重后的选中表情（多选优先，回退单 key）。"""
        merged = list(self.keys) if self.keys else ([self.key] if self.key else [])
        return list(dict.fromkeys(merged))


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
    max_stickers: int = 1,
) -> StickerSelection:
    candidate_count = len(candidates)
    min_count_value = max(1, int(min_count))
    usable_candidates = [
        item
        for item in candidates
        if str(item.get("key") or "").strip()
        and is_sendable_sticker_record(item)
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
        max_stickers=max_stickers,
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

    raw_ids = payload.get("ids")
    if not isinstance(raw_ids, list):
        raw_ids = [payload.get("id") or payload.get("key") or ""]
    selected_ids = [str(item or "").strip() for item in raw_ids if str(item or "").strip()]
    if not selected_ids or any(item.upper() == "NONE" for item in selected_ids):
        return StickerSelection(
            key=None,
            reason=str(payload.get("reason") or ""),
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )

    ref_by_id = {item.id: item for item in candidate_refs}
    allowed_keys = {item.key for item in candidate_refs}

    def _resolve(value: str) -> str | None:
        if value in ref_by_id:
            return ref_by_id[value].key
        if value in allowed_keys:
            return value
        return None

    resolved_keys: list[str] = []
    for selected in selected_ids:
        key = _resolve(selected)
        if key is None:
            logger.warning(
                "invalid sticker selection: id_or_key=%r is not an exact candidate",
                selected,
            )
            continue
        if key not in resolved_keys:
            resolved_keys.append(key)
    if not resolved_keys:
        return StickerSelection(
            key=None,
            reason=str(payload.get("reason") or ""),
            raw_text=raw_text,
            candidate_count=candidate_count,
            usable_count=len(usable_candidates),
        )
    resolved_keys = resolved_keys[: max(1, int(max_stickers))]
    key = resolved_keys[0]

    return StickerSelection(
        key=key,
        keys=tuple(resolved_keys),
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
    max_stickers: int = 1,
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
                "如果没有合适表情，返回 NONE。多数时候选 1 个；仅在情绪强烈或连发更自然时最多选 2 个。"
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
                + '\n\n只返回 JSON：{"ids":["候选 id"]}（1 个，最多 '
                + str(max(1, int(max_stickers)))
                + ' 个；不合适则 {"ids":["NONE"]},"reason":"简短原因"}'
            ),
        },
    ]


def _candidate_media_parts(
    candidates: list[dict[str, Any]],
    *,
    max_candidate_images: int,
) -> list[dict[str, Any]]:
    """候选表情的附图，只用本地仓库副本。

    QQ 的 http 图片 URL 带 rkey 会过期：远程后端（GLM）拉不到就在服务端整单
    400，本地 MLX 下载也会失败。没有本地副本的候选仍留在编号文本列表里，
    选择器靠文字摘要照样能选。
    """
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
        local_uri = _sticker_local_file_uri(item)
        if not local_uri:
            continue
        data = dict(item.get("data") or {})
        data["url"] = urllib.parse.unquote(urllib.parse.urlparse(local_uri).path)
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
    # 加权随机（按 count，封顶 30 防止单张垄断）：兜底也不再永远同一张
    import random

    top = candidates[:5]
    weights = [min(float(getattr(item.item, "get", lambda *_: None)("count") or 1), 30.0) for item in top]
    chosen = random.choices(top, weights=weights, k=1)[0] if len(top) > 1 else top[0]
    return StickerSelection(
        key=chosen.key,
        reason=reason or "fallback weighted-random sticker",
        raw_text=raw_text,
        candidate_count=candidate_count,
        usable_count=len(candidates),
        fallback=True,
    )
