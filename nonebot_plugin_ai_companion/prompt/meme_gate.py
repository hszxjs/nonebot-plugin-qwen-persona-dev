from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .media_text import context_text_from_sample
from .text import normalize_text

CJK_RE = re.compile(r"[\u3400-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def build_meme_candidates_from_samples(
    *,
    samples: Iterable[dict[str, Any]],
    media_assets: Iterable[dict[str, Any]],
    label_limit: int = 36,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    seen_message_ids: set[str] = set()
    seen_media_keys: set[str] = set()

    for item in samples:
        message_id = str(item.get("message_id") or "").strip()
        if message_id:
            if message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
        label = _compact_label(context_text_from_sample(item), limit=label_limit)
        if label:
            _add_candidate_count(
                buckets,
                label=label,
                count=_explicit_count_or_default(item, default=1),
                hint="历史群聊中重复出现过；只有同一话题下才顺手引用",
            )

    for item in media_assets:
        source_key = str(item.get("source_key") or item.get("id") or "").strip()
        if source_key:
            if source_key in seen_media_keys:
                continue
            seen_media_keys.add(source_key)
        label = _compact_label(str(item.get("answer") or ""), limit=label_limit)
        if label:
            _add_candidate_count(
                buckets,
                label=label,
                count=_explicit_count_or_default(item, default=1),
                hint="历史图片或表情语义重复出现过；不确定时忽略",
            )

    return sorted(
        buckets.values(),
        key=lambda item: (int(item.get("recent_count") or 0), str(item.get("label") or "")),
        reverse=True,
    )


def select_active_memes(
    candidates: Iterable[dict[str, Any]],
    *,
    context_text: str,
    min_recent_count: int = 2,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    context = normalize_text(context_text).lower()
    if not context:
        return []

    threshold = max(1, int(min_recent_count))
    selected: list[tuple[int, int, str, dict[str, Any]]] = []
    for candidate in candidates:
        label = normalize_text(str(candidate.get("label") or candidate.get("summary") or ""))
        if not label:
            continue
        recent_count = _recent_count(candidate)
        if recent_count < threshold:
            continue
        keywords = _candidate_keywords(candidate, label)
        if not _matches_context(label=label, keywords=keywords, context=context):
            continue
        payload = {
            "label": label,
            "hint": normalize_text(str(candidate.get("hint") or "")),
            "recent_count": recent_count,
        }
        score = recent_count + len(keywords)
        selected.append((score, recent_count, label, payload))

    selected.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    limit = max(0, int(max_items))
    return [item[3] for item in selected[:limit]]


def _add_candidate_count(
    buckets: dict[str, dict[str, Any]],
    *,
    label: str,
    count: int,
    hint: str,
) -> None:
    bucket = buckets.setdefault(
        label,
        {
            "label": label,
            "hint": hint,
            "recent_count": 0,
        },
    )
    bucket["recent_count"] = int(bucket.get("recent_count") or 0) + max(1, int(count))


def _compact_label(text: str, *, limit: int) -> str:
    value = normalize_text(text)
    if len(value) <= 1:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)] + "..."


def _explicit_count_or_default(candidate: dict[str, Any], *, default: int) -> int:
    for key in ("recent_count", "count", "usage_count"):
        if key not in candidate:
            continue
        try:
            return max(1, int(candidate.get(key) or 0))
        except (TypeError, ValueError):
            return max(1, int(default))
    return max(1, int(default))


def _recent_count(candidate: dict[str, Any]) -> int:
    for key in ("recent_count", "count", "usage_count"):
        try:
            return int(candidate.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return 0


def _candidate_keywords(candidate: dict[str, Any], label: str) -> set[str]:
    raw_keywords = candidate.get("keywords")
    values: list[str] = []
    if isinstance(raw_keywords, (list, tuple, set)):
        values.extend(str(item) for item in raw_keywords)
    elif isinstance(raw_keywords, str):
        values.extend(item for item in re.split(r"[\s,，、/|]+", raw_keywords) if item)
    values.append(label)
    hint = str(candidate.get("hint") or "")
    if hint:
        values.append(hint)

    keywords: set[str] = set()
    for value in values:
        text = normalize_text(value).lower()
        if not text:
            continue
        keywords.update(match.group(0) for match in WORD_RE.finditer(text))
        for match in CJK_RE.finditer(text):
            chunk = match.group(0)
            if len(chunk) <= 6:
                keywords.add(chunk)
            for size in (2, 3, 4):
                if len(chunk) < size:
                    continue
                for index in range(0, len(chunk) - size + 1):
                    keywords.add(chunk[index : index + size])
    return {keyword for keyword in keywords if len(keyword) >= 2}


def _matches_context(*, label: str, keywords: set[str], context: str) -> bool:
    lowered_label = label.lower()
    if len(lowered_label) >= 2 and lowered_label in context:
        return True
    return any(keyword in context for keyword in keywords)
