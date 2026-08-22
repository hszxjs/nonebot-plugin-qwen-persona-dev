"""消息 → 媒体部件解析：OneBot 消息段与 CQ 码统一转为 dict 部件列表。"""

from __future__ import annotations

import re
from typing import Any

CQ_CODE_RE = re.compile(r"\[CQ:(?P<type>[A-Za-z0-9_]+)(?P<params>(?:,[^\]]*)?)\]")


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
