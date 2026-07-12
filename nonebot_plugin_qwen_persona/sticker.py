from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Iterable
from urllib.parse import urlparse

STICKER_KEY_FIELDS = (
    "file_unique",
    "file_id",
    "file",
    "md5",
    "url",
    "emoji_id",
    "id",
    "summary",
)

STICKER_TOKEN_RE = re.compile(r"\[\[sticker:([A-Za-z0-9_.:-]+)\]\]")
ONEBOT_STICKER_TYPES = {"face", "mface", "emoji", "sticker"}
GENERIC_MEDIA_PLACEHOLDERS = {
    "[动画消息]",
    "[动画表情]",
    "[表情消息]",
    "[图片]",
    "[表情]",
}
IMAGE_STICKER_HINT_FIELDS = {
    "emoji_id",
    "emoji_package_id",
    "key",
    "sticker_id",
}
QQ_PRIVATE_STICKER_DOWNLOAD_HOST = "multimedia.nt.qq.com.cn"
QQ_FILE_DIGEST_RE = re.compile(r"^[A-Fa-f0-9]{32}$")
logger = logging.getLogger(__name__)


def extract_plain_text(message: Any) -> str:
    if hasattr(message, "extract_plain_text"):
        return str(message.extract_plain_text())
    return str(message or "")


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    segment_type = getattr(segment, "type", None)
    data = getattr(segment, "data", None)
    if isinstance(segment, dict):
        segment_type = segment.get("type", segment_type)
        data = segment.get("data", data)
    return {"type": str(segment_type or "unknown"), "data": dict(data or {})}


def extract_segments(message: Any) -> list[dict[str, Any]]:
    try:
        return [_segment_to_dict(segment) for segment in message]
    except TypeError:
        return [{"type": "text", "data": {"text": str(message or "")}}]


def _stable_sticker_key(segment_type: str, data: dict[str, Any]) -> str:
    basis: dict[str, str] = {}
    if segment_type == "image":
        file_value = str(data.get("file") or "").strip()
        file_stem = file_value.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if QQ_FILE_DIGEST_RE.fullmatch(file_stem):
            basis = {"qq_file_digest": file_stem.lower()}
    if not basis:
        basis = {
            field: str(data[field])
            for field in STICKER_KEY_FIELDS
            if field in data and data[field] is not None
        }
    if not basis:
        basis = {key: str(value) for key, value in sorted(data.items()) if value is not None}
    payload = json.dumps(
        {"type": segment_type, "basis": basis},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{segment_type}:{digest}"


def is_generic_media_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return normalized in GENERIC_MEDIA_PLACEHOLDERS


def meaningful_protocol_summary(data: dict[str, Any], *, limit: int = 200) -> str:
    value = data.get("summary")
    if value is None:
        return ""
    summary = " ".join(str(value).split())[:limit]
    if not summary or is_generic_media_placeholder(summary):
        return ""
    return summary


def _sticker_summary(segment_type: str, data: dict[str, Any]) -> str:
    summary = meaningful_protocol_summary(data, limit=80)
    if summary:
        return summary
    for field in ("summary", "alt", "name", "text", "id"):
        value = data.get(field)
        if value is not None and str(value).strip():
            summary = str(value).strip().replace("\n", " ")[:80]
            if not is_generic_media_placeholder(summary):
                return summary
    return ""


def _has_animation_file_extension(data: dict[str, Any]) -> bool:
    for field in ("file", "url", "file_id"):
        value = str(data.get(field) or "").strip()
        if not value:
            continue
        path = urlparse(value).path or value
        if path.lower().endswith((".gif", ".webp")):
            return True
    return False


def _is_qq_private_download_url(value: Any) -> bool:
    parsed = urlparse(str(value or ""))
    host = (parsed.hostname or "").lower()
    if host != QQ_PRIVATE_STICKER_DOWNLOAD_HOST:
        return False
    return (parsed.path or "").rstrip("/") == "/download"


def _has_qq_file_digest(data: dict[str, Any]) -> bool:
    file_value = str(data.get("file") or "").strip()
    file_stem = file_value.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return bool(QQ_FILE_DIGEST_RE.fullmatch(file_stem))


def _looks_like_qq_group_animation_sticker_data(data: dict[str, Any]) -> bool:
    if not _is_qq_private_download_url(data.get("url")):
        return False
    subtype = str(data.get("sub_type") or data.get("subType") or "").lower()
    return subtype == "1" and _has_qq_file_digest(data)


def is_transient_image_sticker_data(data: dict[str, Any]) -> bool:
    if not _is_qq_private_download_url(data.get("url")):
        return False
    if is_generic_media_placeholder(str(data.get("summary") or "")):
        return True
    return _looks_like_qq_group_animation_sticker_data(data)


def _has_image_sticker_hint(data: dict[str, Any]) -> bool:
    for field in IMAGE_STICKER_HINT_FIELDS:
        value = str(data.get(field) or "").strip()
        if not value:
            continue
        if field == "summary" and is_generic_media_placeholder(value):
            continue
        return True
    return False


def is_image_sticker_data(data: dict[str, Any]) -> bool:
    if str(data.get("sub_type") or data.get("subType") or "").lower() in {
        "sticker",
        "emoji",
        "mface",
    }:
        return True
    if _has_animation_file_extension(data):
        return True
    if _has_image_sticker_hint(data):
        return True
    return is_transient_image_sticker_data(data)


def sticker_to_record(
    segment: dict[str, Any],
    enabled_types: Iterable[str],
) -> dict[str, Any] | None:
    segment_type = str(segment.get("type") or "unknown")
    data = dict(segment.get("data") or {})
    if segment_type == "image":
        if "image" not in {str(item) for item in enabled_types}:
            return None
        if not is_image_sticker_data(data):
            return None
        key = _stable_sticker_key(segment_type, data)
        return {
            "key": key,
            "type": segment_type,
            "summary": _sticker_summary(segment_type, data),
            "uri": data.get("url") or data.get("file") or data.get("file_id"),
            "data": data,
        }
    if segment_type not in ONEBOT_STICKER_TYPES:
        return None
    if segment_type not in {str(item) for item in enabled_types}:
        return None
    key = _stable_sticker_key(segment_type, data)
    return {
        "key": key,
        "type": segment_type,
        "summary": _sticker_summary(segment_type, data),
        "uri": data.get("url") or data.get("file") or data.get("file_id") or data.get("id"),
        "data": data,
    }


def build_onebot_v11_sticker_segment(message_segment: Any, record: dict[str, Any]) -> Any | None:
    segment_type = str(record.get("type") or "")
    data = dict(record.get("data") or {})
    if segment_type == "image":
        if not is_image_sticker_data(data):
            return None
        uri = record.get("uri") or data.get("url") or data.get("file") or data.get("file_id")
        return message_segment.image(uri) if uri else None
    if segment_type == "face" and data.get("id") is not None:
        return message_segment.face(int(data["id"]))
    if segment_type == "mface":
        segment_data = {
            item: str(data[item])
            for item in ("emoji_id", "emoji_package_id", "key", "summary")
            if data.get(item) is not None
        }
        return message_segment("mface", segment_data) if segment_data else None
    if segment_type in {"emoji", "sticker"}:
        segment_data = {
            str(item): str(value)
            for item, value in data.items()
            if value is not None
        }
        return message_segment(segment_type, segment_data) if segment_data else None
    return None


def build_onebot_v11_sticker_message(
    message: Any,
    message_segment: Any,
    text: str,
    records: Iterable[dict[str, Any]],
) -> Any:
    result = message()
    clean_text = str(text or "").strip()
    if clean_text:
        try:
            result += message_segment.text(clean_text)
        except Exception as exc:
            logger.warning("qwen persona sticker text segment build failed: %r", exc)
    has_sticker = False
    for record in records:
        try:
            segment = build_onebot_v11_sticker_segment(message_segment, record)
        except Exception as exc:
            logger.warning("qwen persona sticker segment build failed: %r", exc)
            continue
        if segment is None:
            continue
        result += segment
        has_sticker = True
    return result if has_sticker else text


def sticker_token(key: str) -> str:
    return f"[[sticker:{key}]]"


def build_training_text(plain_text: str, sticker_keys: Iterable[str]) -> str:
    parts = [extract for extract in [plain_text.strip()] if extract]
    parts.extend(sticker_token(key) for key in sticker_keys)
    return " ".join(parts).strip()


def parse_sticker_tokens(text: str) -> tuple[str, list[str]]:
    keys = STICKER_TOKEN_RE.findall(text or "")
    clean_text = STICKER_TOKEN_RE.sub("", text or "").strip()
    return clean_text, keys
