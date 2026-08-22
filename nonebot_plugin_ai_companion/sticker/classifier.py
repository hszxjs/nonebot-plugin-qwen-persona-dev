from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

ONEBOT_STICKER_TYPES = {"face", "mface", "emoji", "sticker"}
GENERIC_MEDIA_PLACEHOLDERS = {
    "[动画消息]",
    "[动画表情]",
    "[表情消息]",
    "[图片]",
    "[表情]",
}
EXPLICIT_IMAGE_STICKER_FIELDS = {"emoji_id", "emoji_package_id", "sticker_id"}
IMAGE_STICKER_SUBTYPES = {"sticker", "emoji", "mface"}
TEXT_ACCEPT_RE = re.compile(
    r"(表情包|表情|贴纸|贴图|梗图|动图|meme|reaction|sticker|emoji|"
    r"二次元|动漫|动画风格|q版|chibi|颜艺|吐槽|拍桌|狂笑|捂脸|震惊|"
    r"哭笑|滑稽|阴阳怪气)",
    re.IGNORECASE,
)
TEXT_REJECT_RE = re.compile(
    r"(截图|屏幕截图|页面|网页|应用详情|下载量|安装按钮|评分|安全警报|漏洞|"
    r"修复建议|代码|日志|控制台|终端|文档|表格|菜单|设置页|报错|"
    r"错误页面|通知栏|聊天记录|支付|订单|二维码|地图|路线|风景|景色|"
    r"城市夜景|街景|建筑|照片|摄影|实拍|"
    r"screenshot|screen capture|webpage|app page|security alert|"
    r"vulnerability|document|spreadsheet|chart|terminal|console|log|"
    r"invoice|receipt|landscape|scenery|photo)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StickerClassification:
    label: str
    reason: str
    sendable: bool
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "reason": self.reason,
            "sendable": self.sendable,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


def classify_sticker_segment(
    segment: Mapping[str, Any],
    *,
    caption: str = "",
) -> StickerClassification:
    segment_type = str(segment.get("type") or "unknown")
    if segment_type in ONEBOT_STICKER_TYPES:
        return StickerClassification(
            label="protocol_sticker",
            reason=f"onebot {segment_type} segment",
            sendable=True,
            confidence=1.0,
        )
    if segment_type != "image":
        return StickerClassification(
            label="unsupported",
            reason=f"unsupported segment type: {segment_type}",
            sendable=False,
            confidence=1.0,
        )
    return classify_image_sticker_data(dict(segment.get("data") or {}), caption=caption)


def classify_sticker_record(record: Mapping[str, Any]) -> StickerClassification:
    data = dict(record.get("data") or {})
    embedded = data.get("_classification")
    if isinstance(embedded, Mapping) and "sendable" in embedded:
        return StickerClassification(
            label=str(embedded.get("label") or "unknown"),
            reason=str(embedded.get("reason") or ""),
            sendable=bool(embedded.get("sendable")),
            confidence=_optional_float(embedded.get("confidence")),
        )

    segment_type = str(record.get("type") or "unknown")
    if segment_type in ONEBOT_STICKER_TYPES:
        return StickerClassification(
            label="protocol_sticker",
            reason=f"onebot {segment_type} segment",
            sendable=True,
            confidence=1.0,
        )
    if segment_type != "image":
        return StickerClassification(
            label="unsupported",
            reason=f"unsupported segment type: {segment_type}",
            sendable=False,
            confidence=1.0,
        )
    return classify_image_sticker_data(
        data,
        caption=str(record.get("summary") or ""),
    )


def is_sendable_sticker_record(record: Mapping[str, Any]) -> bool:
    return classify_sticker_record(record).sendable


def classify_image_sticker_data(
    data: Mapping[str, Any],
    *,
    caption: str = "",
) -> StickerClassification:
    values = dict(data or {})
    subtype = str(values.get("sub_type") or values.get("subType") or "").strip().lower()
    if subtype in IMAGE_STICKER_SUBTYPES:
        return StickerClassification(
            label="protocol_sticker",
            reason=f"image subtype is {subtype}",
            sendable=True,
            confidence=0.98,
        )

    if _has_explicit_custom_metadata(values):
        return StickerClassification(
            label="custom_sticker",
            reason="explicit custom sticker metadata",
            sendable=True,
            confidence=0.98,
        )

    text = _classification_text(values, caption=caption)
    if text and TEXT_REJECT_RE.search(text):
        return StickerClassification(
            label="screenshot",
            reason="caption or protocol text looks like screenshot/document/plain photo",
            sendable=False,
            confidence=0.9,
        )

    if text and TEXT_ACCEPT_RE.search(text):
        return StickerClassification(
            label="sticker_like_image",
            reason="caption or protocol text describes sticker-like image",
            sendable=True,
            confidence=0.82,
        )

    if _has_animation_file_extension(values):
        return StickerClassification(
            label="animated_image",
            reason="animated image extension without plain-image signals",
            sendable=True,
            confidence=0.62,
        )

    if _is_qq_private_download_url(values.get("url")):
        return StickerClassification(
            label="unknown_qq_image",
            reason="qq private image download lacks sticker-specific metadata",
            sendable=False,
            confidence=0.72,
        )

    return StickerClassification(
        label="plain_image",
        reason="image lacks sticker-specific metadata or caption",
        sendable=False,
        confidence=0.65,
    )


def _classification_text(data: Mapping[str, Any], *, caption: str) -> str:
    chunks = []
    safe_caption = _safe_text(caption)
    if safe_caption:
        chunks.append(safe_caption)
    for field in ("summary", "alt", "name", "text", "title", "description"):
        value = _safe_text(data.get(field))
        if value:
            chunks.append(value)
    return " ".join(chunks)


def _safe_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or _is_generic_media_placeholder(text):
        return ""
    return text[:240]


def _is_generic_media_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return normalized in GENERIC_MEDIA_PLACEHOLDERS


def _has_explicit_custom_metadata(data: Mapping[str, Any]) -> bool:
    if str(data.get("emoji_id") or "").strip() and str(data.get("emoji_package_id") or "").strip():
        return True
    if str(data.get("sticker_id") or "").strip():
        return True
    if str(data.get("key") or "").strip() and str(data.get("url") or "").strip():
        return True
    return False


def _has_animation_file_extension(data: Mapping[str, Any]) -> bool:
    for field in ("file", "url", "file_id", "uri"):
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
    if host != "multimedia.nt.qq.com.cn":
        return False
    return (parsed.path or "").rstrip("/") == "/download"


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
