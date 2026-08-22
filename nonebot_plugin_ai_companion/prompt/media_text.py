"""媒体文本的提示词侧工具：模型输出消毒、样本/学习结果的上下文渲染、caption 净化。"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..context.media_parts import CQ_CODE_RE
from ..providers.generation_safety import is_low_quality_generated_text
from ..sticker.model import STICKER_TOKEN_RE, build_training_text, is_generic_media_placeholder
from .text import clean_media_text

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


def sanitize_model_reply_media_text(text: str) -> str:
    without_cq = CQ_CODE_RE.sub("", text or "")
    without_media_tokens = MEDIA_TOKEN_RE.sub("", without_cq)
    without_artifacts = REPLY_MEDIA_ARTIFACT_RE.sub("", without_media_tokens)
    return clean_media_text(without_artifacts)


def sanitize_media_caption(text: str) -> str:
    cleaned = sanitize_model_reply_media_text(str(text or ""))
    cleaned = re.sub(r"(?:https?://|file://|base64://)\S+", "", cleaned, flags=re.IGNORECASE)
    if is_low_quality_generated_text(cleaned):
        return ""
    if is_generic_media_placeholder(cleaned):
        return ""
    return clean_media_text(cleaned)[:120]


def context_text_from_sample(sample: Mapping[str, Any]) -> str:
    plain_text = clean_media_text(str(sample.get("plain_text") or ""))
    sticker_keys = _safe_sticker_keys(sample.get("sticker_keys"))
    if plain_text:
        return build_training_text(plain_text, sticker_keys)
    fallback_text = context_text_from_training_text(str(sample.get("training_text") or ""))
    if fallback_text:
        return fallback_text
    return build_training_text("", sticker_keys)


def context_text_from_training_text(text: str) -> str:
    without_descriptions = CONTEXT_MEDIA_DESCRIPTION_RE.sub(" ", text or "")
    return clean_media_text(_strip_prompt_unsafe_sticker_tokens(without_descriptions))


def reply_context_text_from_learning(learning: Mapping[str, Any]) -> str:
    plain_text = clean_media_text(str(learning.get("plain_text") or ""))
    training_text = str(learning.get("training_text") or "")
    without_sticker_tokens = STICKER_TOKEN_RE.sub("", training_text)
    media_context = clean_media_text(str(learning.get("media_context_text") or ""))
    if not plain_text and not media_context and not context_text_from_training_text(without_sticker_tokens):
        return ""
    chunks = []
    base = clean_media_text(without_sticker_tokens)
    if base:
        chunks.append(base)
    if media_context:
        chunks.append(media_context)
    return clean_media_text(" ".join(chunks))


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


def parse_sticker_tokens_unique(text: str) -> tuple[str, list[str]]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in STICKER_TOKEN_RE.findall(text or ""):
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    clean_text = STICKER_TOKEN_RE.sub("", text or "").strip()
    return clean_media_text(clean_text), keys


def filter_training_text_sticker_tokens(
    training_text: str,
    stickers: Iterable[Mapping[str, Any]],
) -> str:
    allowed_keys = {
        str(sticker.get("key") or "").strip() for sticker in stickers if str(sticker.get("key") or "").strip()
    }
    base_text, sticker_keys = parse_sticker_tokens_unique(training_text)
    kept_keys = [key for key in sticker_keys if key in allowed_keys]
    return build_training_text(base_text, kept_keys)
