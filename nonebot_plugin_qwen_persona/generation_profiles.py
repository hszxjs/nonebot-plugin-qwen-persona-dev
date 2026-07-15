from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import QwenPersonaConfig
from .prompt_budget import effective_max_tokens


CAPTION_MAX_TOKENS = 64
CAPTION_TEMPERATURE = 0.2
CAPTION_TOP_P = 0.8
CAPTION_REPETITION_PENALTY = 1.15
CAPTION_REPETITION_CONTEXT_SIZE = 128

STICKER_SELECTOR_MIN_TOKENS = 128
STICKER_SELECTOR_MAX_TOKENS = 256
STICKER_SELECTOR_TEMPERATURE = 0.2
STICKER_SELECTOR_TOP_P = 0.8
STICKER_SELECTOR_REPETITION_PENALTY = 1.12
STICKER_SELECTOR_REPETITION_CONTEXT_SIZE = 128


class GenerationPurpose(StrEnum):
    PERSONA_REPLY = "persona_reply"
    MEDIA_CAPTION = "media_caption"
    STICKER_SELECTOR = "sticker_selector"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class GenerationProfile:
    purpose: GenerationPurpose
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    repetition_context_size: int
    reject_repetition: bool = False


def normalize_generation_purpose(purpose: GenerationPurpose | str | None) -> GenerationPurpose:
    if isinstance(purpose, GenerationPurpose):
        return purpose
    if not purpose:
        return GenerationPurpose.PERSONA_REPLY
    try:
        return GenerationPurpose(str(purpose))
    except ValueError:
        return GenerationPurpose.PERSONA_REPLY


def profile_for_purpose(
    config: QwenPersonaConfig,
    purpose: GenerationPurpose | str | None,
) -> GenerationProfile:
    normalized = normalize_generation_purpose(purpose)
    if normalized is GenerationPurpose.MEDIA_CAPTION:
        return _caption_generation_profile(config)
    if normalized is GenerationPurpose.STICKER_SELECTOR:
        return _sticker_selector_generation_profile(config)
    return _persona_generation_profile(config, purpose=normalized)


def _persona_generation_profile(
    config: QwenPersonaConfig,
    *,
    purpose: GenerationPurpose,
) -> GenerationProfile:
    return GenerationProfile(
        purpose=purpose,
        max_tokens=effective_max_tokens(config),
        temperature=float(config.qwen_persona_temperature),
        top_p=float(config.qwen_persona_top_p),
        repetition_penalty=float(config.qwen_persona_repetition_penalty),
        repetition_context_size=int(config.qwen_persona_repetition_context_size),
    )


def _caption_generation_profile(config: QwenPersonaConfig) -> GenerationProfile:
    return GenerationProfile(
        purpose=GenerationPurpose.MEDIA_CAPTION,
        max_tokens=min(effective_max_tokens(config), CAPTION_MAX_TOKENS),
        temperature=CAPTION_TEMPERATURE,
        top_p=CAPTION_TOP_P,
        repetition_penalty=max(
            float(config.qwen_persona_repetition_penalty),
            CAPTION_REPETITION_PENALTY,
        ),
        repetition_context_size=max(
            int(config.qwen_persona_repetition_context_size),
            CAPTION_REPETITION_CONTEXT_SIZE,
        ),
        reject_repetition=True,
    )


def _sticker_selector_generation_profile(config: QwenPersonaConfig) -> GenerationProfile:
    base_tokens = effective_max_tokens(config)
    selector_tokens = max(
        STICKER_SELECTOR_MIN_TOKENS,
        min(base_tokens, STICKER_SELECTOR_MAX_TOKENS),
    )
    return GenerationProfile(
        purpose=GenerationPurpose.STICKER_SELECTOR,
        max_tokens=selector_tokens,
        temperature=STICKER_SELECTOR_TEMPERATURE,
        top_p=STICKER_SELECTOR_TOP_P,
        repetition_penalty=max(
            float(config.qwen_persona_repetition_penalty),
            STICKER_SELECTOR_REPETITION_PENALTY,
        ),
        repetition_context_size=max(
            int(config.qwen_persona_repetition_context_size),
            STICKER_SELECTOR_REPETITION_CONTEXT_SIZE,
        ),
        reject_repetition=True,
    )
