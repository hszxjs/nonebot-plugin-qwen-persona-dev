from __future__ import annotations

from dataclasses import dataclass


DEFAULT_LONG_TEXT_THRESHOLD = 700
DEFAULT_MAX_CHUNK_CHARS = 650
DEFAULT_MAX_NORMAL_LINES = 8


@dataclass(frozen=True)
class ReplyDeliveryDecision:
    use_forward: bool
    forward_chunks: list[str]
    trailing_sticker_keys: list[str]


def plan_reply_delivery(
    text: str,
    sticker_keys: list[str],
    *,
    long_text_threshold: int = DEFAULT_LONG_TEXT_THRESHOLD,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_normal_lines: int = DEFAULT_MAX_NORMAL_LINES,
) -> ReplyDeliveryDecision:
    content = str(text or "")
    use_forward = _should_use_forward(
        content,
        long_text_threshold=long_text_threshold,
        max_normal_lines=max_normal_lines,
    )
    if not use_forward:
        return ReplyDeliveryDecision(
            use_forward=False,
            forward_chunks=[],
            trailing_sticker_keys=[],
        )
    return ReplyDeliveryDecision(
        use_forward=True,
        forward_chunks=split_reply_text_for_forward(content, max_chunk_chars=max_chunk_chars),
        trailing_sticker_keys=[str(key) for key in sticker_keys if str(key).strip()],
    )


def split_reply_text_for_forward(text: str, *, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    content = str(text or "")
    if not content:
        return []
    limit = max(1, int(max_chunk_chars))
    chunks: list[str] = []
    current = ""
    for line in content.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit and current:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks or [content]


def _should_use_forward(
    text: str,
    *,
    long_text_threshold: int,
    max_normal_lines: int,
) -> bool:
    content = str(text or "")
    if not content.strip():
        return False
    if len(content) >= max(1, int(long_text_threshold)):
        return True
    if "```" in content:
        return True
    return len(content.splitlines()) > max(1, int(max_normal_lines))

