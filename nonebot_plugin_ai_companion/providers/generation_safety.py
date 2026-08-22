from __future__ import annotations

from dataclasses import dataclass

GARBLED_TEXT_MARKERS = (
    "<think>",
    "</think>",
    "nodeType",
    "Turing",
    "OrNull",
    "Gam",
    "Unfortunately",
)


@dataclass(frozen=True)
class RepetitiveTailTrim:
    text: str
    changed: bool
    repeated_unit: str = ""
    repeated_count: int = 0
    original_chars: int = 0
    truncated_chars: int = 0


def trim_repetitive_tail(
    text: str,
    *,
    max_unit_chars: int = 8,
    min_repeated_chars: int = 24,
    keep_repeats: int = 2,
) -> RepetitiveTailTrim:
    content = str(text or "")
    if not content:
        return RepetitiveTailTrim(text=content, changed=False)

    best: tuple[int, int, int, str, int, int] | None = None
    max_unit = min(max(1, int(max_unit_chars)), max(1, len(content) // 2))
    for unit_len in range(1, max_unit + 1):
        unit = content[-unit_len:]
        if not unit.strip():
            continue
        pos = len(content)
        count = 0
        while pos >= unit_len and content[pos - unit_len : pos] == unit:
            count += 1
            pos -= unit_len
        repeated_chars = count * unit_len
        min_repeats = 8 if unit_len == 1 else 4
        if count <= keep_repeats or count < min_repeats or repeated_chars < min_repeated_chars:
            continue
        candidate = (repeated_chars, -unit_len, unit_len, unit, count, pos)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return RepetitiveTailTrim(text=content, changed=False)

    repeated_chars, _prefer_shorter_unit, unit_len, unit, count, pos = best
    del unit_len
    trimmed = (content[:pos] + unit * keep_repeats).strip()
    return RepetitiveTailTrim(
        text=trimmed,
        changed=trimmed != content,
        repeated_unit=unit,
        repeated_count=count,
        original_chars=len(content),
        truncated_chars=max(0, len(content) - len(trimmed)),
    )


def is_low_quality_generated_text(text: str) -> bool:
    content = str(text or "")
    if not content.strip():
        return True
    return any(marker in content for marker in GARBLED_TEXT_MARKERS)
