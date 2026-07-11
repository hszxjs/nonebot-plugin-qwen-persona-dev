from __future__ import annotations

from typing import Any, Iterable

LLAMA_RUNTIMES = {"llama_cpp", "external_llama"}


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_llama_runtime(config: Any) -> bool:
    return str(getattr(config, "qwen_persona_runtime", "") or "") in LLAMA_RUNTIMES


def llama_context_size(config: Any) -> int:
    return max(512, _int_value(getattr(config, "qwen_persona_llama_ctx_size", 4096), 4096))


def effective_max_tokens(config: Any) -> int:
    requested = max(1, _int_value(getattr(config, "qwen_persona_max_tokens", 256), 256))
    if not _is_llama_runtime(config):
        return requested
    ctx_size = llama_context_size(config)
    return max(16, min(requested, max(64, ctx_size // 4)))


def effective_context_message_limit(config: Any) -> int:
    requested = max(0, _int_value(getattr(config, "qwen_persona_context_messages", 16), 16))
    if not _is_llama_runtime(config):
        return requested
    ctx_size = llama_context_size(config)
    return min(requested, max(4, ctx_size // 256))


def trim_history_lines_for_context(lines: Iterable[str], config: Any) -> list[str]:
    items = [str(item) for item in lines if str(item).strip()]
    if not _is_llama_runtime(config):
        return items

    budget = max(256, llama_context_size(config) - effective_max_tokens(config) - 512)
    kept_reversed: list[str] = []
    used = 0
    for line in reversed(items):
        cost = len(line) + 1
        if kept_reversed and used + cost > budget:
            break
        kept_reversed.append(line)
        used += cost
        if used >= budget:
            break
    return list(reversed(kept_reversed))
