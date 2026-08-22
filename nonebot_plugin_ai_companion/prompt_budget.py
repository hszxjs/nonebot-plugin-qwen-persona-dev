from __future__ import annotations

from typing import Any, Iterable

DEFAULT_MLX_CONTEXT_SIZE = 4096


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mlx_context_size(config: Any) -> int:
    return max(
        512,
        _int_value(
            getattr(config, "qwen_persona_mlx_lora_max_seq_length", DEFAULT_MLX_CONTEXT_SIZE),
            DEFAULT_MLX_CONTEXT_SIZE,
        ),
    )


def effective_max_tokens(config: Any) -> int:
    requested = max(1, _int_value(getattr(config, "qwen_persona_max_tokens", 256), 256))
    ctx_size = mlx_context_size(config)
    return max(16, min(requested, max(64, ctx_size // 4)))


def effective_context_message_limit(config: Any) -> int:
    requested = max(0, _int_value(getattr(config, "qwen_persona_context_messages", 16), 16))
    ctx_size = mlx_context_size(config)
    return min(requested, max(4, ctx_size // 256))


def trim_history_lines_for_context(lines: Iterable[str], config: Any) -> list[str]:
    items = [str(item) for item in lines if str(item).strip()]
    budget = max(256, mlx_context_size(config) - effective_max_tokens(config) - 512)
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
