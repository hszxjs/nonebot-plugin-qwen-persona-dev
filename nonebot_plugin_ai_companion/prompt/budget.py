"""上下文预算：按后端能力决定历史条数与字符上限。

本地 MLX 受限于推理时的序列长度（几十条就爆），在线 LLM 有远大的上下文窗口，
所以同一个 `ai_companion_context_messages` 在两条路径上的有效值不同。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_MLX_CONTEXT_SIZE = 4096
# 在线后端的字符预算与条数硬上限（避免配置写太大把成本/延迟拉爆）。
ONLINE_CONTEXT_CHAR_BUDGET = 6000
ONLINE_MAX_MESSAGES = 80


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mlx_context_size(config: Any) -> int:
    return max(
        512,
        _int_value(
            getattr(config, "ai_companion_mlx_lora_max_seq_length", DEFAULT_MLX_CONTEXT_SIZE),
            DEFAULT_MLX_CONTEXT_SIZE,
        ),
    )


def effective_max_tokens(config: Any) -> int:
    requested = max(1, _int_value(getattr(config, "ai_companion_max_tokens", 256), 256))
    ctx_size = mlx_context_size(config)
    return max(16, min(requested, max(64, ctx_size // 4)))


def effective_context_message_limit(config: Any, *, online: bool = False) -> int:
    """有效历史条数：在线后端放开序列长度限制，本地按 ctx//256 压。"""
    requested = max(0, _int_value(getattr(config, "ai_companion_context_messages", 16), 16))
    if online:
        return min(requested, ONLINE_MAX_MESSAGES)
    ctx_size = mlx_context_size(config)
    return min(requested, max(4, ctx_size // 256))


def trim_history_lines_for_context(lines: Iterable[str], config: Any, *, online: bool = False) -> list[str]:
    items = [str(item) for item in lines if str(item).strip()]
    if online:
        budget = ONLINE_CONTEXT_CHAR_BUDGET
    else:
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
