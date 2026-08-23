"""模型管理域：各推理后端的精选默认模型列表（前端快速选择候选）。

Qwen3.6 起全系自带视觉能力（不再需要独立的 VL 版本），本地默认档位
以 0.6B（最小可用）与 9B（推荐）为主。
"""

from __future__ import annotations

MLX_MODEL_PRESETS: list[dict[str, str]] = [
    {
        "value": "mlx-community/Qwen3.6-9B-4bit",
        "label": "Qwen 3.6 9B 4bit - recommended",
        "description": "默认 Apple Silicon 目标：全系自带视觉，16GB 统一内存可用。",
    },
    {
        "value": "mlx-community/Qwen3.6-0.6B-4bit",
        "label": "Qwen 3.6 0.6B 4bit - lightest",
        "description": "最小可用档位（同样自带视觉），低配/冒烟测试用。",
    },
]
