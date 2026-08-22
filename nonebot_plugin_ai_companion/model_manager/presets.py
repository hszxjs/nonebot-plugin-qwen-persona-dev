"""模型管理域：各推理后端的精选默认模型列表（前端快速选择候选）。"""

from __future__ import annotations

MLX_MODEL_PRESETS: list[dict[str, str]] = [
    {
        "value": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "label": "Qwen 3.5 VL 9B 4bit - recommended",
        "description": "Default Apple Silicon VLM target for 16GB unified memory.",
    },
    {
        "value": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "label": "Qwen2.5 VL 7B 4bit - lighter fallback",
        "description": "Smaller VL fallback when the 9B model is too tight.",
    },
]
