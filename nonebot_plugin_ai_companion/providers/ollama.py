"""Ollama 本地推理后端（文档三.2 第一优先级）：跨平台 + Metal + 原生视觉。

走 Ollama 的 OpenAI 兼容端点（默认 http://127.0.0.1:11434/v1），
用户可在后台创建多个实例（InferenceBackendConfig）自定 base_url / model。
"""

from __future__ import annotations

from typing import Any

from ..config import AiCompanionConfig
from .base import ProviderCapabilities
from .openai_compat import OpenAICompatBackend
from .registry import register

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.6:9b"

OLLAMA_PRESETS: list[dict[str, str]] = [
    {"value": "qwen3.6:0.6b", "label": "Qwen3.6 0.6B - 最小", "description": "最小可用档位，Qwen3.6 起全系自带视觉"},
    {"value": "qwen3.6:9b", "label": "Qwen3.6 9B - 推荐", "description": "通用量级主力，自带视觉，无需单独的 VL 版本"},
    {"value": "llama3.1:8b", "label": "Llama 3.1 8B", "description": "Meta 开源通用模型（纯文本）"},
    {"value": "gemma3:4b", "label": "Gemma 3 4B（视觉）", "description": "轻量视觉多模态备选"},
    {"value": "gemma3:12b", "label": "Gemma 3 12B（视觉）", "description": "中量级视觉多模态备选"},
    {"value": "deepseek-r1:8b", "label": "DeepSeek-R1 蒸馏 8B", "description": "推理特化蒸馏版"},
    {"value": "minicpm-v:8b", "label": "MiniCPM-V 8B（视觉）", "description": "端侧视觉模型备选"},
]


@register(
    "ollama",
    provider_type="local",
    display_name="Ollama 本地推理",
    docs_url="https://ollama.com/",
)
class OllamaBackend(OpenAICompatBackend):
    def __init__(
        self,
        config: AiCompanionConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        resolved_base = base_url or getattr(config, "ai_companion_ollama_api_url", "") or DEFAULT_BASE_URL
        if not resolved_base.rstrip("/").endswith("/v1"):
            resolved_base = resolved_base.rstrip("/") + "/v1"
        super().__init__(
            provider_name="ollama",
            display_name="Ollama 本地推理",
            base_url=resolved_base,
            model=model or DEFAULT_MODEL,
            api_key=api_key if api_key is not None else "ollama",
            capabilities=ProviderCapabilities(
                vision=True,
                function_calling=True,
                streaming=True,
                lora_hotload=False,
            ),
            default_models=OLLAMA_PRESETS,
            docs_url="https://ollama.com/",
            timeout=180.0,
            extra_body=extra_body,
        )

    def reconfigure(self, config: AiCompanionConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            self.update_settings(
                base_url=config.get("base_url"),
                model=config.get("model"),
                api_key=config.get("api_key"),
                extra_body=config.get("extra_body"),
            )
        else:
            url = getattr(config, "ai_companion_ollama_api_url", "") or DEFAULT_BASE_URL
            if not url.rstrip("/").endswith("/v1"):
                url = url.rstrip("/") + "/v1"
            self.update_settings(base_url=url)
