"""llama.cpp（llama-server）本地推理后端（文档三.2 第一优先级）。

Metal/CUDA/Vulkan/CPU 官方支持，libmtmd 组件支持 LLaVA / MiniCPM-V 等
原生视觉模型。模型由 llama-server 进程加载，这里只对接其 OpenAI 兼容端点
（默认 http://127.0.0.1:8080/v1）。
"""

from __future__ import annotations

from typing import Any

from ..config import AiCompanionConfig
from .base import ProviderCapabilities
from .openai_compat import OpenAICompatBackend
from .registry import register

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "local"

LLAMA_CPP_PRESETS: list[dict[str, str]] = [
    {
        "value": "local",
        "label": "当前 llama-server 已加载模型",
        "description": "model 字段任意，实际用服务端加载的 GGUF",
    },
    {
        "value": "Qwen/Qwen3.6-0.6B-GGUF:Q4_K_M",
        "label": "Qwen3.6 0.6B Q4_K_M",
        "description": "最小档位，Qwen3.6 起自带视觉",
    },
    {
        "value": "Qwen/Qwen3.6-9B-GGUF:Q4_K_M",
        "label": "Qwen3.6 9B Q4_K_M - 推荐",
        "description": "通用量级 GGUF，自带视觉",
    },
    {
        "value": "google/gemma-3-4b-it-GGUF:Q4_K_M",
        "label": "Gemma 3 4B（视觉）",
        "description": "llama.cpp 视觉支持（libmtmd）",
    },
    {"value": "openbmb/MiniCPM-V-2.6-GGUF:Q4_K_M", "label": "MiniCPM-V 2.6（视觉）", "description": "端侧视觉备选"},
]


@register(
    "llama_cpp",
    provider_type="local",
    display_name="llama.cpp (llama-server)",
    docs_url="https://github.com/ggml-org/llama.cpp",
)
class LlamaCppBackend(OpenAICompatBackend):
    def __init__(
        self,
        config: AiCompanionConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        resolved_base = base_url or getattr(config, "ai_companion_llama_cpp_api_url", "") or DEFAULT_BASE_URL
        if not resolved_base.rstrip("/").endswith("/v1"):
            resolved_base = resolved_base.rstrip("/") + "/v1"
        super().__init__(
            provider_name="llama_cpp",
            display_name="llama.cpp (llama-server)",
            base_url=resolved_base,
            model=model or DEFAULT_MODEL,
            api_key=api_key if api_key is not None else "none",
            capabilities=ProviderCapabilities(
                vision=True,
                function_calling=True,
                streaming=True,
                lora_hotload=True,
            ),
            default_models=LLAMA_CPP_PRESETS,
            docs_url="https://github.com/ggml-org/llama.cpp",
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
            url = getattr(config, "ai_companion_llama_cpp_api_url", "") or DEFAULT_BASE_URL
            if not url.rstrip("/").endswith("/v1"):
                url = url.rstrip("/") + "/v1"
            self.update_settings(base_url=url)
