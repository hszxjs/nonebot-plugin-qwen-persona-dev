from __future__ import annotations

import logging
from typing import Any

from ..config import AiCompanionConfig, effective_chat_model
from .base import BaseProvider, ProviderCapabilities
from .mlx_runtime import MlxVLMRuntime
from .registry import create_provider, get_provider_spec, normalize_provider_name

logger = logging.getLogger(__name__)


class QwenClient:
    """推理后端分发器：按 config 经注册表选择本地 MLX 或在线 LLM。

    chat / metrics_snapshot / status_snapshot 全部转发到当前激活的 backend，
    上层（chat 服务、admin_server、observability）无需感知后端差异。
    """

    def __init__(
        self,
        config: AiCompanionConfig,
        *,
        runtime: MlxVLMRuntime | None = None,
        online_backend: Any | None = None,
    ) -> None:
        self.config = config
        # 经注册表构建，保证后端与注册信息一致；允许测试注入替身。
        self.mlx_runtime = runtime or create_provider("mlx", config)
        self.llm_backend = online_backend or create_provider("llm", config)

    def reconfigure(self, config: AiCompanionConfig) -> None:
        prev_backend = normalize_provider_name(self.config.ai_companion_inference_backend)
        self.config = config
        self.mlx_runtime.reconfigure(config)
        self.llm_backend.reconfigure(config)
        new_backend = normalize_provider_name(config.ai_companion_inference_backend)
        if prev_backend != new_backend:
            logger.info("inference backend switched: %s -> %s", prev_backend, new_backend)

    @property
    def _active(self) -> BaseProvider:
        """当前激活的 backend（mlx_runtime 或在线 LLM 后端）。"""
        if normalize_provider_name(self.config.ai_companion_inference_backend) == "llm":
            return self.llm_backend  # type: ignore[return-value]
        return self.mlx_runtime  # type: ignore[return-value]

    def active_backend(self) -> str:
        """返回当前后端标识：'mlx' 或 'llm'（历史值 'glm' 归一为 'llm'）。"""
        return normalize_provider_name(self.config.ai_companion_inference_backend)

    def active_backend_spec(self) -> dict[str, str]:
        """当前后端的注册信息（类型/展示名/文档链接，供前端展示）。"""
        return get_provider_spec(self.active_backend()).to_dict()

    def active_model_name(self) -> str:
        """当前后端实际使用的模型名（供 inference_events 记录）。"""
        return effective_chat_model(self.config)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: Any | str | None = None,
    ) -> str:
        backend = self._active
        # purpose 为 None 时让各 backend 用自己的默认值（MLX 默认 PERSONA_REPLY）。
        if purpose is not None:
            try:
                return await backend.chat(messages, media_parts=media_parts, purpose=purpose)
            except TypeError:
                pass
        return await backend.chat(messages, media_parts=media_parts)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """带 function calling 的对话（仅在线 LLM 后端支持）。

        返回 (文本, tool_calls)；当前后端不支持时抛 NotImplementedError，
        调用方应回退到普通 chat()。
        """
        if self.active_backend() != "llm":
            raise NotImplementedError("tool calls require the online llm backend")
        backend = self.llm_backend
        assert hasattr(backend, "chat_with_tools")
        return await backend.chat_with_tools(messages, tools=tools)

    def metrics_snapshot(self) -> dict[str, Any]:
        return self._active.metrics_snapshot()

    def status_snapshot(self) -> dict[str, Any]:
        return self._active.status_snapshot()

    def capabilities(self) -> ProviderCapabilities:
        return self._active.capabilities()

    def default_models(self) -> list[dict[str, str]]:
        return self._active.default_models()

    async def health_check(self) -> bool:
        return await self._active.health_check()
