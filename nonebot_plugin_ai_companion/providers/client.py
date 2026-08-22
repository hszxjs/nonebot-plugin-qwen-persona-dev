from __future__ import annotations

import logging
from typing import Any

from ..config import QwenPersonaConfig, effective_chat_model
from .glm_backend import GlmBackend
from .mlx_runtime import MlxVLMRuntime

logger = logging.getLogger(__name__)


class QwenClient:
    """推理后端分发器：按 config 选择本地 MLX 或 GLM 在线。

    chat / metrics_snapshot / status_snapshot 全部转发到当前激活的 backend，
    上层（reply_planner、admin_server、inference_events）无需感知后端差异。
    """

    def __init__(
        self,
        config: QwenPersonaConfig,
        *,
        runtime: MlxVLMRuntime | None = None,
        glm_backend: GlmBackend | None = None,
    ) -> None:
        self.config = config
        self.mlx_runtime = runtime or MlxVLMRuntime(config)
        self.glm_backend = glm_backend or GlmBackend(config)

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        prev_backend = self.config.qwen_persona_inference_backend
        self.config = config
        self.mlx_runtime.reconfigure(config)
        self.glm_backend.reconfigure(config)
        new_backend = config.qwen_persona_inference_backend
        if prev_backend != new_backend:
            logger.info(
                "inference backend switched: %s -> %s", prev_backend, new_backend
            )

    @property
    def _active(self) -> Any:
        """当前激活的 backend（mlx_runtime 或 glm_backend）。"""
        if self.config.qwen_persona_inference_backend == "glm":
            return self.glm_backend
        return self.mlx_runtime

    def active_backend(self) -> str:
        """返回当前后端标识：'mlx' 或 'glm'。"""
        return self.config.qwen_persona_inference_backend

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

    def metrics_snapshot(self) -> dict[str, Any]:
        return self._active.metrics_snapshot()

    def status_snapshot(self) -> dict[str, Any]:
        return self._active.status_snapshot()
