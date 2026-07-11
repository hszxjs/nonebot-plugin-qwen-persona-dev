from __future__ import annotations

from typing import Any

from .config import QwenPersonaConfig
from .mlx_runtime import MlxVLMRuntime


class QwenClient:
    def __init__(
        self,
        config: QwenPersonaConfig,
        *,
        runtime: MlxVLMRuntime | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or MlxVLMRuntime(config)

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        self.config = config
        self.runtime.reconfigure(config)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
    ) -> str:
        return await self.runtime.chat(messages, media_parts=media_parts)

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.runtime.metrics_snapshot()
