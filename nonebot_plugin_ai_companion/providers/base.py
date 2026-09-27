"""Provider 抽象：本地推理与在线 API 统一实现同一接口。

上层业务（chat 服务、表情选择、后台识图、admin API）只依赖
`BaseProvider` 与 `ProviderCapabilities`，不感知 local/remote 差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..config import AiCompanionConfig
    from .generation_profiles import GenerationPurpose


@dataclass(frozen=True)
class ProviderCapabilities:
    """后端能力声明：供前端展示与健康徽章、决策层能力判断。"""

    vision: bool = False
    function_calling: bool = False
    streaming: bool = False
    lora_hotload: bool = False


class BaseProvider(Protocol):
    """所有推理后端（本地 MLX / 在线 LLM / Ollama / llama.cpp）的统一接口。"""

    @property
    def provider_name(self) -> str: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: GenerationPurpose | str = ...,
    ) -> str: ...

    def metrics_snapshot(self) -> dict[str, Any]: ...

    def status_snapshot(self) -> dict[str, Any]: ...

    def reconfigure(self, config: AiCompanionConfig) -> None: ...

    def capabilities(self) -> ProviderCapabilities: ...

    def default_models(self) -> list[dict[str, str]]: ...

    async def health_check(self) -> bool: ...
