"""推理后端注册表。

后端类通过 ``@register(name=..., provider_type=...)`` 自注册；新增后端
（如第三期的 Ollama / llama.cpp Adapter）只需新增模块并在其中使用装饰器，
不改动本文件。注册表与具体实现解耦，禁止反向 import。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseProvider

PROVIDER_TYPES = ("local", "remote")


class ProviderRegistrationError(ValueError):
    """注册信息非法（重名冲突、类型不合法等）。"""


class ProviderNotFoundError(KeyError):
    """按名查找后端时未注册。"""


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    provider_type: str
    display_name: str
    docs_url: str = ""
    factory: Callable[..., Any] | None = field(default=None)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "provider_type": self.provider_type,
            "display_name": self.display_name,
            "docs_url": self.docs_url,
        }


_REGISTRY: dict[str, ProviderSpec] = {}

# 历史名兼容：在线 LLM 后端早期叫 "glm"，旧配置/旧实例行里可能仍是这个值。
PROVIDER_ALIASES: dict[str, str] = {"glm": "llm"}


def normalize_provider_name(name: str) -> str:
    """把历史别名归一到当前注册名（"glm" → "llm"）。"""
    key = str(name or "").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def register(
    name: str,
    *,
    provider_type: str,
    display_name: str | None = None,
    docs_url: str = "",
) -> Callable[[type], type]:
    if provider_type not in PROVIDER_TYPES:
        raise ProviderRegistrationError(
            f"provider {name!r}: provider_type must be one of {PROVIDER_TYPES}, got {provider_type!r}"
        )

    def decorator(cls: type) -> type:
        spec = ProviderSpec(
            name=name,
            provider_type=provider_type,
            display_name=display_name or name,
            docs_url=docs_url,
            factory=cls,
        )
        existing = _REGISTRY.get(name)
        if existing is not None and existing.factory is not cls:
            raise ProviderRegistrationError(f"provider {name!r} already registered by {existing.factory!r}")
        _REGISTRY[name] = spec
        return cls

    return decorator


def get_provider_spec(name: str) -> ProviderSpec:
    try:
        return _REGISTRY[normalize_provider_name(name)]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ProviderNotFoundError(f"provider {name!r} not registered (available: {available})") from None


def provider_specs(*, provider_type: str | None = None) -> list[ProviderSpec]:
    specs = list(_REGISTRY.values())
    if provider_type is not None:
        specs = [spec for spec in specs if spec.provider_type == provider_type]
    return sorted(specs, key=lambda spec: spec.name)


def create_provider(name: str, config: Any) -> BaseProvider:
    spec = get_provider_spec(name)
    if spec.factory is None:
        raise ProviderNotFoundError(f"provider {name!r} has no factory")
    return spec.factory(config)


def reset_registry_for_tests() -> None:
    """仅供测试隔离使用：清空注册表后需重新导入后端模块重建。"""
    _REGISTRY.clear()
