"""推理后端实例管理层（文档三.3/三.4）。

- `InferenceBackendConfig`：数据库持久化的后端实例配置（同一后端类型可建多个，
  供故障降级链按 priority 排序使用）
- `RoleProviderBinding`：按用途角色（chat / vision / data_cleaning）绑定
  Provider 实例 + 模型 + 降级实例
- `ProviderInstanceManager`：DB 行 → 后端实例（缓存，配置变更失效）
- `ProviderResolver`：角色 → 实例解析；无绑定时回退到旧配置路径（QwenClient /
  MLX 运行时 / 辅助 LLM），保证未配置时行为与二期完全一致
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..config import AiCompanionConfig
from ..memory.db import PersonaDatabase
from .client import QwenClient
from .glm_backend import GlmBackend
from .llama_cpp import LlamaCppBackend
from .mlx_runtime import MlxVLMRuntime
from .ollama import OllamaBackend
from .registry import ProviderNotFoundError

logger = logging.getLogger(__name__)

PROVIDER_ROLES: tuple[str, ...] = ("chat", "vision", "data_cleaning")


class InferenceBackendConfig(BaseModel):
    id: str
    backend_type: str
    display_name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    priority: int = 100
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RoleProviderBinding(BaseModel):
    role: str
    provider_id: str
    model: str = ""
    fallback_provider_id: str | None = None
    updated_at: float = Field(default_factory=time.time)


class InferenceBackendRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def list_configs(self, *, enabled_only: bool = False) -> list[InferenceBackendConfig]:
        def _sync() -> list[InferenceBackendConfig]:
            self.db.ensure_schema()
            where = "WHERE enabled = 1" if enabled_only else ""
            with self.db.connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM inference_backend_configs {where} ORDER BY priority ASC, created_at ASC"
                ).fetchall()
            return [self._row_to_config(row) for row in rows]

        return await asyncio.to_thread(_sync)

    async def get_config(self, config_id: str) -> InferenceBackendConfig | None:
        def _sync() -> InferenceBackendConfig | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute("SELECT * FROM inference_backend_configs WHERE id = ?", (config_id,)).fetchone()
            return self._row_to_config(row) if row else None

        return await asyncio.to_thread(_sync)

    async def upsert_config(self, config: InferenceBackendConfig) -> None:
        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO inference_backend_configs (
                        id, backend_type, display_name, config_json, enabled, priority,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        backend_type=excluded.backend_type,
                        display_name=excluded.display_name,
                        config_json=excluded.config_json,
                        enabled=excluded.enabled,
                        priority=excluded.priority,
                        updated_at=excluded.updated_at
                    """,
                    (
                        config.id,
                        config.backend_type,
                        config.display_name,
                        json.dumps(config.settings, ensure_ascii=False, default=str),
                        1 if config.enabled else 0,
                        int(config.priority),
                        config.created_at,
                        config.updated_at,
                    ),
                )

        await asyncio.to_thread(_sync)

    async def delete_config(self, config_id: str) -> bool:
        def _sync() -> bool:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute("DELETE FROM inference_backend_configs WHERE id = ?", (config_id,))
                # 引用该实例的角色绑定一并清空，避免悬空
                conn.execute(
                    """
                    UPDATE role_provider_bindings
                    SET provider_id = '', fallback_provider_id = NULL
                    WHERE provider_id = ? OR fallback_provider_id = ?
                    """,
                    (config_id, config_id),
                )
            return bool(cursor.rowcount)

        return await asyncio.to_thread(_sync)

    @staticmethod
    def _row_to_config(row: Any) -> InferenceBackendConfig:
        data = dict(row)
        try:
            settings = json.loads(data.get("config_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            settings = {}
        return InferenceBackendConfig(
            id=data["id"],
            backend_type=data["backend_type"],
            display_name=data["display_name"],
            settings=settings if isinstance(settings, dict) else {},
            enabled=bool(data.get("enabled")),
            priority=int(data.get("priority") or 100),
            created_at=float(data.get("created_at") or 0),
            updated_at=float(data.get("updated_at") or 0),
        )


class RoleProviderBindingRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def list_bindings(self) -> list[RoleProviderBinding]:
        def _sync() -> list[RoleProviderBinding]:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                rows = conn.execute("SELECT * FROM role_provider_bindings").fetchall()
            return [
                RoleProviderBinding(
                    role=row["role"],
                    provider_id=row["provider_id"],
                    model=row["model"] or "",
                    fallback_provider_id=row["fallback_provider_id"],
                    updated_at=float(row["updated_at"] or 0),
                )
                for row in rows
            ]

        return await asyncio.to_thread(_sync)

    async def get_binding(self, role: str) -> RoleProviderBinding | None:
        for binding in await self.list_bindings():
            if binding.role == role:
                return binding
        return None

    async def set_binding(self, binding: RoleProviderBinding) -> None:
        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO role_provider_bindings (role, provider_id, model, fallback_provider_id, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(role) DO UPDATE SET
                        provider_id=excluded.provider_id,
                        model=excluded.model,
                        fallback_provider_id=excluded.fallback_provider_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        binding.role,
                        binding.provider_id,
                        binding.model,
                        binding.fallback_provider_id,
                        binding.updated_at,
                    ),
                )

        await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# 实例构建与缓存
# ---------------------------------------------------------------------------


def build_backend_from_config(
    config: InferenceBackendConfig,
) -> Any:
    """按 backend_type + settings 构建后端实例。未知类型抛 ProviderNotFoundError。"""
    settings = dict(config.settings or {})
    backend_type = config.backend_type
    if backend_type == "ollama":
        return OllamaBackend(
            base_url=settings.get("base_url"),
            model=settings.get("model"),
            api_key=settings.get("api_key"),
            extra_body=settings.get("extra_body"),
        )
    if backend_type == "llama_cpp":
        return LlamaCppBackend(
            base_url=settings.get("base_url"),
            model=settings.get("model"),
            api_key=settings.get("api_key"),
            extra_body=settings.get("extra_body"),
        )
    if backend_type == "glm":
        return GlmBackend(
            AiCompanionConfig(
                ai_companion_online_api_url=settings.get("base_url")
                or "https://api.z.ai/api/coding/paas/v4/chat/completions",
                ai_companion_online_api_key=settings.get("api_key") or "",
                ai_companion_online_model=settings.get("model") or "glm-5.2",
            )
        )
    raise ProviderNotFoundError(f"no instance builder for backend type {backend_type!r}")


@dataclass
class BackendInstance:
    config: InferenceBackendConfig
    provider: Any

    @property
    def id(self) -> str:
        return self.config.id

    def chat_model(self, model_override: str = "") -> str:
        """实例实际使用的模型：绑定覆盖 > 实例设置 > 后端默认。"""
        if model_override:
            return model_override
        return str(self.config.settings.get("model") or "")


class ProviderInstanceManager:
    def __init__(self, repo: InferenceBackendRepo) -> None:
        self.repo = repo
        self._cache: dict[str, BackendInstance] = {}

    async def list_instances(self, *, enabled_only: bool = True) -> list[BackendInstance]:
        configs = await self.repo.list_configs(enabled_only=enabled_only)
        return [self._instance_for(config) for config in configs]

    async def get_instance(self, config_id: str) -> BackendInstance | None:
        config = await self.repo.get_config(config_id)
        if config is None:
            return None
        return self._instance_for(config)

    async def invalidate(self, config_id: str | None = None) -> None:
        if config_id is None:
            self._cache.clear()
        else:
            self._cache.pop(config_id, None)

    def _instance_for(self, config: InferenceBackendConfig) -> BackendInstance:
        cached = self._cache.get(config.id)
        if cached is not None and cached.config.updated_at == config.updated_at:
            return cached
        instance = BackendInstance(config=config, provider=build_backend_from_config(config))
        self._cache[config.id] = instance
        return instance


class ProviderResolver:
    """角色 → Provider 解析（文档三.4 细粒度角色化绑定）。

    无绑定时回退旧路径：chat=QwenClient、vision=MLX 运行时、data_cleaning=辅助 LLM。
    """

    def __init__(
        self,
        *,
        config: AiCompanionConfig,
        client: QwenClient,
        mlx_runtime: MlxVLMRuntime,
        instances: ProviderInstanceManager,
        bindings: RoleProviderBindingRepo,
    ) -> None:
        self.config = config
        self.client = client
        self.mlx_runtime = mlx_runtime
        self.instances = instances
        self.bindings = bindings

    async def resolve(self, role: str) -> Any:
        """返回该角色的 provider（BackendInstance.provider 或旧路径对象）。"""
        binding = await self.bindings.get_binding(role)
        if binding and binding.provider_id:
            instance = await self.instances.get_instance(binding.provider_id)
            if instance is not None:
                return instance.provider
            logger.warning("role %r bound to missing/disabled provider %r; fallback", role, binding.provider_id)
        if role == "chat":
            return self.client
        if role == "vision":
            return self.mlx_runtime
        return None  # data_cleaning 默认走辅助 LLM（aux_llm），调用方自行处理

    async def resolve_with_meta(self, role: str) -> tuple[Any, str, str]:
        """返回 (provider, model_override, provider_id)；无绑定时 provider_id 为空。"""
        binding = await self.bindings.get_binding(role)
        if binding and binding.provider_id:
            instance = await self.instances.get_instance(binding.provider_id)
            if instance is not None:
                return instance.provider, binding.model or "", instance.id
        if role == "chat":
            return self.client, "", ""
        if role == "vision":
            return self.mlx_runtime, "", ""
        return None, "", ""

    async def chat(self, messages: list[dict[str, Any]], *, media_parts: list[dict[str, Any]] | None = None) -> str:
        """chat 角色的统一调用：绑定实例（可带模型覆盖）→ 失败按降级链重试 → 旧路径。"""
        binding = await self.bindings.get_binding("chat")
        tried: list[str] = []
        if binding and binding.provider_id:
            candidates: list[tuple[str, str]] = [(binding.provider_id, binding.model or "")]
            if binding.fallback_provider_id:
                candidates.append((binding.fallback_provider_id, ""))
            for provider_id, model_override in candidates:
                instance = await self.instances.get_instance(provider_id)
                if instance is None:
                    continue
                tried.append(provider_id)
                provider = instance.provider
                instance_model = str(getattr(instance.config, "settings", {}).get("model") or "")
                payload_model = model_override or instance_model
                try:
                    return await _chat_with_model(provider, messages, media_parts=media_parts, model=payload_model)
                except Exception as exc:  # noqa: BLE001 — 降级链：实例失败换下一个
                    logger.warning("chat provider %s failed (%r), trying fallback", provider_id, exc)
            # 显式绑定但实例全失败：仍走旧路径兜底（比直接报错更稳）
            logger.warning("all bound chat providers failed; falling back to legacy client")
        return await self.client.chat(messages, media_parts=media_parts)


async def _chat_with_model(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    media_parts: list[dict[str, Any]] | None,
    model: str,
) -> str:
    """带模型覆盖调用实例：临时切换 model 字段（实例为独享对象，无并发冲突）。"""
    if not model or getattr(provider, "model", None) == model:
        return await provider.chat(messages, media_parts=media_parts)
    previous = getattr(provider, "model", None)
    try:
        if hasattr(provider, "update_settings"):
            provider.update_settings(model=model)
        return await provider.chat(messages, media_parts=media_parts)
    finally:
        if previous is not None and hasattr(provider, "update_settings"):
            provider.update_settings(model=previous)


def new_backend_config(
    *,
    backend_type: str,
    display_name: str,
    settings: dict[str, Any] | None = None,
    enabled: bool = True,
    priority: int = 100,
) -> InferenceBackendConfig:
    return InferenceBackendConfig(
        id=uuid4().hex,
        backend_type=backend_type,
        display_name=display_name,
        settings=dict(settings or {}),
        enabled=enabled,
        priority=priority,
    )
