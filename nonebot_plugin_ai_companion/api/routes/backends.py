"""推理后端实例与角色绑定路由（文档三.3/三.4）。

- /api/backends：InferenceBackendConfig CRUD + 健康检查 + 注册表可用类型
- /api/provider-bindings：chat / vision / data_cleaning 三角色绑定（含降级实例）
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...providers.instances import (
    PROVIDER_ROLES,
    InferenceBackendConfig,
    ProviderInstanceManager,
    RoleProviderBinding,
    RoleProviderBindingRepo,
    new_backend_config,
)
from ...providers.registry import ProviderNotFoundError, provider_specs

logger = logging.getLogger(__name__)


class BackendPayload(BaseModel):
    backend_type: str
    display_name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    priority: int = 100


class BindingPayload(BaseModel):
    provider_id: str
    model: str = ""
    fallback_provider_id: str | None = None


def build_router(
    *,
    auth,
    instances: ProviderInstanceManager | None,
    bindings: RoleProviderBindingRepo | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    def require() -> tuple[ProviderInstanceManager, RoleProviderBindingRepo]:
        if instances is None or bindings is None:
            raise HTTPException(status_code=503, detail="provider instance manager unavailable")
        return instances, bindings

    @router.get("/backends")
    async def list_backends() -> dict[str, Any]:
        manager, _binding_repo = require()
        configs = await manager.repo.list_configs()
        items = []
        for config in configs:
            spec_snapshot: dict[str, Any] = {}
            try:
                from ...providers.registry import get_provider_spec

                spec_snapshot = get_provider_spec(config.backend_type).to_dict()
            except ProviderNotFoundError:
                spec_snapshot = {"name": config.backend_type, "provider_type": "unknown"}
            items.append({**config.model_dump(), "spec": spec_snapshot})
        return {
            "items": items,
            "total": len(items),
            "available_types": [spec.to_dict() for spec in provider_specs()],
        }

    @router.post("/backends", response_model=None)
    async def create_backend(payload: BackendPayload) -> dict[str, Any]:
        manager, _binding_repo = require()
        try:
            from ...providers.registry import get_provider_spec

            spec = get_provider_spec(payload.backend_type)
        except ProviderNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        config = new_backend_config(
            backend_type=payload.backend_type,
            display_name=payload.display_name.strip() or spec.display_name,
            settings=payload.settings,
            enabled=payload.enabled,
            priority=payload.priority,
        )
        try:
            await manager.repo.upsert_config(config)
        except Exception as exc:  # noqa: BLE001 — 构建失败说明 settings 非法
            raise HTTPException(status_code=400, detail=f"backend build failed: {exc}") from exc
        await manager.invalidate(config.id)
        return {"item": config.model_dump()}

    @router.patch("/backends/{backend_id}", response_model=None)
    async def update_backend(backend_id: str, payload: BackendPayload) -> dict[str, Any]:
        manager, _binding_repo = require()
        existing = await manager.repo.get_config(backend_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="backend config not found")
        updated = InferenceBackendConfig(
            id=existing.id,
            backend_type=payload.backend_type,
            display_name=payload.display_name.strip() or existing.display_name,
            settings=dict(payload.settings),
            enabled=payload.enabled,
            priority=payload.priority,
            created_at=existing.created_at,
            updated_at=time.time(),
        )
        await manager.repo.upsert_config(updated)
        await manager.invalidate(backend_id)
        return {"item": updated.model_dump()}

    @router.delete("/backends/{backend_id}")
    async def delete_backend(backend_id: str) -> dict[str, int]:
        manager, _binding_repo = require()
        if not await manager.repo.delete_config(backend_id):
            raise HTTPException(status_code=404, detail="backend config not found")
        await manager.invalidate(backend_id)
        return {"deleted": 1}

    @router.post("/backends/{backend_id}/health-check")
    async def health_check_backend(backend_id: str) -> dict[str, Any]:
        manager, _binding_repo = require()
        instance = await manager.get_instance(backend_id)
        if instance is None:
            raise HTTPException(status_code=404, detail="backend config not found")
        healthy = await instance.provider.health_check()
        return {
            "id": backend_id,
            "healthy": bool(healthy),
            "status": instance.provider.status_snapshot(),
        }

    @router.get("/provider-bindings")
    async def list_bindings() -> dict[str, Any]:
        _manager, binding_repo = require()
        items = await binding_repo.list_bindings()
        return {
            "items": [item.model_dump() for item in items],
            "roles": list(PROVIDER_ROLES),
        }

    @router.put("/provider-bindings/{role}", response_model=None)
    async def set_binding(role: str, payload: BindingPayload) -> dict[str, Any]:
        manager, binding_repo = require()
        if role not in PROVIDER_ROLES:
            raise HTTPException(status_code=400, detail=f"unknown role: {role}")
        if payload.fallback_provider_id and await manager.repo.get_config(payload.fallback_provider_id) is None:
            raise HTTPException(status_code=400, detail="fallback provider not found")
        binding = RoleProviderBinding(
            role=role,
            provider_id=payload.provider_id,
            model=payload.model,
            fallback_provider_id=payload.fallback_provider_id,
            updated_at=time.time(),
        )
        await binding_repo.set_binding(binding)
        return {"binding": binding.model_dump()}

    return router
