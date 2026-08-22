"""角色卡管理路由：CRUD / 克隆 / 启停 / 会话绑定 / member_mimic 生成与刷新 / 工具清单。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...providers.aux_llm import AuxLlmError
from ...role_card.model import RoleCardCreate, RoleType
from ...role_card.service import RoleCardService
from ...tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CardPayload(BaseModel):
    name: str
    role_type: RoleType = RoleType.BOT_PERSONA
    system_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    avatar: str = ""
    enabled: bool = True
    target_user_id: str | None = None


class BindPayload(BaseModel):
    group_id: str | None = None
    user_id: str | None = None


class MemberMimicPayload(BaseModel):
    user_id: str
    group_id: str | None = None


class ClonePayload(BaseModel):
    name: str


def _card_payload(card: Any, bindings: dict[str, str]) -> dict[str, Any]:
    data = card.model_dump()
    data["bindings"] = [session_key for session_key, role_card_id in bindings.items() if role_card_id == card.id]
    return data


def build_router(
    *,
    auth,
    role_card_service: RoleCardService | None,
    tool_registry: ToolRegistry | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    def require_service() -> RoleCardService:
        if role_card_service is None:
            raise HTTPException(status_code=503, detail="role card service unavailable")
        return role_card_service

    @router.get("/role-cards")
    async def list_cards() -> dict[str, Any]:
        service = require_service()
        cards = await service.list_cards()
        bindings = {b.session_key: b.role_card_id for b in await service.list_bindings()}
        items = [_card_payload(card, bindings) for card in cards]
        return {
            "items": items,
            "total": len(items),
            "role_types": [str(role) for role in RoleType],
        }

    @router.post("/role-cards", response_model=None)
    async def create_card(payload: CardPayload) -> dict[str, Any]:
        service = require_service()
        card = await service.create_card(
            RoleCardCreate(
                name=payload.name.strip() or "未命名",
                role_type=payload.role_type,
                system_prompt=payload.system_prompt,
                style_tags=payload.style_tags,
                tone_examples=payload.tone_examples,
                allowed_tools=payload.allowed_tools,
                avatar=payload.avatar,
                enabled=payload.enabled,
                target_user_id=payload.target_user_id,
            )
        )
        return {"item": card.model_dump()}

    @router.patch("/role-cards/{card_id}", response_model=None)
    async def update_card(card_id: str, payload: CardPayload) -> dict[str, Any]:
        service = require_service()
        updated = await service.update_card(card_id, RoleCardCreate(**payload.model_dump()))
        if updated is None:
            raise HTTPException(status_code=404, detail="role card not found")
        return {"item": updated.model_dump()}

    @router.delete("/role-cards/bindings")
    async def unbind(payload: BindPayload) -> dict[str, int]:
        service = require_service()
        unbound = await service.unbind_session(group_id=payload.group_id, user_id=payload.user_id)
        return {"unbound": 1 if unbound else 0}

    @router.delete("/role-cards/{card_id}")
    async def delete_card(card_id: str) -> dict[str, int]:
        service = require_service()
        if not await service.delete_card(card_id):
            raise HTTPException(status_code=404, detail="role card not found")
        return {"deleted": 1}

    @router.post("/role-cards/{card_id}/clone", response_model=None)
    async def clone_card(card_id: str, payload: ClonePayload) -> dict[str, Any]:
        service = require_service()
        clone = await service.clone_card(card_id, name=payload.name.strip() or "克隆卡")
        if clone is None:
            raise HTTPException(status_code=404, detail="role card not found")
        return {"item": clone.model_dump()}

    @router.post("/role-cards/{card_id}/refresh-style", response_model=None)
    async def refresh_style(card_id: str) -> dict[str, Any]:
        service = require_service()
        try:
            refreshed = await service.refresh_member_mimic_style(card_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AuxLlmError as exc:
            raise HTTPException(status_code=502, detail=f"style analysis failed: {exc}") from exc
        return {"item": refreshed.model_dump()}

    @router.post("/role-cards/member-mimic", response_model=None)
    async def generate_member_mimic(payload: MemberMimicPayload) -> dict[str, Any]:
        service = require_service()
        try:
            card = await service.generate_member_mimic(
                user_id=str(payload.user_id).strip(),
                group_id=payload.group_id,
            )
        except AuxLlmError as exc:
            raise HTTPException(status_code=502, detail=f"style analysis failed: {exc}") from exc
        return {"item": card.model_dump()}

    @router.post("/role-cards/{card_id}/bind", response_model=None)
    async def bind_card(card_id: str, payload: BindPayload) -> dict[str, Any]:
        service = require_service()
        binding = await service.bind_session(
            role_card_id=card_id,
            group_id=payload.group_id,
            user_id=payload.user_id,
        )
        if binding is None:
            raise HTTPException(status_code=404, detail="role card not found")
        return {"binding": binding.model_dump()}

    @router.get("/tools")
    async def list_tools() -> dict[str, Any]:
        if tool_registry is None:
            return {"items": [], "total": 0}
        items = [tool.to_dict() for tool in tool_registry.list_tools()]
        return {"items": items, "total": len(items)}

    return router
