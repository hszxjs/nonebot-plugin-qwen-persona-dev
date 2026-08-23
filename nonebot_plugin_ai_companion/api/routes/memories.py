"""长期记忆与 LoRA adapter 管理路由（文档六/十一节）。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...finetune.adapters import LoraAdapterRepo, detect_language
from ...memory.longterm import DataCleaningChat, LongTermMemoryRepo, extract_memories
from ...providers.aux_llm import AuxLlmError

logger = logging.getLogger(__name__)


class ExtractPayload(BaseModel):
    user_id: str


class AdapterRegisterPayload(BaseModel):
    role_card_id: str
    source_language: str = "zh"
    path: str
    note: str = ""
    activate: bool = False


class AdapterRollbackPayload(BaseModel):
    role_card_id: str
    source_language: str


def build_router(
    *,
    auth,
    store: Any | None,
    config: Any | None,
    data_cleaning_chat: DataCleaningChat | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    def repos() -> tuple[LongTermMemoryRepo, LoraAdapterRepo]:
        if store is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        return store.long_term_memories, store.lora_adapters

    # ---- 长期记忆 ----
    @router.get("/memories")
    async def list_memories(user_id: str = "", include_decayed: bool = False, limit: int = 200) -> dict[str, Any]:
        memories, _adapters = repos()
        items = await memories.list_memories(
            user_id=user_id or None,
            include_decayed=include_decayed,
            limit=limit,
        )
        return {"items": items, "total": len(items)}

    @router.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: int) -> dict[str, int]:
        memories, _adapters = repos()
        if not await memories.delete_memory(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"deleted": 1}

    @router.post("/memories/extract", response_model=None)
    async def extract(payload: ExtractPayload) -> dict[str, Any]:
        if store is None or config is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        try:
            written = await extract_memories(
                config,
                store.samples,
                user_id=payload.user_id.strip(),
                memories=store.long_term_memories,
                data_cleaning_chat=data_cleaning_chat,
            )
        except AuxLlmError as exc:
            raise HTTPException(status_code=502, detail=f"memory extraction failed: {exc}") from exc
        return {"written": written, "count": len(written)}

    @router.post("/memories/decay")
    async def decay(half_life_days: float = 30.0) -> dict[str, Any]:
        memories, _adapters = repos()
        return await memories.decay(half_life_days=half_life_days)

    # ---- LoRA adapter 版本管理 ----
    @router.get("/lora-adapters")
    async def list_adapters(role_card_id: str = "") -> dict[str, Any]:
        _memories, adapters = repos()
        items = await adapters.list_adapters(role_card_id=role_card_id or None)
        return {"items": [item.model_dump() for item in items], "total": len(items)}

    @router.post("/lora-adapters", response_model=None)
    async def register_adapter(payload: AdapterRegisterPayload) -> dict[str, Any]:
        _memories, adapters = repos()
        adapter = await adapters.register_adapter(
            role_card_id=payload.role_card_id,
            source_language=payload.source_language or detect_language(payload.note),
            path=payload.path,
            note=payload.note,
            activate=payload.activate,
        )
        return {"item": adapter.model_dump()}

    @router.post("/lora-adapters/{adapter_id}/activate", response_model=None)
    async def activate_adapter(adapter_id: str) -> dict[str, Any]:
        _memories, adapters = repos()
        activated = await adapters.activate(adapter_id)
        if activated is None:
            raise HTTPException(status_code=404, detail="adapter not found")
        return {"item": activated.model_dump()}

    @router.post("/lora-adapters/rollback", response_model=None)
    async def rollback_adapter(payload: AdapterRollbackPayload) -> dict[str, Any]:
        _memories, adapters = repos()
        rolled = await adapters.rollback(role_card_id=payload.role_card_id, source_language=payload.source_language)
        if rolled is None:
            raise HTTPException(status_code=409, detail="no earlier version to rollback")
        return {"item": rolled.model_dump()}

    @router.delete("/lora-adapters/{adapter_id}")
    async def delete_adapter(adapter_id: str) -> dict[str, int]:
        _memories, adapters = repos()
        if not await adapters.delete_adapter(adapter_id):
            raise HTTPException(status_code=404, detail="adapter not found")
        return {"deleted": 1}

    return router
