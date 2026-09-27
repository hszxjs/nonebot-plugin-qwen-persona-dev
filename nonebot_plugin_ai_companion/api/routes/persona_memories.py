"""群记忆 / 成员记忆路由：查看、编辑、删除、按成员清除、手动抽取与衰减。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...memory.persona_memory import SENSITIVITIES

logger = logging.getLogger(__name__)


class MemoryPatchPayload(BaseModel):
    content: str | None = None
    importance: float | None = None
    sensitivity: str | None = None
    restore: bool = False  # 把已过期的记忆恢复注入


class MemoryPurgePayload(BaseModel):
    group_id: str
    user_id: str


class MemoryExtractPayload(BaseModel):
    group_id: str = ""
    force: bool = True
    max_turns: int = 300


class MemoryDecayPayload(BaseModel):
    half_life_days: float | None = None


def build_router(
    *,
    auth,
    store: Any | None,
    config: Any | None,
    data_cleaning_chat: Any | None,
) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    def memory_repo():
        repo = getattr(store, "persona_memory", None) if store is not None else None
        if repo is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        return repo

    @router.get("/persona-memories")
    async def list_memories(
        scope: str = "group",
        group_id: str = "",
        user_id: str = "",
        limit: int = 200,
        include_decayed: bool = False,
    ) -> dict[str, Any]:
        repo = memory_repo()
        clean_scope = "member" if str(scope).lower() == "member" else "group"
        items = await repo.list_memories(
            scope=clean_scope,
            group_id=group_id or None,
            user_id=user_id or None,
            limit=min(max(1, int(limit)), 500),
            include_decayed=bool(include_decayed),
        )
        return {"items": [item.to_dict() for item in items], "scope": clean_scope}

    @router.get("/persona-memories/groups")
    async def list_memory_groups() -> dict[str, Any]:
        """记忆页的群筛选：时间线里的群 + 已禁用抽取的群。"""
        if store is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        groups = sorted({str(item) for item in await store.conversation.list_session_groups()})
        disabled = sorted(
            {str(item) for item in getattr(config, "ai_companion_memory_disabled_groups", []) if str(item).strip()}
        )
        return {"groups": groups, "disabled": disabled, "sensitivities": list(SENSITIVITIES)}

    @router.patch("/persona-memories/{scope}/{memory_id}")
    async def update_memory(scope: str, memory_id: int, payload: MemoryPatchPayload) -> dict[str, Any]:
        repo = memory_repo()
        clean_scope = "member" if str(scope).lower() == "member" else "group"
        updated = await repo.update_memory(
            scope=clean_scope,
            memory_id=memory_id,
            content=payload.content,
            importance=payload.importance,
            sensitivity=payload.sensitivity,
            restore=bool(payload.restore),
        )
        if not updated:
            raise HTTPException(status_code=404, detail="memory not found or nothing to update")
        return {"ok": True}

    @router.delete("/persona-memories/{scope}/{memory_id}")
    async def delete_memory(scope: str, memory_id: int) -> dict[str, Any]:
        repo = memory_repo()
        clean_scope = "member" if str(scope).lower() == "member" else "group"
        deleted = await repo.delete_memory(scope=clean_scope, memory_id=memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory not found")
        return {"ok": True, "deleted": 1}

    @router.post("/persona-memories/purge")
    async def purge_member(payload: MemoryPurgePayload) -> dict[str, Any]:
        """一键清除某成员在本群的痕迹：记忆 + 时间线原话 + 昵称。

        只删记忆是不够的：原话留在时间线里会被下一轮回复当历史注入，
        抽取也会从同样的内容里把记忆重新长出来。
        """
        repo = memory_repo()
        if not payload.group_id.strip() or not payload.user_id.strip():
            raise HTTPException(status_code=400, detail="group_id and user_id are required")
        deleted = await repo.purge_member(group_id=payload.group_id, user_id=payload.user_id)
        cleared_turns = 0
        if store is not None:
            cleared_turns = await store.conversation.anonymize_user_turns(
                user_id=payload.user_id, scope_group_id=payload.group_id
            )
            await store.member_names.delete_name(group_id=payload.group_id, user_id=payload.user_id)
            # 抽取进度回退到"已读但无内容"：下次抽取看到的是空文本，不会再长出同样的记忆
            state = (await repo.get_extraction_states()).get(f"group:{payload.group_id}")
            if state is not None:
                await repo.mark_extracted(
                    session_key=f"group:{payload.group_id}",
                    last_turn_id=state.last_turn_id,
                    ran_at=state.last_run_at,
                )
        return {"ok": True, "deleted": deleted, "cleared_turns": cleared_turns}

    @router.post("/persona-memories/extract")
    async def extract_now(payload: MemoryExtractPayload) -> dict[str, Any]:
        """手动触发一次增量抽取（默认忽略阈值）；group_id 为空则跑所有群。"""
        if store is None or data_cleaning_chat is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        from ...memory.extractor import run_memory_extraction_round

        disabled = {str(item) for item in getattr(config, "ai_companion_memory_disabled_groups", [])}
        target = payload.group_id.strip()
        if target and target in disabled:
            raise HTTPException(status_code=400, detail=f"group {target} is excluded from memory")
        results = await run_memory_extraction_round(
            conversation=store.conversation,
            memories=store.persona_memory,
            names_repo=store.member_names,
            data_cleaning_chat=data_cleaning_chat,
            min_new_turns=1 if payload.force else int(getattr(config, "ai_companion_memory_extract_min_turns", 30)),
            interval_seconds=0.0
            if payload.force
            else float(getattr(config, "ai_companion_memory_extract_interval_seconds", 1800)),
            max_turns=min(max(50, int(payload.max_turns)), 1000),
            # 指定群时只跑该群（其余全部跳过）；不指定时保留禁用名单
            disabled_groups=({f"group:{target}"} if target else disabled),
        )
        saved = sum(int(item.get("group_saved", 0)) + int(item.get("member_saved", 0)) for item in results)
        failed = [item for item in results if item.get("error")]
        return {"results": results, "saved": saved, "failed": len(failed)}

    @router.post("/persona-memories/consolidate")
    async def consolidate(persona_chat: Any | None = None) -> dict[str, Any]:
        """一次性整理：合并近义重复、淘汰临时状态噪声（软淘汰，可恢复）。"""
        if store is None or data_cleaning_chat is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        from ...memory.consolidate import consolidate_all

        reports = await consolidate_all(
            repo=store.persona_memory,
            chat=data_cleaning_chat,
            conversation=store.conversation,
        )
        merged = sum(int(item.get("merged", 0)) for item in reports.values())
        retired = sum(int(item.get("retired", 0)) for item in reports.values())
        return {"groups": len(reports), "merged": merged, "retired": retired, "details": reports}

    @router.post("/persona-memories/decay")
    async def decay_memories(payload: MemoryDecayPayload | None = None) -> dict[str, Any]:
        repo = memory_repo()
        half_life = payload.half_life_days if payload is not None and payload.half_life_days else None
        if half_life is None:
            half_life = float(getattr(config, "ai_companion_memory_half_life_days", 30.0))
        return await repo.decay(half_life_days=float(half_life))

    return router


__all__ = ["build_router"]
