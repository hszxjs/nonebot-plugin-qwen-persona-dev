"""会话时间线路由：把群成员昵称回填进 member_names（历史渲染用真名而非尾号）。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)


def names_from_group_member_list(members: Any) -> dict[str, str]:
    """OneBot get_group_member_list → {user_id: 显示名}；群名片优先、其次昵称。"""
    if not isinstance(members, list):
        return {}
    names: dict[str, str] = {}
    for item in members:
        if not isinstance(item, dict):
            continue
        user_id = str(item.get("user_id") or "").strip()
        name = str(item.get("card") or "").strip() or str(item.get("nickname") or "").strip()
        if user_id and name:
            names[user_id] = name
    return names


def build_router(*, auth, store: Any | None) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    @router.post("/conversation/refresh-names")
    async def refresh_names(group_id: str = "") -> dict[str, Any]:
        """拉取群成员列表写入昵称表；不传 group_id 时刷新时间线里出现过的所有群。"""
        if store is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        try:
            from nonebot import get_bot
        except Exception as exc:  # pragma: no cover - nonebot 缺失
            raise HTTPException(status_code=503, detail=f"nonebot unavailable: {exc}") from exc

        groups = [group_id.strip()] if group_id.strip() else await _timeline_groups(store)
        if not groups:
            return {"groups": 0, "names": 0, "details": {}}

        try:
            bot = get_bot()
        except Exception as exc:  # 无在线 bot 时由前端提示稍后再试
            raise HTTPException(status_code=503, detail=f"no online bot: {exc}") from exc

        details: dict[str, int] = {}
        total = 0
        for gid in groups:
            try:
                members = await bot.call_api("get_group_member_list", group_id=int(gid))
            except Exception as exc:  # noqa: BLE001 — 单群失败不影响其他群
                logger.warning(f"refresh names failed for group {gid}: {exc!r}")
                details[gid] = 0
                continue
            names = names_from_group_member_list(members)
            for user_id, name in names.items():
                await store.member_names.upsert_name(group_id=gid, user_id=user_id, name=name)
            details[gid] = len(names)
            total += len(names)
            logger.info(f"refresh names: group {gid} -> {len(names)} names")
        return {"groups": len(groups), "names": total, "details": details}

    return router


async def _timeline_groups(store: Any) -> list[str]:
    """时间线里出现过的群号（去重，含返回 None 的兜底）。"""
    repo = getattr(store, "conversation", None)
    if repo is None:
        return []
    rows = await repo.list_session_groups()
    return [str(item) for item in rows if str(item).strip()]
