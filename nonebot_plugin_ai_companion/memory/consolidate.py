"""存量记忆整理：用模型把同一件事的多种说法合并，并淘汰临时状态噪声。

为什么需要一次性整理：语义去重（existing_id 更新）只对之后的新抽取生效，
之前几轮积累下来的近义重复（"雷雷=战争雷霆"的 5-6 个变体、一个人的 6 条拉货状态）
得清一次，否则会一直占着注入名额。

安全边界：只"软淘汰"（decayed=1，可恢复、可删），不物理删除；合并只改内容与重要度。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .persona_memory import SENSITIVITIES, SENSITIVITY_NEUTRAL, PersonaMemoryRepo

logger = logging.getLogger(__name__)

ConsolidateChat = Callable[[str], Awaitable[str]]

CONSOLIDATE_PROMPT = """你在整理一份记忆列表，目的是去掉重复与噪声，保留真正有用的信息。

规则：
1. 同一件事的多种说法（例如"「雷雷」指战争雷霆"和"群内用雷雷指代战争雷霆游戏"）合并成一条，
   保留信息最全的说法，重要度取其中最大值。
2. 临时状态、流水账、一次性情绪（"这周在拉货""今天很累"）如果没什么长期价值，放进 drop。
3. 合并时 ids 列出被合并的编号（包含要保留的那条），content 写合并后的最终说法（20 字内）。
4. 不新增任何列表里没有的信息；不确定是否重复就保持原样（不要放进任何一组）。
5. 只输出 JSON，不要解释：
{"groups": [{"ids": [3, 7], "kind": "梗", "content": "…", "importance": 4, "sensitivity": "neutral"}],
 "drop": [12, 15]}
"""


def parse_consolidation(raw: str, *, known_ids: set[int]) -> tuple[list[dict[str, Any]], list[int]]:
    """解析整理结果：只接受列表里出现过的编号，防幻觉。"""
    text = str(raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return [], []
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return [], []
    groups: list[dict[str, Any]] = []
    for item in payload.get("groups") or []:
        if not isinstance(item, dict):
            continue
        ids = [int(value) for value in (item.get("ids") or []) if str(value).strip().lstrip("-").isdigit()]
        ids = [value for value in ids if value in known_ids]
        content = " ".join(str(item.get("content") or "").split()).strip()
        if len(ids) < 2 or not content:
            continue
        sensitivity = str(item.get("sensitivity") or "").strip().lower()
        try:
            importance = min(5.0, max(1.0, float(item.get("importance") or 3.0)))
        except (TypeError, ValueError):
            importance = 3.0
        groups.append(
            {
                "ids": ids,
                "keep": ids[0],
                "content": content[:120],
                "importance": importance,
                "sensitivity": sensitivity if sensitivity in SENSITIVITIES else SENSITIVITY_NEUTRAL,
                "kind": str(item.get("kind") or "").strip()[:12],
            }
        )
    drop = [int(value) for value in (payload.get("drop") or []) if str(value).strip().lstrip("-").isdigit()]
    drop = [value for value in drop if value in known_ids]
    return groups, drop


async def consolidate_scope(
    *,
    repo: PersonaMemoryRepo,
    chat: ConsolidateChat,
    scope: str,
    group_id: str,
    user_id: str = "",
    min_rows: int = 4,
) -> dict[str, int]:
    """整理单个 (scope, group, user?) 的记忆：合并近义 + 淘汰噪声。"""
    rows = await repo.list_memories(scope=scope, group_id=group_id or None, user_id=user_id or None, limit=200)
    if len(rows) < min_rows:
        return {"merged": 0, "retired": 0, "scanned": len(rows)}
    known_ids = {row.id for row in rows}
    listing = "\n".join(
        f"#{row.id} [{row.kind}/{row.sensitivity}/重要度{row.importance:.0f}] {row.content}" for row in rows
    )
    raw = await chat(CONSOLIDATE_PROMPT + "\n\n【记忆列表】\n" + listing)
    groups, drop = parse_consolidation(raw, known_ids=known_ids)

    merged = retired = 0
    for item in groups:
        keep = int(item["keep"])
        changed = await repo.update_memory(
            scope=scope,
            memory_id=keep,
            content=str(item["content"]),
            importance=float(item["importance"]),
            sensitivity=str(item["sensitivity"]),
        )
        merged += 1 if changed else 0
        for other in item["ids"]:
            if other != keep:
                retired += 1 if await repo.retire_memory(scope=scope, memory_id=other) else 0
    for memory_id in drop:
        retired += 1 if await repo.retire_memory(scope=scope, memory_id=memory_id) else 0
    return {"merged": merged, "retired": retired, "scanned": len(rows)}


async def consolidate_all(
    *,
    repo: PersonaMemoryRepo,
    chat: ConsolidateChat,
    conversation: Any | None = None,
    min_rows: int = 4,
) -> dict[str, Any]:
    """整理所有群 + 所有成员的记忆（运维手动触发一次即可）。"""
    groups: list[str] = []
    if conversation is not None:
        groups = [str(item) for item in await conversation.list_session_groups()]
    for row in await repo.list_memories(scope="group", limit=500):
        if row.group_id and row.group_id not in groups:
            groups.append(row.group_id)
    reports: dict[str, Any] = {}
    for group_id in groups:
        reports[f"group:{group_id}"] = await consolidate_scope(
            repo=repo, chat=chat, scope="group", group_id=group_id, min_rows=min_rows
        )
        member_ids: list[str] = []
        for row in await repo.list_memories(scope="member", group_id=group_id, limit=500):
            if row.user_id and row.user_id not in member_ids:
                member_ids.append(row.user_id)
        for user_id in member_ids:
            reports[f"member:{group_id}:{user_id}"] = await consolidate_scope(
                repo=repo, chat=chat, scope="member", group_id=group_id, user_id=user_id, min_rows=min_rows
            )
    if conversation is not None:
        for user_id in await conversation.list_private_users():
            reports[f"member::private:{user_id}"] = await consolidate_scope(
                repo=repo, chat=chat, scope="member", group_id="", user_id=str(user_id), min_rows=min_rows
            )
    return reports


__all__ = [
    "CONSOLIDATE_PROMPT",
    "ConsolidateChat",
    "consolidate_all",
    "consolidate_scope",
    "parse_consolidation",
]
