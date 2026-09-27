"""记忆抽取：把新产生的会话 turn 批量喂给辅助模型，落成群记忆 / 成员记忆。

批量增量策略（已确认）：某会话累计够 min_new_turns 条新 turn、或距上次抽取超过
interval_seconds 且有新内容时才跑一次，避免每条消息一次 LLM 调用。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..conversation.model import ConversationTurn
from ..conversation.repo import ConversationRepo, MemberNameRepo
from .persona_memory import (
    ALL_KINDS,
    SENSITIVITIES,
    SENSITIVITY_NEUTRAL,
    PersonaMemoryRepo,
)

logger = logging.getLogger(__name__)

DataCleaningChat = Callable[[str], Awaitable[str]]

EXTRACT_PROMPT = """你在维护一个 QQ 群的长期记忆库。下面是该群最近的对话记录（「昵称：内容」，
「你：」是机器人自己说的话）。请提取值得长期记住的信息，输出 JSON。

只提取这几类：
- 群层面（scope=group）：群的设定与规矩、稳定事实、群内黑话/梗的含义、重要事件与约定。
- 成员层面（scope=member）：某个成员的称呼/外号、稳定偏好、群内身份、与其他成员的关系、他身上的重要事件。

规则：
1. 忽略闲聊噪声、临时情绪、一次性玩笑；只记以后还用得上的。
2. 不记真实姓名/住址/电话/单位/经济状况等隐私信息；不记负面评价与八卦评判。
3. 每条 content 用一句话、20 字以内、自带主语（成员条目前面不用写名字），只写人话层面的事实，
   例如「喜欢在周末开黑」；不要写成「该成员经常使用某指令」这类机制描述。
4. importance 取 1-5（5=长期稳定且重要）。sensitivity 取：
   neutral（中性事实/梗/约定）、relation（成员之间的关系）、emotion（情绪与心理状态）、private（敏感隐私）。
5. 不要记录：机器人自身的运行规则/概率/触发方式、插件命令的用法与参数、以及"谁说了什么"的一次性流水。
   群里玩插件或游戏这类只概括成一句「本群在玩什么」，不要写具体命令、参数与机制。
6. 成员条目的 user_id 只能填对话记录开头给出的说话人编号（M1/M2…），
   不要编造编号，也不要把编号写进 content。
7. 没有值得记的就返回空数组。

只输出 JSON，不要解释。格式：
{"memories": [{"scope": "group", "kind": "梗", "content": "...", "importance": 3, "sensitivity": "neutral"},
              {"scope": "member", "user_id": "12345", "kind": "偏好", "content": "...", "importance": 2, "sensitivity": "neutral"}]}
"""


def parse_memories(raw: str) -> list[dict[str, Any]]:
    """解析抽取结果：容错 JSON 围栏/前后缀，过滤非法条目。

    只做白名单校验，不猜语义；成员条目缺 user_id 直接丢弃（防幻觉造人）。
    """
    text = str(raw or "").strip()
    if not text:
        return []
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    items = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope") or "").strip().lower()
        if scope not in {"group", "member"}:
            continue
        content = " ".join(str(item.get("content") or "").split()).strip()
        if not content:
            continue
        kind = str(item.get("kind") or "").strip()[:12]
        if kind not in ALL_KINDS:
            kind = "事实" if scope == "group" else "偏好"
        sensitivity = str(item.get("sensitivity") or "").strip().lower()
        if sensitivity not in SENSITIVITIES:
            sensitivity = SENSITIVITY_NEUTRAL
        try:
            importance = min(5.0, max(1.0, float(item.get("importance") or 1.0)))
        except (TypeError, ValueError):
            importance = 1.0
        entry: dict[str, Any] = {
            "scope": scope,
            "kind": kind,
            "content": content[:120],
            "importance": importance,
            "sensitivity": sensitivity,
        }
        if scope == "member":
            user_id = str(item.get("user_id") or "").strip()
            if not user_id:
                continue
            entry["user_id"] = user_id
        cleaned.append(entry)
    return cleaned


def build_batch_context(
    turns: list[ConversationTurn],
    *,
    names: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """渲染批上下文：说话人用编号（M1/M2…）标注，返回 (行, 编号→真实 QQ 映射)。

    提示词里只出现编号：模型不需要（也不该）看到 QQ 号，归属由编号映射回真实 user_id。
    """
    from ..conversation.window import render_turn_line, safe_speaker_name

    user_by_alias: dict[str, str] = {}
    label_by_user: dict[str, str] = {}
    for turn in turns:
        if turn.is_bot:
            continue
        user_id = str(turn.user_id or "").strip()
        if not user_id or user_id in label_by_user:
            continue
        alias = f"M{len(user_by_alias) + 1}"
        user_by_alias[alias] = user_id
        label_by_user[user_id] = f"{alias}（{safe_speaker_name(turn.speaker_name or names.get(user_id, ''), user_id=user_id)}）"

    lines: list[str] = []
    for turn in turns:
        if turn.is_bot:
            line = render_turn_line(turn, names=names)
        else:
            user_id = str(turn.user_id or "").strip()
            label = label_by_user.get(user_id, "M?")
            body = render_turn_line(turn, names=names)
            body = body.split("：", 1)[1] if "：" in body else body
            line = f"{label}：{body}" if body else ""
        if line:
            lines.append(line)
    return lines, user_by_alias


async def extract_session_memories(
    *,
    session_key: str,
    group_id: str,
    turns: list[ConversationTurn],
    names: dict[str, str],
    memories: PersonaMemoryRepo,
    data_cleaning_chat: DataCleaningChat,
    chat: DataCleaningChat | None = None,
    max_turns: int = 200,
) -> dict[str, int]:
    """单会话增量抽取：读取新 turn → 调辅助模型 → 写两张记忆表。"""
    batch = [turn for turn in turns if not turn.is_bot][-max_turns:]
    if not batch:
        return {"group_saved": 0, "member_saved": 0, "skipped": 0}

    lines, user_by_alias = build_batch_context(batch, names=names)
    prompt = EXTRACT_PROMPT + "\n\n【对话记录】\n" + "\n".join(lines[-max_turns:])
    runner = chat or data_cleaning_chat
    raw = await runner(prompt)
    entries = parse_memories(raw)

    group_saved = member_saved = skipped = 0
    for entry in entries:
        if entry["scope"] == "group":
            saved = await memories.add_group_memory(
                group_id=group_id,
                kind=str(entry["kind"]),
                content=str(entry["content"]),
                importance=float(entry["importance"]),
                sensitivity=str(entry["sensitivity"]),
            )
            group_saved += 1 if saved else 0
        else:
            alias = str(entry.get("user_id") or "").strip().upper()
            user_id = user_by_alias.get(alias, "")
            if not user_id:
                skipped += 1  # 批里没有这个编号：视为幻觉，丢弃
                continue
            saved = await memories.add_member_memory(
                group_id=group_id,
                user_id=user_id,
                kind=str(entry["kind"]),
                content=str(entry["content"]),
                importance=float(entry["importance"]),
                sensitivity=str(entry["sensitivity"]),
            )
            member_saved += 1 if saved else 0
    return {"group_saved": group_saved, "member_saved": member_saved, "skipped": skipped}


async def run_memory_extraction_round(
    *,
    conversation: ConversationRepo,
    memories: PersonaMemoryRepo,
    names_repo: MemberNameRepo,
    data_cleaning_chat: DataCleaningChat,
    min_new_turns: int,
    interval_seconds: float,
    max_turns: int = 200,
    now: float | None = None,
    disabled_groups: set[str] | None = None,
) -> list[dict[str, Any]]:
    """扫描所有会话，给满足条件的会话跑一次抽取。返回每会话的结果摘要。"""
    current = float(now if now is not None else time.time())
    states = await memories.get_extraction_states()
    results: list[dict[str, Any]] = []
    skip = {str(item) for item in (disabled_groups or set())}
    for group_id in await conversation.list_session_groups():
        if str(group_id) in skip:
            continue
        session_key = f"group:{group_id}"
        state = states.get(session_key)
        last_turn_id = state.last_turn_id if state else 0
        pending = await conversation.turns_after(session_key=session_key, after_id=last_turn_id, limit=max_turns)
        if not pending:
            continue
        waited = current - (state.last_run_at if state else 0.0)
        if len(pending) < max(1, int(min_new_turns)) and waited < max(60.0, float(interval_seconds)):
            continue
        names = await names_repo.get_names(
            group_id=group_id,
            user_ids=[str(turn.user_id or "") for turn in pending if str(turn.user_id or "").strip()],
        )
        try:
            outcome = await extract_session_memories(
                session_key=session_key,
                group_id=group_id,
                turns=pending,
                names=names,
                memories=memories,
                data_cleaning_chat=data_cleaning_chat,
                max_turns=max_turns,
            )
        except Exception as exc:  # noqa: BLE001 — 抽取失败不影响其它会话，下次再试
            logger.warning(f"memory extraction failed for {session_key}: {exc!r}")
            results.append({"session_key": session_key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        await memories.mark_extracted(
            session_key=session_key,
            last_turn_id=int(pending[-1].id or last_turn_id),
            ran_at=current,
        )
        outcome["session_key"] = session_key
        outcome["turns"] = len(pending)
        results.append(outcome)
        logger.info(f"memory extraction {session_key}: turns={len(pending)} result={outcome}")
    return results


__all__ = [
    "EXTRACT_PROMPT",
    "build_batch_context",
    "extract_session_memories",
    "parse_memories",
    "run_memory_extraction_round",
]
