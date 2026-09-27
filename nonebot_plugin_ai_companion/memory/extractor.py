"""记忆抽取：把新产生的会话 turn 批量喂给辅助模型，落成群记忆 / 成员记忆。

批量增量策略（已确认）：某会话累计够 min_new_turns 条新 turn、或距上次抽取超过
interval_seconds 且有新内容时才跑一次，避免每条消息一次 LLM 调用。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..conversation.model import ConversationTurn
from ..conversation.repo import ConversationRepo, MemberNameRepo
from .bot_meta import looks_like_bot_meta
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
1. 只记**稳定特征**：长期偏好、身份/职业、称呼外号、人际关系、群规与群梗。
   临时状态（"这周在拉货""最近没回家""今天在忙"）只在明确会持续一段时间时记为 kind=事件，
   不要记成偏好或身份；一次性玩笑、流水账、情绪波动都不记。
2. 不记真实姓名/住址/电话/单位/经济状况等隐私信息；不记负面评价与八卦评判。
3. 每条 content 用一句话、20 字以内、自带主语（成员条目前面不用写名字），只写人话层面的事实，
   例如「喜欢在周末开黑」；不要写成「该成员经常使用某指令」这类机制描述。
4. importance 取 1-5（5=长期稳定且重要）。sensitivity 取：
   neutral（中性事实/梗/约定）、relation（成员之间的关系）、emotion（情绪与心理状态）、private（敏感隐私）。
5. 不要记录：机器人自身的运行规则/概率/触发方式、插件命令的用法与参数、以及"谁说了什么"的一次性流水。
   成员条目的主语必须是这个人本身：像"群友常约一起打本"这种是群层面事实，应记为 scope=group，不要挂在个人名下。
   群里玩插件或游戏这类只概括成一句「本群在玩什么」，不要写具体命令、参数与机制。
6. 成员条目的 user_id 只能填对话记录开头给出的说话人编号（M1/M2…），
   不要编造编号，也不要把编号写进 content。
7. 如果一条新信息其实是在修正/补充下面的「已有记忆」，就填 existing_id 表示更新它，
   不要再新增一条重复的（同一件事只能有一条记忆，也不要同一事实既记"事实"又记"梗"）。
8. 没有值得记的就返回空数组。

只输出 JSON，不要解释。格式：
{"memories": [{"scope": "group", "kind": "梗", "content": "...", "importance": 3, "sensitivity": "neutral"},
              {"scope": "member", "user_id": "M1", "kind": "偏好", "content": "...", "importance": 2, "sensitivity": "neutral"},
              {"scope": "group", "existing_id": 12, "kind": "梗", "content": "（更新后的说法）", "importance": 4, "sensitivity": "neutral"}]}
"""


_GROUP_SUBJECT_RE = re.compile(r"^(本群|群内|群里|群友|群中|群里的人)|(^群[^友]?[的是在])")


def _looks_like_group_fact(content: str) -> bool:
    """判断一条"成员记忆"其实是不是群层面事实（主语是群而不是这个人）。"""
    return bool(_GROUP_SUBJECT_RE.search(" ".join(str(content or "").split())))


def parse_ok(raw: str) -> bool:
    """模型是否给出了可解析的 JSON 结构（用来判断"没有记忆"还是"根本没答对"）。"""
    import json as _json

    text = str(raw or "").strip()
    if not text:
        return False
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        payload = _json.loads(text[start : end + 1])
    except (_json.JSONDecodeError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("memories"), list)


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
        if not content or looks_like_bot_meta(content, scope=scope):
            continue  # 群层面挡机制描述；成员层面只挡"机器人自己怎么运行"
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
        raw_existing = item.get("existing_id")
        if raw_existing is not None:
            try:
                existing_id = int(str(raw_existing).strip())
                if existing_id > 0:
                    entry["existing_id"] = existing_id
            except (TypeError, ValueError):
                pass
        if scope == "member":
            user_id = str(item.get("user_id") or "").strip()
            if not user_id:
                continue
            if _looks_like_group_fact(content):
                continue  # "群友常约…"这类主语是群，不该挂在个人名下
            entry["user_id"] = user_id
        cleaned.append(entry)
    return cleaned


# 失败退避：同一会话连续失败时按 5/10/20…分钟退避，避免每隔几分钟重复烧钱与刷日志
_FAILURE_BACKOFF: dict[str, tuple[int, float]] = {}


def _failure_backoff_until(session_key: str, now: float) -> float:
    count, until = _FAILURE_BACKOFF.get(session_key, (0, 0.0))
    return until if count else 0.0


def _record_failure(session_key: str, now: float) -> None:
    count, _until = _FAILURE_BACKOFF.get(session_key, (0, 0.0))
    count += 1
    delay = min(3600.0, 300.0 * (2 ** (count - 1)))
    _FAILURE_BACKOFF[session_key] = (count, now + delay)


def _clear_failure(session_key: str) -> None:
    _FAILURE_BACKOFF.pop(session_key, None)


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
        label_by_user[user_id] = (
            f"{alias}（{safe_speaker_name(turn.speaker_name or names.get(user_id, ''), user_id=user_id)}）"
        )

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
    private_user_id: str = "",
) -> dict[str, int]:
    """单会话增量抽取：读取新 turn → 调辅助模型 → 写两张记忆表。

    private_user_id 非空表示这是私聊会话：只写成员记忆、group_id 记为空串
    （「私聊里说过的事」独立于任何群），群层面的条目直接丢弃。
    """
    batch = [turn for turn in turns if not turn.is_bot][-max_turns:]
    if not batch:
        return {"group_saved": 0, "member_saved": 0, "skipped": 0}

    lines, user_by_alias = build_batch_context(batch, names=names)

    # 已有记忆：给模型看 id，让它选择"更新"而不是重复新增（语义去重只有模型做得到）
    existing_blocks: list[str] = []
    existing_ids: dict[int, str] = {}
    if not private_user_id:
        group_rows = await memories.list_memories(scope="group", group_id=group_id, limit=30)
        if group_rows:
            existing_blocks.append("群记忆：")
            for record in group_rows:
                existing_ids[record.id] = "group"
                existing_blocks.append(f"  #{record.id} [{record.kind}] {record.content}")
    member_ids = list(user_by_alias.values())
    for user_id in member_ids:
        rows = await memories.list_memories(
            scope="member", group_id=("" if private_user_id else group_id), user_id=user_id, limit=20
        )
        if not rows:
            continue
        alias = next((k for k, v in user_by_alias.items() if v == user_id), user_id)
        existing_blocks.append(f"{alias} 的成员记忆：")
        for record in rows:
            existing_ids[record.id] = "member"
            existing_blocks.append(f"  #{record.id} [{record.kind}] {record.content}")

    prompt = EXTRACT_PROMPT
    if existing_blocks:
        prompt += "\n\n【已有记忆】（重复的请用 existing_id 更新，不要新增）\n" + "\n".join(existing_blocks)
    prompt += "\n\n【对话记录】\n" + "\n".join(lines[-max_turns:])
    runner = chat or data_cleaning_chat
    raw = await runner(prompt)
    entries = parse_memories(raw)
    parsed = parse_ok(raw)

    group_saved = member_saved = skipped = updated = 0
    for entry in entries:
        existing_id = int(entry.get("existing_id") or 0)
        if existing_id and existing_ids.get(existing_id):
            changed = await memories.update_memory(
                scope=existing_ids[existing_id],
                memory_id=existing_id,
                content=str(entry["content"]),
                importance=float(entry["importance"]),
                sensitivity=str(entry["sensitivity"]),
            )
            updated += 1 if changed else 0
            continue
        if entry["scope"] == "group":
            if private_user_id:
                skipped += 1  # 私聊不是群，不产生群记忆
                continue
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
                group_id="" if private_user_id else group_id,
                user_id=user_id,
                kind=str(entry["kind"]),
                content=str(entry["content"]),
                importance=float(entry["importance"]),
                sensitivity=str(entry["sensitivity"]),
            )
            member_saved += 1 if saved else 0
    return {
        "group_saved": group_saved,
        "member_saved": member_saved,
        "updated": updated,
        "skipped": skipped,
        "parse_ok": parsed,
    }


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
    sessions = [f"group:{gid}" for gid in await conversation.list_session_groups()]
    sessions.extend(f"private:{uid}" for uid in await conversation.list_private_users())
    for session_key in sessions:
        is_private = session_key.startswith("private:")
        scope_id = session_key.split(":", 1)[1]
        # 配置里写的是群号，这里也接受 "group:xxx" 形式，两种都认
        if session_key in skip or scope_id in skip:
            continue
        state = states.get(session_key)
        last_turn_id = state.last_turn_id if state else 0
        pending = await conversation.turns_after(session_key=session_key, after_id=last_turn_id, limit=max_turns)
        if not pending:
            continue
        waited = current - (state.last_run_at if state else 0.0)
        if len(pending) < max(1, int(min_new_turns)) and waited < max(60.0, float(interval_seconds)):
            continue
        names = await names_repo.get_names(
            group_id=scope_id if not is_private else None,
            user_ids=[str(turn.user_id or "") for turn in pending if str(turn.user_id or "").strip()],
        )
        backoff_until = _failure_backoff_until(session_key, current)
        if backoff_until > current:
            continue  # 上一轮失败：退避期内不重复烧钱
        try:
            outcome = await extract_session_memories(
                session_key=session_key,
                group_id="" if is_private else scope_id,
                private_user_id=scope_id if is_private else "",
                turns=pending,
                names=names,
                memories=memories,
                data_cleaning_chat=data_cleaning_chat,
                max_turns=max_turns,
            )
        except Exception as exc:  # noqa: BLE001 — 抽取失败不影响其它会话，下次再试
            _record_failure(session_key, current)
            logger.warning(f"memory extraction failed for {session_key}: {exc!r}")
            results.append({"session_key": session_key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not outcome.get("parse_ok", True):
            # 模型没给出可解析结构：不推进游标，退避后重试同一批（否则这批对话被静默跳过）
            _record_failure(session_key, current)
            logger.warning(f"memory extraction unparsable for {session_key}; cursor stays put")
            results.append({"session_key": session_key, "error": "unparsable output"})
            continue
        _clear_failure(session_key)
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
    "parse_ok",
    "extract_session_memories",
    "parse_memories",
    "run_memory_extraction_round",
]
