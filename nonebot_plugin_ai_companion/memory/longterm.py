"""长期结构化记忆（文档六节）：事实/称呼/偏好/事件，带重要度与时效性衰减。

写入经 data_cleaning 角色（aux LLM 或绑定实例）产出结构化 JSON；
定期去重与遗忘（decay）；prompt 记忆注入段从这里取数。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import AiCompanionConfig
from ..memory.db import PersonaDatabase
from ..memory.samples import ChatSampleRepo
from ..prompt.media_text import context_text_from_sample
from ..providers.aux_llm import AuxLlmError, aux_chat_json, parse_json_object

KINDS: tuple[str, ...] = ("称呼", "偏好", "重要事件", "事实")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_EXTRACT_PROMPT = """下面是群友 {user_id} 的近期消息（每行一条）。

{messages}

请从中提取关于这位群友的长期记忆（称呼、稳定的偏好、重要事件、可靠事实），
忽略一次性的闲聊内容。字符串值内部不要使用英文双引号。严格输出 JSON 数组（没有可提取项则输出 []）：
[{{"kind": "称呼|偏好|重要事件|事实", "content": "一句话事实", "importance": 1到5}}]
最多 10 条。"""

DataCleaningChat = Callable[[str], Awaitable[str]]


class LongTermMemoryRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def upsert(
        self,
        *,
        user_id: str,
        kind: str,
        content: str,
        importance: float,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        """写入一条记忆；与既有未衰减记忆高度相似时合并（刷新 last_seen、取更高重要度）。"""
        content = str(content or "").strip()
        if not content:
            raise ValueError("empty memory content")
        importance = min(5.0, max(1.0, float(importance or 1.0)))

        def _sync() -> dict[str, Any]:
            self.db.ensure_schema()
            now = float(created_at if created_at is not None else time.time())
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, content, importance FROM long_term_memories
                    WHERE user_id = ? AND kind = ? AND decayed = 0
                    """,
                    (user_id, kind),
                ).fetchall()
                for row in rows:
                    if _similar(str(row["content"]), content):
                        conn.execute(
                            """
                            UPDATE long_term_memories
                            SET importance = MAX(importance, ?), last_seen_at = ?
                            WHERE id = ?
                            """,
                            (importance, now, row["id"]),
                        )
                        return {"id": row["id"], "merged": True}
                cursor = conn.execute(
                    """
                    INSERT INTO long_term_memories (
                        user_id, kind, content, importance, created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, kind, content, importance, now, now),
                )
                return {"id": cursor.lastrowid, "merged": False}

        return await asyncio.to_thread(_sync)

    async def list_memories(
        self,
        *,
        user_id: str | None = None,
        include_decayed: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        def _sync() -> list[dict[str, Any]]:
            self.db.ensure_schema()
            clauses = [] if include_decayed else ["decayed = 0"]
            params: list[Any] = []
            if user_id:
                clauses.append("user_id = ?")
                params.append(user_id)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(max(1, min(int(limit), 1000)))
            with self.db.connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM long_term_memories {where}
                    ORDER BY importance DESC, last_seen_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(_sync)

    async def prompt_lines(self, *, user_id: str, limit: int = 6) -> list[str]:
        """记忆注入段（prompt memory segment）取数：top 重要度、未衰减。"""
        memories = await self.list_memories(user_id=user_id, limit=limit)
        return [f"{item['kind']}｜{item['content']}（重要度 {item['importance']:.0f}/5）" for item in memories]

    async def delete_memory(self, memory_id: int) -> bool:
        def _sync() -> bool:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute("DELETE FROM long_term_memories WHERE id = ?", (int(memory_id),))
            return bool(cursor.rowcount)

        return await asyncio.to_thread(_sync)

    async def decay(self, *, half_life_days: float = 30.0, threshold: float = 0.4) -> dict[str, int]:
        """遗忘：重要度随 last_seen 距今按半衰期衰减，低于阈值的标记 decayed。"""

        def _sync() -> dict[str, int]:
            self.db.ensure_schema()
            now = time.time()
            half_life_seconds = max(1.0, float(half_life_days) * 86400.0)
            decayed_count = 0
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, importance, last_seen_at FROM long_term_memories WHERE decayed = 0"
                ).fetchall()
                for row in rows:
                    age = max(0.0, now - float(row["last_seen_at"] or row["created_at"] or 0))
                    effective = float(row["importance"]) * 0.5 ** (age / half_life_seconds)
                    if effective < threshold:
                        conn.execute("UPDATE long_term_memories SET decayed = 1 WHERE id = ?", (row["id"],))
                        decayed_count += 1
                    else:
                        conn.execute(
                            "UPDATE long_term_memories SET importance = ? WHERE id = ?",
                            (round(effective, 3), row["id"]),
                        )
            return {"decayed": decayed_count, "total": len(rows)}

        return await asyncio.to_thread(_sync)


def _similar(left: str, right: str) -> bool:
    """去重相似度：字符 2-gram Jaccard ≥ 0.6 视为同一条。"""
    left_norm = "".join(_CJK_RE.findall(left)) or left.strip()
    right_norm = "".join(_CJK_RE.findall(right)) or right.strip()
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    def grams(value: str) -> set[str]:
        return {value[i : i + 2] for i in range(len(value) - 1)} or {value}

    intersection = len(grams(left_norm) & grams(right_norm))
    union = len(grams(left_norm) | grams(right_norm))
    return union > 0 and intersection / union >= 0.6


async def extract_memories(
    config: AiCompanionConfig,
    samples_repo: ChatSampleRepo,
    *,
    user_id: str,
    memories: LongTermMemoryRepo,
    data_cleaning_chat: DataCleaningChat | None = None,
    sample_limit: int = 100,
) -> list[dict[str, Any]]:
    """从该群友近期样本提取结构化记忆并入库；返回本次新写入条目。"""
    samples = await samples_repo.recent_samples_by_user(user_id=user_id, limit=sample_limit)
    lines: list[str] = []
    for sample in reversed(samples):
        text = context_text_from_sample(sample)
        if text:
            lines.append(text[:120])
    if len(lines) < 5:
        raise AuxLlmError(f"not enough samples for memory extraction (got {len(lines)})")

    prompt = _EXTRACT_PROMPT.format(user_id=user_id, messages="\n".join(lines[-150:]))
    if data_cleaning_chat is not None:
        reply = str(await data_cleaning_chat(prompt)).strip()
        data = parse_json_object(reply) if reply.startswith("{") else _parse_json_array(reply)
    else:
        # 记忆提取输出是 JSON 数组：包一层对象走 aux_chat_json（含容错解析与一次重试）
        wrapped = await aux_chat_json(
            config,
            prompt=prompt + '\n请把 JSON 数组包成 {"items": [...]} 对象输出。',
            model=config.ai_companion_member_mimic_model,
            max_tokens=800,
            temperature=0.0,
            timeout_seconds=30.0,
        )
        data = wrapped.get("items") if isinstance(wrapped.get("items"), list) else []
    written: list[dict[str, Any]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        content = str(item.get("content") or "").strip()
        if kind not in KINDS or not content:
            continue
        try:
            importance = float(item.get("importance") or 2.0)
        except (TypeError, ValueError):
            importance = 2.0
        result = await memories.upsert(user_id=user_id, kind=kind, content=content, importance=importance)
        written.append({"kind": kind, "content": content, "importance": importance, **result})
    return written


def _parse_json_array(text: str) -> list[Any]:
    body = str(text or "").strip()
    start = body.find("[")
    end = body.rfind("]")
    if start >= 0 and end > start:
        body = body[start : end + 1]
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AuxLlmError(f"memory json parse failed: {exc!r}") from exc
    return data if isinstance(data, list) else []
