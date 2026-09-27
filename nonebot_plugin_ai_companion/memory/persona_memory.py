"""群记忆 / 成员记忆仓库：两表分工、敏感度、重要度衰减、抽取进度。

- `group_memories`：这个群的事（设定/事实/梗/事件/约定），按 group_id 作用域。
- `member_memories`：某个人的事（称呼/偏好/身份/关系/事件），按 (group_id, user_id)。
- 注入分级：群聊只允许 neutral/relation，私聊才允许 emotion/private（见 `prompt_lines_for_session`）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .db import PersonaDatabase

logger = logging.getLogger(__name__)

# 记忆类别：抽取端只做白名单校验，不强行归类（LLM 输出多样，过窄会大量丢弃）。
GROUP_KINDS = ("设定", "事实", "梗", "事件", "约定")
MEMBER_KINDS = ("称呼", "偏好", "身份", "关系", "事件")
ALL_KINDS = tuple(dict.fromkeys((*GROUP_KINDS, *MEMBER_KINDS)))

SENSITIVITY_NEUTRAL = "neutral"
SENSITIVITY_RELATION = "relation"
SENSITIVITY_EMOTION = "emotion"
SENSITIVITY_PRIVATE = "private"
SENSITIVITIES = (SENSITIVITY_NEUTRAL, SENSITIVITY_RELATION, SENSITIVITY_EMOTION, SENSITIVITY_PRIVATE)
# 群聊（公开场合）允许注入的敏感度：@ 不等于私密，情绪/隐私类只在私聊注入。
GROUP_VISIBLE_SENSITIVITIES = (SENSITIVITY_NEUTRAL, SENSITIVITY_RELATION)

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_DECAY_FLOOR = 0.4
DEDUP_SIMILARITY = 0.6


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    scope: str  # "group" | "member"
    group_id: str
    user_id: str = ""
    kind: str = ""
    content: str = ""
    importance: float = 1.0
    sensitivity: str = SENSITIVITY_NEUTRAL
    created_at: float = 0.0
    last_seen_at: float = 0.0
    decayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "group_id": self.group_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "content": self.content,
            "importance": round(float(self.importance), 2),
            "sensitivity": self.sensitivity,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "decayed": bool(self.decayed),
        }


@dataclass
class ExtractionState:
    session_key: str
    last_turn_id: int = 0
    last_run_at: float = 0.0
    updated_at: float = 0.0
    new_turns: int = 0
    pending_turns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _normalize_content(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _similar(left: str, right: str) -> float:
    """字符 2-gram Jaccard 相似度（沿用长期记忆的去重口径，中文短句够用）。"""
    a = _normalize_content(left)
    b = _normalize_content(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    gram_a = {a[i : i + 2] for i in range(len(a) - 1)}
    gram_b = {b[i : i + 2] for i in range(len(b) - 1)}
    if not gram_a or not gram_b:
        return 0.0
    return len(gram_a & gram_b) / len(gram_a | gram_b)


class PersonaMemoryRepo:
    """群记忆 / 成员记忆的统一仓库（两表共享一套去重、衰减、注入逻辑）。"""

    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    # ---- 写入 ----
    async def add_group_memory(
        self,
        *,
        group_id: str,
        kind: str,
        content: str,
        importance: float = 1.0,
        sensitivity: str = SENSITIVITY_NEUTRAL,
    ) -> bool:
        return await asyncio.to_thread(
            self._add_sync, "group", str(group_id or ""), "", kind, content, importance, sensitivity
        )

    async def add_member_memory(
        self,
        *,
        group_id: str,
        user_id: str,
        kind: str,
        content: str,
        importance: float = 1.0,
        sensitivity: str = SENSITIVITY_NEUTRAL,
    ) -> bool:
        return await asyncio.to_thread(
            self._add_sync, "member", str(group_id or ""), str(user_id or ""), kind, content, importance, sensitivity
        )

    def _add_sync(
        self,
        scope: str,
        group_id: str,
        user_id: str,
        kind: str,
        content: str,
        importance: float,
        sensitivity: str,
    ) -> bool:
        text = _normalize_content(content)
        if not text:
            return False
        if scope == "group" and not group_id:
            return False  # 群记忆必须有群号
        if scope == "member" and not user_id:
            return False
        # 注意：成员记忆允许 group_id 为空串——表示「私聊里形成」的记忆，与任何群无关
        clean_kind = str(kind or "").strip()[:12] or "事实"
        clean_sensitivity = sensitivity if sensitivity in SENSITIVITIES else SENSITIVITY_NEUTRAL
        try:
            importance_value = min(5.0, max(1.0, float(importance)))
        except (TypeError, ValueError):
            importance_value = 1.0

        self.db.ensure_schema()
        now = time.time()
        table = "group_memories" if scope == "group" else "member_memories"
        where = "group_id = ?" if scope == "group" else "group_id = ? AND user_id = ?"
        params: tuple[Any, ...] = (group_id,) if scope == "group" else (group_id, user_id)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT id, content, importance FROM {table} WHERE {where} AND kind = ? AND decayed = 0",
                (*params, clean_kind),
            ).fetchall()
            for row in rows:
                if _similar(str(row["content"]), text) >= DEDUP_SIMILARITY:
                    # 合并：重要度取大、刷新 last_seen（不新增行）
                    conn.execute(
                        f"UPDATE {table} SET importance = ?, last_seen_at = ? WHERE id = ?",
                        (max(importance_value, float(row["importance"] or 1.0)), now, int(row["id"])),
                    )
                    return False
            if scope == "member":
                conn.execute(
                    """
                    INSERT INTO member_memories (
                        group_id, user_id, kind, content, importance, sensitivity, created_at, last_seen_at, decayed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (group_id, user_id, clean_kind, text, importance_value, clean_sensitivity, now, now),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO group_memories (
                        group_id, kind, content, importance, sensitivity, created_at, last_seen_at, decayed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (group_id, clean_kind, text, importance_value, clean_sensitivity, now, now),
                )
        return True

    # ---- 查询 ----
    async def list_memories(
        self,
        *,
        scope: str = "group",
        group_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        include_decayed: bool = False,
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._list_sync, scope, group_id, user_id, max(1, int(limit)), include_decayed)

    def _list_sync(
        self,
        scope: str,
        group_id: str | None,
        user_id: str | None,
        limit: int,
        include_decayed: bool,
    ) -> list[MemoryRecord]:
        self.db.ensure_schema()
        table = "group_memories" if scope == "group" else "member_memories"
        clauses: list[str] = []
        params: list[Any] = []
        if group_id:
            clauses.append("group_id = ?")
            params.append(str(group_id))
        if scope == "member" and user_id:
            clauses.append("user_id = ?")
            params.append(str(user_id))
        if not include_decayed:
            clauses.append("decayed = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM {table} {where}
                ORDER BY importance DESC, last_seen_at DESC LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_record(row, scope) for row in rows]

    @staticmethod
    def _row_to_record(row: Any, scope: str) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            scope=scope,
            group_id=str(row["group_id"] or ""),
            user_id=str(row["user_id"] or "") if scope == "member" else "",
            kind=str(row["kind"] or ""),
            content=str(row["content"] or ""),
            importance=float(row["importance"] or 1.0),
            sensitivity=str(row["sensitivity"] or SENSITIVITY_NEUTRAL),
            created_at=float(row["created_at"] or 0.0),
            last_seen_at=float(row["last_seen_at"] or 0.0),
            decayed=bool(row["decayed"]),
        )

    async def prompt_lines_for_session(
        self,
        *,
        group_id: str | None,
        user_ids: list[str],
        include_sensitive: bool,
        group_limit: int = 5,
        member_limit: int = 4,
    ) -> list[str]:
        """拼注入用的记忆行：群记忆 + 在场成员记忆。

        群聊（include_sensitive=False）只给 neutral/relation；私聊给全部敏感度。
        """
        return await asyncio.to_thread(
            self._prompt_lines_sync, group_id, list(user_ids), include_sensitive, group_limit, member_limit
        )

    def _prompt_lines_sync(
        self,
        group_id: str | None,
        user_ids: list[str],
        include_sensitive: bool,
        group_limit: int,
        member_limit: int,
    ) -> list[str]:
        if not group_id:
            return []
        allowed = list(SENSITIVITIES) if include_sensitive else list(GROUP_VISIBLE_SENSITIVITIES)
        placeholders = ",".join("?" for _ in allowed)
        lines: list[str] = []
        self.db.ensure_schema()
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT kind, content, importance FROM group_memories
                WHERE group_id = ? AND decayed = 0 AND sensitivity IN ({placeholders})
                ORDER BY importance DESC, last_seen_at DESC LIMIT ?
                """,
                (group_id, *allowed, max(1, int(group_limit))),
            ).fetchall()
            for row in rows:
                lines.append(f"本群{row['kind']}｜{row['content']}（重要度 {float(row['importance']):.0f}/5）")
            clean_ids = [str(item) for item in user_ids if str(item).strip()]
            if clean_ids:
                id_placeholders = ",".join("?" for _ in clean_ids)
                rows = conn.execute(
                    f"""
                    SELECT user_id, kind, content, importance FROM member_memories
                    WHERE group_id = ? AND decayed = 0 AND sensitivity IN ({placeholders})
                      AND user_id IN ({id_placeholders})
                    ORDER BY importance DESC, last_seen_at DESC LIMIT ?
                    """,
                    (group_id, *allowed, *clean_ids, max(1, int(member_limit))),
                ).fetchall()
                for row in rows:
                    names = conn.execute(
                        "SELECT name FROM member_names WHERE group_id = ? AND user_id = ?",
                        (group_id, str(row["user_id"])),
                    ).fetchone()
                    who = str(names["name"]) if names and names["name"] else f"群友{str(row['user_id'])[-4:]}"
                    lines.append(f"{who}的{row['kind']}｜{row['content']}（重要度 {float(row['importance']):.0f}/5）")
        return lines

    async def prompt_lines_for_private_member(
        self,
        *,
        user_id: str,
        include_sensitive: bool = True,
        limit: int = 5,
    ) -> list[str]:
        """私聊场景：把这个人跨群的成员记忆聚合出来（私聊才允许情绪/隐私类）。

        成员记忆天然带群号（它是在某个群里形成的），私聊没有群作用域，
        所以这里按 user_id 跨群取，渲染时带上来源群，避免张冠李戴。
        """
        return await asyncio.to_thread(
            self._prompt_lines_for_private_member_sync, str(user_id or ""), bool(include_sensitive), max(1, int(limit))
        )

    def _prompt_lines_for_private_member_sync(self, user_id: str, include_sensitive: bool, limit: int) -> list[str]:
        if not user_id:
            return []
        allowed = list(SENSITIVITIES) if include_sensitive else list(GROUP_VISIBLE_SENSITIVITIES)
        placeholders = ",".join("?" for _ in allowed)
        self.db.ensure_schema()
        lines: list[str] = []
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT group_id, kind, content, importance FROM member_memories
                WHERE user_id = ? AND decayed = 0 AND sensitivity IN ({placeholders})
                ORDER BY importance DESC, last_seen_at DESC LIMIT ?
                """,
                (user_id, *allowed, limit),
            ).fetchall()
        for row in rows:
            origin = str(row["group_id"] or "")
            if origin:
                lines.append(f"你在群 {origin} 认识的 TA 的{row['kind']}｜{row['content']}")
            else:
                lines.append(f"你们私聊里聊过的 TA 的{row['kind']}｜{row['content']}")
        return lines

    # ---- 维护 ----
    async def update_memory(
        self,
        *,
        scope: str,
        memory_id: int,
        content: str | None = None,
        importance: float | None = None,
        sensitivity: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._update_sync, scope, int(memory_id), content, importance, sensitivity)

    def _update_sync(
        self,
        scope: str,
        memory_id: int,
        content: str | None,
        importance: float | None,
        sensitivity: str | None,
    ) -> bool:
        updates: dict[str, Any] = {}
        if content is not None:
            text = _normalize_content(content)
            if not text:
                return False
            updates["content"] = text
        if importance is not None:
            try:
                updates["importance"] = min(5.0, max(1.0, float(importance)))
            except (TypeError, ValueError):
                return False
        if sensitivity is not None:
            if sensitivity not in SENSITIVITIES:
                return False
            updates["sensitivity"] = sensitivity
        if not updates:
            return False
        table = "group_memories" if scope == "group" else "member_memories"
        assignments = ", ".join(f"{key} = ?" for key in updates)
        self.db.ensure_schema()
        with self.db.connection() as conn:
            cursor = conn.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                (*updates.values(), memory_id),
            )
            return bool(cursor.rowcount)

    async def delete_memory(self, *, scope: str, memory_id: int) -> bool:
        return await asyncio.to_thread(self._delete_sync, scope, int(memory_id))

    def _delete_sync(self, scope: str, memory_id: int) -> bool:
        table = "group_memories" if scope == "group" else "member_memories"
        self.db.ensure_schema()
        with self.db.connection() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))
            return bool(cursor.rowcount)

    async def purge_member(self, *, group_id: str, user_id: str) -> int:
        """一键清除某成员的记忆（可删可控的兜底手段）。"""
        return await asyncio.to_thread(self._purge_member_sync, str(group_id or ""), str(user_id or ""))

    def _purge_member_sync(self, group_id: str, user_id: str) -> int:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            cursor = conn.execute("DELETE FROM member_memories WHERE group_id = ? AND user_id = ?", (group_id, user_id))
            return int(cursor.rowcount or 0)

    async def decay(
        self,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        floor: float = DEFAULT_DECAY_FLOOR,
    ) -> dict[str, int]:
        """按半衰期衰减重要度，低于阈值标记 decayed（幂等可重复跑）。"""
        return await asyncio.to_thread(self._decay_sync, float(half_life_days), float(floor))

    def _decay_sync(self, half_life_days: float, floor: float) -> dict[str, int]:
        half_life_seconds = max(3600.0, half_life_days * 86400.0)
        now = time.time()
        touched = decayed_total = 0
        self.db.ensure_schema()
        with self.db.connection() as conn:
            for table in ("group_memories", "member_memories"):
                rows = conn.execute(
                    f"SELECT id, importance, created_at, last_seen_at FROM {table} WHERE decayed = 0"
                ).fetchall()
                for row in rows:
                    created_at = float(row["created_at"] or now)
                    last_seen_at = float(row["last_seen_at"] or created_at)
                    age = max(0.0, now - created_at)
                    decayed_value = float(row["importance"] or 1.0) * (0.5 ** (age / half_life_seconds))
                    fresh_value = float(row["importance"] or 1.0) * (
                        0.5 ** (max(0.0, now - last_seen_at) / half_life_seconds)
                    )
                    value = max(decayed_value, fresh_value)
                    if value < floor:
                        conn.execute(f"UPDATE {table} SET decayed = 1 WHERE id = ?", (int(row["id"]),))
                        decayed_total += 1
                    else:
                        conn.execute(f"UPDATE {table} SET importance = ? WHERE id = ?", (value, int(row["id"])))
                        touched += 1
        return {"updated": touched, "decayed": decayed_total}

    # ---- 抽取进度 ----
    async def get_extraction_states(self) -> dict[str, ExtractionState]:
        return await asyncio.to_thread(self._get_extraction_states_sync)

    def _get_extraction_states_sync(self) -> dict[str, ExtractionState]:
        self.db.ensure_schema()
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM memory_extraction_state").fetchall()
        return {
            str(row["session_key"]): ExtractionState(
                session_key=str(row["session_key"]),
                last_turn_id=int(row["last_turn_id"] or 0),
                last_run_at=float(row["last_run_at"] or 0.0),
                updated_at=float(row["updated_at"] or 0.0),
            )
            for row in rows
        }

    async def mark_extracted(self, *, session_key: str, last_turn_id: int, ran_at: float | None = None) -> None:
        await asyncio.to_thread(self._mark_extracted_sync, str(session_key), int(last_turn_id), ran_at)

    def _mark_extracted_sync(self, session_key: str, last_turn_id: int, ran_at: float | None) -> None:
        now = time.time()
        self.db.ensure_schema()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_extraction_state (session_key, last_turn_id, last_run_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    last_turn_id = excluded.last_turn_id,
                    last_run_at = excluded.last_run_at,
                    updated_at = excluded.updated_at
                """,
                (session_key, last_turn_id, float(ran_at if ran_at is not None else now), now),
            )


__all__ = [
    "ALL_KINDS",
    "DEFAULT_DECAY_FLOOR",
    "DEFAULT_HALF_LIFE_DAYS",
    "ExtractionState",
    "GROUP_KINDS",
    "GROUP_VISIBLE_SENSITIVITIES",
    "MEMBER_KINDS",
    "MemoryRecord",
    "PersonaMemoryRepo",
    "SENSITIVITIES",
    "SENSITIVITY_EMOTION",
    "SENSITIVITY_NEUTRAL",
    "SENSITIVITY_PRIVATE",
    "SENSITIVITY_RELATION",
]
