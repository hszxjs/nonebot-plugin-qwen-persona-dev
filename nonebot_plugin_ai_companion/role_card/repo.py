"""角色卡域持久化：role_cards 与 session_role_bindings 仓库（增量表，见 memory/db.py DDL）。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from ..memory.db import PersonaDatabase
from .model import RoleCard, RoleCardCreate, RoleType, SessionRoleBinding, binding_row_to_model

_LIST_FIELDS = (
    "style_tags",
    "tone_examples",
    "catchphrases",
    "topic_preferences",
    "allowed_tools",
)


class RoleCardRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def list_cards(self, *, role_type: RoleType | None = None) -> list[RoleCard]:
        def _sync() -> list[RoleCard]:
            self.db.ensure_schema()
            where = "WHERE role_type = ?" if role_type else ""
            params: tuple[Any, ...] = (str(role_type),) if role_type else ()
            with self.db.connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM role_cards {where} ORDER BY updated_at DESC",
                    params,
                ).fetchall()
            return [self._row_to_card(row) for row in rows]

        return await asyncio.to_thread(_sync)

    async def get_card(self, card_id: str) -> RoleCard | None:
        def _sync() -> RoleCard | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute("SELECT * FROM role_cards WHERE id = ?", (card_id,)).fetchone()
            return self._row_to_card(row) if row else None

        return await asyncio.to_thread(_sync)

    async def get_member_mimic_by_user(self, user_id: str) -> RoleCard | None:
        def _sync() -> RoleCard | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM role_cards
                    WHERE role_type = ? AND target_user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (RoleType.MEMBER_MIMIC, str(user_id)),
                ).fetchone()
            return self._row_to_card(row) if row else None

        return await asyncio.to_thread(_sync)

    async def create_card(self, payload: RoleCardCreate) -> RoleCard:
        card = RoleCard(
            id=uuid4().hex,
            role_type=payload.role_type,
            name=payload.name,
            system_prompt=payload.system_prompt,
            style_tags=list(payload.style_tags),
            tone_examples=list(payload.tone_examples),
            allowed_tools=list(payload.allowed_tools),
            avatar=payload.avatar,
            enabled=payload.enabled,
            target_user_id=payload.target_user_id,
        )
        await self._upsert(card)
        return card

    async def update_card(self, card_id: str, payload: RoleCardCreate) -> RoleCard | None:
        existing = await self.get_card(card_id)
        if existing is None:
            return None
        updated = existing.model_copy(
            update={
                "name": payload.name,
                "system_prompt": payload.system_prompt,
                "style_tags": list(payload.style_tags),
                "tone_examples": list(payload.tone_examples),
                "allowed_tools": list(payload.allowed_tools),
                "avatar": payload.avatar,
                "enabled": payload.enabled,
                "updated_at": time.time(),
            }
        )
        await self._upsert(updated)
        return updated

    async def patch_style_fields(
        self,
        card_id: str,
        *,
        system_prompt: str | None = None,
        style_tags: list[str] | None = None,
        tone_examples: list[str] | None = None,
        catchphrases: list[str] | None = None,
        topic_preferences: list[str] | None = None,
        emotion_baseline: str | None = None,
        relation_profile: str | None = None,
    ) -> RoleCard | None:
        """member_mimic 风格刷新：仅更新风格字段，不动身份与权限字段。"""
        existing = await self.get_card(card_id)
        if existing is None:
            return None
        updates: dict[str, Any] = {"updated_at": time.time()}
        if system_prompt is not None:
            updates["system_prompt"] = system_prompt
        if style_tags is not None:
            updates["style_tags"] = style_tags
        if tone_examples is not None:
            updates["tone_examples"] = tone_examples
        if catchphrases is not None:
            updates["catchphrases"] = catchphrases
        if topic_preferences is not None:
            updates["topic_preferences"] = topic_preferences
        if emotion_baseline is not None:
            updates["emotion_baseline"] = emotion_baseline
        if relation_profile is not None:
            updates["relation_profile"] = relation_profile
        updated = existing.model_copy(update=updates)
        await self._upsert(updated)
        return updated

    async def set_enabled(self, card_id: str, *, enabled: bool) -> RoleCard | None:
        existing = await self.get_card(card_id)
        if existing is None:
            return None
        updated = existing.model_copy(update={"enabled": enabled, "updated_at": time.time()})
        await self._upsert(updated)
        return updated

    async def delete_card(self, card_id: str) -> bool:
        def _sync() -> bool:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute("DELETE FROM role_cards WHERE id = ?", (card_id,))
                # 绑定到该卡的会话一并解绑，避免悬空引用。
                conn.execute("DELETE FROM session_role_bindings WHERE role_card_id = ?", (card_id,))
            return bool(cursor.rowcount)

        return await asyncio.to_thread(_sync)

    async def clone_card(self, card_id: str, *, name: str) -> RoleCard | None:
        existing = await self.get_card(card_id)
        if existing is None:
            return None
        clone = existing.model_copy(
            update={
                "id": uuid4().hex,
                "name": name,
                "enabled": False,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        await self._upsert(clone)
        return clone

    async def _upsert(self, card: RoleCard) -> None:
        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO role_cards (
                        id, role_type, name, system_prompt, style_tags, tone_examples,
                        catchphrases, topic_preferences, emotion_baseline, relation_profile,
                        allowed_tools, avatar, enabled, target_user_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        system_prompt=excluded.system_prompt,
                        style_tags=excluded.style_tags,
                        tone_examples=excluded.tone_examples,
                        catchphrases=excluded.catchphrases,
                        topic_preferences=excluded.topic_preferences,
                        emotion_baseline=excluded.emotion_baseline,
                        relation_profile=excluded.relation_profile,
                        allowed_tools=excluded.allowed_tools,
                        avatar=excluded.avatar,
                        enabled=excluded.enabled,
                        target_user_id=excluded.target_user_id,
                        updated_at=excluded.updated_at
                    """,
                    self._card_to_row(card),
                )

        await asyncio.to_thread(_sync)

    def _card_to_row(self, card: RoleCard) -> tuple[Any, ...]:
        return (
            card.id,
            str(card.role_type),
            card.name,
            card.system_prompt,
            json.dumps(card.style_tags, ensure_ascii=False),
            json.dumps(card.tone_examples, ensure_ascii=False),
            json.dumps(card.catchphrases, ensure_ascii=False),
            json.dumps(card.topic_preferences, ensure_ascii=False),
            card.emotion_baseline,
            card.relation_profile,
            json.dumps(card.allowed_tools, ensure_ascii=False),
            card.avatar,
            1 if card.enabled else 0,
            card.target_user_id,
            card.created_at,
            card.updated_at,
        )

    def _row_to_card(self, row: Any) -> RoleCard:
        data = dict(row)
        for field in _LIST_FIELDS:
            try:
                parsed = json.loads(data.get(field) or "[]")
            except (TypeError, json.JSONDecodeError):
                parsed = []
            data[field] = parsed if isinstance(parsed, list) else []
        data["enabled"] = bool(data.get("enabled"))
        return RoleCard.model_validate(data)


class SessionRoleBindingRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def list_bindings(self) -> list[SessionRoleBinding]:
        def _sync() -> list[SessionRoleBinding]:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                rows = conn.execute("SELECT session_key, role_card_id, bound_at FROM session_role_bindings").fetchall()
            return [binding_row_to_model(row) for row in rows]

        return await asyncio.to_thread(_sync)

    async def get_binding(self, session_key: str) -> SessionRoleBinding | None:
        def _sync() -> SessionRoleBinding | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT session_key, role_card_id, bound_at
                    FROM session_role_bindings
                    WHERE session_key = ?
                    """,
                    (session_key,),
                ).fetchone()
            return binding_row_to_model(row) if row else None

        return await asyncio.to_thread(_sync)

    async def bind(self, session_key: str, *, role_card_id: str) -> SessionRoleBinding:
        binding = SessionRoleBinding(
            session_key=session_key,
            role_card_id=role_card_id,
            bound_at=time.time(),
        )

        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO session_role_bindings (session_key, role_card_id, bound_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET
                        role_card_id=excluded.role_card_id,
                        bound_at=excluded.bound_at
                    """,
                    (binding.session_key, binding.role_card_id, binding.bound_at),
                )

        await asyncio.to_thread(_sync)
        return binding

    async def unbind(self, session_key: str) -> bool:
        def _sync() -> bool:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute("DELETE FROM session_role_bindings WHERE session_key = ?", (session_key,))
            return bool(cursor.rowcount)

        return await asyncio.to_thread(_sync)
