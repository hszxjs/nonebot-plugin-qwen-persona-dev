"""LoRA adapter 版本管理（文档十一节）：与「角色卡 + 源语言」绑定。

同一角色卡可按语种分别训练多个 adapter；推理时按消息语言选择匹配的
adapter，未匹配回退基座（不加载 adapter）。激活/回滚 = 切换 active 标记
并同步到 MLX 运行时的 adapter 路径。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..memory.db import PersonaDatabase

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def detect_language(text: str) -> str:
    """粗粒度语言检测：zh / en / zh-mixed-en / unknown。"""
    value = str(text or "")
    if not value.strip():
        return "unknown"
    cjk = len(_CJK_RE.findall(value))
    latin = len(_LATIN_WORD_RE.findall(value))
    if cjk and latin >= 2:
        return "zh-mixed-en"
    if cjk:
        return "zh"
    if latin:
        return "en"
    return "unknown"


class LoraAdapter(BaseModel):
    id: str
    role_card_id: str
    source_language: str
    version: int
    path: str
    note: str = ""
    active: bool = False
    created_at: float = Field(default_factory=time.time)


class LoraAdapterRepo:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def list_adapters(self, *, role_card_id: str | None = None) -> list[LoraAdapter]:
        def _sync() -> list[LoraAdapter]:
            self.db.ensure_schema()
            where = "WHERE role_card_id = ?" if role_card_id else ""
            params: tuple[Any, ...] = (role_card_id,) if role_card_id else ()
            with self.db.connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM lora_adapters {where}
                    ORDER BY role_card_id, source_language, version DESC
                    """,
                    params,
                ).fetchall()
            return [self._row_to_adapter(row) for row in rows]

        return await asyncio.to_thread(_sync)

    async def register_adapter(
        self,
        *,
        role_card_id: str,
        source_language: str,
        path: str,
        note: str = "",
        activate: bool = False,
    ) -> LoraAdapter:
        """登记新版本（版本号在「角色卡+语种」维度自增）。"""

        def _sync() -> LoraAdapter:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(version) FROM lora_adapters
                    WHERE role_card_id = ? AND source_language = ?
                    """,
                    (role_card_id, source_language),
                ).fetchone()
                next_version = int(row[0] or 0) + 1
                adapter = LoraAdapter(
                    id=uuid4().hex,
                    role_card_id=role_card_id,
                    source_language=source_language,
                    version=next_version,
                    path=str(path),
                    note=str(note or ""),
                    active=False,
                )
                if activate:
                    conn.execute(
                        "UPDATE lora_adapters SET active = 0 WHERE role_card_id = ? AND source_language = ?",
                        (role_card_id, source_language),
                    )
                    adapter = adapter.model_copy(update={"active": True})
                conn.execute(
                    """
                    INSERT INTO lora_adapters (
                        id, role_card_id, source_language, version, path, note, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        adapter.id,
                        adapter.role_card_id,
                        adapter.source_language,
                        adapter.version,
                        adapter.path,
                        adapter.note,
                        1 if adapter.active else 0,
                        adapter.created_at,
                    ),
                )
            return adapter

        return await asyncio.to_thread(_sync)

    async def get_adapter(self, adapter_id: str) -> LoraAdapter | None:
        def _sync() -> LoraAdapter | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute("SELECT * FROM lora_adapters WHERE id = ?", (adapter_id,)).fetchone()
            return self._row_to_adapter(row) if row else None

        return await asyncio.to_thread(_sync)

    async def find_active(self, *, role_card_id: str, source_language: str) -> LoraAdapter | None:
        adapters = await self.list_adapters(role_card_id=role_card_id)
        for adapter in adapters:
            if adapter.source_language == source_language and adapter.active:
                return adapter
        return None

    async def activate(self, adapter_id: str) -> LoraAdapter | None:
        """激活指定版本（同卡同语种其余版本自动停用）；返回激活后的 adapter。"""
        target = await self.get_adapter(adapter_id)
        if target is None:
            return None

        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE lora_adapters SET active = 0 WHERE role_card_id = ? AND source_language = ?",
                    (target.role_card_id, target.source_language),
                )
                conn.execute("UPDATE lora_adapters SET active = 1 WHERE id = ?", (adapter_id,))

        await asyncio.to_thread(_sync)
        updated = await self.get_adapter(adapter_id)
        assert updated is not None
        return updated

    async def rollback(self, *, role_card_id: str, source_language: str) -> LoraAdapter | None:
        """回滚：激活「版本号小于当前激活版本」的最高版本；无更早版本返回 None。"""
        adapters = [
            adapter
            for adapter in await self.list_adapters(role_card_id=role_card_id)
            if adapter.source_language == source_language
        ]
        current = next((adapter for adapter in adapters if adapter.active), None)
        if current is None:
            return None
        earlier = [adapter for adapter in adapters if adapter.version < current.version]
        if not earlier:
            return None
        return await self.activate(earlier[0].id)  # list 倒序，earlier[0] 即最近的历史版本

    async def delete_adapter(self, adapter_id: str) -> bool:
        def _sync() -> bool:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                cursor = conn.execute("DELETE FROM lora_adapters WHERE id = ?", (adapter_id,))
            return bool(cursor.rowcount)

        return await asyncio.to_thread(_sync)

    @staticmethod
    def _row_to_adapter(row: Any) -> LoraAdapter:
        data = dict(row)
        data["active"] = bool(data.get("active"))
        return LoraAdapter.model_validate(data)
