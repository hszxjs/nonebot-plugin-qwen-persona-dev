"""消息溯源仓库（文档十三节）：MessageLog 记录 trigger/reply 平台消息 ID 及关联上下文。

- trigger_message_id：触发本次回复的原始消息平台 ID
- reply_message_id：Bot 回复产生的消息平台 ID（/trace 据此反查）
- 关联当时生效的角色卡、决策原因、Provider/模型、耗时、贴纸
- 环形淘汰：按配置的保留条数滚动清理
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ..memory.db import PersonaDatabase

DEFAULT_MAX_RECORDS = 1000


class MessageLogRepo:
    def __init__(
        self,
        db: PersonaDatabase,
        *,
        max_records: int = DEFAULT_MAX_RECORDS,
        bus: Any | None = None,
    ) -> None:
        self.db = db
        self.max_records = max(1, int(max_records))
        self.bus = bus

    def set_max_records(self, max_records: int) -> None:
        self.max_records = max(1, int(max_records))

    async def record(
        self,
        *,
        platform: str | None,
        group_id: str | None,
        user_id: str | None,
        trigger_message_id: str | None,
        reply_message_id: str | None,
        reply_kind: str = "chat",
        trigger_reason: str = "",
        decision_detail: dict[str, Any] | None = None,
        role_card_id: str | None = None,
        role_card_name: str | None = None,
        provider: str = "",
        model: str = "",
        latency_seconds: float | None = None,
        status: str = "success",
        sticker_keys: list[str] | None = None,
        input_text: str = "",
        output_text: str = "",
        created_at: float | None = None,
    ) -> dict[str, Any]:
        item = {
            "created_at": float(created_at if created_at is not None else time.time()),
            "platform": platform,
            "group_id": group_id,
            "user_id": user_id,
            "trigger_message_id": str(trigger_message_id or "") or None,
            "reply_message_id": str(reply_message_id or "") or None,
            "reply_kind": str(reply_kind or "chat"),
            "trigger_reason": str(trigger_reason or ""),
            "decision_detail": decision_detail if isinstance(decision_detail, dict) else {},
            "role_card_id": role_card_id,
            "role_card_name": role_card_name,
            "provider": str(provider or ""),
            "model": str(model or ""),
            "latency_seconds": None if latency_seconds is None else float(latency_seconds),
            "status": str(status or "success"),
            "sticker_keys": [str(key) for key in sticker_keys or []],
            "input_preview": _preview(input_text),
            "output_preview": _preview(output_text),
        }

        def _sync() -> None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO message_logs (
                        created_at, platform, group_id, user_id, trigger_message_id,
                        reply_message_id, reply_kind, trigger_reason, decision_detail,
                        role_card_id, role_card_name, provider, model, latency_seconds,
                        status, sticker_keys, input_preview, output_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["created_at"],
                        item["platform"],
                        item["group_id"],
                        item["user_id"],
                        item["trigger_message_id"],
                        item["reply_message_id"],
                        item["reply_kind"],
                        item["trigger_reason"],
                        json.dumps(item["decision_detail"], ensure_ascii=False, default=str),
                        item["role_card_id"],
                        item["role_card_name"],
                        item["provider"],
                        item["model"],
                        item["latency_seconds"],
                        item["status"],
                        json.dumps(item["sticker_keys"], ensure_ascii=False),
                        item["input_preview"],
                        item["output_preview"],
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM message_logs
                    WHERE id NOT IN (
                        SELECT id FROM message_logs ORDER BY created_at DESC LIMIT ?
                    )
                    """,
                    (self.max_records,),
                )

        await asyncio.to_thread(_sync)
        if self.bus is not None:
            self.bus.publish({"type": "message", "item": self._with_time(item)})
        return self._with_time(item)

    async def list_logs(
        self,
        *,
        kind: str = "",
        group_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        def _sync() -> dict[str, Any]:
            self.db.ensure_schema()
            clauses: list[str] = []
            params: list[Any] = []
            if kind and kind != "all":
                clauses.append("reply_kind = ?")
                params.append(kind)
            if group_id:
                clauses.append("group_id = ?")
                params.append(group_id)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            with self.db.connection() as conn:
                total = int(conn.execute(f"SELECT COUNT(*) FROM message_logs {where}", params).fetchone()[0])
                rows = conn.execute(
                    f"""
                    SELECT * FROM message_logs {where}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*params, max(1, min(int(limit or 100), 500)), max(0, int(offset or 0))),
                ).fetchall()
            return {"items": [self._row_to_item(row) for row in rows], "total": total}

        return await asyncio.to_thread(_sync)

    async def find_by_reply_message_id(self, reply_message_id: str) -> dict[str, Any] | None:
        return await self._find_by_id_field("reply_message_id", reply_message_id)

    async def find_by_trigger_message_id(self, trigger_message_id: str) -> dict[str, Any] | None:
        return await self._find_by_id_field("trigger_message_id", trigger_message_id)

    async def _find_by_id_field(self, field: str, value: str) -> dict[str, Any] | None:
        assert field in {"reply_message_id", "trigger_message_id"}

        def _sync() -> dict[str, Any] | None:
            self.db.ensure_schema()
            with self.db.connection() as conn:
                row = conn.execute(
                    f"SELECT * FROM message_logs WHERE {field} = ? ORDER BY created_at DESC LIMIT 1",
                    (str(value or ""),),
                ).fetchone()
            return self._row_to_item(row) if row else None

        return await asyncio.to_thread(_sync)

    def _row_to_item(self, row: Any) -> dict[str, Any]:
        data = dict(row)
        data["decision_detail"] = self.db.json_object(data.pop("decision_detail", "{}"))
        data["sticker_keys"] = [str(key) for key in self.db.json_list(data.get("sticker_keys")) if str(key).strip()]
        return self._with_time(data)

    @staticmethod
    def _with_time(item: dict[str, Any]) -> dict[str, Any]:
        created_at = float(item.get("created_at") or 0)
        item["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at))
        return item


def _preview(text: str, *, limit: int = 240) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


def format_trace(item: dict[str, Any]) -> str:
    """把 MessageLog 记录渲染为 /trace 回复文本。"""
    latency = item.get("latency_seconds")
    latency_text = f"{float(latency):.1f}s" if latency is not None else "-"
    card = item.get("role_card_name") or "默认人格"
    stickers = item.get("sticker_keys") or []
    lines = [
        "溯源结果：",
        f"触发原因：{item.get('trigger_reason') or '-'}",
        f"回复类型：{item.get('reply_kind') or '-'}",
        f"角色卡：{card}",
        f"Provider：{item.get('provider') or '-'} / {item.get('model') or '-'}",
        f"耗时：{latency_text}",
        f"携带表情：{len(stickers)} 个" if stickers else "携带表情：无",
    ]
    detail = item.get("decision_detail") or {}
    if detail:
        lines.append(f"决策明细：{json.dumps(detail, ensure_ascii=False)[:200]}")
    preview = str(item.get("output_preview") or "").strip()
    if preview:
        lines.append(f"回复预览：{preview[:120]}")
    return "\n".join(lines)


def quoted_message_id(event: Any) -> str | None:
    """提取事件引用消息的平台 ID（reply 对象或 reply 消息段；/trace、/添加表情 用）。"""
    reply_obj = getattr(event, "reply", None)
    if reply_obj is not None:
        quoted = getattr(reply_obj, "message_id", None)
        if quoted:
            return str(quoted)
    try:
        for segment in event.get_message():
            if str(getattr(segment, "type", "")) == "reply":
                value = str(segment.data.get("id") or "")
                if value:
                    return value
    except Exception:
        return None
    return None
