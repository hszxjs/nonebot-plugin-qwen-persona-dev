"""会话时间线域：群/私聊的对话记录是"谁在什么时候说了什么"的一等公民。

与 `chat_samples` 的区别：samples 是**训练语料**（去重、跳过 bot 自己的话、
可被数据集导出消费）；conversation_turns 是**对话历史**（保留顺序与说话人、
bot 自己的回复也在里面、按会话键作用域），供回复时拼上下文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SPEAKER_MEMBER = "member"
SPEAKER_BOT = "bot"


def session_key_for(*, group_id: str | None, user_id: str | None) -> str:
    """会话键：群聊以 group 为主，私聊以 private:user 为主（与角色卡绑定同构）。"""
    if group_id:
        return f"group:{group_id}"
    if user_id:
        return f"private:{user_id}"
    return "private:unknown"


@dataclass(frozen=True)
class ConversationTurn:
    session_key: str
    speaker_kind: str
    text: str = ""
    group_id: str | None = None
    user_id: str | None = None
    speaker_name: str = ""
    sticker_keys: list[str] = field(default_factory=list)
    trigger_reason: str = ""
    reply_to_message_id: str = ""
    source_message_id: str = ""
    created_at: float = field(default_factory=time.time)
    id: int | None = None

    @property
    def is_bot(self) -> bool:
        return self.speaker_kind == SPEAKER_BOT

    def to_row(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "group_id": self.group_id,
            "user_id": self.user_id,
            "speaker_kind": self.speaker_kind,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "sticker_keys": list(self.sticker_keys),
            "trigger_reason": self.trigger_reason,
            "reply_to_message_id": self.reply_to_message_id,
            "source_message_id": self.source_message_id,
            "created_at": float(self.created_at),
        }


def turn_from_row(row: Any) -> ConversationTurn:
    import json

    try:
        raw_keys = row["sticker_keys"]
    except (KeyError, IndexError):
        raw_keys = "[]"
    try:
        keys = [str(item) for item in json.loads(raw_keys or "[]")]
    except (TypeError, ValueError):
        keys = []
    return ConversationTurn(
        id=int(row["id"]) if row["id"] is not None else None,
        session_key=str(row["session_key"] or ""),
        group_id=row["group_id"],
        user_id=row["user_id"],
        speaker_kind=str(row["speaker_kind"] or SPEAKER_MEMBER),
        speaker_name=str(row["speaker_name"] or ""),
        text=str(row["text"] or ""),
        sticker_keys=keys,
        trigger_reason=str(row["trigger_reason"] or ""),
        reply_to_message_id=str(row["reply_to_message_id"] or ""),
        source_message_id=str(row["source_message_id"] or ""),
        created_at=float(row["created_at"] or 0.0),
    )
