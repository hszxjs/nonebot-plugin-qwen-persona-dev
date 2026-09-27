from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..prompt.text import normalize_text


@dataclass(frozen=True)
class MessageContextEnvelope:
    current_text: str
    quoted_texts: list[str]
    forwarded_texts: list[str]
    media_parts: list[dict[str, Any]]
    sticker_keys: list[str]
    forced_by_at: bool
    context_sources: list[str]
    # 主语指代：当前发言者、被@的人、被引用消息发送者
    sender_name: str = ""
    at_targets: list[dict[str, str]] = field(default_factory=list)
    reply_sender: dict[str, str] | None = None


def build_message_context(
    *,
    message: Any,
    event: Any,
    current_text: str,
    media_parts: list[dict[str, Any]],
    sticker_keys: list[str],
    forced_by_at: bool,
    at_nicknames: dict[str, str | None] | None = None,
    reply_nickname: str | None = None,
    sender_name: str = "",
) -> MessageContextEnvelope:
    quoted_texts = _texts_from_reply(event)
    forwarded_texts = _texts_from_forwarded_nodes(message)
    at_targets = _at_targets_from_message(message, at_nicknames or {})
    reply_sender = _reply_sender_from_event(event, reply_nickname)
    resolved_sender = sender_name or sender_name_from_event(event)
    sources: list[str] = []
    if normalize_text(current_text):
        sources.append("current")
    if quoted_texts:
        sources.append("quoted")
    if forwarded_texts:
        sources.append("forwarded")
    if media_parts:
        sources.append("media")
    if sticker_keys:
        sources.append("sticker")
    if forced_by_at:
        sources.append("force_reply")
    if at_targets:
        sources.append("at")
    if reply_sender:
        sources.append("reply_to")

    return MessageContextEnvelope(
        current_text=normalize_text(current_text),
        quoted_texts=quoted_texts,
        forwarded_texts=forwarded_texts,
        media_parts=list(media_parts),
        sticker_keys=[str(key) for key in sticker_keys if str(key).strip()],
        forced_by_at=bool(forced_by_at),
        context_sources=sources,
        sender_name=resolved_sender,
        at_targets=at_targets,
        reply_sender=reply_sender,
    )


def _texts_from_reply(event: Any) -> list[str]:
    reply = _get_field(event, "reply", None)
    if reply is None:
        return []
    message = _get_field(reply, "message", None)
    return _texts_from_message(message)


def _at_targets_from_message(
    message: Any,
    nicknames: dict[str, str | None],
) -> list[dict[str, str]]:
    """提取消息里所有 @ 的目标。返回 [{"qq": "12345", "name": "小明"}]。"""
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for segment in _iter_segments(message):
        if str(_get_field(segment, "type", "")).lower() != "at":
            continue
        data = _get_field(segment, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        qq = str(data.get("qq") or "").strip()
        if not qq or qq in seen:
            continue
        seen.add(qq)
        # 优先用预解析的昵称，其次用 OneBot 段里的 name 字段，最后留空（显示 qq）
        name = nicknames.get(qq) or str(data.get("name") or "").strip()
        targets.append({"qq": qq, "name": name or qq})
    return targets


def reply_sender_from_event(event: Any, nickname: str | None = None) -> dict[str, str] | None:
    """被引用消息的发送者（谁被回复了）；判定"有人在回复 bot 自己"也用它。"""
    return _reply_sender_from_event(event, nickname)


def _reply_sender_from_event(
    event: Any,
    nickname: str | None,
) -> dict[str, str] | None:
    """提取被引用消息的发送者。"""
    reply = _get_field(event, "reply", None)
    if reply is None:
        return None
    sender = _get_field(reply, "sender", None)
    user_id = ""
    nick = ""
    if isinstance(sender, dict):
        user_id = str(sender.get("user_id") or sender.get("qq") or "").strip()
        nick = str(sender.get("card") or sender.get("nickname") or "").strip()
    elif sender is not None:
        user_id = str(_get_field(sender, "user_id", "") or _get_field(sender, "qq", "") or "").strip()
        nick = str(_get_field(sender, "card", "") or _get_field(sender, "nickname", "") or "").strip()
    if not user_id and not nick:
        return None
    # 预解析的昵称优先
    final_nick = nickname or nick
    return {"user_id": user_id, "nickname": final_nick}


def sender_name_from_event(event: Any) -> str:
    """提取当前消息发送者的昵称（群名片优先）。会话时间线的昵称快照也用它。"""
    sender = _get_field(event, "sender", None)
    if isinstance(sender, dict):
        return str(sender.get("card") or sender.get("nickname") or "").strip()
    if sender is not None:
        return str(_get_field(sender, "card", "") or _get_field(sender, "nickname", "") or "").strip()
    return ""


def _texts_from_forwarded_nodes(message: Any) -> list[str]:
    texts: list[str] = []
    for segment in _iter_segments(message):
        if str(_get_field(segment, "type", "")).lower() != "node":
            continue
        data = _get_field(segment, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        for text_key in ("content", "message"):
            texts.extend(_texts_from_message(data.get(text_key)))
    return _dedupe_texts(texts)


def _texts_from_message(message: Any) -> list[str]:
    texts: list[str] = []
    if message is None:
        return texts
    if isinstance(message, str):
        cleaned = normalize_text(message)
        return [cleaned] if cleaned else []
    for segment in _iter_segments(message):
        segment_type = str(_get_field(segment, "type", "")).lower()
        data = _get_field(segment, "data", {}) or {}
        if segment_type == "text":
            text = data.get("text") if isinstance(data, dict) else None
            cleaned = normalize_text(str(text or ""))
            if cleaned:
                texts.append(cleaned)
            continue
        if segment_type == "node" and isinstance(data, dict):
            texts.extend(_texts_from_message(data.get("content") or data.get("message")))
    return _dedupe_texts(texts)


def _iter_segments(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, dict):
        return [message]
    if isinstance(message, str):
        return [{"type": "text", "data": {"text": message}}]
    try:
        return list(message)
    except TypeError:
        return [{"type": "text", "data": {"text": str(message or "")}}]


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _dedupe_texts(texts: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = normalize_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
