"""OneBot v11 平台投递：戳一戳、合并转发发送、表情消息构造。

所有 OneBot v11 专用代码收口在本模块，跨平台投递在第二期扩展。
"""

from __future__ import annotations

import logging
from typing import Any

from ..sticker.model import build_onebot_v11_sticker_message

logger = logging.getLogger(__name__)


def _str_id(value: Any) -> str:
    return str(value or "").strip()


def _int_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def onebot_numeric_id(value: Any) -> Any:
    text = str(value or "").strip()
    if text.isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return value


def is_poke_at_self(event: Any, self_id: str | int | None) -> bool:
    if _str_id(getattr(event, "notice_type", "")) != "notify":
        return False
    if _str_id(getattr(event, "sub_type", "")) != "poke":
        return False
    if not _str_id(getattr(event, "user_id", "")):
        return False
    target_id = _str_id(getattr(event, "target_id", ""))
    wanted = _str_id(self_id or getattr(event, "self_id", ""))
    return bool(target_id and wanted and target_id == wanted)


async def send_counter_poke(bot: Any, event: Any) -> bool:
    user_id = _int_id(getattr(event, "user_id", None))
    if user_id is None:
        return False

    group_id = _int_id(getattr(event, "group_id", None))
    if group_id is not None:
        api_candidates = (
            ("group_poke", {"group_id": group_id, "user_id": user_id}),
            ("send_group_poke", {"group_id": group_id, "user_id": user_id}),
            ("send_poke", {"group_id": group_id, "user_id": user_id}),
        )
    else:
        api_candidates = (
            ("friend_poke", {"user_id": user_id}),
            ("send_friend_poke", {"user_id": user_id}),
            ("send_poke", {"user_id": user_id}),
        )

    failures: list[str] = []
    for api_name, params in api_candidates:
        try:
            await bot.call_api(api_name, **params)
            return True
        except Exception as exc:
            failures.append(f"{api_name}: {type(exc).__name__}: {exc}")
            continue
    if failures:
        logger.warning("qwen persona counter poke failed: %s", "; ".join(failures))
    return False


async def send_onebot_v11_forward_reply(
    bot: Any,
    event: Any,
    chunks: list[str],
    *,
    self_id: str | None,
) -> bool:
    """以合并转发消息发送长回复；非 OneBot v11 事件或发送失败返回 False。"""
    if "onebot.v11" not in event.__class__.__module__:
        return False
    if not chunks or not hasattr(bot, "call_api"):
        return False
    try:
        from nonebot.adapters.onebot.v11 import Message, MessageSegment
    except Exception:
        return False

    try:
        node_user_id = onebot_numeric_id(self_id or "0")
        nickname = str(getattr(bot, "nickname", "") or self_id or "bot")
        nodes = []
        for chunk in chunks:
            content = Message()
            content += MessageSegment.text(chunk)
            nodes.append(
                MessageSegment.node_custom(
                    user_id=node_user_id,
                    nickname=nickname,
                    content=content,
                )
            )

        group_id = getattr(event, "group_id", None)
        if group_id is not None:
            await bot.call_api(
                "send_group_forward_msg",
                group_id=onebot_numeric_id(group_id),
                messages=nodes,
            )
            return True
        try:
            user_id = event.get_user_id()
        except Exception:
            user_id = getattr(event, "user_id", None)
        if user_id is None:
            return False
        await bot.call_api(
            "send_private_forward_msg",
            user_id=onebot_numeric_id(user_id),
            messages=nodes,
        )
        return True
    except Exception as exc:
        logger.warning(f"ai companion merged reply send failed: {exc!r}")
        return False


async def build_sticker_reply_message(
    event: Any,
    text: str,
    sticker_keys: list[str],
    *,
    get_sticker,
    enable_sending_stickers: bool,
    max_reply_stickers: int,
) -> Any:
    """构造携带表情的消息；非 OneBot v11 或未启用表情发送时原样返回文本。"""
    if not enable_sending_stickers or not sticker_keys:
        return text

    try:
        from nonebot.adapters.onebot.v11 import Message, MessageSegment
    except Exception:
        return text

    if "onebot.v11" not in event.__class__.__module__:
        return text

    records: list[dict[str, Any]] = []
    for key in sticker_keys[:max_reply_stickers]:
        record = await get_sticker(key)
        if not record:
            logger.warning(f"ai companion sticker key missing: {key}")
            continue
        records.append(record)

    return build_onebot_v11_sticker_message(Message, MessageSegment, text, records)
