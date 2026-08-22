"""合并转发消息展开：调 get_forward_msg API 获取转发内容，提取文字+图片。

OneBot V11 里群友发的合并转发到达 bot 时通常是一个 forward 段带 id（resid），
实际内容需调 get_forward_msg API 展开。本模块负责展开 + 提取文字和图片段，
供主回复路径接入 prompt。

容错：API 失败/超时返回空，绝不阻塞主回复。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# get_forward_msg 调用超时（秒）。
_FORWARD_API_TIMEOUT = 8.0
# 图片段类型。
_MEDIA_TYPES = {"image", "mface", "emoji", "sticker"}


async def resolve_forward(
    bot: Any,
    message: Any,
    *,
    max_messages: int = 20,
    max_chars: int = 1500,
) -> tuple[list[str], list[dict[str, Any]]]:
    """展开合并转发，返回 (forward_texts, forward_media_parts)。

    forward_texts: 每条转发消息的文字（带说话人，如 "张三：xxx"）。
    forward_media_parts: 转发里的图片段列表。
    没有转发段或展开失败时返回 ([], [])。
    """
    forward_ids = _extract_forward_ids(message)
    if not forward_ids:
        return [], []

    texts: list[str] = []
    media_parts: list[dict[str, Any]] = []
    total_chars = 0

    for fid in forward_ids:
        try:
            nodes = await asyncio.wait_for(
                _call_get_forward_msg(bot, fid),
                timeout=_FORWARD_API_TIMEOUT,
            )
        except TimeoutError:
            logger.info("forward_resolver: get_forward_msg timeout id=%s", fid)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.info("forward_resolver: get_forward_msg failed id=%s err=%r", fid, exc)
            continue

        for node in nodes[:max_messages]:
            if total_chars >= max_chars:
                break
            sender_name = _node_sender_name(node)
            node_text, node_media = _extract_node_content(node)
            if node_text:
                # 截断单条过长文字
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                if len(node_text) > remaining:
                    node_text = node_text[:remaining]
                line = f"{sender_name}：{node_text}" if sender_name else node_text
                texts.append(line)
                total_chars += len(line)
            media_parts.extend(node_media)
        if total_chars >= max_chars:
            texts.append("…（转发内容过长，已截断）")
            break

    return texts, media_parts


def _extract_forward_ids(message: Any) -> list[str]:
    """从 message 里提取所有 forward 段的 id（resid）。"""
    ids: list[str] = []
    segments = _iter_segments(message)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("type") or "").lower() != "forward":
            continue
        data = seg.get("data") or {}
        if not isinstance(data, dict):
            continue
        # OneBot V11 字段名：id 或 message_id（不同适配器有差异）
        for field in ("id", "message_id", "resId"):
            value = str(data.get(field) or "").strip()
            if value:
                ids.append(value)
                break
    return ids


async def _call_get_forward_msg(bot: Any, forward_id: str) -> list[Any]:
    """调 get_forward_msg API，返回 messages 列表。"""
    if not hasattr(bot, "call_api"):
        return []
    # OneBot V11 标准参数名是 message_id，部分适配器也认 id
    try:
        result = await bot.call_api("get_forward_msg", message_id=forward_id)
    except Exception:
        # 兜底：试 id 参数
        result = await bot.call_api("get_forward_msg", id=forward_id)
    # 返回结构通常是 {"message": [...]} 或直接是列表
    if isinstance(result, dict):
        messages = result.get("message") or result.get("messages") or []
    elif isinstance(result, list):
        messages = result
    else:
        messages = []
    return messages if isinstance(messages, list) else []


def _node_sender_name(node: Any) -> str:
    """从转发节点提取说话人昵称。"""
    if not isinstance(node, dict):
        return ""
    # 直接 sender 字段
    sender = node.get("sender")
    if isinstance(sender, dict):
        name = str(sender.get("card") or sender.get("nickname") or "").strip()
        if name:
            return name
    # 有些适配器把昵称放顶层
    for field in ("nickname", "name", "card"):
        value = str(node.get(field) or "").strip()
        if value:
            return value
    return ""


def _extract_node_content(node: Any) -> tuple[str, list[dict[str, Any]]]:
    """从转发节点提取文字和图片段。

    OneBot V11 转发节点结构：{"type":"node","data":{"content":[消息段...], ...}}
    或已展开的 {"sender":..., "content":[...], ...}
    """
    if not isinstance(node, dict):
        return "", []
    # content 在 data 里（标准 node 段）或顶层（部分适配器）
    data = node.get("data") if isinstance(node.get("data"), dict) else node
    content = data.get("content") or data.get("message")
    segments = _iter_segments(content)

    text_parts: list[str] = []
    media: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type") or "").lower()
        seg_data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
        if seg_type == "text":
            t = str(seg_data.get("text") or "").strip()
            if t:
                text_parts.append(t)
        elif seg_type in _MEDIA_TYPES:
            media.append({"type": seg_type, "data": dict(seg_data)})
    return " ".join(text_parts).strip(), media


def _iter_segments(message: Any) -> list[Any]:
    """把 message 归一成段列表。兼容 Message 对象/列表/单段。"""
    if message is None:
        return []
    if isinstance(message, list):
        return message
    if isinstance(message, dict):
        return [message]
    if isinstance(message, str):
        return [{"type": "text", "data": {"text": message}}]
    # nonebot Message 对象（可迭代）
    try:
        return list(message)
    except TypeError:
        return [{"type": "text", "data": {"text": str(message or "")}}]


def count_forward_segments(message: Any) -> int:
    """统计消息里 forward 段的数量（用于判断是否含转发）。"""
    return len(_extract_forward_ids(message))
