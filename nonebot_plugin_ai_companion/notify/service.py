"""「有人找你」离线信箱：静默累积 → 目标上线时一次性投递。

流程（用户确认的模型）：
1. 群里有人 @ / 引用了某个 10 分钟没说话的群友 → 不打扰任何人，
   把该消息（含图片，触发当下下载）记进信箱；
2. 期间再次被找 → 继续累积；
3. 目标自己在群里发了消息（=上线）→ bot @ 他，把每条触发的上下 ±3 条
   （相连的区间合并成连续段）画成一张卡片发出去，然后清空信箱。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher

from ..bot_env import BotEnvironment
from ..config import AiCompanionConfig
from ..context.envelope import (
    MessageContextEnvelope,
    build_message_context,
    reply_sender_from_event,
    sender_name_from_event,
)
from ..context.media_parts import message_to_parts
from ..conversation.window import safe_speaker_name
from ..observability.message_log import MessageLogRepo
from ..prompt.text import normalize_text
from ..store import PersonaStore
from .detector import detect_ping_candidates
from .digest import merge_context_blocks
from .inbox import PingInboxRepo
from .render import (
    delete_attachments,
    image_bytes_from_part,
    render_digest_png,
    save_attachment,
    save_card_png,
)

logger = logging.getLogger(__name__)

# 单目标信箱容量：超过后丢最老的（防刷屏式灌爆）
DEFAULT_MAX_PENDING = 30


def split_pending(pending: list[Any], *, now: float, grace_seconds: float) -> tuple[list[Any], list[Any]]:
    """把信箱记录分成（错过，在场看到）。

    @ 之后目标在宽限期内开过口 → 那条算"在场看到"（作废）；
    超过宽限期都没开口 → 算"错过"（待投递）。
    """
    if grace_seconds <= 0:
        return list(pending), []
    missed: list[Any] = []
    fresh: list[Any] = []
    for entry in pending:
        (missed if now - float(entry.created_at or 0.0) >= grace_seconds else fresh).append(entry)
    return missed, fresh


class PingService:
    def __init__(
        self,
        *,
        config: AiCompanionConfig,
        env: BotEnvironment,
        store: PersonaStore,
        message_logs: MessageLogRepo,
    ) -> None:
        self.config = config
        self.env = env
        self.store = store
        self.message_logs = message_logs
        self.inbox = PingInboxRepo(store.db)

    # ---- 主入口 ----
    async def handle_message(self, bot: Bot, event: Event, matcher: Matcher) -> None:
        if not bool(getattr(self.config, "ai_companion_ping_enabled", True)):
            return
        ids = self.env.dump_event_ids(event)
        group_id = ids.get("group_id")
        if not group_id or not self.env.group_enabled(group_id):
            return  # 私聊不存在"找人"
        if self.env.is_self_event(bot, ids):
            return
        session_key = f"group:{group_id}"
        sender_user_id = str(ids.get("user_id") or "")

        # 1) 投递判定：目标自己发了消息。
        #    @ 之后过了宽限期（默认 10 分钟）B 才开口 → 这些是"错过"的，投递；
        #    宽限期内就开口 → 他当时在场看到了，那些记录直接作废。
        pending = await self.inbox.list_for_target(session_key=session_key, target_user_id=sender_user_id)
        if pending:
            grace = float(getattr(self.config, "ai_companion_ping_grace_seconds", 600) or 0)
            missed, fresh = split_pending(pending, now=time.time(), grace_seconds=grace)
            if fresh:
                await self.inbox.delete_entries([entry.id for entry in fresh])
                for entry in fresh:
                    delete_attachments(entry.attachments)
                await self._record_group(
                    group_id,
                    ids,
                    sender_user_id,
                    status="dropped_seen_live",
                    detail={"count": len(fresh)},
                )
            if missed:
                await self._deliver(
                    bot,
                    matcher,
                    group_id=group_id,
                    ids=ids,
                    target_user_id=sender_user_id,
                    pending=missed,
                    exclude_message_id=str(ids.get("message_id") or ""),
                )

        # 2) 触发累积：消息里有 @ 或引用才继续（无条件记，宽限期内开口的会在下次说话时作废）
        try:
            message = event.get_message()
        except Exception:
            return
        has_at = any(str(part.get("type") or "") == "at" for part in message_to_parts(message))
        has_reply = getattr(event, "reply", None) is not None
        if not has_at and not has_reply:
            return

        envelope = await self._build_envelope(bot, event, message, group_id=group_id)
        decision = detect_ping_candidates(
            envelope=envelope,
            sender_user_id=sender_user_id,
            bot_self_id=self.env.bot_self_id(bot),
            quote_enabled=bool(getattr(self.config, "ai_companion_ping_quote_enabled", True)),
        )
        if not decision.candidates:
            return

        for candidate in decision.candidates:
            await self._accumulate(
                group_id=group_id, session_key=session_key, ids=ids, candidate=candidate, event=event
            )

    # ---- 累积 ----
    async def _accumulate(
        self,
        *,
        group_id: str,
        session_key: str,
        ids: dict[str, str | None],
        candidate: Any,
        event: Event,
    ) -> None:
        max_pending = int(
            getattr(self.config, "ai_companion_ping_max_pending", DEFAULT_MAX_PENDING) or DEFAULT_MAX_PENDING
        )
        existing = await self.inbox.list_for_target(session_key=session_key, target_user_id=candidate.target_user_id)
        if len(existing) >= max_pending:
            dropped = existing[0]
            await self.inbox.delete_entries([dropped.id])
            delete_attachments(dropped.attachments)
            logger.info(f"ping inbox full for {candidate.target_user_id}: dropped oldest trigger")

        text = envelope_text_of(event)
        if candidate.reason == "quote" and not text:
            try:
                from ..context.envelope import _texts_from_reply

                text = " ".join(_texts_from_reply(event))[:200]
            except Exception:  # noqa: BLE001
                text = ""
        # 图片触发当下下载落盘（QQ URL 会过期）
        attachments: list[str] = []
        for index, part in enumerate(
            p for p in message_to_parts(event.get_message()) if str(p.get("type") or "") == "image"
        ):
            image_bytes = image_bytes_from_part(part)
            path = save_attachment(
                self.config.ai_companion_data_dir, image_bytes, message_id=str(ids.get("message_id") or ""), index=index
            )
            if path is not None:
                attachments.append(str(path))
        await self.inbox.add(
            session_key=session_key,
            group_id=group_id,
            target_user_id=candidate.target_user_id,
            finder_user_id=str(candidate.finder_user_id or ""),
            reason=candidate.reason,
            source_message_id=str(ids.get("message_id") or ""),
            text=text,
            attachments=attachments,
        )
        await self._record(group_id, ids, candidate, status="accumulated", input_text=text)

    # ---- 投递 ----
    async def _deliver(
        self,
        bot: Bot,
        matcher: Matcher,
        *,
        group_id: str,
        ids: dict[str, str | None],
        target_user_id: str,
        pending: list[Any],
        exclude_message_id: str = "",
    ) -> None:
        session_key = f"group:{group_id}"
        started = time.perf_counter()
        name_rows = await self.store.member_names.get_names(group_id=group_id, user_ids=[target_user_id])
        target_name = safe_speaker_name(name_rows.get(target_user_id, ""), user_id=target_user_id)

        # 上下文：最近一段时间线（锚点在最近 200 条里找足够覆盖 ±3 的合并窗口）
        turns = await self.store.conversation.recent_turns(session_key=session_key, limit=200)
        turn_dicts = [
            {
                "turn": turn,
                "source_message_id": turn.source_message_id,
                "created_at": turn.created_at,
                "thread_id": turn.thread_id,
            }
            for turn in turns
        ]
        anchors = [(entry.source_message_id, entry.created_at) for entry in pending]
        blocks, dropped = merge_context_blocks(
            turn_dicts,
            anchors,
            context_lines=int(getattr(self.config, "ai_companion_ping_context_lines", 3) or 3),
            max_lines=int(getattr(self.config, "ai_companion_ping_max_lines", 40) or 40),
        )
        if exclude_message_id:
            # 目标"上线"的那条消息本身不算错过的内容
            for block in blocks:
                block.turns = [
                    item for item in block.turns if str(item.get("source_message_id") or "") != exclude_message_id
                ]
            blocks = [block for block in blocks if block.turns]

        user_ids = [target_user_id, *[entry.finder_user_id for entry in pending if entry.finder_user_id]]
        names = await self.store.member_names.get_names(group_id=group_id, user_ids=user_ids)
        finder_names = {
            safe_speaker_name(names.get(entry.finder_user_id, ""), user_id=entry.finder_user_id) for entry in pending
        }
        finder_names.discard("")
        finder_count = len(finder_names) or len(pending)
        entry_count = len(pending)
        title = (
            f"{target_name}同学，有 {finder_count} 人找过你" if finder_count > 1 else f"{target_name}同学，有人找过你"
        )
        if entry_count > finder_count:
            title += f"（共 {entry_count} 次）"
        subtitle = "以下是你不在时被找的原话和上下文："

        attachments_by_message_id: dict[str, list[str]] = {}
        all_attachments: list[str] = []
        for entry in pending:
            if entry.attachments:
                attachments_by_message_id.setdefault(entry.source_message_id, []).extend(entry.attachments)
                all_attachments.extend(entry.attachments)

        png = b""
        try:
            png = await render_digest_png(
                title=title,
                subtitle=subtitle,
                blocks=blocks,
                names=names,
                attachments_by_message_id=attachments_by_message_id,
                dropped_lines=dropped,
            )
        except Exception as exc:  # noqa: BLE001 — 渲染失败降级纯文本
            logger.warning(f"ping digest render failed: {exc!r}")
        latency = time.perf_counter() - started

        try:
            from nonebot.adapters.onebot.v11 import Message, MessageSegment

            message = Message()
            if bool(getattr(self.config, "ai_companion_ping_at_target", True)):
                message += MessageSegment.at(int(target_user_id))
            if png:
                path = save_card_png(self.config.ai_companion_data_dir, png, target_user_id=target_user_id)
                message += MessageSegment.image(f"file://{path}")
            else:
                finder_summary = "、".join(sorted(finder_names)[:5])
                message += MessageSegment.text(f" {title}。{finder_summary} 找过你，详见上面的聊天记录。")
            sent = await matcher.send(message)
        except Exception as exc:  # noqa: BLE001 — 发送失败：信箱保留，等目标下一条消息再试
            logger.warning(f"ping digest send failed: {exc!r}")
            await self._record_group(
                group_id,
                ids,
                target_user_id,
                status="send_error",
                detail={"error": f"{type(exc).__name__}: {exc}", "pending": len(pending)},
            )
            return
        # 投递成功：清空信箱与附件
        await self.inbox.delete_entries([entry.id for entry in pending])
        delete_attachments(all_attachments)
        reply_id = sent.get("message_id") if isinstance(sent, dict) else None
        await self._record_group(
            group_id,
            ids,
            target_user_id,
            status="delivered",
            latency=latency,
            reply_message_id=str(reply_id) if reply_id else None,
            detail={"pending": len(pending), "blocks": len(blocks), "dropped": dropped},
        )

    # ---- 辅助 ----
    async def _build_envelope(
        self,
        bot: Bot,
        event: Event,
        message: Any,
        *,
        group_id: str,
    ) -> MessageContextEnvelope:
        at_parts = [part for part in message_to_parts(message) if str(part.get("type") or "") == "at"]
        at_nickname_rows = await self._resolve_at_names(
            bot, group_id, [str(part.get("data", {}).get("qq") or "") for part in at_parts]
        )
        reply_sender = reply_sender_from_event(event)
        reply_nickname = ""
        if reply_sender and reply_sender.get("user_id"):
            rows = await self.store.member_names.get_names(group_id=group_id, user_ids=[str(reply_sender["user_id"])])
            reply_nickname = rows.get(str(reply_sender["user_id"]), "")
        media_parts = [part for part in message_to_parts(message) if str(part.get("type") or "") != "text"]
        return build_message_context(
            message=message,
            event=event,
            current_text=envelope_text_of(event),
            media_parts=media_parts,
            sticker_keys=[],
            forced_by_at=False,
            at_nicknames=at_nickname_rows,
            reply_nickname=reply_nickname or None,
            sender_name=sender_name_from_event(event),
        )

    async def _resolve_at_names(self, bot: Bot, group_id: str, qq_list: list[str]) -> dict[str, str | None]:
        clean = [qq for qq in {item.strip() for item in qq_list} if qq and qq != "all"]
        if not clean:
            return {}
        from ..context.nicknames import NicknameResolver

        resolver = NicknameResolver(bot)
        return await resolver.resolve_many(group_id=group_id, user_ids=clean)

    async def _record(
        self,
        group_id: str,
        ids: dict[str, str | None],
        candidate: Any,
        *,
        status: str,
        input_text: str = "",
        latency: float | None = None,
        reply_message_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await self._record_group(
            group_id,
            ids,
            candidate.target_user_id,
            reason=getattr(candidate, "reason", "at"),
            status=status,
            input_text=input_text,
            latency=latency,
            reply_message_id=reply_message_id,
            detail=detail,
        )

    async def _record_group(
        self,
        group_id: str,
        ids: dict[str, str | None],
        target_user_id: str,
        *,
        reason: str = "at",
        status: str,
        input_text: str = "",
        latency: float | None = None,
        reply_message_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self.message_logs.record(
                platform=ids.get("platform"),
                group_id=group_id,
                user_id=target_user_id,
                trigger_message_id=ids.get("message_id"),
                reply_message_id=reply_message_id,
                reply_kind="ping",
                trigger_reason=f"ping:{reason}",
                decision_detail=detail or {"reason": status},
                status=status,
                input_text=input_text[:200],
                output_text=f"提醒 {target_user_id}" if status in {"delivered", "send_error"} else "",
            )
        except Exception as exc:  # noqa: BLE001 — 留痕失败不影响主流程
            logger.debug(f"ping message log skipped: {exc!r}")


def envelope_text_of(event: Event) -> str:
    """当前消息的纯文本（卡片里展示的原话）。"""
    try:
        return normalize_text(
            " ".join(
                str(part.get("text") or "")
                for part in message_to_parts(event.get_message())
                if part.get("type") == "text"
            )
        )
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["PingService"]
