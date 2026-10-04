"""找人提醒服务：判定 → 风控 → 收集 → 渲染 → 发送 → 留痕。

风控（全部内存态，进程重启归零）：
- 每目标冷却：同一个人 10 分钟内不被重复提醒（冷却期内的找人对会记为 skipped）；
- 每群每小时上限：防刷屏；
- 在场判定：目标 10 分钟内在本群说过话 → 人还在，不提醒（用户已确认 10 分钟）。
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
    sender_name_from_event,
)
from ..context.media_parts import message_to_parts
from ..conversation.window import safe_speaker_name
from ..observability.message_log import MessageLogRepo
from ..prompt.text import normalize_text
from ..store import PersonaStore
from .detector import detect_ping_candidates, target_last_spoke_age
from .model import FinderEntry, PingCard
from .render import image_bytes_from_part, render_card_png, save_card_png

logger = logging.getLogger(__name__)


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
        self._group_hourly: dict[str, list[float]] = {}
        self._target_last_card: dict[str, float] = {}

    # ---- 风控 ----
    def _target_blocked(self, group_id: str, target_user_id: str, now: float) -> str:
        cooldown = float(self.config.ai_companion_ping_target_cooldown_seconds or 0)
        if cooldown > 0:
            last = self._target_last_card.get(f"{group_id}:{target_user_id}", 0.0)
            if now - last < cooldown:
                return "target_cooldown"
        cutoff = now - 3600.0
        stamps = [stamp for stamp in self._group_hourly.get(group_id, []) if stamp >= cutoff]
        self._group_hourly[group_id] = stamps
        limit = int(self.config.ai_companion_ping_hourly_limit or 0)
        if limit > 0 and len(stamps) >= limit:
            return "group_hourly_limit"
        return ""

    def _mark_sent(self, group_id: str, target_user_id: str, now: float) -> None:
        self._target_last_card[f"{group_id}:{target_user_id}"] = now
        self._group_hourly.setdefault(group_id, []).append(now)

    # ---- 主入口 ----
    async def handle_message(
        self,
        bot: Bot,
        event: Event,
        matcher: Matcher,
    ) -> None:
        if not bool(getattr(self.config, "ai_companion_ping_enabled", True)):
            return
        ids = self.env.dump_event_ids(event)
        group_id = ids.get("group_id")
        if not group_id:  # 私聊不存在"找人"
            return
        if not self.env.group_enabled(group_id):
            return
        try:
            message = event.get_message()
        except Exception:
            return
        at_parts = [part for part in message_to_parts(message) if str(part.get("type") or "") == "at"]
        reply_present = getattr(event, "reply", None) is not None
        if not at_parts and not reply_present:  # 早退：绝大多数消息到这里就结束
            return
        if self.env.is_self_event(bot, ids):
            return

        sender_user_id = ids.get("user_id")
        media_parts = [part for part in message_to_parts(message) if str(part.get("type") or "") != "text"]
        at_nicknames = await self._resolve_at_names(
            bot, group_id, [str(p.get("data", {}).get("qq") or "") for p in at_parts]
        )
        reply_nickname = None
        envelope: MessageContextEnvelope = build_message_context(
            message=message,
            event=event,
            current_text=normalize_text(
                " ".join(
                    str(part.get("text") or "") for part in message_to_parts(message) if part.get("type") == "text"
                )
            ),
            media_parts=media_parts,
            sticker_keys=[],
            forced_by_at=False,
            at_nicknames=at_nicknames,
            reply_nickname=reply_nickname,
            sender_name=sender_name_from_event(event),
        )
        decision = detect_ping_candidates(
            envelope=envelope,
            sender_user_id=sender_user_id,
            bot_self_id=self.env.bot_self_id(bot),
            quote_enabled=bool(getattr(self.config, "ai_companion_ping_quote_enabled", True)),
        )
        if not decision.candidates:
            return

        now = time.time()
        idle_seconds = float(self.config.ai_companion_ping_idle_seconds or 0)
        for candidate in decision.candidates:
            blocked = self._target_blocked(group_id, candidate.target_user_id, now)
            if blocked:
                await self._record(group_id, ids, candidate, status=f"skipped_{blocked}")
                continue
            if idle_seconds > 0:
                turns = await self.store.conversation.recent_turns(session_key=f"group:{group_id}", limit=30)
                age = target_last_spoke_age(
                    [
                        {
                            "user_id": turn.user_id,
                            "speaker_kind": turn.speaker_kind,
                            "created_at": turn.created_at,
                        }
                        for turn in turns
                    ],
                    target_user_id=candidate.target_user_id,
                    now=now,
                )
                if age is not None and age < idle_seconds:
                    await self._record(group_id, ids, candidate, status="skipped_target_active")
                    continue
            await self._send_card(bot, event, matcher, group_id=group_id, ids=ids, candidate=candidate, now=now)

    async def _resolve_at_names(self, bot: Bot, group_id: str, qq_list: list[str]) -> dict[str, str | None]:
        clean = [qq for qq in {item.strip() for item in qq_list} if qq and qq != "all"]
        if not clean:
            return {}
        from ..context.nicknames import NicknameResolver

        resolver = NicknameResolver(bot)
        return await resolver.resolve_many(group_id=group_id, user_ids=clean)

    async def _send_card(
        self,
        bot: Bot,
        event: Event,
        matcher: Matcher,
        *,
        group_id: str,
        ids: dict[str, str | None],
        candidate: Any,
        now: float,
    ) -> None:
        started = time.perf_counter()
        target_name_row = await self.store.member_names.get_names(
            group_id=group_id, user_ids=[candidate.target_user_id]
        )
        target_name = safe_speaker_name(
            target_name_row.get(candidate.target_user_id, ""), user_id=candidate.target_user_id
        )
        finder_name = safe_speaker_name(candidate.finder_name or "", user_id=candidate.finder_user_id)
        if not candidate.finder_name and candidate.finder_user_id:
            finder_rows = await self.store.member_names.get_names(
                group_id=group_id, user_ids=[candidate.finder_user_id]
            )
            finder_name = safe_speaker_name(
                finder_rows.get(candidate.finder_user_id, ""), user_id=candidate.finder_user_id
            )

        card = PingCard(group_id=group_id, target_user_id=candidate.target_user_id, target_name=target_name)
        finder = FinderEntry(
            user_id=candidate.finder_user_id or "",
            name=finder_name,
            text=envelope_text_of(event),
            reason=candidate.reason,
        )
        card.finders.append(finder)

        # 图片触发当下下载（QQ URL 会过期），正文取当前消息文本
        from ..context.media_parts import message_to_parts

        media_parts = [part for part in message_to_parts(event.get_message()) if str(part.get("type") or "") == "image"]
        image_lists = [[data for data in (image_bytes_from_part(part) for part in media_parts) if data]]
        if candidate.reason == "quote" and not finder.text:
            # 引用场景：卡片里展示"被引用的那句话"才有意义
            try:
                from ..context.envelope import _texts_from_reply

                finder.text = " ".join(_texts_from_reply(event))[:160]
            except Exception:  # noqa: BLE001
                pass

        png = b""
        try:
            png = await render_card_png(card, image_bytes_list=image_lists)
        except Exception as exc:  # noqa: BLE001 — 渲染失败降级纯文本
            logger.warning(f"ping card render failed: {exc!r}")
        latency = time.perf_counter() - started

        try:
            if png:
                path = save_card_png(self.config.ai_companion_data_dir, png, target_user_id=candidate.target_user_id)
                from nonebot.adapters.onebot.v11 import Message, MessageSegment

                message = Message()
                if bool(getattr(self.config, "ai_companion_ping_at_target", True)):
                    message += MessageSegment.at(int(candidate.target_user_id))
                message += MessageSegment.image(f"file://{path}")
                sent = await matcher.send(message)
            else:
                from nonebot.adapters.onebot.v11 import Message, MessageSegment

                message = Message()
                if bool(getattr(self.config, "ai_companion_ping_at_target", True)):
                    message += MessageSegment.at(int(candidate.target_user_id))
                message += MessageSegment.text(f" {card.title}（详见上面的消息）")
                sent = await matcher.send(message)
        except Exception as exc:  # noqa: BLE001 — 发送失败留痕
            logger.warning(f"ping card send failed: {exc!r}")
            await self._record(
                group_id,
                ids,
                candidate,
                status="send_error",
                input_text=finder.text,
                latency=latency,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._mark_sent(group_id, candidate.target_user_id, now)
        reply_id = sent.get("message_id") if isinstance(sent, dict) else None
        await self._record(
            group_id,
            ids,
            candidate,
            status="success",
            input_text=finder.text,
            latency=latency,
            reply_message_id=str(reply_id) if reply_id else None,
        )

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
        try:
            await self.message_logs.record(
                platform=ids.get("platform"),
                group_id=group_id,
                user_id=candidate.target_user_id,
                trigger_message_id=ids.get("message_id"),
                reply_message_id=reply_message_id,
                reply_kind="ping",
                trigger_reason=f"ping:{candidate.reason}",
                decision_detail=detail or {"reason": status},
                status=status,
                input_text=input_text[:200],
                output_text=f"提醒 {candidate.target_user_id}" if status == "success" else "",
            )
        except Exception as exc:  # noqa: BLE001 — 留痕失败不影响主流程
            logger.debug(f"ping message log skipped: {exc!r}")


def envelope_text_of(event: Event) -> str:
    """当前消息的纯文本（找人卡片里展示的原话）。"""
    try:
        from ..context.media_parts import message_to_parts
        from ..prompt.text import normalize_text

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
