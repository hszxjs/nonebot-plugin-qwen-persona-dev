"""成就卡服务：成员发言使总数跨过阈值 → 渲染成就卡发进群。"""

from __future__ import annotations

import logging
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher

from ..bot_env import BotEnvironment
from ..config import AiCompanionConfig
from ..conversation.window import safe_speaker_name
from ..observability.message_log import MessageLogRepo
from ..store import PersonaStore
from .milestone import (
    fetch_avatar,
    fetch_member_stats,
    is_milestone,
    render_milestone_png,
    save_milestone_png,
)

logger = logging.getLogger(__name__)


class MilestoneService:
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
        self._group_names: dict[str, str] = {}

    async def handle_message(self, bot: Bot, event: Event, matcher: Matcher) -> None:
        if not bool(getattr(self.config, "ai_companion_milestone_enabled", True)):
            return
        ids = self.env.dump_event_ids(event)
        group_id = ids.get("group_id")
        if not group_id or not self.env.group_enabled(group_id):
            return
        if self.env.is_self_event(bot, ids):
            return
        user_id = str(ids.get("user_id") or "")
        if not user_id:
            return

        # 计数在学习 matcher（priority 97）之后执行：这条消息已经进了时间线
        configured = [int(item) for item in getattr(self.config, "ai_companion_milestone_list", [])]
        stats = await fetch_member_stats(self.store, group_id=group_id, user_id=user_id)
        if stats is None:
            return
        total = int(stats["total"])
        if not is_milestone(total, configured):
            return

        # 显示名：昵称表 → 兜底「群友xxxx」；QQ 号绝不进卡片
        names = await self.store.member_names.get_names(group_id=group_id, user_ids=[user_id])
        name = safe_speaker_name(names.get(user_id, ""), user_id=user_id)

        group_label = await self._group_label(bot, group_id)
        avatar = await fetch_avatar(user_id)
        png = b""
        render_error = ""
        try:
            # 渲染失败要有可读日志：build 与 screenshot 分开试
            png = await render_milestone_png(
                badge="群聊成就",
                name=name,
                count=total,
                rank=int(stats["rank"]),
                total_members=int(stats["total_members"]),
                sticker_count=int(stats["sticker_count"]),
                avatar=avatar,
                footer_left=f"< {group_label} >",
            )
        except Exception as exc:  # noqa: BLE001 — 渲染失败降级纯文本
            render_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"milestone card render failed: {exc!r}")

        from nonebot.adapters.onebot.v11 import Message, MessageSegment

        try:
            if png:
                path = save_milestone_png(self.config.ai_companion_data_dir, png, user_id=user_id)
                await matcher.send(Message(MessageSegment.image(f"file://{path}")))
            else:
                await matcher.send(
                    Message(
                        MessageSegment.at(int(user_id))
                        + MessageSegment.text(
                            f" 🏆 群聊成就：{name} 总发言突破 {total}！"
                            f"群内排名 #{stats['rank']}，超过 "
                            f"{(stats['total_members'] - stats['rank']) / max(1, stats['total_members']) * 100:.2f}%"
                            " 的活跃群友。"
                        )
                    )
                )
        except Exception as exc:  # noqa: BLE001 — 发送失败留痕
            logger.warning(f"milestone card send failed: {exc!r}")
            await self._record(group_id, ids, user_id, total, status="send_error", detail={"error": str(exc)})
            return
        await self._record(
            group_id,
            ids,
            user_id,
            total,
            status="success",
            detail={"rank": stats["rank"], "members": stats["total_members"], "render_error": render_error},
        )

    async def _group_label(self, bot: Bot, group_id: str) -> str:
        """群名[群号]，进程内缓存。"""
        if group_id in self._group_names:
            return f"{self._group_names[group_id]}[{group_id}]"
        group_name = ""
        try:
            info = await bot.call_api("get_group_info", group_id=int(group_id))
            group_name = str((info or {}).get("group_name") or "").strip()
        except Exception as exc:  # noqa: BLE001 — 拿不到群名就用群号
            logger.debug(f"get_group_info failed for {group_id}: {exc!r}")
        self._group_names[group_id] = group_name
        return f"{group_name}[{group_id}]" if group_name else f"群 {group_id}"

    async def _record(
        self,
        group_id: str,
        ids: dict[str, str | None],
        user_id: str,
        count: int,
        *,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self.message_logs.record(
                platform=ids.get("platform"),
                group_id=group_id,
                user_id=user_id,
                trigger_message_id=ids.get("message_id"),
                reply_message_id=None,
                reply_kind="milestone",
                trigger_reason=f"milestone:{count}",
                decision_detail=detail or {},
                status=status,
                input_text=f"总发言 {count}",
                output_text=f"成就卡 {count}" if status == "success" else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"milestone message log skipped: {exc!r}")


__all__ = ["MilestoneService"]
