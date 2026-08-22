"""学习管线编排：被动学习每条消息（文本/表情/模式）。"""

from __future__ import annotations

from typing import Any

from nonebot.adapters import Bot, Event

from .bot_env import BotEnvironment
from .decision.gates import should_force_reply, should_learn_text
from .decision.repeat import ConversationPatternTracker
from .memory.ingest import filter_training_text_sticker_tokens, prepare_message_learning
from .observability.events import safe_record_sticker_event
from .sticker.repo_files import record_sticker_repo_manifest
from .store import PersonaStore

try:  # logger 由 nonebot 提供；独立导入时退回标准 logging 便于单测
    from nonebot import logger as nb_logger

    logger = nb_logger  # type: ignore[assignment]
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


class LearningService:
    def __init__(
        self,
        *,
        env: BotEnvironment,
        store: PersonaStore,
        conversation_pattern_tracker: ConversationPatternTracker,
    ) -> None:
        self.env = env
        self.config = env.config
        self.store = store
        self.conversation_pattern_tracker = conversation_pattern_tracker

    async def handle_learn(self, bot: Bot, event: Event) -> None:
        if not self.config.ai_companion_enable_learning:
            return
        try:
            message = event.get_message()
        except Exception:
            return

        ids = self.env.dump_event_ids(event)
        if not self.env.group_enabled(ids["group_id"]):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return
        if self.env.is_self_event(bot, ids):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return

        learning = await prepare_message_learning(
            message,
            sticker_segment_types=self.config.ai_companion_sticker_segment_types,
            captioner=None,
            image_fallback_text="",
            sticker_fallback_text="",
        )
        plain_text = str(learning["plain_text"])
        raw_message_text = str(message)
        pattern_allowed = bool(ids["group_id"])
        if should_force_reply(
            event,
            ids,
            message,
            bot_self_id=self.env.bot_self_id(bot),
            to_me=self.env.is_to_me(event, plain_text),
        ):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            pattern_allowed = False
        if self.env.is_command_message(plain_text) or self.env.is_command_message(raw_message_text):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return
        if self.env.is_blacklisted(ids["user_id"], ids["group_id"], plain_text + "\n" + raw_message_text):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return

        await self._record_learning_sticker_events(ids, list(learning.get("sticker_events") or []))

        stickers = list(learning["stickers"])
        training_text = str(learning["training_text"])
        decision = should_learn_text(
            training_text,
            min_chars=self.config.ai_companion_min_sample_chars,
            max_chars=self.config.ai_companion_max_sample_chars,
            prefixes=self.env.command_start_values(),
            regexes=self.env.command_regex_values(),
        )

        if not decision.allowed and not stickers:
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return

        await record_sticker_repo_manifest(
            repo_dir=self.config.ai_companion_sticker_repo_dir,
            group_id=ids["group_id"],
            user_id=ids["user_id"],
            stickers=stickers,
        )
        training_text = filter_training_text_sticker_tokens(training_text, stickers)
        decision = should_learn_text(
            training_text,
            min_chars=self.config.ai_companion_min_sample_chars,
            max_chars=self.config.ai_companion_max_sample_chars,
            prefixes=self.env.command_start_values(),
            regexes=self.env.command_regex_values(),
        )
        if not decision.allowed and not stickers:
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return

        await self.store.record_sample(
            platform=ids["platform"],
            user_id=ids["user_id"],
            group_id=ids["group_id"],
            message_id=ids["message_id"],
            plain_text=plain_text,
            training_text=training_text,
            raw_segments=list(learning["raw_segments"]),
            stickers=stickers,
            learn_reason=decision.reason,
        )
        if not pattern_allowed:
            return
        candidate = self.conversation_pattern_tracker.observe(
            group_id=ids["group_id"],
            user_id=ids["user_id"],
            message=message,
            sticker_segment_types=self.config.ai_companion_sticker_segment_types,
            plain_text=plain_text,
            sticker_keys=[
                str(sticker.get("key") or "").strip() for sticker in stickers if str(sticker.get("key") or "").strip()
            ],
        )
        if candidate is None:
            return
        await self.store.record_conversation_pattern(
            group_id=ids["group_id"],
            trigger_signature=candidate.trigger.signature,
            trigger_text=candidate.trigger.plain_text,
            response_signature=candidate.response.signature,
            response_text=candidate.response.plain_text,
            response_sticker_keys=candidate.response.sticker_keys,
        )

    async def _record_learning_sticker_events(
        self,
        ids: dict[str, str | None],
        events: list[dict[str, Any]],
    ) -> None:
        for event in events:
            await safe_record_sticker_event(
                self.store,
                logger=logger,
                event_type=str(event.get("event_type") or "classifier"),
                status=str(event.get("status") or "unknown"),
                key=str(event.get("key") or "") or None,
                group_id=ids["group_id"],
                user_id=ids["user_id"],
                message=str(event.get("message") or ""),
                details=dict(event.get("details") or {}),
            )
