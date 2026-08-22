"""回复主管线编排：从消息接入到回复发送的完整流程。

matcher 层只做路由，业务逻辑在此与各领域模块中。
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher

from .bot_env import BotEnvironment
from .config import AiCompanionConfig
from .context.envelope import MessageContextEnvelope, _at_targets_from_message, build_message_context
from .context.forward import count_forward_segments, resolve_forward
from .context.media_parts import message_to_parts
from .context.nicknames import NicknameResolver
from .context.relevance import check_media_relevance
from .contracts import MediaCaptioner
from .decision.gates import (
    ReplyDecision,
    is_private_chat,
    should_attempt_reply,
    should_force_reply,
    should_use_sticker_reply,
)
from .decision.repeat import (
    ConversationPatternTracker,
    RepeatTracker,
    build_conversation_pattern_message,
)
from .delivery.onebot_v11 import build_sticker_reply_message, send_onebot_v11_forward_reply
from .delivery.plan import plan_reply_delivery
from .memory.ingest import prepare_message_learning
from .observability.debug_log import append_caption_debug_log
from .observability.events import safe_record_sticker_event
from .observability.inference_events import record_inference_event
from .prompt.budget import effective_context_message_limit, trim_history_lines_for_context
from .prompt.builder import build_system_prompt
from .prompt.media_text import (
    context_text_from_sample,
    sanitize_model_reply_media_text,
)
from .prompt.meme_gate import build_meme_candidates_from_samples, select_active_memes
from .prompt.planner import ReplyIntent, ReplyPlan, build_reply_plan
from .prompt.text import (
    is_command_like,
    is_echo_reply,
    is_question_deflection_reply,
    is_training_noise,
    sanitize_reply,
)
from .providers.client import QwenClient
from .providers.mlx_runtime import MlxVLMRuntime
from .sticker.analysis import analyze_media_parts, should_analyze_background_media
from .sticker.model import extract_plain_text, parse_sticker_tokens
from .sticker.selector import select_reply_sticker
from .store import PersonaStore
from .tools.registry import ToolRegistry

try:  # logger 由 nonebot 提供；独立导入时退回标准 logging 便于单测
    from nonebot import logger as nb_logger

    logger = nb_logger  # type: ignore[assignment]
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


class ChatService:
    def __init__(
        self,
        *,
        env: BotEnvironment,
        store: PersonaStore,
        client: QwenClient,
        mlx_runtime: MlxVLMRuntime,
        captioner: MediaCaptioner | None,
        nickname_resolver: NicknameResolver,
        repeat_tracker: RepeatTracker,
        conversation_pattern_tracker: ConversationPatternTracker,
        tool_registry: ToolRegistry,
    ) -> None:
        self.env = env
        self.config: AiCompanionConfig = env.config
        self.store = store
        self.client = client
        self.mlx_runtime = mlx_runtime
        self.captioner = captioner
        self.nickname_resolver = nickname_resolver
        self.repeat_tracker = repeat_tracker
        self.conversation_pattern_tracker = conversation_pattern_tracker
        self.tool_registry = tool_registry
        self.background_media_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    async def handle_chat(self, bot: Bot, event: Event, matcher: Matcher) -> None:
        try:
            message = event.get_message()
        except Exception:
            return

        ids = self.env.dump_event_ids(event)
        if self.env.is_self_event(bot, ids):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return
        plain_text = extract_plain_text(message)
        # 合并转发：展开 forward 段，提取文字拼进 plain_text，图片段单独收集。
        if count_forward_segments(message) > 0:
            try:
                f_texts, f_media = await resolve_forward(
                    bot,
                    message,
                    max_messages=int(self.config.ai_companion_forward_max_messages),
                    max_chars=int(self.config.ai_companion_forward_max_chars),
                )
                if f_texts:
                    forward_block = "【合并转发】\n" + "\n".join(f_texts)
                    plain_text = (plain_text + "\n" + forward_block).strip() if plain_text else forward_block
                    logger.info(
                        "ai companion forward expanded: texts=%d media=%d",
                        len(f_texts),
                        len(f_media),
                    )
            except Exception as exc:  # noqa: BLE001 — 展开失败不阻塞回复
                logger.debug("forward expand skipped: %r", exc)
        reply_input_text, incoming_sticker_keys, media_parts, media_context_text = await self._build_reply_input_text(
            message, plain_text
        )
        # 图片智能控制：图片超过阈值则丢弃图片识别（清空 media_context_text），只保留文字。
        image_count = sum(1 for p in media_parts if str(p.get("type") or "") in {"image", "mface", "emoji", "sticker"})
        image_limit = int(self.config.ai_companion_media_image_limit)
        if image_count > image_limit and media_context_text:
            logger.info(
                "ai companion media: %d images > limit %d, dropping captions",
                image_count,
                image_limit,
            )
            media_context_text = ""
        if self._should_schedule_background_media_analysis(bot, event, plain_text, ids, media_parts):
            self._schedule_background_media_analysis(ids, media_parts)
        if await self._send_repeat_if_needed(bot, event, ids, message, matcher):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            matcher.stop_propagation()
            return
        if await self._send_pattern_reply_if_known(bot, event, ids, message, plain_text, matcher):
            matcher.stop_propagation()
            return
        reply_decision = self._reply_decision(bot, event, plain_text, ids)
        if not reply_decision.allowed:
            return

        # 主语指代：预解析 @ 目标和被引用消息发送者的昵称，让 bot 知道谁在对谁说话。
        at_nicknames: dict[str, str | None] = {}
        reply_nickname: str | None = None
        group_id = ids.get("group_id")
        try:
            at_qqs = [t["qq"] for t in _at_targets_from_message(message, {}) if t.get("qq")]
            if at_qqs and group_id:
                at_nicknames = await self.nickname_resolver.resolve_many(bot, group_id, at_qqs)
            reply_obj = getattr(event, "reply", None)
            reply_sender_id = ""
            if reply_obj is not None:
                reply_sender_obj = getattr(reply_obj, "sender", None)
                if isinstance(reply_sender_obj, dict):
                    reply_sender_id = str(reply_sender_obj.get("user_id") or reply_sender_obj.get("qq") or "")
                elif reply_sender_obj is not None:
                    reply_sender_id = str(
                        getattr(reply_sender_obj, "user_id", "") or getattr(reply_sender_obj, "qq", "")
                    )
            if reply_sender_id and group_id:
                reply_nickname = await self.nickname_resolver.resolve(bot, group_id, reply_sender_id)
        except Exception as exc:  # noqa: BLE001 — 昵称解析失败不阻塞回复
            logger.debug("nickname pre-resolve skipped: %r", exc)

        envelope = build_message_context(
            message=message,
            event=event,
            current_text=reply_input_text,
            media_parts=media_parts,
            sticker_keys=incoming_sticker_keys,
            forced_by_at=reply_decision.reason == "force_reply",
            at_nicknames=at_nicknames,
            reply_nickname=reply_nickname,
        )
        sticker_reply_allowed = should_use_sticker_reply(
            incoming_has_sticker=bool(incoming_sticker_keys),
            text_sticker_probability=self.config.ai_companion_text_sticker_reply_probability,
            random_value=random.random(),
        )
        reply_plan = await self._build_model_reply_plan(
            envelope,
            ids["group_id"],
            ids=ids,
            sticker_reply_allowed=sticker_reply_allowed,
            incoming_has_sticker=bool(incoming_sticker_keys),
            media_parts=media_parts,
            media_context_text=media_context_text,
        )
        inference_started = time.perf_counter()
        raw_reply = ""
        if reply_plan.intent != ReplyIntent.STICKER_REQUEST:
            try:
                raw_reply = await self.client.chat(reply_plan.messages, media_parts=reply_plan.media_parts)
            except Exception as exc:
                await record_inference_event(
                    store=self.store,
                    ids=ids,
                    trigger_reason="auto_reply",
                    model=self.config.ai_companion_mlx_model,
                    adapter=self.config.ai_companion_mlx_adapter_path,
                    started_at=inference_started,
                    finished_at=time.perf_counter(),
                    input_text=reply_input_text,
                    output_text="",
                    media_count=len(media_parts),
                    status="failed",
                    error_summary=f"{type(exc).__name__}: {exc}",
                    tps=None,
                )
                logger.warning(f"ai companion chat failed: {exc!r}")
                return

            metrics = self.client.metrics_snapshot() if hasattr(self.client, "metrics_snapshot") else {"last": None}
            last_metric = metrics.get("last") if isinstance(metrics, dict) else None
            tps = last_metric.get("tps") if isinstance(last_metric, dict) else None
            await record_inference_event(
                store=self.store,
                ids=ids,
                trigger_reason="auto_reply",
                model=(
                    self.client.active_model_name()
                    if hasattr(self.client, "active_model_name")
                    else self.config.ai_companion_mlx_model
                ),
                adapter=(self.config.ai_companion_mlx_adapter_path if self.client.active_backend() == "mlx" else ""),
                started_at=inference_started,
                finished_at=time.perf_counter(),
                input_text=reply_input_text,
                output_text=raw_reply,
                media_count=len(media_parts),
                status="success",
                error_summary="",
                tps=tps,
            )

        raw_reply = sanitize_model_reply_media_text(raw_reply)
        reply_text, generated_sticker_keys = parse_sticker_tokens(raw_reply)
        if generated_sticker_keys:
            logger.warning(
                "ai companion generated sticker tokens ignored: "
                + ",".join(generated_sticker_keys[: self.config.ai_companion_max_reply_stickers])
            )
        clean_text = sanitize_reply(
            reply_text,
            prefixes=self.env.command_start_values(),
            regexes=self.env.command_regex_values(),
        )
        clean_text = self.env.remove_blacklisted_reply_lines(clean_text)
        if is_echo_reply(clean_text, reply_input_text) or is_question_deflection_reply(clean_text, reply_input_text):
            clean_text = ""
        sticker_keys: list[str] = []
        if reply_plan.allow_sticker and (
            clean_text or incoming_sticker_keys or reply_plan.intent == ReplyIntent.STICKER_REQUEST
        ):
            sticker_key = await self._select_reply_sticker_key(
                group_id=ids["group_id"],
                user_id=ids["user_id"],
                trigger_text=reply_input_text,
                reply_text=clean_text,
                intent=reply_plan.intent,
                incoming_sticker_keys=incoming_sticker_keys,
            )
            if sticker_key:
                sticker_keys = [sticker_key]
        if not clean_text and not sticker_keys:
            return

        if not await self._send_adapter_reply(
            bot=bot, event=event, matcher=matcher, text=clean_text, sticker_keys=sticker_keys
        ):
            return
        self.conversation_pattern_tracker.reset(ids["group_id"])
        matcher.stop_propagation()

    # ------------------------------------------------------------------
    # 决策与输入构建
    # ------------------------------------------------------------------
    def _active_reply_captioner(self) -> MediaCaptioner | None:
        """主回复路径的图片 captioner：优先动漫识别，禁用则返回 None。"""
        if getattr(self.config, "ai_companion_anime_caption_enabled", True):
            return self.captioner
        return None

    def _reply_decision(
        self,
        bot: Bot,
        event: Event,
        plain_text: str,
        ids: dict[str, str | None],
    ) -> ReplyDecision:
        private_chat = is_private_chat(event, ids)
        try:
            message = event.get_message()
        except Exception:
            message = ()
        force_reply = should_force_reply(
            event,
            ids,
            message,
            bot_self_id=self.env.bot_self_id(bot),
            to_me=self.env.is_to_me(event, plain_text),
        )
        return should_attempt_reply(
            enable_reply=self.config.ai_companion_enable_reply,
            group_enabled=private_chat or self.env.group_enabled(ids["group_id"]),
            self_event=self.env.is_self_event(bot, ids),
            blacklisted=self.env.is_blacklisted(ids["user_id"], ids["group_id"], plain_text),
            command_like=self.env.is_command_message(plain_text),
            force_reply=force_reply,
            has_text=bool(str(plain_text or "").strip()) or force_reply,
            reply_probability=self.config.ai_companion_reply_probability,
            follow_reply_probability=self.config.ai_companion_follow_reply_probability,
            random_value=random.random(),
        )

    def _should_schedule_background_media_analysis(
        self,
        bot: Bot,
        event: Event,
        plain_text: str,
        ids: dict[str, str | None],
        media_parts: list[dict[str, Any]],
    ) -> bool:
        private_chat = is_private_chat(event, ids)
        try:
            message = event.get_message()
        except Exception:
            message = ()
        force_reply = should_force_reply(
            event,
            ids,
            message,
            bot_self_id=self.env.bot_self_id(bot),
            to_me=self.env.is_to_me(event, plain_text),
        )
        decision = should_analyze_background_media(
            enable_learning=self.config.ai_companion_enable_learning,
            group_enabled=private_chat or self.env.group_enabled(ids["group_id"]),
            self_event=self.env.is_self_event(bot, ids),
            blacklisted=self.env.is_blacklisted(ids["user_id"], ids["group_id"], plain_text),
            command_like=self.env.is_command_message(plain_text),
            force_reply=force_reply,
            has_media=any(str(part.get("type") or "") == "image" for part in media_parts),
            background_busy=bool(self.background_media_tasks),
        )
        return decision.allowed

    async def _build_reply_input_text(
        self,
        message: Any,
        plain_text: str,
    ) -> tuple[str, list[str], list[dict[str, Any]], str]:
        active_captioner = self._active_reply_captioner()
        logger.info(
            "ai companion reply media: captioner=%s, anime_enabled=%s, media_segments=%d",
            type(active_captioner).__name__ if active_captioner else "None",
            getattr(self.config, "ai_companion_anime_caption_enabled", True),
            sum(
                1
                for p in message_to_parts(message)
                if str(p.get("type") or "") in {"image", "mface", "emoji", "sticker"}
            ),
        )
        try:
            learning = await prepare_message_learning(
                message,
                sticker_segment_types=self.config.ai_companion_sticker_segment_types,
                captioner=active_captioner,
                image_fallback_text="",
                sticker_fallback_text="",
            )
        except Exception as exc:
            logger.debug(f"ai companion reply media context skipped: {exc!r}")
            return (
                plain_text,
                [],
                [part for part in message_to_parts(message) if part.get("type") != "text"],
                "",
            )

        sticker_keys = [
            str(item.get("key") or "").strip()
            for item in learning.get("stickers") or []
            if str(item.get("key") or "").strip()
        ]
        media_context_text = str(learning.get("media_context_text") or "")
        return plain_text, sticker_keys, list(learning.get("media_parts") or []), media_context_text

    # ------------------------------------------------------------------
    # 提示词计划
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            self.config,
            tool_registry=self.tool_registry,
            command_prefixes=self.env.command_start_values(),
        )

    @staticmethod
    def _sample_fingerprint(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(item.get("created_at") or ""),
            str(item.get("group_id") or ""),
            str(item.get("user_id") or ""),
            str(item.get("plain_text") or ""),
            str(item.get("training_text") or ""),
        )

    def _merge_context_samples(
        self,
        group_samples: list[dict[str, Any]],
        global_samples: list[dict[str, Any]],
        *,
        group_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        limit = max(0, int(limit))
        if limit <= 0:
            return []
        merged: list[dict[str, Any]] = list(group_samples[:limit])
        seen = {self._sample_fingerprint(item) for item in merged}
        other_group_samples = [
            item for item in global_samples if str(item.get("group_id") or "") != str(group_id or "")
        ]
        for item in other_group_samples:
            fingerprint = self._sample_fingerprint(item)
            if fingerprint in seen:
                continue
            merged.append(item)
            seen.add(fingerprint)
            if len(merged) >= limit:
                break
        if len(merged) < limit:
            for item in global_samples:
                fingerprint = self._sample_fingerprint(item)
                if fingerprint in seen:
                    continue
                merged.append(item)
                seen.add(fingerprint)
                if len(merged) >= limit:
                    break
        merged.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
        return merged[:limit]

    async def _recent_context_samples(self, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        scope = str(getattr(self.config, "ai_companion_context_scope", "mixed") or "mixed")
        if scope == "group":
            return await self.store.recent_samples(group_id=group_id, limit=limit)
        if scope == "global" or not group_id:
            return await self.store.recent_samples(group_id=None, limit=limit)

        group_limit = max(1, limit // 2)
        group_samples, global_samples = await asyncio.gather(
            self.store.recent_samples(group_id=group_id, limit=group_limit),
            self.store.recent_samples(group_id=None, limit=max(limit * 3, limit + group_limit)),
        )
        return self._merge_context_samples(
            group_samples,
            global_samples,
            group_id=group_id,
            limit=limit,
        )

    def _is_context_noise(self, text: str) -> bool:
        return is_training_noise(text) or is_command_like(
            text,
            prefixes=self.env.command_start_values(),
            regexes=self.env.command_regex_values(),
        )

    async def _build_model_reply_plan(
        self,
        envelope: MessageContextEnvelope,
        group_id: str | None,
        *,
        ids: dict[str, str | None],
        sticker_reply_allowed: bool,
        incoming_has_sticker: bool,
        media_parts: list[dict[str, Any]],
        media_context_text: str = "",
    ) -> ReplyPlan:
        context_limit = effective_context_message_limit(self.config)
        query_text = self._context_query_text(envelope)
        history, relevant_samples, relevant_media_assets = await asyncio.gather(
            self._recent_context_samples(group_id, context_limit),
            self._search_relevant_context_samples(group_id, query_text, max(3, context_limit // 2)),
            self._search_relevant_media_assets(group_id, query_text, max(2, context_limit // 3)),
        )

        conversation_lines = self._conversation_lines_from_envelope(envelope)
        # 图文相关性判断：若图片描述与对话文字无关，丢弃图片描述只保留文字。
        if media_context_text and getattr(self.config, "ai_companion_media_relevance_check", True):
            dialogue_text = envelope.current_text
            if not dialogue_text and conversation_lines:
                dialogue_text = " ".join(conversation_lines)
            if dialogue_text:
                relevant = await check_media_relevance(self.config, media_context_text, dialogue_text)
                if not relevant:
                    logger.info("ai companion media: caption dropped (irrelevant to text)")
                    media_context_text = ""
        if media_context_text:
            conversation_lines.append(media_context_text)
        # 诊断：记录 caption 是否进了 conversation_lines
        append_caption_debug_log(
            f"PROMPT media_context_text={media_context_text!r} conversation_lines_count={len(conversation_lines)}",
            data_dir=self.config.ai_companion_data_dir,
        )
        for item in relevant_samples:
            context_text = context_text_from_sample(item)
            context_text, _sticker_keys = parse_sticker_tokens(context_text)
            if self._is_context_noise(context_text):
                continue
            if context_text:
                conversation_lines.append(f"相关历史：{context_text}")
        for item in relevant_media_assets:
            answer = str(item.get("answer") or "").strip()
            if answer and not self._is_context_noise(answer):
                conversation_lines.append(f"历史图片理解：{answer}")

        style_memory_lines: list[str] = []
        current_message_id = str(ids.get("message_id") or "")
        for item in reversed(history):
            if current_message_id and str(item.get("message_id") or "") == current_message_id:
                continue
            context_text = context_text_from_sample(item)
            context_text, _sticker_keys = parse_sticker_tokens(context_text)
            if self._is_context_noise(context_text):
                continue
            if context_text:
                style_memory_lines.append(context_text)
        conversation_lines = trim_history_lines_for_context(conversation_lines, self.config)
        style_memory_lines = trim_history_lines_for_context(style_memory_lines, self.config)
        active_memes = select_active_memes(
            build_meme_candidates_from_samples(
                samples=[*history, *relevant_samples],
                media_assets=relevant_media_assets,
            ),
            context_text=query_text,
            min_recent_count=self.config.ai_companion_min_meme_recent_count,
            max_items=3,
        )

        return build_reply_plan(
            system_prompt=self._build_system_prompt(),
            conversation_lines=conversation_lines,
            style_memory_lines=style_memory_lines,
            active_memes=active_memes,
            current_text=envelope.current_text,
            media_parts=media_parts,
            sticker_reply_allowed=sticker_reply_allowed,
            incoming_has_sticker=incoming_has_sticker,
        )

    @staticmethod
    def _context_query_text(envelope: MessageContextEnvelope) -> str:
        chunks = [envelope.current_text]
        chunks.extend(envelope.quoted_texts)
        chunks.extend(envelope.forwarded_texts)
        return " ".join(chunk for chunk in chunks if chunk).strip()

    async def _search_relevant_context_samples(
        self,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        scope = str(getattr(self.config, "ai_companion_context_scope", "mixed") or "mixed")
        search_group_id = None if scope == "global" else group_id
        try:
            return await self.store.search_context_samples(
                group_id=search_group_id,
                query=query,
                limit=limit,
            )
        except Exception as exc:
            logger.warning(f"ai companion context search failed: {exc!r}")
            return []

    async def _search_relevant_media_assets(
        self,
        group_id: str | None,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        scope = str(getattr(self.config, "ai_companion_context_scope", "mixed") or "mixed")
        search_group_id = None if scope == "global" else group_id
        try:
            return await self.store.search_media_assets(
                group_id=search_group_id,
                query=query,
                limit=limit,
            )
        except Exception as exc:
            logger.warning(f"ai companion media context search failed: {exc!r}")
            return []

    @staticmethod
    def _conversation_lines_from_envelope(envelope: MessageContextEnvelope) -> list[str]:
        lines: list[str] = []
        # 主语：让 bot 知道是谁在说话、@ 了谁、回复了谁
        if envelope.sender_name:
            lines.append(f"说话人：{envelope.sender_name}")
        if envelope.reply_sender:
            reply_name = envelope.reply_sender.get("nickname") or envelope.reply_sender.get("user_id") or "某人"
            for text in envelope.quoted_texts:
                lines.append(f"回复{reply_name}的消息：{text}")
            if not envelope.quoted_texts:
                lines.append(f"当前消息回复了{reply_name}。")
        else:
            for text in envelope.quoted_texts:
                lines.append(f"引用消息：{text}")
        if envelope.at_targets:
            names = "、".join(t.get("name") or t.get("qq") or "?" for t in envelope.at_targets)
            lines.append(f"当前消息@了：{names}")
        for text in envelope.forwarded_texts:
            lines.append(f"合并转发：{text}")
        if envelope.sticker_keys:
            lines.append("当前消息包含群聊表情。")
        if envelope.media_parts:
            lines.append("当前消息包含图片或表情媒体，必要时直接看图回答。")
        return lines

    # ------------------------------------------------------------------
    # 后台媒体分析
    # ------------------------------------------------------------------
    def _schedule_background_media_analysis(
        self,
        ids: dict[str, str | None],
        media_parts: list[dict[str, Any]],
    ) -> None:
        image_parts = [part for part in media_parts if str(part.get("type") or "") == "image"]
        if not image_parts:
            return
        if self.background_media_tasks:
            return
        task = asyncio.create_task(self._run_background_media_analysis(ids=dict(ids), media_parts=image_parts))
        self.background_media_tasks.add(task)

        def _discard(done: asyncio.Task[None]) -> None:
            self.background_media_tasks.discard(done)

        task.add_done_callback(_discard)

    async def _run_background_media_analysis(
        self,
        *,
        ids: dict[str, str | None],
        media_parts: list[dict[str, Any]],
    ) -> None:
        if not await self._wait_for_generation_idle(timeout_seconds=12.0):
            await safe_record_sticker_event(
                self.store,
                logger=logger,
                event_type="media_analysis",
                status="skipped",
                group_id=ids.get("group_id"),
                user_id=ids.get("user_id"),
                message="background media analysis skipped: generation busy",
                details={"media_count": len(media_parts)},
            )
            return
        try:
            result = await analyze_media_parts(
                store=self.store,
                captioner=self.mlx_runtime,
                parts=media_parts,
                group_id=ids.get("group_id"),
                user_id=ids.get("user_id"),
                sticker_segment_types=set(self.config.ai_companion_sticker_segment_types),
            )
        except Exception as exc:
            logger.warning(f"ai companion background media analysis failed: {exc!r}")
            await safe_record_sticker_event(
                self.store,
                logger=logger,
                event_type="media_analysis",
                status="failed",
                group_id=ids.get("group_id"),
                user_id=ids.get("user_id"),
                message=f"{type(exc).__name__}: {exc}",
                details={"media_count": len(media_parts)},
            )
            return
        for event in result.events:
            await safe_record_sticker_event(
                self.store,
                logger=logger,
                event_type=str(event.get("event_type") or "media_analysis"),
                status=str(event.get("status") or "unknown"),
                key=str(event.get("key") or "") or None,
                group_id=ids.get("group_id"),
                user_id=ids.get("user_id"),
                message=str(event.get("message") or ""),
                details=dict(event.get("details") or {}),
            )

    async def _wait_for_generation_idle(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            try:
                generation = dict(self.mlx_runtime.status_snapshot().get("generation") or {})
                active = int(generation.get("active") or 0)
                queued = int(generation.get("queued") or 0)
            except Exception:
                return True
            if active <= 0 and queued <= 0:
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    f"ai companion background media analysis skipped: generation busy active={active}, queued={queued}"
                )
                return False
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # 复读与模式回复
    # ------------------------------------------------------------------
    async def _send_repeat_if_needed(
        self,
        bot: Bot,
        event: Event,
        ids: dict[str, str | None],
        message: Any,
        matcher: Matcher,
    ) -> bool:
        if not self.config.ai_companion_enable_repeat:
            return False
        candidate = self.repeat_tracker.observe(
            group_id=ids["group_id"],
            message=message,
            sticker_segment_types=self.config.ai_companion_sticker_segment_types,
            min_count=self.config.ai_companion_repeat_min_count,
            cooldown_seconds=self.config.ai_companion_repeat_cooldown_seconds,
        )
        if not candidate:
            return False
        if random.random() >= self.config.ai_companion_repeat_probability:
            return False
        if self.env.is_command_message(candidate.plain_text):
            return False
        if self.env.is_blacklisted(ids["user_id"], ids["group_id"], candidate.plain_text):
            return False
        reply_text = self.env.remove_blacklisted_reply_lines(candidate.plain_text)
        if not reply_text and not candidate.sticker_keys:
            return False
        return await self._send_adapter_reply(
            bot=bot, event=event, matcher=matcher, text=reply_text, sticker_keys=candidate.sticker_keys
        )

    async def _send_pattern_reply_if_known(
        self,
        bot: Bot,
        event: Event,
        ids: dict[str, str | None],
        message: Any,
        plain_text: str,
        matcher: Matcher,
    ) -> bool:
        if not self.config.ai_companion_enable_reply:
            return False
        if not ids["group_id"] or not self.env.group_enabled(ids["group_id"]):
            return False
        if self.env.is_self_event(bot, ids):
            return False
        if self.env.is_command_message(plain_text) or self.env.is_blacklisted(
            ids["user_id"], ids["group_id"], plain_text
        ):
            self.conversation_pattern_tracker.reset(ids["group_id"])
            return False
        if self.config.ai_companion_pattern_reply_threshold <= 0:
            return False

        trigger = build_conversation_pattern_message(
            user_id=ids["user_id"] or "",
            message=message,
            sticker_segment_types=self.config.ai_companion_sticker_segment_types,
        )
        if trigger is None:
            return False
        pattern = await self.store.get_conversation_pattern(
            group_id=ids["group_id"],
            trigger_signature=trigger.signature,
            min_count=self.config.ai_companion_pattern_reply_threshold,
        )
        if not pattern:
            return False

        reply_text = sanitize_reply(
            str(pattern.get("response_text") or ""),
            prefixes=self.env.command_start_values(),
            regexes=self.env.command_regex_values(),
        )
        reply_text = self.env.remove_blacklisted_reply_lines(reply_text)
        sticker_keys = [str(key) for key in pattern.get("response_sticker_keys", []) if str(key).strip()]
        if not reply_text and not sticker_keys:
            return False

        if not await self._send_adapter_reply(
            bot=bot, event=event, matcher=matcher, text=reply_text, sticker_keys=sticker_keys
        ):
            return False
        self.conversation_pattern_tracker.reset(ids["group_id"])
        return True

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------
    async def _send_adapter_reply(
        self,
        *,
        bot: Bot | None,
        event: Event,
        matcher: Matcher,
        text: str,
        sticker_keys: list[str],
    ) -> bool:
        delivery = plan_reply_delivery(text, sticker_keys)
        if delivery.use_forward and bot is not None:
            if await send_onebot_v11_forward_reply(
                bot, event, delivery.forward_chunks, self_id=self.env.bot_self_id(bot)
            ):
                if delivery.trailing_sticker_keys:
                    sticker_reply = await build_sticker_reply_message(
                        event,
                        "",
                        delivery.trailing_sticker_keys,
                        get_sticker=self.store.get_sticker,
                        enable_sending_stickers=self.config.ai_companion_enable_sending_stickers,
                        max_reply_stickers=self.config.ai_companion_max_reply_stickers,
                    )
                    if not (isinstance(sticker_reply, str) and not sticker_reply.strip()):
                        await matcher.send(sticker_reply)
                return True

        reply = await build_sticker_reply_message(
            event,
            text,
            sticker_keys,
            get_sticker=self.store.get_sticker,
            enable_sending_stickers=self.config.ai_companion_enable_sending_stickers,
            max_reply_stickers=self.config.ai_companion_max_reply_stickers,
        )
        if isinstance(reply, str) and not reply.strip():
            return False
        await matcher.send(reply)
        return True

    # ------------------------------------------------------------------
    # 表情选择
    # ------------------------------------------------------------------
    async def _select_reply_sticker_key(
        self,
        *,
        group_id: str | None,
        user_id: str | None,
        trigger_text: str,
        reply_text: str,
        intent: ReplyIntent,
        incoming_sticker_keys: list[str],
    ) -> str | None:
        candidates = await self.store.top_stickers(
            group_id=group_id,
            limit=self.config.ai_companion_sticker_prompt_limit,
        )
        seen = {str(item.get("key") or "").strip() for item in candidates}
        for key in incoming_sticker_keys:
            if key in seen:
                continue
            record = await self.store.get_sticker(key)
            if record:
                candidates.insert(0, record)
                seen.add(key)
            else:
                logger.warning(f"ai companion incoming sticker key missing: {key}")
        # 现状保持：表情选择器固定使用本地 MLX 运行时（GLM 模式下也走本地视觉）。
        selection = await select_reply_sticker(
            self.mlx_runtime,
            candidates=candidates,
            trigger_text=trigger_text,
            reply_text=reply_text,
            intent=intent,
            min_count=self.config.ai_companion_min_reply_sticker_count,
        )
        await safe_record_sticker_event(
            self.store,
            logger=logger,
            event_type="selector",
            status="selected" if selection.key else "none",
            key=selection.key,
            group_id=group_id,
            user_id=user_id,
            message=selection.reason or ("selected sticker" if selection.key else "no sticker selected"),
            details={
                "intent": str(getattr(intent, "value", intent)),
                "candidate_count": selection.candidate_count,
                "usable_count": selection.usable_count,
                "min_count": self.config.ai_companion_min_reply_sticker_count,
                "fallback": selection.fallback,
                "trigger_text": trigger_text[:240],
                "reply_text": reply_text[:240],
                "raw_text": selection.raw_text[:500],
            },
        )
        return selection.key

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def shutdown(self) -> None:
        for task in list(self.background_media_tasks):
            task.cancel()
        if self.background_media_tasks:
            await asyncio.gather(*self.background_media_tasks, return_exceptions=True)
            self.background_media_tasks.clear()
