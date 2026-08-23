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
from .observability.message_log import MessageLogRepo
from .prompt.budget import effective_context_message_limit, trim_history_lines_for_context
from .prompt.builder import PromptContext, build_system_prompt, effective_segment_order
from .prompt.media_text import (
    context_text_from_sample,
    sanitize_model_reply_media_text,
)
from .prompt.meme_gate import build_meme_candidates_from_samples, select_active_memes
from .prompt.planner import ReplyIntent, ReplyPlan, build_reply_plan
from .prompt.text import (
    extract_pseudo_tool_reply,
    is_command_like,
    is_echo_reply,
    is_question_deflection_reply,
    is_training_noise,
    sanitize_reply,
)
from .providers.client import QwenClient
from .providers.instances import ProviderResolver
from .providers.mlx_runtime import MlxVLMRuntime
from .role_card.model import RoleCard
from .role_card.service import RoleCardService
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
        role_card_service: RoleCardService | None = None,
        provider_resolver: ProviderResolver | None = None,
        message_log_repo: MessageLogRepo | None = None,
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
        self.role_card_service = role_card_service
        self.provider_resolver = provider_resolver
        self.message_logs = message_log_repo
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
        active_card = None
        if reply_plan.intent != ReplyIntent.STICKER_REQUEST:
            try:
                allowed_tools: list[str] | None = None
                if self.role_card_service is not None:
                    card, _binding = await self.role_card_service.get_effective_card(
                        group_id=ids.get("group_id"), user_id=ids.get("user_id")
                    )
                    active_card = card
                    if card is not None and card.allowed_tools:
                        allowed_tools = list(card.allowed_tools)
                raw_reply = await self._generate_reply(reply_plan, allowed_tools=allowed_tools)
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
        # 模型可能在文本里"表演"工具调用（FC 关闭时）：提取真实答案
        raw_reply = extract_pseudo_tool_reply(raw_reply)
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
            sticker_keys = await self._select_reply_sticker_keys(
                group_id=ids["group_id"],
                user_id=ids["user_id"],
                trigger_text=reply_input_text,
                reply_text=clean_text,
                intent=reply_plan.intent,
                incoming_sticker_keys=incoming_sticker_keys,
            )
        if not clean_text and not sticker_keys:
            await self._record_message_log(
                kind="chat",
                ids=ids,
                trigger_reason=reply_decision.reason,
                decision_detail={"reason": reply_decision.reason, "stage": "skipped_empty_reply"},
                role_card=active_card,
                provider=self._provider_label(),
                latency_seconds=time.perf_counter() - inference_started,
                status="skipped_empty_reply",
                input_text=reply_input_text,
                output_text=raw_reply,
            )
            return

        try:
            sent = await self._send_adapter_reply(
                bot=bot, event=event, matcher=matcher, text=clean_text, sticker_keys=sticker_keys
            )
        except Exception as exc:  # noqa: BLE001 — 发送异常留痕，不再无声消失
            logger.warning(f"ai companion reply send raised: {exc!r}")
            await self._record_message_log(
                kind="chat",
                ids=ids,
                trigger_reason=reply_decision.reason,
                decision_detail={"reason": reply_decision.reason, "error": f"{type(exc).__name__}: {exc}"},
                role_card=active_card,
                provider=self._provider_label(),
                latency_seconds=time.perf_counter() - inference_started,
                status="send_error",
                input_text=reply_input_text,
                output_text=clean_text,
                sticker_keys=sticker_keys,
            )
            return
        provider_label = self._provider_label()
        if self.provider_resolver is not None:
            try:
                _binding = await self.provider_resolver.bindings.get_binding("chat")
                if _binding and _binding.provider_id:
                    provider_label = f"instance:{_binding.provider_id}"
            except Exception as exc:  # noqa: BLE001 — 标签仅供展示
                logger.debug("provider label resolve skipped: %r", exc)
        await self._record_message_log(
            kind="chat",
            ids=ids,
            trigger_reason=reply_decision.reason,
            decision_detail={"reason": reply_decision.reason, "allowed": True},
            reply_message_id=sent.get("reply_message_id"),
            role_card=active_card,
            provider=provider_label,
            latency_seconds=time.perf_counter() - inference_started,
            status="success" if sent.get("ok") else "send_failed",
            input_text=reply_input_text,
            output_text=clean_text,
            sticker_keys=sticker_keys,
        )
        if not sent.get("ok"):
            return
        self.conversation_pattern_tracker.reset(ids["group_id"])
        matcher.stop_propagation()

    async def _vision_provider(self) -> Any:
        """vision 角色解析：绑定实例优先，否则回退本地 MLX 运行时（三期解决）。"""
        if self.provider_resolver is None:
            return self.mlx_runtime
        provider, _model, _pid = await self.provider_resolver.resolve_with_meta("vision")
        return provider if provider is not None else self.mlx_runtime

    # ------------------------------------------------------------------
    # 推理（含工具调用回路，配置默认关闭）
    # ------------------------------------------------------------------
    async def _generate_reply(
        self,
        reply_plan: ReplyPlan,
        *,
        allowed_tools: list[str] | None,
    ) -> str:
        tools_payload = (
            self.tool_registry.openai_tools_payload(allowed_tools)
            if self.config.ai_companion_enable_tool_calls and self.tool_registry.has_tools
            else []
        )
        # chat 角色绑定实例：优先走解析器（含降级链与模型覆盖）
        if self.provider_resolver is not None and await self._chat_instance_provider() is not None:
            provider = await self._chat_instance_provider()
            if tools_payload and hasattr(provider, "chat_with_tools"):
                try:
                    return await self._run_tool_loop(
                        messages=reply_plan.messages,
                        media_parts=reply_plan.media_parts,
                        tools_payload=tools_payload,
                        provider=provider,
                    )
                except NotImplementedError:
                    pass
            return await self.provider_resolver.chat(reply_plan.messages, media_parts=reply_plan.media_parts)
        if not tools_payload:
            return await self.client.chat(reply_plan.messages, media_parts=reply_plan.media_parts)
        try:
            return await self._run_tool_loop(
                messages=reply_plan.messages,
                media_parts=reply_plan.media_parts,
                tools_payload=tools_payload,
            )
        except NotImplementedError:
            logger.info("ai companion tool calls unsupported by backend, falling back to plain chat")
            return await self.client.chat(reply_plan.messages, media_parts=reply_plan.media_parts)

    async def _chat_instance_provider(self) -> Any | None:
        """chat 角色绑定的实例 provider（未绑定/实例缺失返回 None）。"""
        if self.provider_resolver is None:
            return None
        binding = await self.provider_resolver.bindings.get_binding("chat")
        if not binding or not binding.provider_id:
            return None
        instance = await self.provider_resolver.instances.get_instance(binding.provider_id)
        return instance.provider if instance is not None else None

    async def _run_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        media_parts: list[dict[str, Any]] | None,
        tools_payload: list[dict[str, Any]],
        provider: Any | None = None,
    ) -> str:
        """function calling 回路：模型请求工具 → 执行 → 结果回填 → 二次推理。

        轮次上限与超时由配置约束，防止死循环（文档四节）。
        """
        import json as _json

        max_rounds = max(1, int(self.config.ai_companion_tool_max_rounds or 1))
        tool_timeout = float(self.config.ai_companion_tool_call_timeout_seconds or 15.0)
        result_cap = int(self.config.ai_companion_tool_result_max_chars or 1200)
        conversation = [dict(message) for message in messages]
        chat_tools = provider.chat_with_tools if provider is not None else self.client.chat_with_tools
        final_chat = provider.chat if provider is not None else self.client.chat
        for _round in range(max_rounds):
            content, tool_calls = await chat_tools(conversation, tools=tools_payload)
            if not tool_calls:
                return content
            conversation.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})
            for call in tool_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = _json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                tool_started = time.perf_counter()
                try:
                    result = await asyncio.wait_for(self.tool_registry.execute(name, arguments), timeout=tool_timeout)
                except TimeoutError:
                    logger.warning("ai companion tool %s timed out (%.0fs)", name, tool_timeout)
                    await self._record_tool_call(name, ok=False, latency=time.perf_counter() - tool_started, call=call)
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id") or name),
                            "content": f"工具 {name} 执行超时",
                        }
                    )
                    continue
                logger.info("ai companion tool call %s ok=%s", name, result.ok)
                await self._record_tool_call(
                    name,
                    ok=result.ok,
                    latency=time.perf_counter() - tool_started,
                    call=call,
                    result=result,
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or name),
                        "content": result.render(max_chars=result_cap),
                    }
                )
        # 超过轮次上限：普通调用收敛出最终文本
        return await final_chat(conversation, media_parts=None)

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
    def _build_system_prompt(
        self,
        *,
        card: RoleCard | None = None,
        user_nickname: str | None = None,
        memory_lines: list[str] | None = None,
    ) -> str:
        identity_text = None
        if card is not None and self.role_card_service is not None:
            identity_text = self.role_card_service.identity_prompt(card, user_nickname=user_nickname)
        allowed_tools = list(card.allowed_tools) if card is not None and card.allowed_tools else None
        return build_system_prompt(
            PromptContext(
                config=self.config,
                tool_registry=self.tool_registry,
                command_prefixes=self.env.command_start_values(),
                identity_text=identity_text,
                allowed_tools=allowed_tools,
                memory_lines=list(memory_lines or []),
            )
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
        # 角色卡：会话绑定的卡渲染身份段；切换绑定后短期上下文从 bound_at 重新计。
        card, binding = (
            await self.role_card_service.get_effective_card(group_id=group_id, user_id=ids.get("user_id"))
            if self.role_card_service is not None
            else (None, None)
        )
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
        bound_at = float(binding.bound_at) if binding is not None else None
        for item in reversed(history):
            if current_message_id and str(item.get("message_id") or "") == current_message_id:
                continue
            if bound_at is not None and binding is not None and card is not None:
                try:
                    if float(item.get("created_at") or 0) < bound_at:
                        continue  # 切换角色卡前的旧上下文不进新人格的短期记忆
                except (TypeError, ValueError):
                    pass
            context_text = context_text_from_sample(item)
            context_text, _sticker_keys = parse_sticker_tokens(context_text)
            if self._is_context_noise(context_text):
                continue
            if context_text:
                style_memory_lines.append(context_text)
        conversation_lines = trim_history_lines_for_context(conversation_lines, self.config)
        style_memory_lines = trim_history_lines_for_context(style_memory_lines, self.config)
        # 记忆注入段（文档六/十节）：启用 memory 段时取发送者长期记忆（未衰减 top）
        memory_lines: list[str] = []
        if "memory" in effective_segment_order(self.config) and ids.get("user_id"):
            try:
                memory_lines = await self.store.long_term_memories.prompt_lines(user_id=str(ids["user_id"]))
            except Exception as exc:  # noqa: BLE001 — 记忆取数失败不阻塞回复
                logger.debug("long term memory fetch skipped: %r", exc)

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
            system_prompt=self._build_system_prompt(
                card=card,
                user_nickname=envelope.sender_name,
                memory_lines=memory_lines,
            ),
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
                captioner=await self._vision_provider(),
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
        sent = await self._send_adapter_reply(
            bot=bot, event=event, matcher=matcher, text=reply_text, sticker_keys=candidate.sticker_keys
        )
        await self._record_message_log(
            kind="repeat",
            ids=ids,
            trigger_reason="repeat",
            reply_message_id=sent.get("reply_message_id"),
            status="success" if sent.get("ok") else "send_failed",
            input_text=candidate.plain_text,
            output_text=reply_text,
            sticker_keys=list(candidate.sticker_keys),
        )
        return bool(sent.get("ok"))

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

        sent = await self._send_adapter_reply(
            bot=bot, event=event, matcher=matcher, text=reply_text, sticker_keys=sticker_keys
        )
        await self._record_message_log(
            kind="pattern",
            ids=ids,
            trigger_reason="conversation_pattern",
            reply_message_id=sent.get("reply_message_id"),
            status="success" if sent.get("ok") else "send_failed",
            input_text=plain_text,
            output_text=reply_text,
            sticker_keys=sticker_keys,
        )
        if not sent.get("ok"):
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
    ) -> dict[str, Any]:
        """发送回复；返回 {ok, reply_message_id}（id 供 MessageLog 溯源）。"""
        delivery = plan_reply_delivery(text, sticker_keys)
        if delivery.use_forward and bot is not None:
            sent = await send_onebot_v11_forward_reply(
                bot, event, delivery.forward_chunks, self_id=self.env.bot_self_id(bot)
            )
            if sent is not False and sent is not None:
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
                return {"ok": True, "reply_message_id": _extract_message_id(sent)}

        reply = await build_sticker_reply_message(
            event,
            text,
            sticker_keys,
            get_sticker=self.store.get_sticker,
            enable_sending_stickers=self.config.ai_companion_enable_sending_stickers,
            max_reply_stickers=self.config.ai_companion_max_reply_stickers,
        )
        if isinstance(reply, str) and not reply.strip():
            return {"ok": False, "reply_message_id": None}
        sent = await matcher.send(reply)
        return {"ok": True, "reply_message_id": _extract_message_id(sent)}

    # ------------------------------------------------------------------
    # 表情选择
    # ------------------------------------------------------------------
    async def _select_reply_sticker_keys(
        self,
        *,
        group_id: str | None,
        user_id: str | None,
        trigger_text: str,
        reply_text: str,
        intent: ReplyIntent,
        incoming_sticker_keys: list[str],
    ) -> list[str]:
        """选择回复携带的表情（多选，上限 max_reply_stickers）。

        通道优先级：vision 角色（默认本地 MLX）→ 失败时用当前 chat provider 做
        纯文本选择（候选带编号+摘要，文本模型可选）→ 全失败走加权随机兜底。
        """
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

        max_stickers = max(1, int(self.config.ai_companion_max_reply_stickers or 1))
        selector_kwargs = dict(
            candidates=candidates,
            trigger_text=trigger_text,
            reply_text=reply_text,
            intent=intent,
            min_count=self.config.ai_companion_min_reply_sticker_count,
            max_stickers=max_stickers,
        )

        selection = None
        selection_error = ""
        try:
            selection = await select_reply_sticker(await self._vision_provider(), **selector_kwargs)
        except Exception as exc:  # noqa: BLE001 — vision 通道异常时降级文本通道
            selection_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"ai companion sticker selection vision channel failed: {exc!r}")

        if selection is not None and not selection.fallback and selection.key:
            # vision 通道真实成功（非兜底）——直接采用
            pass
        else:
            # vision 不可用/失败/兜底：尝试当前 chat provider 的纯文本选择
            try:
                text_selection = await select_reply_sticker(self.client, **selector_kwargs)
                if text_selection is not None and text_selection.key and not text_selection.fallback or selection is None:
                    selection = text_selection
            except Exception as exc:  # noqa: BLE001 — 文本通道也失败则保留兜底
                logger.warning(f"ai companion sticker selection text channel failed: {exc!r}")

        if selection is None:
            selection_keys: list[str] = []
        else:
            selection_keys = selection.selected_keys[:max_stickers]

        await safe_record_sticker_event(
            self.store,
            logger=logger,
            event_type="selector",
            status="selected" if selection_keys else "none",
            key=selection_keys[0] if selection_keys else None,
            group_id=group_id,
            user_id=user_id,
            message=selection.reason if selection else selection_error or "no sticker selected",
            details={
                "intent": str(getattr(intent, "value", intent)),
                "candidate_count": selection.candidate_count if selection else 0,
                "usable_count": selection.usable_count if selection else 0,
                "min_count": self.config.ai_companion_min_reply_sticker_count,
                "fallback": bool(selection and selection.fallback),
                "selected_count": len(selection_keys),
                "trigger_text": trigger_text[:240],
                "reply_text": reply_text[:240],
                "raw_text": (selection.raw_text[:500] if selection else ""),
            },
        )
        return selection_keys

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def _record_message_log(
        self,
        *,
        kind: str,
        ids: dict[str, str | None],
        trigger_reason: str = "",
        decision_detail: dict[str, Any] | None = None,
        reply_message_id: str | None = None,
        role_card: Any = None,
        provider: str = "",
        latency_seconds: float | None = None,
        status: str = "success",
        input_text: str = "",
        output_text: str = "",
        sticker_keys: list[str] | None = None,
    ) -> None:
        if self.message_logs is None:
            return
        try:
            await self.message_logs.record(
                platform=ids.get("platform"),
                group_id=ids.get("group_id"),
                user_id=ids.get("user_id"),
                trigger_message_id=ids.get("message_id"),
                reply_message_id=reply_message_id,
                reply_kind=kind,
                trigger_reason=trigger_reason,
                decision_detail=decision_detail,
                role_card_id=getattr(role_card, "id", None),
                role_card_name=getattr(role_card, "name", None),
                provider=provider or self._provider_label(),
                model=self._provider_model_label(),
                latency_seconds=latency_seconds,
                status=status,
                sticker_keys=sticker_keys,
                input_text=input_text,
                output_text=output_text,
            )
        except Exception as exc:  # noqa: BLE001 — 溯源写入失败不影响回复
            logger.warning(f"ai companion message log record failed: {exc!r}")

    async def _record_tool_call(
        self,
        name: str,
        *,
        ok: bool,
        latency: float,
        call: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        try:
            await self.store.tool_call_logs.record(
                name=name,
                ok=ok,
                latency_seconds=latency,
                group_id=None,
                arguments=(call or {}).get("function", {}).get("arguments"),
                result=getattr(result, "payload", None),
            )
        except Exception as exc:  # noqa: BLE001 — 日志失败不影响主流程
            logger.debug("tool call log record failed: %r", exc)

    def _provider_label(self) -> str:
        """同步可得的 Provider 标签（实例绑定标签由 chat 路径异步补全）。"""
        try:
            return str(self.client.active_backend())
        except Exception:  # noqa: BLE001 — 标签仅供展示
            return ""

    def _provider_model_label(self) -> str:
        try:
            return self.client.active_model_name()
        except Exception:  # noqa: BLE001
            return ""

    async def shutdown(self) -> None:
        for task in list(self.background_media_tasks):
            task.cancel()
        if self.background_media_tasks:
            await asyncio.gather(*self.background_media_tasks, return_exceptions=True)
            self.background_media_tasks.clear()


def _extract_message_id(send_result: Any) -> str | None:
    """从 matcher.send / call_api 返回值提取平台消息 ID（OneBot v11 dict.message_id）。"""
    if isinstance(send_result, dict):
        value = send_result.get("message_id")
        return None if value is None else str(value)
    if send_result is None or isinstance(send_result, (str, bool)):
        return None
    value = getattr(send_result, "message_id", None)
    return None if value is None else str(value)
