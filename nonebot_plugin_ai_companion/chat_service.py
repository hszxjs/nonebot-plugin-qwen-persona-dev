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
from .conversation.model import SPEAKER_BOT, session_key_for
from .conversation.window import collect_user_ids, render_history_lines, safe_speaker_name
from .decision.gates import (
    ReplyDecision,
    is_private_chat,
    should_attempt_reply,
    should_force_reply,
    should_use_sticker_reply,
)
from .decision.interjection import InterjectionInput, InterjectionLimiter, weighted_probability
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
        self.interjection_limiter = InterjectionLimiter(
            cooldown_seconds=float(getattr(self.config, "ai_companion_group_cooldown_seconds", 45)),
            hourly_limit=int(getattr(self.config, "ai_companion_group_hourly_limit", 20)),
            heat_window_seconds=float(getattr(self.config, "ai_companion_heat_window_seconds", 120)),
        )

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
        reply_decision = self._reply_decision(bot, event, plain_text, ids, media_parts)
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

        quote_target = self._reply_quote_target(ids)
        try:
            sent = await self._send_adapter_reply(
                bot=bot,
                event=event,
                matcher=matcher,
                text=clean_text,
                sticker_keys=sticker_keys,
                reply_to_message_id=quote_target,
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
        await self._record_bot_turn(
            ids=ids,
            text=clean_text,
            sticker_keys=sticker_keys,
            trigger_reason=reply_decision.reason,
            reply_to_message_id=str(ids.get("message_id") or ""),
        )
        if ids.get("group_id"):
            self.interjection_limiter.observe_reply(
                group_id=str(ids["group_id"]), user_id=str(ids.get("user_id") or "")
            )
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
        try:
            return await self._generate_reply_online(reply_plan, allowed_tools=allowed_tools)
        except Exception as exc:  # noqa: BLE001 — 在线端点失败：按配置降级到本地 MLX
            fallback = await self._fallback_to_local(reply_plan, error=exc)
            if fallback is None:
                raise
            return fallback

    async def _fallback_to_local(self, reply_plan: ReplyPlan, *, error: Exception) -> str | None:
        """在线端点挂了（429/超时/断链）时用本地 MLX 兜底；未加载/超时则放弃。

        不主动拉起大模型（首次加载要几分钟，会把回复队列卡死），
        只给已经加载好的运行时机会；超时上限由配置控制。
        """
        if not bool(getattr(self.config, "ai_companion_fallback_to_local", True)):
            return None
        if not self._online_backend_active():
            return None
        runtime = self.mlx_runtime
        if runtime is None or not hasattr(runtime, "chat"):
            return None
        try:
            if str(runtime.status_snapshot().get("state") or "") != "loaded":
                logger.warning(f"ai companion online backend failed ({error!r}); local mlx not loaded, skipping fallback")
                return None
        except Exception:  # noqa: BLE001 — 状态取不到就按未加载处理
            return None
        timeout = float(getattr(self.config, "ai_companion_fallback_timeout_seconds", 120.0) or 120.0)
        logger.warning(f"ai companion online backend failed ({error!r}); falling back to local MLX")
        try:
            return await asyncio.wait_for(
                runtime.chat(reply_plan.messages, media_parts=reply_plan.media_parts),
                timeout=timeout,
            )
        except Exception as fallback_exc:  # noqa: BLE001 — 兜底也失败就向上抛原始错误
            logger.warning(f"ai companion local fallback failed: {fallback_exc!r}")
            return None

    async def _generate_reply_online(
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
        """主回复路径的图片 captioner：通用视觉理解（类型/图内文字/槽点），禁用则 None。"""
        if not getattr(self.config, "ai_companion_anime_caption_enabled", True):
            return None
        captioner: Any = self.captioner

        class _GeneralReactionCaptioner:
            """MediaCaptioner 适配：走 caption_media_general（打趣素材向）。"""

            async def caption_media(self, part: dict[str, Any]) -> str:
                return await captioner.caption_media_general(part)

        return _GeneralReactionCaptioner()

    def _reply_decision(
        self,
        bot: Bot,
        event: Event,
        plain_text: str,
        ids: dict[str, str | None],
        media_parts: list[dict[str, Any]] | None = None,
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
        # 纯图片消息也算"有内容"（打趣回复的前提）；普通表情(face/mface)不算，避免刷触发
        has_media = any(str(part.get("type") or "") == "image" for part in (media_parts or []))
        group_id = ids.get("group_id")
        if group_id:
            self.interjection_limiter.observe_message(group_id=group_id, user_id=str(ids.get("user_id") or ""))
        weighted = weighted_probability(
            InterjectionInput(
                base_probability=float(self.config.ai_companion_reply_probability),
                addressed_recently=False,
                heat_messages=self.interjection_limiter.heat_messages(group_id=group_id or ""),
                heat_min_messages=int(getattr(self.config, "ai_companion_heat_min_messages", 5)),
                same_speaker_as_last_reply=self.interjection_limiter.same_speaker_as_last_reply(
                    group_id=group_id or "", user_id=str(ids.get("user_id") or "")
                ),
            )
        )
        decision = should_attempt_reply(
            enable_reply=self.config.ai_companion_enable_reply,
            group_enabled=private_chat or self.env.group_enabled(group_id),
            self_event=self.env.is_self_event(bot, ids),
            blacklisted=self.env.is_blacklisted(ids["user_id"], group_id, plain_text),
            command_like=self.env.is_command_message(plain_text),
            force_reply=force_reply,
            has_text=bool(str(plain_text or "").strip()) or force_reply,
            reply_probability=weighted.probability,
            follow_reply_probability=self.config.ai_companion_follow_reply_probability,
            random_value=random.random(),
            has_media=has_media,
        )
        if decision.allowed and group_id and not force_reply:
            # 主动插话受限流约束：冷却期内 / 超过每小时上限就闭嘴（被叫的不受限制）
            blocked = self.interjection_limiter.block_reason(group_id=group_id)
            if blocked:
                return ReplyDecision(False, f"rate_limited:{blocked}")
            decision = ReplyDecision(True, f"{decision.reason}[{weighted.describe()}]")
        return decision

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
    def _reply_mode(self, *, group_id: str | None, envelope: MessageContextEnvelope) -> str:
        """成员模式 vs 助手模式：私聊、被 @、被回复 → 助手口气；群里未被叫 → 成员口气。"""
        if group_id is None:
            return "assistant"
        if envelope.forced_by_at:
            return "assistant"
        reply_sender = envelope.reply_sender or {}
        self_id = self.env.cached_self_id()
        if reply_sender and self_id and str(reply_sender.get("user_id") or "") == self_id:
            return "assistant"
        return "member"

    def _build_system_prompt(
        self,
        *,
        card: RoleCard | None = None,
        user_nickname: str | None = None,
        memory_lines: list[str] | None = None,
        scene_lines: list[str] | None = None,
        mode: str = "member",
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
                scene_lines=list(scene_lines or []),
                mode=str(mode or "member"),
            )
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
        context_limit = effective_context_message_limit(self.config, online=self._online_backend_active())
        query_text = self._context_query_text(envelope)
        # 角色卡：会话绑定的卡渲染身份段与工具白名单（不改历史作用域）。
        card, binding = (
            await self.role_card_service.get_effective_card(group_id=group_id, user_id=ids.get("user_id"))
            if self.role_card_service is not None
            else (None, None)
        )
        # 会话时间线：只取本会话（群 or 私聊）的对话，带说话人；bot 自己的话也在里面。
        # 不再按角色卡绑定时刻砍历史：绑定决定"用哪张卡"，历史是事实（换人格不该失忆）。
        session_key = session_key_for(group_id=group_id, user_id=ids.get("user_id"))
        history_turns = await self.store.conversation.recent_turns(
            session_key=session_key,
            limit=context_limit,
        )
        history_names = await self.store.member_names.get_names(
            group_id=group_id,
            user_ids=collect_user_ids(history_turns),
        )
        history_lines = render_history_lines(
            history_turns,
            names=history_names,
            exclude_message_ids=[str(ids.get("message_id") or "")],
        )
        relevant_samples, relevant_media_assets = await asyncio.gather(
            self._search_relevant_context_samples(
                group_id, query_text, max(3, context_limit // 2), ids_user_id=ids.get("user_id")
            ),
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
            # 图内容打趣指引：像群友一样就图吐槽，不要机械复述描述
            conversation_lines.append("（上面是当前消息图片的理解素材：优先自然地就图内容吐槽/玩梗，别复述描述文字）")
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

        conversation_lines = trim_history_lines_for_context(
            conversation_lines, self.config, online=self._online_backend_active()
        )
        history_lines = trim_history_lines_for_context(history_lines, self.config, online=self._online_backend_active())
        # 记忆注入：群记忆 + 在场成员（说话人 / 被@ / 被引用）的记忆，按敏感度分级。
        # 群聊是公开场合：即使被 @ 也只注入 neutral/relation；私聊才注入情绪与隐私类。
        memory_lines: list[str] = []
        if "memory" in effective_segment_order(self.config) and bool(
            getattr(self.config, "ai_companion_memory_enabled", True)
        ):
            member_ids = [str(ids.get("user_id") or "")]
            member_ids.extend(str(target.get("qq") or "") for target in envelope.at_targets)
            if envelope.reply_sender:
                member_ids.append(str(envelope.reply_sender.get("user_id") or ""))
            try:
                if group_id is None:
                    # 私聊没有群作用域：聚合这个人在各个群里的成员记忆（含敏感类）
                    memory_lines = await self.store.persona_memory.prompt_lines_for_private_member(
                        user_id=str(ids.get("user_id") or ""),
                        include_sensitive=True,
                    )
                else:
                    memory_lines = await self.store.persona_memory.prompt_lines_for_session(
                        group_id=group_id,
                        user_ids=[item for item in member_ids if item],
                        include_sensitive=False,
                    )
            except Exception as exc:  # noqa: BLE001 — 记忆取数失败不阻塞回复
                logger.debug("persona memory fetch skipped: %r", exc)

        # 滚动摘要：跨小时的话题背景（scene 段）
        scene_lines: list[str] = []
        if group_id and "scene" in effective_segment_order(self.config) and bool(
            getattr(self.config, "ai_companion_summary_enabled", True)
        ):
            try:
                summary_record = await self.store.group_summaries.get(group_id=str(group_id))
                if summary_record and summary_record.summary.strip():
                    scene_lines = [f"本群最近话题摘要：{summary_record.summary.strip()}"]
            except Exception as exc:  # noqa: BLE001 — 摘要取不到不阻塞回复
                logger.debug("group summary fetch skipped: %r", exc)

        active_memes = select_active_memes(
            build_meme_candidates_from_samples(
                samples=list(relevant_samples),
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
                scene_lines=scene_lines,
                mode=self._reply_mode(group_id=group_id, envelope=envelope),
            ),
            conversation_lines=conversation_lines,
            history_lines=history_lines,
            active_memes=active_memes,
            current_text=envelope.current_text,
            media_parts=media_parts,
            sticker_reply_allowed=sticker_reply_allowed,
            incoming_has_sticker=incoming_has_sticker,
        )

    def _online_backend_active(self) -> bool:
        """当前是否走在线 LLM 后端（决定上下文条数与字符预算）。"""
        try:
            return self.client.active_backend() == "llm"
        except Exception:  # noqa: BLE001 — 判定失败按本地保守预算
            return False

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
        *,
        ids_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        global_scope = str(getattr(self.config, "ai_companion_context_scope", "") or "") == "global"
        try:
            return await self.store.search_context_samples(
                group_id=group_id,
                user_id=str(ids_user_id or "") or None,
                query=query,
                limit=limit,
                max_age_hours=int(self.config.ai_companion_relevant_context_max_age_hours or 0),
                global_scope=global_scope,
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
        global_scope = str(getattr(self.config, "ai_companion_context_scope", "") or "") == "global"
        try:
            return await self.store.search_media_assets(
                group_id=group_id,
                query=query,
                limit=limit,
                max_age_hours=int(self.config.ai_companion_relevant_context_max_age_hours or 0),
                global_scope=global_scope,
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
            # 名字永远不回落成 QQ 号：解析不到就用「群友xxxx」，避免数字 id 被复述出来
            reply_name = safe_speaker_name(
                str(envelope.reply_sender.get("nickname") or ""),
                user_id=str(envelope.reply_sender.get("user_id") or ""),
            )
            for text in envelope.quoted_texts:
                lines.append(f"回复{reply_name}的消息：{text}")
            if not envelope.quoted_texts:
                lines.append(f"当前消息回复了{reply_name}。")
        else:
            for text in envelope.quoted_texts:
                lines.append(f"引用消息：{text}")
        if envelope.at_targets:
            names = "、".join(
                safe_speaker_name(str(target.get("name") or ""), user_id=str(target.get("qq") or ""))
                for target in envelope.at_targets
            )
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
            bot=bot,
            event=event,
            matcher=matcher,
            text=reply_text,
            sticker_keys=candidate.sticker_keys,
            reply_to_message_id=self._reply_quote_target(ids),
        )
        if sent.get("ok"):
            await self._record_bot_turn(
                ids=ids,
                text=reply_text,
                sticker_keys=list(candidate.sticker_keys),
                trigger_reason="repeat",
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
            bot=bot,
            event=event,
            matcher=matcher,
            text=reply_text,
            sticker_keys=sticker_keys,
            reply_to_message_id=self._reply_quote_target(ids),
        )
        if sent.get("ok"):
            await self._record_bot_turn(
                ids=ids,
                text=reply_text,
                sticker_keys=list(sticker_keys),
                trigger_reason="conversation_pattern",
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
    async def _record_bot_turn(
        self,
        *,
        ids: dict[str, str | None],
        text: str,
        sticker_keys: list[str],
        trigger_reason: str,
        reply_to_message_id: str = "",
    ) -> None:
        """把 bot 这条回复写进会话时间线：下一轮它才看得见自己说过什么。"""
        try:
            await self.store.conversation.record_turn(
                session_key=session_key_for(group_id=ids.get("group_id"), user_id=ids.get("user_id")),
                speaker_kind=SPEAKER_BOT,
                text=str(text or ""),
                group_id=ids.get("group_id"),
                user_id=None,
                speaker_name="",
                sticker_keys=[str(key) for key in (sticker_keys or []) if str(key).strip()],
                trigger_reason=str(trigger_reason or ""),
                reply_to_message_id=str(reply_to_message_id or ""),
            )
        except Exception as exc:  # noqa: BLE001 — 时间线写失败不影响发送结果
            logger.debug("bot turn record skipped: %r", exc)

    def _reply_quote_target(self, ids: dict[str, str | None]) -> str | None:
        """引用回复的目标：默认引用触发它的那条消息；开关关闭或 id 不可引用时返回 None。"""
        if not bool(getattr(self.config, "ai_companion_reply_quote", True)):
            return None
        message_id = str(ids.get("message_id") or "").strip()
        return message_id if message_id.isdigit() else None

    async def _send_adapter_reply(
        self,
        *,
        bot: Bot | None,
        event: Event,
        matcher: Matcher,
        text: str,
        sticker_keys: list[str],
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        """发送回复；返回 {ok, reply_message_id}（id 供 MessageLog 溯源）。

        长文合并转发无法挂引用段，那条路径不带引用。
        """
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
                        reply_to_message_id=reply_to_message_id,
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
            reply_to_message_id=reply_to_message_id,
        )
        if isinstance(reply, str) and not reply.strip():
            return {"ok": False, "reply_message_id": None}
        try:
            sent = await matcher.send(reply)
        except Exception as exc:  # noqa: BLE001 — 表情资源过期（NapCat 下载 400 等）不应连带丢掉文字
            if sticker_keys and str(text or "").strip():
                logger.warning(f"ai companion sticker send failed ({type(exc).__name__}: {exc}), retrying text-only")
                sent = await matcher.send(text)
            else:
                raise
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
        selection_channel = ""
        try:
            selection = await select_reply_sticker(await self._vision_provider(), **selector_kwargs)
            selection_channel = "vision"
        except Exception as exc:  # noqa: BLE001 — vision 通道异常时降级文本通道
            selection_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"ai companion sticker selection vision channel failed: {exc!r}")

        if selection is None or selection.fallback or not selection.key:
            # vision 不可用/失败/兜底：尝试当前 chat provider 的纯文本选择
            try:
                text_selection = await select_reply_sticker(self.client, **selector_kwargs)
                text_real = text_selection is not None and text_selection.key and not text_selection.fallback
                if text_real or selection is None:
                    selection = text_selection
                    selection_channel = "text"
            except Exception as exc:  # noqa: BLE001 — 文本通道也失败则保留兜底
                selection_channel = "vision(fallback)"
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
                "channel": selection_channel or "none",
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
