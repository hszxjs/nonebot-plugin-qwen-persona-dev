from __future__ import annotations

import asyncio
import random
import re
import time
from pathlib import Path
from typing import Any

from nonebot import get_driver, logger, on_command, on_message, on_notice, require
from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

from .admin_server import AdminServer
from .anime_captioner import AnimeCaptioner
from .command_discovery import discover_picmenu_command_regexes
from .config import QwenPersonaConfig, load_plugin_config, normalize_matcher_priorities
from .context.envelope import MessageContextEnvelope, build_message_context
from .context.envelope import _at_targets_from_message  # noqa: F401 — 仅供主语提取预扫描
from .context.forward import count_forward_segments, resolve_forward
from .context.media_parts import message_to_parts
from .context.nicknames import nickname_resolver
from .decision.gates import (
    ReplyDecision,
    is_private_chat,
    should_attempt_reply,
    should_force_reply,
    should_learn_text,
    should_use_sticker_reply,
)
from .decision.repeat import (
    ConversationPatternTracker,
    RepeatTracker,
    build_conversation_pattern_message,
)
from .delivery.onebot_v11 import is_poke_at_self, send_counter_poke
from .delivery.plan import plan_reply_delivery
from .finetune.trainer import PersonaTrainer, parse_time_slot
from .memory.ingest import filter_training_text_sticker_tokens, prepare_message_learning
from .observability.inference_events import record_inference_event
from .prompt.budget import effective_context_message_limit, trim_history_lines_for_context
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
from .runtime_config import (
    apply_config_values,
    config_to_dict,
    load_config_overrides,
    save_config_overrides,
)
from .sticker.analysis import analyze_media_parts, should_analyze_background_media
from .sticker.model import (
    build_onebot_v11_sticker_message,
    extract_plain_text,
    parse_sticker_tokens,
)
from .sticker.repo_files import cleanup_sticker_repo_manifest, record_sticker_repo_manifest
from .sticker.selector import select_reply_sticker
from .store import PersonaStore
from .tools.registry import ToolRegistry

__plugin_meta__ = PluginMetadata(
    name="AI Companion",
    description="AI 对话伙伴：本地 MLX / 在线 GLM 双推理后端，学习群友语气与表情包，支持 LoRA 训练。",
    usage=(
        "配置推理后端后，@机器人即可聊天。\n"
        "命令：人格状态、人格训练（SUPERUSER）。"
    ),
    type="application",
    homepage="https://github.com/hszxjs/nonebot-plugin-qwen-persona-dev",
    supported_adapters=None,
    config=QwenPersonaConfig,
)

driver = get_driver()
plugin_config = load_plugin_config(driver.config)
try:
    apply_config_values(plugin_config, load_config_overrides(plugin_config))
except Exception as exc:
    logger.warning(f"qwen persona admin config load failed: {exc!r}")
priority_changes = normalize_matcher_priorities(plugin_config)
if priority_changes:
    logger.warning(
        "qwen persona matcher priorities normalized: "
        f"learning={plugin_config.qwen_persona_learning_priority}, "
        f"reply={plugin_config.qwen_persona_reply_priority}"
    )
store = PersonaStore(Path(plugin_config.qwen_persona_data_dir))
mlx_runtime = MlxVLMRuntime(plugin_config)
client = QwenClient(plugin_config, runtime=mlx_runtime)
anime_captioner = AnimeCaptioner(plugin_config)
trainer = PersonaTrainer(plugin_config, store)


def _active_reply_captioner() -> Any:
    """主回复路径的图片 captioner：优先动漫识别，禁用则返回 None。"""
    if getattr(plugin_config, "qwen_persona_anime_caption_enabled", True):
        return anime_captioner
    return None
repeat_tracker = RepeatTracker()
conversation_pattern_tracker = ConversationPatternTracker()
tool_registry = ToolRegistry()
admin_server = AdminServer(
    config=plugin_config,
    save_callback=lambda values: _save_admin_config(values),
    logger=logger,
    mlx_runtime=mlx_runtime,
    store=store,
    qwen_client=client,
    trainer=trainer,
)
_schedule_installed = False
_background_media_tasks: set[asyncio.Task[None]] = set()


_CONFIG_RESTART_REQUIRED = {
    "qwen_persona_mlx_model",
    "qwen_persona_mlx_adapter_path",
    "qwen_persona_mlx_auto_load",
    "qwen_persona_mlx_trust_remote_code",
    "qwen_persona_model_dir",
    "qwen_persona_admin_enabled",
    "qwen_persona_admin_host",
    "qwen_persona_admin_port",
    "qwen_persona_data_dir",
    "qwen_persona_train_times",
    "qwen_persona_enable_training",
    "qwen_persona_learning_priority",
    "qwen_persona_reply_priority",
}


def _save_admin_config(values: dict[str, Any]) -> dict[str, Any]:
    changed = apply_config_values(plugin_config, values)
    changed.extend(normalize_matcher_priorities(plugin_config))
    changed = list(dict.fromkeys(changed))
    save_config_overrides(plugin_config, config_to_dict(plugin_config))
    mlx_runtime.reconfigure(plugin_config)
    client.reconfigure(plugin_config)
    restart_required = [name for name in changed if name in _CONFIG_RESTART_REQUIRED]
    return {"changed": changed, "restart_required": restart_required}


def _dump_event_ids(event: Event) -> dict[str, str | None]:
    def as_str(value: Any) -> str | None:
        return None if value is None else str(value)

    try:
        user_id = event.get_user_id()
    except Exception:
        user_id = getattr(event, "user_id", None)

    return {
        "platform": getattr(event, "platform", None) or event.__class__.__module__,
        "user_id": as_str(user_id),
        "group_id": as_str(
            getattr(event, "group_id", None)
            or getattr(event, "guild_id", None)
            or getattr(event, "channel_id", None)
        ),
        "message_id": as_str(getattr(event, "message_id", None)),
    }


def _bot_self_id(bot: Bot) -> str | None:
    value = str(getattr(bot, "self_id", "") or "")
    return value or None


def _is_self_event(bot: Bot, ids: dict[str, str | None]) -> bool:
    self_id = _bot_self_id(bot)
    return bool(self_id and ids.get("user_id") == self_id)


def _group_enabled(group_id: str | None) -> bool:
    enabled = {str(item) for item in plugin_config.qwen_persona_enabled_groups}
    return not enabled or (group_id is not None and group_id in enabled)


def _command_start_values() -> list[str]:
    values = list(plugin_config.qwen_persona_command_prefixes)
    values.extend(str(item) for item in getattr(driver.config, "command_start", set()) or set())
    return values


def _command_regex_values() -> list[str]:
    regexes = list(plugin_config.qwen_persona_command_regexes)
    if plugin_config.qwen_persona_auto_read_picmenu_config:
        regexes.extend(
            discover_picmenu_command_regexes(
                driver.config,
                config_keys=plugin_config.qwen_persona_picmenu_config_keys,
                command_keys=plugin_config.qwen_persona_picmenu_command_keys,
                prefixes=_command_start_values(),
            )
        )
    return list(dict.fromkeys(regexes))


def _is_blacklisted(user_id: str | None, group_id: str | None, text: str) -> bool:
    if user_id and user_id in {str(item) for item in plugin_config.qwen_persona_blacklist_user_ids}:
        return True
    if group_id and group_id in {str(item) for item in plugin_config.qwen_persona_blacklist_group_ids}:
        return True
    for word in plugin_config.qwen_persona_blacklist_words:
        word = str(word).strip()
        if word and word in text:
            return True
    for pattern in plugin_config.qwen_persona_blacklist_regexes:
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


def _remove_blacklisted_reply_lines(text: str) -> str:
    kept: list[str] = []
    for line in (text or "").splitlines():
        if not _is_blacklisted(None, None, line):
            kept.append(line)
    return "\n".join(kept).strip()


def _is_command_message(text: str) -> bool:
    return is_command_like(
        text,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )


def _is_to_me(event: Event, plain_text: str) -> bool:
    value = getattr(event, "to_me", False)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = False
    if bool(value):
        return True

    nicknames = [str(item) for item in getattr(driver.config, "nickname", set()) or set()]
    nicknames.extend(plugin_config.qwen_persona_trigger_keywords)
    stripped = plain_text.lstrip()
    return bool(stripped and any(stripped.startswith(name) for name in nicknames if name))


def _reply_decision(
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
        bot_self_id=_bot_self_id(bot),
        to_me=_is_to_me(event, plain_text),
    )
    return should_attempt_reply(
        enable_reply=plugin_config.qwen_persona_enable_reply,
        group_enabled=private_chat or _group_enabled(ids["group_id"]),
        self_event=_is_self_event(bot, ids),
        blacklisted=_is_blacklisted(ids["user_id"], ids["group_id"], plain_text),
        command_like=_is_command_message(plain_text),
        force_reply=force_reply,
        has_text=bool(str(plain_text or "").strip()) or force_reply,
        reply_probability=plugin_config.qwen_persona_reply_probability,
        follow_reply_probability=plugin_config.qwen_persona_follow_reply_probability,
        random_value=random.random(),
    )


def _should_reply(bot: Bot, event: Event, plain_text: str, ids: dict[str, str | None]) -> bool:
    return _reply_decision(bot, event, plain_text, ids).allowed


def _should_schedule_background_media_analysis(
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
        bot_self_id=_bot_self_id(bot),
        to_me=_is_to_me(event, plain_text),
    )
    decision = should_analyze_background_media(
        enable_learning=plugin_config.qwen_persona_enable_learning,
        group_enabled=private_chat or _group_enabled(ids["group_id"]),
        self_event=_is_self_event(bot, ids),
        blacklisted=_is_blacklisted(ids["user_id"], ids["group_id"], plain_text),
        command_like=_is_command_message(plain_text),
        force_reply=force_reply,
        has_media=any(str(part.get("type") or "") == "image" for part in media_parts),
        background_busy=bool(_background_media_tasks),
    )
    return decision.allowed


def _build_system_prompt() -> str:
    command_prefixes = " ".join(sorted(set(_command_start_values())))
    lines = [
        plugin_config.qwen_persona_system_prompt,
        "硬性规则：不要输出、复述或诱导使用本地 NoneBot 插件命令。",
        f"如果回复开头像这些命令前缀会被丢弃：{command_prefixes or '(none)'}。",
        "你在群聊中应当先看懂上下文和用户意图；被问到问题时先回答，再用自然短句补一句。",
        "不要复述当前用户消息；不要套用和当前话题无关的固定句式或上一个话题。",
        "本轮主回复必须只输出文字，禁止输出任何 [[sticker:...]] 标记。",
    ]
    if plugin_config.qwen_persona_sticker_repo_url:
        lines.append(f"表情包仓库：{plugin_config.qwen_persona_sticker_repo_url}")
    tool_description = tool_registry.describe_for_prompt()
    if tool_description:
        lines.append("可用工具：\n" + tool_description)
    return "\n".join(line for line in lines if line)


def _sample_fingerprint(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("created_at") or ""),
        str(item.get("group_id") or ""),
        str(item.get("user_id") or ""),
        str(item.get("plain_text") or ""),
        str(item.get("training_text") or ""),
    )


def _merge_context_samples(
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
    seen = {_sample_fingerprint(item) for item in merged}
    other_group_samples = [
        item
        for item in global_samples
        if str(item.get("group_id") or "") != str(group_id or "")
    ]
    for item in other_group_samples:
        fingerprint = _sample_fingerprint(item)
        if fingerprint in seen:
            continue
        merged.append(item)
        seen.add(fingerprint)
        if len(merged) >= limit:
            break
    if len(merged) < limit:
        for item in global_samples:
            fingerprint = _sample_fingerprint(item)
            if fingerprint in seen:
                continue
            merged.append(item)
            seen.add(fingerprint)
            if len(merged) >= limit:
                break
    merged.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    return merged[:limit]


async def _recent_context_samples(group_id: str | None, limit: int) -> list[dict[str, Any]]:
    scope = str(getattr(plugin_config, "qwen_persona_context_scope", "mixed") or "mixed")
    if scope == "group":
        return await store.recent_samples(group_id=group_id, limit=limit)
    if scope == "global" or not group_id:
        return await store.recent_samples(group_id=None, limit=limit)

    group_limit = max(1, limit // 2)
    group_samples, global_samples = await asyncio.gather(
        store.recent_samples(group_id=group_id, limit=group_limit),
        store.recent_samples(group_id=None, limit=max(limit * 3, limit + group_limit)),
    )
    return _merge_context_samples(
        group_samples,
        global_samples,
        group_id=group_id,
        limit=limit,
    )


def _is_context_noise(text: str) -> bool:
    return is_training_noise(text) or is_command_like(
        text,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )


async def _build_model_reply_plan(
    envelope: MessageContextEnvelope,
    group_id: str | None,
    *,
    ids: dict[str, str | None],
    sticker_reply_allowed: bool,
    incoming_has_sticker: bool,
    media_parts: list[dict[str, Any]],
    media_context_text: str = "",
) -> ReplyPlan:
    context_limit = effective_context_message_limit(plugin_config)
    query_text = _context_query_text(envelope)
    history, relevant_samples, relevant_media_assets = await asyncio.gather(
        _recent_context_samples(group_id, context_limit),
        _search_relevant_context_samples(group_id, query_text, max(3, context_limit // 2)),
        _search_relevant_media_assets(group_id, query_text, max(2, context_limit // 3)),
    )

    conversation_lines = _conversation_lines_from_envelope(envelope)
    # 图文相关性判断：若图片描述与对话文字无关，丢弃图片描述只保留文字。
    if media_context_text and getattr(plugin_config, "qwen_persona_media_relevance_check", True):
        # 取对话文字（envelope.current_text + 历史样本）用于相关性对比
        dialogue_text = envelope.current_text
        if not dialogue_text and conversation_lines:
            dialogue_text = " ".join(conversation_lines)
        if dialogue_text:
            relevant = await _check_media_relevance(media_context_text, dialogue_text)
            if not relevant:
                logger.info("qwen persona media: caption dropped (irrelevant to text)")
                media_context_text = ""
    if media_context_text:
        conversation_lines.append(media_context_text)
    # 诊断：记录 caption 是否进了 conversation_lines
    try:
        from pathlib import Path as _P
        _p = _P("data/qwen_persona/caption_debug.log")
        _p.parent.mkdir(parents=True, exist_ok=True)
        import time as _t
        with _p.open("a", encoding="utf-8") as _f:
            _f.write(_t.strftime("%Y-%m-%d %H:%M:%S ") + f"PROMPT media_context_text={media_context_text!r} conversation_lines_count={len(conversation_lines)}\n")
    except OSError:
        pass
    for item in relevant_samples:
        context_text = context_text_from_sample(item)
        context_text, _sticker_keys = parse_sticker_tokens(context_text)
        if _is_context_noise(context_text):
            continue
        if context_text:
            conversation_lines.append(f"相关历史：{context_text}")
    for item in relevant_media_assets:
        answer = str(item.get("answer") or "").strip()
        if answer and not _is_context_noise(answer):
            conversation_lines.append(f"历史图片理解：{answer}")

    style_memory_lines: list[str] = []
    current_message_id = str(ids.get("message_id") or "")
    for item in reversed(history):
        if current_message_id and str(item.get("message_id") or "") == current_message_id:
            continue
        context_text = context_text_from_sample(item)
        context_text, _sticker_keys = parse_sticker_tokens(context_text)
        if _is_context_noise(context_text):
            continue
        if context_text:
            style_memory_lines.append(context_text)
    conversation_lines = trim_history_lines_for_context(conversation_lines, plugin_config)
    style_memory_lines = trim_history_lines_for_context(style_memory_lines, plugin_config)
    active_memes = select_active_memes(
        build_meme_candidates_from_samples(
            samples=[*history, *relevant_samples],
            media_assets=relevant_media_assets,
        ),
        context_text=query_text,
        min_recent_count=plugin_config.qwen_persona_min_meme_recent_count,
        max_items=3,
    )

    return build_reply_plan(
        system_prompt=_build_system_prompt(),
        conversation_lines=conversation_lines,
        style_memory_lines=style_memory_lines,
        active_memes=active_memes,
        current_text=envelope.current_text,
        media_parts=media_parts,
        sticker_reply_allowed=sticker_reply_allowed,
        incoming_has_sticker=incoming_has_sticker,
    )


def _context_query_text(envelope: MessageContextEnvelope) -> str:
    chunks = [envelope.current_text]
    chunks.extend(envelope.quoted_texts)
    chunks.extend(envelope.forwarded_texts)
    return " ".join(chunk for chunk in chunks if chunk).strip()


async def _search_relevant_context_samples(
    group_id: str | None,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not query:
        return []
    scope = str(getattr(plugin_config, "qwen_persona_context_scope", "mixed") or "mixed")
    search_group_id = None if scope == "global" else group_id
    try:
        return await store.search_context_samples(
            group_id=search_group_id,
            query=query,
            limit=limit,
        )
    except Exception as exc:
        logger.warning(f"qwen persona context search failed: {exc!r}")
        return []


async def _search_relevant_media_assets(
    group_id: str | None,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not query:
        return []
    scope = str(getattr(plugin_config, "qwen_persona_context_scope", "mixed") or "mixed")
    search_group_id = None if scope == "global" else group_id
    try:
        return await store.search_media_assets(
            group_id=search_group_id,
            query=query,
            limit=limit,
        )
    except Exception as exc:
        logger.warning(f"qwen persona media context search failed: {exc!r}")
        return []


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


async def _check_media_relevance(media_context_text: str, conversation_text: str) -> bool:
    """判断图片描述与文字内容是否相关。返回 True=相关保留，False=无关丢弃。

    用 glm-4.5-flash 做轻量判断（免费/便宜），max_tokens=20 只回是/否。
    纯图消息（无文字）跳过判断直接返回 True。
    """
    if not media_context_text or not conversation_text:
        return True  # 无文字可对比，保留图片描述
    try:
        import httpx
        # 复用在线推理配置，但强制用便宜的 flash 模型做判断
        url = (plugin_config.qwen_persona_online_api_url or "").strip()
        # 视觉模型走 paas/v4，文本判断也走同平台；用 coding 端点的 glm-5.2 也行
        # 优先用 online 端点（用户已配的 GLM 推理），降级用 lora_helper
        if not url:
            url = (plugin_config.qwen_persona_lora_helper_api_url or "").strip()
        key = (plugin_config.qwen_persona_online_api_key or "").strip()
        if not key:
            key = (plugin_config.qwen_persona_lora_helper_api_key or "").strip()
        if not url or not key:
            return True  # 没配置无法判断，保守保留
        # 判断模型：online 模型通常是 glm-5.2（贵），判断用 flash 更省
        judge_model = "glm-4.5-flash"
        prompt = (
            "判断下面【图片描述】和【对话文字】是否讨论同一话题/角色/场景。\n"
            f"【图片描述】{media_context_text[:200]}\n"
            f"【对话文字】{conversation_text[:300]}\n"
            "只回复一个字：相关 或 无关"
        )
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": judge_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0},
            )
        if r.status_code != 200:
            return True  # 判断失败保守保留
        content = str((r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        relevant = "无关" not in content  # 含"无关"判为不相关，其余视为相关
        logger.info("qwen persona media relevance: judge=%r relevant=%s", content[:20], relevant)
        return relevant
    except Exception as exc:  # noqa: BLE001
        logger.debug("media relevance check skipped: %r", exc)
        return True  # 出错保守保留


async def _build_reply_input_text(
    message: Any,
    plain_text: str,
) -> tuple[str, list[str], list[dict[str, Any]], str]:
    active_captioner = _active_reply_captioner()
    logger.info(
        "qwen persona reply media: captioner=%s, anime_enabled=%s, media_segments=%d",
        type(active_captioner).__name__ if active_captioner else "None",
        getattr(plugin_config, "qwen_persona_anime_caption_enabled", True),
        sum(1 for p in message_to_parts(message) if str(p.get("type") or "") in {"image", "mface", "emoji", "sticker"}),
    )
    try:
        learning = await prepare_message_learning(
            message,
            sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
            captioner=active_captioner,
            image_fallback_text="",
            sticker_fallback_text="",
        )
    except Exception as exc:
        logger.debug(f"qwen persona reply media context skipped: {exc!r}")
        return plain_text, [], [part for part in message_to_parts(message) if part.get("type") != "text"], ""

    sticker_keys = [
        str(item.get("key") or "").strip()
        for item in learning.get("stickers") or []
        if str(item.get("key") or "").strip()
    ]
    media_context_text = str(learning.get("media_context_text") or "")
    return plain_text, sticker_keys, list(learning.get("media_parts") or []), media_context_text


def _schedule_background_media_analysis(
    ids: dict[str, str | None],
    media_parts: list[dict[str, Any]],
) -> None:
    image_parts = [part for part in media_parts if str(part.get("type") or "") == "image"]
    if not image_parts:
        return
    if _background_media_tasks:
        return
    task = asyncio.create_task(
        _run_background_media_analysis(ids=dict(ids), media_parts=image_parts)
    )
    _background_media_tasks.add(task)

    def _discard(done: asyncio.Task[None]) -> None:
        _background_media_tasks.discard(done)

    task.add_done_callback(_discard)


async def _run_background_media_analysis(
    *,
    ids: dict[str, str | None],
    media_parts: list[dict[str, Any]],
) -> None:
    if not await _wait_for_generation_idle(timeout_seconds=12.0):
        await _record_sticker_event(
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
            store=store,
            captioner=mlx_runtime,
            parts=media_parts,
            group_id=ids.get("group_id"),
            user_id=ids.get("user_id"),
            sticker_segment_types=set(plugin_config.qwen_persona_sticker_segment_types),
        )
    except Exception as exc:
        logger.warning(f"qwen persona background media analysis failed: {exc!r}")
        await _record_sticker_event(
            event_type="media_analysis",
            status="failed",
            group_id=ids.get("group_id"),
            user_id=ids.get("user_id"),
            message=f"{type(exc).__name__}: {exc}",
            details={"media_count": len(media_parts)},
        )
        return
    for event in result.events:
        await _record_sticker_event(
            event_type=str(event.get("event_type") or "media_analysis"),
            status=str(event.get("status") or "unknown"),
            key=str(event.get("key") or "") or None,
            group_id=ids.get("group_id"),
            user_id=ids.get("user_id"),
            message=str(event.get("message") or ""),
            details=dict(event.get("details") or {}),
        )


async def _wait_for_generation_idle(*, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            generation = dict(mlx_runtime.status_snapshot().get("generation") or {})
            active = int(generation.get("active") or 0)
            queued = int(generation.get("queued") or 0)
        except Exception:
            return True
        if active <= 0 and queued <= 0:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "qwen persona background media analysis skipped: "
                f"generation busy active={active}, queued={queued}"
            )
            return False
        await asyncio.sleep(1.0)


async def _send_repeat_if_needed(
    bot: Bot,
    event: Event,
    ids: dict[str, str | None],
    message: Any,
) -> bool:
    if not plugin_config.qwen_persona_enable_repeat:
        return False
    candidate = repeat_tracker.observe(
        group_id=ids["group_id"],
        message=message,
        sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
        min_count=plugin_config.qwen_persona_repeat_min_count,
        cooldown_seconds=plugin_config.qwen_persona_repeat_cooldown_seconds,
    )
    if not candidate:
        return False
    if random.random() >= plugin_config.qwen_persona_repeat_probability:
        return False
    if _is_command_message(candidate.plain_text):
        return False
    if _is_blacklisted(ids["user_id"], ids["group_id"], candidate.plain_text):
        return False
    reply_text = _remove_blacklisted_reply_lines(candidate.plain_text)
    if not reply_text and not candidate.sticker_keys:
        return False
    return await _send_adapter_reply(
        bot=bot,
        event=event,
        text=reply_text,
        sticker_keys=candidate.sticker_keys,
    )


async def _send_pattern_reply_if_known(
    bot: Bot,
    event: Event,
    ids: dict[str, str | None],
    message: Any,
    plain_text: str,
) -> bool:
    if not plugin_config.qwen_persona_enable_reply:
        return False
    if not ids["group_id"] or not _group_enabled(ids["group_id"]):
        return False
    if _is_self_event(bot, ids):
        return False
    if _is_command_message(plain_text) or _is_blacklisted(
        ids["user_id"], ids["group_id"], plain_text
    ):
        conversation_pattern_tracker.reset(ids["group_id"])
        return False
    if plugin_config.qwen_persona_pattern_reply_threshold <= 0:
        return False

    trigger = build_conversation_pattern_message(
        user_id=ids["user_id"] or "",
        message=message,
        sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
    )
    if trigger is None:
        return False
    pattern = await store.get_conversation_pattern(
        group_id=ids["group_id"],
        trigger_signature=trigger.signature,
        min_count=plugin_config.qwen_persona_pattern_reply_threshold,
    )
    if not pattern:
        return False

    reply_text = sanitize_reply(
        str(pattern.get("response_text") or ""),
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )
    reply_text = _remove_blacklisted_reply_lines(reply_text)
    sticker_keys = [
        str(key)
        for key in pattern.get("response_sticker_keys", [])
        if str(key).strip()
    ]
    if not reply_text and not sticker_keys:
        return False

    if not await _send_adapter_reply(
        bot=bot,
        event=event,
        text=reply_text,
        sticker_keys=sticker_keys,
    ):
        return False
    conversation_pattern_tracker.reset(ids["group_id"])
    return True


async def _send_adapter_reply(
    *,
    bot: Bot | None,
    event: Event,
    text: str,
    sticker_keys: list[str],
) -> bool:
    delivery = plan_reply_delivery(text, sticker_keys)
    if delivery.use_forward and bot is not None:
        if await _send_onebot_v11_forward_reply(bot, event, delivery.forward_chunks):
            if delivery.trailing_sticker_keys:
                sticker_reply = await _build_adapter_reply(
                    event,
                    "",
                    delivery.trailing_sticker_keys,
                )
                if not (isinstance(sticker_reply, str) and not sticker_reply.strip()):
                    await persona_chat.send(sticker_reply)
            return True

    reply = await _build_adapter_reply(event, text, sticker_keys)
    if isinstance(reply, str) and not reply.strip():
        return False
    await persona_chat.send(reply)
    return True


async def _send_onebot_v11_forward_reply(
    bot: Bot,
    event: Event,
    chunks: list[str],
) -> bool:
    if "onebot.v11" not in event.__class__.__module__:
        return False
    if not chunks or not hasattr(bot, "call_api"):
        return False
    try:
        from nonebot.adapters.onebot.v11 import Message, MessageSegment
    except Exception:
        return False

    try:
        node_user_id = _onebot_numeric_id(_bot_self_id(bot) or "0")
        nickname = str(getattr(bot, "nickname", "") or _bot_self_id(bot) or "bot")
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
                group_id=_onebot_numeric_id(group_id),
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
            user_id=_onebot_numeric_id(user_id),
            messages=nodes,
        )
        return True
    except Exception as exc:
        logger.warning(f"qwen persona merged reply send failed: {exc!r}")
        return False


def _onebot_numeric_id(value: Any) -> Any:
    text = str(value or "").strip()
    if text.isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return value


async def _build_adapter_reply(event: Event, text: str, sticker_keys: list[str]) -> Any:
    if not plugin_config.qwen_persona_enable_sending_stickers or not sticker_keys:
        return text

    try:
        from nonebot.adapters.onebot.v11 import Message, MessageSegment
    except Exception:
        return text

    if "onebot.v11" not in event.__class__.__module__:
        return text

    records: list[dict[str, Any]] = []
    for key in sticker_keys[: plugin_config.qwen_persona_max_reply_stickers]:
        record = await store.get_sticker(key)
        if not record:
            logger.warning(f"qwen persona sticker key missing: {key}")
            continue
        records.append(record)

    return build_onebot_v11_sticker_message(Message, MessageSegment, text, records)


async def _record_sticker_event(
    *,
    event_type: str,
    status: str,
    key: str | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
    message: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    if not hasattr(store, "record_sticker_event"):
        return
    try:
        await store.record_sticker_event(
            event_type=event_type,
            status=status,
            key=key,
            group_id=group_id,
            user_id=user_id,
            message=message,
            details=details or {},
        )
    except Exception as exc:
        logger.warning(f"qwen persona sticker event record failed: {exc!r}")


async def _record_learning_sticker_events(
    ids: dict[str, str | None],
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        await _record_sticker_event(
            event_type=str(event.get("event_type") or "classifier"),
            status=str(event.get("status") or "unknown"),
            key=str(event.get("key") or "") or None,
            group_id=ids["group_id"],
            user_id=ids["user_id"],
            message=str(event.get("message") or ""),
            details=dict(event.get("details") or {}),
        )


async def _select_reply_sticker_key(
    *,
    group_id: str | None,
    user_id: str | None,
    trigger_text: str,
    reply_text: str,
    intent: ReplyIntent,
    incoming_sticker_keys: list[str],
) -> str | None:
    candidates = await store.top_stickers(
        group_id=group_id,
        limit=plugin_config.qwen_persona_sticker_prompt_limit,
    )
    seen = {str(item.get("key") or "").strip() for item in candidates}
    for key in incoming_sticker_keys:
        if key in seen:
            continue
        record = await store.get_sticker(key)
        if record:
            candidates.insert(0, record)
            seen.add(key)
        else:
            logger.warning(f"qwen persona incoming sticker key missing: {key}")
    selection = await select_reply_sticker(
        mlx_runtime,
        candidates=candidates,
        trigger_text=trigger_text,
        reply_text=reply_text,
        intent=intent,
        min_count=plugin_config.qwen_persona_min_reply_sticker_count,
    )
    await _record_sticker_event(
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
            "min_count": plugin_config.qwen_persona_min_reply_sticker_count,
            "fallback": selection.fallback,
            "trigger_text": trigger_text[:240],
            "reply_text": reply_text[:240],
            "raw_text": selection.raw_text[:500],
        },
    )
    return selection.key


def _install_schedule() -> None:
    global _schedule_installed
    if _schedule_installed or not plugin_config.qwen_persona_enable_training:
        return

    try:
        require("nonebot_plugin_apscheduler")
        from nonebot_plugin_apscheduler import scheduler
    except Exception as exc:
        logger.warning(f"qwen persona training schedule disabled: {exc!r}")
        return

    for slot in plugin_config.qwen_persona_train_times:
        try:
            hour, minute = parse_time_slot(slot)
        except ValueError as exc:
            logger.warning(f"Invalid qwen_persona_train_times item {slot!r}: {exc}")
            continue

        scheduler.add_job(
            _run_scheduled_training,
            "cron",
            hour=hour,
            minute=minute,
            id=f"qwen_persona_lora_{hour:02d}_{minute:02d}",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info(f"qwen persona LoRA training scheduled at {hour:02d}:{minute:02d}")

    _schedule_installed = True


@driver.on_startup
async def _startup() -> None:
    await store.init()
    try:
        empty_cleanup = await store.cleanup_empty_chat_artifacts()
        if empty_cleanup.get("samples_deleted") or empty_cleanup.get("patterns_deleted"):
            logger.info(
                "qwen persona empty chat cleanup: "
                f"samples_deleted={empty_cleanup.get('samples_deleted', 0)}, "
                f"patterns_deleted={empty_cleanup.get('patterns_deleted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona empty chat cleanup skipped: {exc!r}")
    try:
        noise_cleanup = await store.cleanup_training_noise(
            command_prefixes=_command_start_values(),
            command_regexes=_command_regex_values(),
        )
        if noise_cleanup.get("samples_deleted") or noise_cleanup.get("patterns_deleted"):
            logger.info(
                "qwen persona training noise cleanup: "
                f"samples_deleted={noise_cleanup.get('samples_deleted', 0)}, "
                f"patterns_deleted={noise_cleanup.get('patterns_deleted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona training noise cleanup skipped: {exc!r}")
    try:
        cleanup = await store.cleanup_legacy_media_artifacts()
        if cleanup.get("deleted_image_stickers") or cleanup.get("updated_samples"):
            logger.info(
                "qwen persona legacy media cleanup: "
                f"deleted_image_stickers={cleanup.get('deleted_image_stickers', 0)}, "
                f"updated_samples={cleanup.get('updated_samples', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona legacy media cleanup skipped: {exc!r}")
    try:
        summary_cleanup = await store.normalize_sticker_summaries_from_protocol()
        if summary_cleanup.get("updated"):
            logger.info(
                "qwen persona sticker summary cleanup: "
                f"updated={summary_cleanup.get('updated', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona sticker summary cleanup skipped: {exc!r}")
    try:
        repo_cleanup = await cleanup_sticker_repo_manifest(
            plugin_config.qwen_persona_sticker_repo_dir
        )
        if (
            repo_cleanup.get("removed")
            or repo_cleanup.get("media_deleted")
            or repo_cleanup.get("summary_updated")
        ):
            logger.info(
                "qwen persona sticker repo cleanup: "
                f"removed={repo_cleanup.get('removed', 0)}, "
                f"media_deleted={repo_cleanup.get('media_deleted', 0)}, "
                f"summary_updated={repo_cleanup.get('summary_updated', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona sticker repo cleanup skipped: {exc!r}")
    try:
        backfill = await store.backfill_stickers_from_raw_segments(
            plugin_config.qwen_persona_sticker_segment_types
        )
        if backfill.get("samples_updated") or backfill.get("stickers_upserted"):
            logger.info(
                "qwen persona sticker backfill: "
                f"samples_updated={backfill.get('samples_updated', 0)}, "
                f"stickers_upserted={backfill.get('stickers_upserted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona sticker backfill skipped: {exc!r}")
    try:
        dedupe = await store.deduplicate_stickers_by_canonical_key(
            plugin_config.qwen_persona_sticker_segment_types
        )
        if any(dedupe.values()):
            logger.info(
                "qwen persona sticker dedupe: "
                f"merged_stickers={dedupe.get('merged_stickers', 0)}, "
                f"rekeyed_stickers={dedupe.get('rekeyed_stickers', 0)}, "
                f"rewritten_samples={dedupe.get('rewritten_samples', 0)}"
            )
    except Exception as exc:
        logger.warning(f"qwen persona sticker dedupe skipped: {exc!r}")
    try:
        admin_server.start()
    except Exception as exc:
        logger.exception(f"qwen persona admin server failed to start: {exc!r}")
    if plugin_config.qwen_persona_mlx_auto_load:
        try:
            load_result = await mlx_runtime.load()
            logger.info(
                "qwen persona mlx runtime load result: "
                f"{load_result.get('status')}, model={load_result.get('model')}"
            )
        except Exception as exc:
            logger.exception(f"qwen persona mlx runtime auto load failed: {exc!r}")
    _install_schedule()


@driver.on_shutdown
async def _shutdown() -> None:
    for task in list(_background_media_tasks):
        task.cancel()
    if _background_media_tasks:
        await asyncio.gather(*_background_media_tasks, return_exceptions=True)
        _background_media_tasks.clear()
    await mlx_runtime.close()
    admin_server.stop()


async def _learn_from_event(bot: Bot, event: Event) -> None:
    if not plugin_config.qwen_persona_enable_learning:
        return
    try:
        message = event.get_message()
    except Exception:
        return

    ids = _dump_event_ids(event)
    if not _group_enabled(ids["group_id"]):
        conversation_pattern_tracker.reset(ids["group_id"])
        return
    if _is_self_event(bot, ids):
        conversation_pattern_tracker.reset(ids["group_id"])
        return

    learning = await prepare_message_learning(
        message,
        sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
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
        bot_self_id=_bot_self_id(bot),
        to_me=_is_to_me(event, plain_text),
    ):
        conversation_pattern_tracker.reset(ids["group_id"])
        pattern_allowed = False
    if _is_command_message(plain_text) or _is_command_message(raw_message_text):
        conversation_pattern_tracker.reset(ids["group_id"])
        return
    if _is_blacklisted(ids["user_id"], ids["group_id"], plain_text + "\n" + raw_message_text):
        conversation_pattern_tracker.reset(ids["group_id"])
        return

    await _record_learning_sticker_events(ids, list(learning.get("sticker_events") or []))

    stickers = list(learning["stickers"])
    training_text = str(learning["training_text"])
    decision = should_learn_text(
        training_text,
        min_chars=plugin_config.qwen_persona_min_sample_chars,
        max_chars=plugin_config.qwen_persona_max_sample_chars,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )

    if not decision.allowed and not stickers:
        conversation_pattern_tracker.reset(ids["group_id"])
        return

    await record_sticker_repo_manifest(
        repo_dir=plugin_config.qwen_persona_sticker_repo_dir,
        group_id=ids["group_id"],
        user_id=ids["user_id"],
        stickers=stickers,
    )
    training_text = filter_training_text_sticker_tokens(training_text, stickers)
    decision = should_learn_text(
        training_text,
        min_chars=plugin_config.qwen_persona_min_sample_chars,
        max_chars=plugin_config.qwen_persona_max_sample_chars,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )
    if not decision.allowed and not stickers:
        conversation_pattern_tracker.reset(ids["group_id"])
        return

    await store.record_sample(
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
    candidate = conversation_pattern_tracker.observe(
        group_id=ids["group_id"],
        user_id=ids["user_id"],
        message=message,
        sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
        plain_text=plain_text,
        sticker_keys=[
            str(sticker.get("key") or "").strip()
            for sticker in stickers
            if str(sticker.get("key") or "").strip()
        ],
    )
    if candidate is None:
        return
    await store.record_conversation_pattern(
        group_id=ids["group_id"],
        trigger_signature=candidate.trigger.signature,
        trigger_text=candidate.trigger.plain_text,
        response_signature=candidate.response.signature,
        response_text=candidate.response.plain_text,
        response_sticker_keys=candidate.response.sticker_keys,
    )


persona_learn = on_message(priority=plugin_config.qwen_persona_learning_priority, block=False)


@persona_learn.handle()
async def _learn_message(bot: Bot, event: Event) -> None:
    await _learn_from_event(bot, event)


persona_chat = on_message(priority=plugin_config.qwen_persona_reply_priority, block=False)


@persona_chat.handle()
async def _chat(bot: Bot, event: Event, matcher: Matcher) -> None:
    try:
        message = event.get_message()
    except Exception:
        return

    ids = _dump_event_ids(event)
    if _is_self_event(bot, ids):
        conversation_pattern_tracker.reset(ids["group_id"])
        return
    plain_text = extract_plain_text(message)
    # 合并转发：展开 forward 段，提取文字拼进 plain_text，图片段单独收集。
    forward_texts_extra: list[str] = []
    if count_forward_segments(message) > 0:
        try:
            f_texts, f_media = await resolve_forward(
                bot,
                message,
                max_messages=int(plugin_config.qwen_persona_forward_max_messages),
                max_chars=int(plugin_config.qwen_persona_forward_max_chars),
            )
            if f_texts:
                forward_texts_extra = f_texts
                forward_block = "【合并转发】\n" + "\n".join(f_texts)
                plain_text = (plain_text + "\n" + forward_block).strip() if plain_text else forward_block
                logger.info("qwen persona forward expanded: texts=%d media=%d", len(f_texts), len(f_media))
        except Exception as exc:  # noqa: BLE001
            logger.debug("forward expand skipped: %r", exc)
    reply_input_text, incoming_sticker_keys, media_parts, media_context_text = await _build_reply_input_text(message, plain_text)
    # 图片智能控制：图片超过阈值则丢弃图片识别（media_context_text 清空），只保留文字。
    image_count = sum(1 for p in media_parts if str(p.get("type") or "") in {"image", "mface", "emoji", "sticker"})
    image_limit = int(plugin_config.qwen_persona_media_image_limit)
    if image_count > image_limit and media_context_text:
        logger.info("qwen persona media: %d images > limit %d, dropping captions", image_count, image_limit)
        media_context_text = ""
    if _should_schedule_background_media_analysis(bot, event, plain_text, ids, media_parts):
        _schedule_background_media_analysis(ids, media_parts)
    if await _send_repeat_if_needed(bot, event, ids, message):
        conversation_pattern_tracker.reset(ids["group_id"])
        matcher.stop_propagation()
        return
    if await _send_pattern_reply_if_known(bot, event, ids, message, plain_text):
        matcher.stop_propagation()
        return
    reply_decision = _reply_decision(bot, event, plain_text, ids)
    if not reply_decision.allowed:
        return

    # 主语指代：预解析 @ 目标和被引用消息发送者的昵称，让 bot 知道谁在对谁说话。
    at_nicknames: dict[str, str | None] = {}
    reply_nickname: str | None = None
    group_id = ids.get("group_id")
    try:
        at_qqs = [t["qq"] for t in _at_targets_from_message(message, {}) if t.get("qq")]
        if at_qqs and group_id:
            at_nicknames = await nickname_resolver.resolve_many(bot, group_id, at_qqs)
        reply_obj = getattr(event, "reply", None)
        reply_sender_id = ""
        if reply_obj is not None:
            reply_sender_obj = getattr(reply_obj, "sender", None)
            if isinstance(reply_sender_obj, dict):
                reply_sender_id = str(reply_sender_obj.get("user_id") or reply_sender_obj.get("qq") or "")
            elif reply_sender_obj is not None:
                reply_sender_id = str(getattr(reply_sender_obj, "user_id", "") or getattr(reply_sender_obj, "qq", ""))
        if reply_sender_id and group_id:
            reply_nickname = await nickname_resolver.resolve(bot, group_id, reply_sender_id)
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
        text_sticker_probability=plugin_config.qwen_persona_text_sticker_reply_probability,
        random_value=random.random(),
    )
    reply_plan = await _build_model_reply_plan(
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
            raw_reply = await client.chat(reply_plan.messages, media_parts=reply_plan.media_parts)
        except Exception as exc:
            await record_inference_event(
                store=store,
                ids=ids,
                trigger_reason="auto_reply",
                model=plugin_config.qwen_persona_mlx_model,
                adapter=plugin_config.qwen_persona_mlx_adapter_path,
                started_at=inference_started,
                finished_at=time.perf_counter(),
                input_text=reply_input_text,
                output_text="",
                media_count=len(media_parts),
                status="failed",
                error_summary=f"{type(exc).__name__}: {exc}",
                tps=None,
            )
            logger.warning(f"qwen persona chat failed: {exc!r}")
            return

        metrics = client.metrics_snapshot() if hasattr(client, "metrics_snapshot") else {"last": None}
        last_metric = metrics.get("last") if isinstance(metrics, dict) else None
        tps = last_metric.get("tps") if isinstance(last_metric, dict) else None
        await record_inference_event(
            store=store,
            ids=ids,
            trigger_reason="auto_reply",
            model=client.active_model_name() if hasattr(client, "active_model_name") else plugin_config.qwen_persona_mlx_model,
            adapter=plugin_config.qwen_persona_mlx_adapter_path if client.active_backend() == "mlx" else "",
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
            "qwen persona generated sticker tokens ignored: "
            + ",".join(generated_sticker_keys[: plugin_config.qwen_persona_max_reply_stickers])
        )
    clean_text = sanitize_reply(
        reply_text,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )
    clean_text = _remove_blacklisted_reply_lines(clean_text)
    if is_echo_reply(clean_text, reply_input_text) or is_question_deflection_reply(clean_text, reply_input_text):
        clean_text = ""
    sticker_keys: list[str] = []
    if reply_plan.allow_sticker and (
        clean_text or incoming_sticker_keys or reply_plan.intent == ReplyIntent.STICKER_REQUEST
    ):
        sticker_key = await _select_reply_sticker_key(
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

    if not await _send_adapter_reply(bot=bot, event=event, text=clean_text, sticker_keys=sticker_keys):
        return
    conversation_pattern_tracker.reset(ids["group_id"])
    matcher.stop_propagation()


persona_poke = on_notice(priority=plugin_config.qwen_persona_reply_priority - 1, block=False)


@persona_poke.handle()
async def _counter_poke(bot: Bot, event: Event, matcher: Matcher) -> None:
    if not is_poke_at_self(event, _bot_self_id(bot)):
        return
    try:
        sent = await send_counter_poke(bot, event)
    except Exception as exc:
        logger.debug(f"qwen persona counter poke failed: {exc!r}")
        return
    if sent:
        matcher.stop_propagation()


persona_stats = on_command("人格状态", aliases={"qwen人格状态", "群友人格状态"}, priority=5, block=True)


@persona_stats.handle()
async def _stats() -> None:
    stats = await store.stats()
    await persona_stats.finish(
        "群友人格记忆："
        f"\n样本 {stats['samples']} 条"
        f"\n表情包 {stats['stickers']} 个"
        f"\n训练记录 {stats['training_runs']} 次"
    )


persona_train = on_command(
    "人格训练",
    aliases={"qwen人格训练", "群友人格训练"},
    permission=SUPERUSER,
    priority=5,
    block=True,
)


@persona_train.handle()
async def _manual_train(args: Any = CommandArg()) -> None:
    reason = (args.extract_plain_text().strip() if args else "") or "manual"
    result = await trainer.run_training(reason=reason)
    await persona_train.finish(
        f"训练任务 {result.status}"
        f"\n样本 {result.sample_count} 条"
        f"\n数据集 {result.dataset_path or '-'}"
        f"\n输出 {result.output_dir or '-'}"
    )

persona_model_status = on_command(
    "人格模型状态",
    aliases={"qwen模型状态", "群友人格模型状态"},
    priority=5,
    block=True,
)


@persona_model_status.handle()
async def _model_status() -> None:
    snapshot = mlx_runtime.status_snapshot()
    await persona_model_status.finish(
        "人格模型："
        f"\n运行时 MLX-VLM"
        f"\n状态 {snapshot.get('state')}"
        f"\n模型 {snapshot.get('model') or '-'}"
        f"\nAdapter {snapshot.get('adapter_path') or '-'}"
        f"\n错误 {snapshot.get('last_error') or '-'}"
    )


persona_model_deploy = on_command(
    "人格部署模型",
    aliases={"qwen部署模型", "群友人格部署模型"},
    permission=SUPERUSER,
    priority=5,
    block=True,
)


@persona_model_deploy.handle()
async def _manual_model_deploy() -> None:
    result = await mlx_runtime.load(force=True)
    await persona_model_deploy.finish(
        f"模型加载 {result.get('status')}"
        f"\n运行时 MLX-VLM"
        f"\n模型 {result.get('model') or '-'}"
        f"\nAdapter {result.get('adapter_path') or '-'}"
    )


async def _run_scheduled_training() -> None:
    try:
        result = await trainer.run_training(reason="scheduled")
        logger.info(
            "qwen persona scheduled training finished: "
            f"{result.status}, samples={result.sample_count}"
        )
    except Exception as exc:
        logger.exception(f"qwen persona scheduled training failed: {exc!r}")










