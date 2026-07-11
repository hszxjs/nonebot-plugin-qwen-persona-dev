from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path
from typing import Any

from nonebot import get_driver, logger, on_command, on_message, on_notice, require
from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

from .admin_web import AdminWebServer
from .command_discovery import discover_picmenu_command_regexes
from .config import QwenPersonaConfig, load_plugin_config, normalize_matcher_priorities
from .filters import (
    is_echo_reply,
    is_command_like,
    is_private_chat,
    is_question_deflection_reply,
    is_training_noise,
    sanitize_reply,
    should_force_reply,
    should_learn_text,
    should_random_reply,
    should_use_sticker_reply,
)
from .media import (
    context_text_from_sample,
    filter_training_text_sticker_tokens,
    prepare_message_learning,
    reply_context_text_from_learning,
    sanitize_model_reply_media_text,
)
from .model_manager import ModelManager
from .poke import is_poke_at_self, send_counter_poke
from .prompt_budget import effective_context_message_limit, trim_history_lines_for_context
from .qwen_client import QwenClient
from .repeat import (
    ConversationPatternTracker,
    RepeatTracker,
    build_conversation_pattern_message,
)
from .runtime_config import (
    apply_config_values,
    config_to_dict,
    load_config_overrides,
    save_config_overrides,
)
from .sticker import (
    build_onebot_v11_sticker_message,
    extract_plain_text,
    parse_sticker_tokens,
)
from .sticker_repo import cleanup_sticker_repo_manifest, record_sticker_repo_manifest
from .storage import PersonaStore
from .trainer import PersonaTrainer, parse_time_slot
from .vision_manager import VisionManager

__plugin_meta__ = PluginMetadata(
    name="Qwen Persona",
    description="调用本地千问模型聊天，学习群友语气和表情包，并定时导出 LoRA 数据。",
    usage=(
        "配置本地 OpenAI-compatible API 后，@机器人即可聊天。\n"
        "命令：人格状态、人格训练（SUPERUSER）。"
    ),
    type="application",
    homepage="https://github.com/local/nonebot-plugin-qwen-persona",
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
client = QwenClient(plugin_config)
trainer = PersonaTrainer(plugin_config, store)
model_manager = ModelManager(plugin_config)
vision_manager = VisionManager(plugin_config)
repeat_tracker = RepeatTracker()
conversation_pattern_tracker = ConversationPatternTracker()
admin_web = AdminWebServer(
    config=plugin_config,
    save_callback=lambda values: _save_admin_config(values),
    logger=logger,
    model_manager=model_manager,
    vision_manager=vision_manager,
    store=store,
    qwen_client=client,
    trainer=trainer,
)
_schedule_installed = False


_CONFIG_RESTART_REQUIRED = {
    "qwen_persona_runtime",
    "qwen_persona_llama_server_path",
    "qwen_persona_llama_auto_install",
    "qwen_persona_llama_model",
    "qwen_persona_llama_model_file",
    "qwen_persona_llama_host",
    "qwen_persona_llama_port",
    "qwen_persona_llama_ctx_size",
    "qwen_persona_llama_extra_args",
    "qwen_persona_api_key",
    "qwen_persona_model_dir",
    "qwen_persona_admin_enabled",
    "qwen_persona_admin_host",
    "qwen_persona_admin_port",
    "qwen_persona_data_dir",
    "qwen_persona_train_times",
    "qwen_persona_enable_training",
    "qwen_persona_learning_priority",
    "qwen_persona_reply_priority",
    "qwen_persona_vision_enabled",
    "qwen_persona_vision_auto_start",
    "qwen_persona_vision_python",
    "qwen_persona_vision_host",
    "qwen_persona_vision_port",
    "qwen_persona_vision_model",
}


def _save_admin_config(values: dict[str, Any]) -> dict[str, Any]:
    changed = apply_config_values(plugin_config, values)
    changed.extend(normalize_matcher_priorities(plugin_config))
    changed = list(dict.fromkeys(changed))
    save_config_overrides(plugin_config, config_to_dict(plugin_config))
    model_manager.reconfigure(plugin_config)
    vision_manager.reconfigure(plugin_config)
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


def _should_reply(bot: Bot, event: Event, plain_text: str, ids: dict[str, str | None]) -> bool:
    if not plugin_config.qwen_persona_enable_reply:
        return False
    private_chat = is_private_chat(event, ids)
    if not private_chat and not _group_enabled(ids["group_id"]):
        return False
    if _is_self_event(bot, ids):
        return False
    if _is_command_message(plain_text):
        return False
    if _is_blacklisted(ids["user_id"], ids["group_id"], plain_text):
        return False
    try:
        message = event.get_message()
    except Exception:
        message = ()
    if should_force_reply(
        event,
        ids,
        message,
        bot_self_id=_bot_self_id(bot),
        to_me=_is_to_me(event, plain_text),
    ):
        return True
    probability = plugin_config.qwen_persona_reply_probability
    if probability <= 0:
        probability = plugin_config.qwen_persona_follow_reply_probability
    return should_random_reply(plain_text, probability=probability, random_value=random.random())


def _build_system_prompt(
    sticker_lines: list[str],
    *,
    sticker_reply_allowed: bool,
    incoming_has_sticker: bool,
) -> str:
    command_prefixes = " ".join(sorted(set(_command_start_values())))
    lines = [
        plugin_config.qwen_persona_system_prompt,
        "硬性规则：不要输出、复述或诱导使用本地 NoneBot 插件命令。",
        f"如果回复开头像这些命令前缀会被丢弃：{command_prefixes or '(none)'}。",
        "你在群聊中应当自然、短句、像真人接话，不要解释自己是模型。",
        "不要复述当前用户消息；如果不知道怎么答，也要换一种自然接话方式。",
    ]
    if plugin_config.qwen_persona_sticker_repo_url:
        lines.append(f"表情包仓库：{plugin_config.qwen_persona_sticker_repo_url}")
    if sticker_reply_allowed and sticker_lines:
        if incoming_has_sticker:
            lines.append("当前用户发来的是表情。本轮只输出一个可用表情 token，不要附加文字。")
        else:
            lines.append("本轮已按概率选择表情回复。只输出一个可用表情 token，不要附加文字。")
        lines.append(
            "可用表情格式：[[sticker:<key>]]。只能使用下面列出的 key。"
        )
        lines.append("\n".join(sticker_lines))
    else:
        lines.append("本轮必须只回复文字，禁止输出任何 [[sticker:...]] 标记。")
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


async def _build_model_messages(
    event: Event,
    plain_text: str,
    group_id: str | None,
    *,
    sticker_reply_allowed: bool,
    incoming_has_sticker: bool,
) -> list[dict[str, str]]:
    history = await _recent_context_samples(
        group_id,
        effective_context_message_limit(plugin_config),
    )
    stickers = (
        await store.top_stickers(
            group_id=group_id,
            limit=plugin_config.qwen_persona_sticker_prompt_limit,
        )
        if sticker_reply_allowed
        else []
    )
    sticker_lines = [
        f"- key={item['key']} summary={item.get('summary') or item['type']} count={item['count']}"
        for item in stickers
    ]
    history_lines: list[str] = []
    for item in reversed(history):
        context_text = context_text_from_sample(item)
        if not sticker_reply_allowed:
            context_text, _sticker_keys = parse_sticker_tokens(context_text)
        if _is_context_noise(context_text):
            continue
        if context_text:
            history_lines.append(f"{item.get('user_id') or 'someone'}: {context_text}")
    history_lines = trim_history_lines_for_context(history_lines, plugin_config)

    user_parts = []
    if history_lines:
        user_parts.append("最近群聊样本：\n" + "\n".join(history_lines))
    user_parts.append(f"当前消息：{plain_text.strip() or '[有人在叫你]'}")
    if sticker_reply_allowed and sticker_lines:
        user_parts.append("只选择并输出一个最贴合当前消息的可用表情 token，不要输出文字。")
    else:
        user_parts.append("请自然接一句，只输出文字，不要输出命令或表情 token。")
    user_parts.append("/no_think\n只输出最终要发到群里的内容，不要写思考过程。")
    return [
        {
            "role": "system",
            "content": _build_system_prompt(
                sticker_lines,
                sticker_reply_allowed=sticker_reply_allowed,
                incoming_has_sticker=incoming_has_sticker,
            ),
        },
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


async def _build_reply_input_text(message: Any, plain_text: str) -> tuple[str, list[str]]:
    try:
        learning = await prepare_message_learning(
            message,
            sticker_segment_types=plugin_config.qwen_persona_sticker_segment_types,
            captioner=vision_manager if plugin_config.qwen_persona_vision_enabled else None,
            image_fallback_text=plugin_config.qwen_persona_vision_image_fallback_text,
            sticker_fallback_text=plugin_config.qwen_persona_vision_sticker_fallback_text,
        )
    except Exception as exc:
        logger.debug(f"qwen persona reply media context skipped: {exc!r}")
        return plain_text, []

    reply_context = reply_context_text_from_learning(learning)
    sticker_keys = [
        str(item.get("key") or "").strip()
        for item in learning.get("stickers") or []
        if str(item.get("key") or "").strip()
    ]
    return reply_context or plain_text, sticker_keys


async def _send_repeat_if_needed(event: Event, ids: dict[str, str | None], message: Any) -> bool:
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
    reply = await _build_adapter_reply(event, reply_text, candidate.sticker_keys)
    await persona_chat.send(reply)
    return True


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

    reply = await _build_adapter_reply(event, reply_text, sticker_keys)
    if isinstance(reply, str) and not reply.strip():
        return False
    await persona_chat.send(reply)
    conversation_pattern_tracker.reset(ids["group_id"])
    return True


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
            continue
        records.append(record)

    return build_onebot_v11_sticker_message(Message, MessageSegment, text, records)


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
        admin_web.set_event_loop(asyncio.get_running_loop())
        admin_web.start()
    except Exception as exc:
        logger.exception(f"qwen persona admin web failed to start: {exc!r}")
    try:
        deploy_result = await model_manager.ensure_ready()
        if deploy_result.status not in {"disabled", "present"}:
            logger.info(
                "qwen persona model deploy result: "
                f"{deploy_result.status}, repo={deploy_result.repo_id or '-'}, "
                f"dir={deploy_result.model_dir}"
            )
    except Exception as exc:
        logger.exception(f"qwen persona model auto deploy failed: {exc!r}")
    try:
        vision_result = await vision_manager.ensure_ready()
        if vision_result.status not in {"vision_disabled", "vision_present"}:
            logger.info(
                "qwen persona vision deploy result: "
                f"{vision_result.status}, endpoint={vision_result.endpoint}"
            )
    except Exception as exc:
        logger.exception(f"qwen persona vision auto deploy failed: {exc!r}")
    _install_schedule()


@driver.on_shutdown
async def _shutdown() -> None:
    await vision_manager.close()
    await model_manager.close()
    admin_web.stop()


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
        captioner=vision_manager if plugin_config.qwen_persona_vision_enabled else None,
        image_fallback_text=plugin_config.qwen_persona_vision_image_fallback_text,
        sticker_fallback_text=plugin_config.qwen_persona_vision_sticker_fallback_text,
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
    if await _send_repeat_if_needed(event, ids, message):
        conversation_pattern_tracker.reset(ids["group_id"])
        matcher.stop_propagation()
        return
    if await _send_pattern_reply_if_known(bot, event, ids, message, plain_text):
        matcher.stop_propagation()
        return
    if not _should_reply(bot, event, plain_text, ids):
        return

    reply_input_text, incoming_sticker_keys = await _build_reply_input_text(message, plain_text)
    sticker_reply_allowed = should_use_sticker_reply(
        incoming_has_sticker=bool(incoming_sticker_keys),
        text_sticker_probability=plugin_config.qwen_persona_text_sticker_reply_probability,
        random_value=random.random(),
    )
    model_messages = await _build_model_messages(event, reply_input_text, ids["group_id"],
        sticker_reply_allowed=sticker_reply_allowed,
        incoming_has_sticker=bool(incoming_sticker_keys),
    )
    try:
        raw_reply = await client.chat(model_messages)
    except Exception as exc:
        logger.warning(f"qwen persona chat failed: {exc!r}")
        return

    raw_reply = sanitize_model_reply_media_text(raw_reply)
    reply_text, sticker_keys = parse_sticker_tokens(raw_reply)
    clean_text = sanitize_reply(
        reply_text,
        prefixes=_command_start_values(),
        regexes=_command_regex_values(),
    )
    clean_text = _remove_blacklisted_reply_lines(clean_text)
    if is_echo_reply(clean_text, reply_input_text) or is_question_deflection_reply(clean_text, reply_input_text):
        clean_text = ""
        sticker_keys = []
    if not sticker_reply_allowed:
        sticker_keys = []
    elif incoming_sticker_keys and not sticker_keys:
        sticker_keys = incoming_sticker_keys[:1]
    if not clean_text and not sticker_keys:
        return

    reply = await _build_adapter_reply(event, clean_text, sticker_keys)
    if isinstance(reply, str) and not reply.strip():
        return
    await persona_chat.send(reply)
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
    last = model_manager.last_result
    await persona_model_status.finish(
        "人格模型："
        f"\n目录 {model_manager.model_dir}"
        f"\n已存在 {'是' if model_manager.model_present() else '否'}"
        f"\n上次部署 {last.status if last else '-'}"
        f"\n仓库 {last.repo_id if last and last.repo_id else '-'}"
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
    result = await model_manager.ensure_ready(force=True)
    await persona_model_deploy.finish(
        f"模型部署 {result.status}"
        f"\n目录 {result.model_dir}"
        f"\n仓库 {result.repo_id or '-'}"
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










