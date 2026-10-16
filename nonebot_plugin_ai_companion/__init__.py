"""nonebot_plugin_ai_companion：AI 对话伙伴插件入口。

matcher 层只做路由 + 依赖装配；业务编排在 chat_service / learning_service，
领域逻辑在各领域子包（见 docs/architecture.md）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nonebot import get_driver, logger, on_command, on_message, on_notice
from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

from .anime_captioner import AnimeCaptioner
from .api.server import AdminServer
from .bot_env import BotEnvironment
from .chat_service import ChatService
from .config import AiCompanionConfig, load_plugin_config, normalize_matcher_priorities
from .context.nicknames import NicknameResolver
from .decision.repeat import ConversationPatternTracker, RepeatTracker
from .delivery.onebot_v11 import is_poke_at_self, send_counter_poke
from .finetune.schedule import install_training_schedule
from .finetune.trainer import PersonaTrainer
from .learning_service import LearningService
from .model_manager.manager import ModelManager
from .notify.milestone_service import MilestoneService
from .notify.service import PingService
from .observability.bus import EventBus
from .observability.message_log import MessageLogRepo, format_trace, quoted_message_id
from .providers.client import QwenClient
from .providers.instances import (
    InferenceBackendRepo,
    ProviderInstanceManager,
    ProviderResolver,
    RoleProviderBindingRepo,
)
from .providers.mlx_runtime import MlxVLMRuntime
from .role_card.repo import RoleCardRepo, SessionRoleBindingRepo
from .role_card.service import RoleCardService
from .runtime_config import (
    apply_config_values,
    config_to_dict,
    load_config_overrides,
    save_config_overrides,
)
from .sticker.model import sticker_to_record
from .sticker.repo_files import cleanup_sticker_repo_manifest
from .store import PersonaStore
from .tools.builtins import register_builtin_tools
from .tools.registry import ToolRegistry

__plugin_meta__ = PluginMetadata(
    name="AI Companion",
    description="AI 对话伙伴：本地 MLX / 在线 LLM 双推理后端，学习群友语气与表情包，支持 LoRA 训练。",
    usage=("配置推理后端后，@机器人即可聊天。\n命令：人格状态、人格训练（SUPERUSER）。"),
    type="application",
    homepage="https://github.com/hszxjs/nonebot-plugin-qwen-persona-dev",
    supported_adapters=None,
    config=AiCompanionConfig,
)

# ---------------------------------------------------------------------------
# 依赖装配（DI 根）
# ---------------------------------------------------------------------------
driver = get_driver()
plugin_config = load_plugin_config(driver.config)
try:
    apply_config_values(plugin_config, load_config_overrides(plugin_config))
except Exception as exc:
    logger.warning(f"ai companion admin config load failed: {exc!r}")
priority_changes = normalize_matcher_priorities(plugin_config)
if priority_changes:
    logger.warning(
        "ai companion matcher priorities normalized: "
        f"learning={plugin_config.ai_companion_learning_priority}, "
        f"reply={plugin_config.ai_companion_reply_priority}"
    )

env = BotEnvironment(driver.config, plugin_config)
store = PersonaStore(Path(plugin_config.ai_companion_data_dir))
mlx_runtime = MlxVLMRuntime(plugin_config)
client = QwenClient(plugin_config, runtime=mlx_runtime)
anime_captioner = AnimeCaptioner(plugin_config)
nickname_resolver = NicknameResolver()
repeat_tracker = RepeatTracker()
conversation_pattern_tracker = ConversationPatternTracker()
tool_registry = ToolRegistry()
register_builtin_tools(tool_registry, config=plugin_config, store=store)
model_manager_service = ModelManager(plugin_config)
event_bus = EventBus()
store.attach_event_bus(event_bus)
store.message_logs.set_max_records(plugin_config.ai_companion_log_max_records)
message_log_repo = MessageLogRepo(
    store.db,
    max_records=plugin_config.ai_companion_log_max_records,
    bus=event_bus,
)
backend_repo = InferenceBackendRepo(store.db)
provider_binding_repo = RoleProviderBindingRepo(store.db)
provider_instances = ProviderInstanceManager(backend_repo)
provider_resolver = ProviderResolver(
    config=plugin_config,
    client=client,
    mlx_runtime=mlx_runtime,
    instances=provider_instances,
    bindings=provider_binding_repo,
)


async def _proactive_reply(group_id: str) -> str:
    """会话监控页"手动触发接话"：取该群最近上下文生成一条主动消息并发送。"""
    from nonebot import get_bot

    bot = get_bot()  # 无在线 bot 时抛 ValueError，由路由转 503
    samples = await store.recent_samples(group_id=group_id, limit=10)
    from .prompt.media_text import context_text_from_sample

    history_lines = [context_text_from_sample(item) for item in reversed(samples)]
    history_lines = [line for line in history_lines if line][-8:]
    prompt_lines = [
        "你在群聊中参与对话。下面是最近的群聊记录，请自然地接一句话活跃气氛（不要复述、不要命令、不要提你是 AI）：",
        *[f"上一句：{line}" for line in history_lines],
    ]
    reply = await client.chat([{"role": "user", "content": "\n".join(prompt_lines)}])
    from .prompt.text import sanitize_reply

    clean = sanitize_reply(reply, prefixes=env.command_start_values())
    if not clean:
        raise RuntimeError("generated empty proactive reply")
    result = await bot.call_api("send_group_msg", group_id=int(group_id), message=clean)
    return str((result or {}).get("message_id") or "")


async def _auto_deploy_adapter(adapter) -> None:
    """LoRA 训练后自动部署：激活版本 + 持久化 adapter 路径 + MLX 模式热重载。"""
    plugin_config.ai_companion_mlx_adapter_path = str(adapter.path)
    save_config_overrides(plugin_config, config_to_dict(plugin_config))
    logger.info(
        f"ai companion lora adapter v{adapter.version} deployed: {adapter.path} "
        f"({adapter.role_card_id}/{adapter.source_language})"
    )
    if plugin_config.ai_companion_inference_backend == "mlx":
        reload_result = await mlx_runtime.load(force=True)
        logger.info(
            "ai companion mlx runtime adapter reload: "
            f"{reload_result.get('status')}, adapter={reload_result.get('adapter_path')}"
        )


async def _data_cleaning_chat(prompt: str) -> str:
    """data_cleaning 角色实例对话；未绑定回退辅助 LLM（aux_llm）。"""
    from .providers.aux_llm import aux_chat

    provider, _model, _pid = await provider_resolver.resolve_with_meta("data_cleaning")
    if provider is not None:
        return await provider.chat([{"role": "user", "content": prompt}])
    return await aux_chat(
        plugin_config,
        prompt=prompt,
        model=plugin_config.ai_companion_member_mimic_model,
        max_tokens=900,
        temperature=0.2,
        timeout_seconds=30.0,
    )


role_card_service = RoleCardService(
    config=plugin_config,
    cards=RoleCardRepo(store.db),
    bindings=SessionRoleBindingRepo(store.db),
    samples=store.samples,
    data_cleaning_chat=_data_cleaning_chat,
)
trainer = PersonaTrainer(plugin_config, store, deploy_callback=_auto_deploy_adapter)

chat_service = ChatService(
    env=env,
    store=store,
    client=client,
    mlx_runtime=mlx_runtime,
    captioner=anime_captioner,
    nickname_resolver=nickname_resolver,
    repeat_tracker=repeat_tracker,
    conversation_pattern_tracker=conversation_pattern_tracker,
    tool_registry=tool_registry,
    role_card_service=role_card_service,
    provider_resolver=provider_resolver,
    message_log_repo=message_log_repo,
)
learning_service = LearningService(
    env=env,
    store=store,
    conversation_pattern_tracker=conversation_pattern_tracker,
)


def _save_admin_config(values: dict[str, Any]) -> dict[str, Any]:
    changed = apply_config_values(plugin_config, values)
    changed.extend(normalize_matcher_priorities(plugin_config))
    changed = list(dict.fromkeys(changed))
    save_config_overrides(plugin_config, config_to_dict(plugin_config))
    mlx_runtime.reconfigure(plugin_config)
    client.reconfigure(plugin_config)
    store.apply_memory_limits(
        member_limit=plugin_config.ai_companion_memory_member_limit,
        group_limit=plugin_config.ai_companion_memory_group_limit,
    )
    restart_required = [name for name in changed if name in _CONFIG_RESTART_REQUIRED]
    return {"changed": changed, "restart_required": restart_required}


admin_server = AdminServer(
    config=plugin_config,
    save_callback=lambda values: _save_admin_config(values),
    logger=logger,
    mlx_runtime=mlx_runtime,
    store=store,
    qwen_client=client,
    trainer=trainer,
    role_card_service=role_card_service,
    tool_registry=tool_registry,
    provider_instances=provider_instances,
    provider_bindings=provider_binding_repo,
    model_manager=model_manager_service,
    event_bus=event_bus,
    env=env,
    data_cleaning_chat=_data_cleaning_chat,
    proactive_sender=_proactive_reply,
)

_CONFIG_RESTART_REQUIRED = {
    "ai_companion_mlx_model",
    "ai_companion_mlx_adapter_path",
    "ai_companion_mlx_auto_load",
    "ai_companion_mlx_trust_remote_code",
    "ai_companion_model_dir",
    "ai_companion_admin_enabled",
    "ai_companion_admin_host",
    "ai_companion_admin_port",
    "ai_companion_data_dir",
    "ai_companion_train_times",
    "ai_companion_enable_training",
    "ai_companion_learning_priority",
    "ai_companion_reply_priority",
}
_schedule_installed = False
_memory_loop_started = False


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
@driver.on_startup
async def _startup() -> None:
    await store.init()
    store.apply_memory_limits(
        member_limit=plugin_config.ai_companion_memory_member_limit,
        group_limit=plugin_config.ai_companion_memory_group_limit,
    )
    try:
        empty_cleanup = await store.cleanup_empty_chat_artifacts()
        if empty_cleanup.get("samples_deleted") or empty_cleanup.get("patterns_deleted"):
            logger.info(
                "ai companion empty chat cleanup: "
                f"samples_deleted={empty_cleanup.get('samples_deleted', 0)}, "
                f"patterns_deleted={empty_cleanup.get('patterns_deleted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion empty chat cleanup skipped: {exc!r}")
    try:
        noise_cleanup = await store.cleanup_training_noise(
            command_prefixes=env.command_start_values(),
            command_regexes=env.command_regex_values(),
        )
        if noise_cleanup.get("samples_deleted") or noise_cleanup.get("patterns_deleted"):
            logger.info(
                "ai companion training noise cleanup: "
                f"samples_deleted={noise_cleanup.get('samples_deleted', 0)}, "
                f"patterns_deleted={noise_cleanup.get('patterns_deleted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion training noise cleanup skipped: {exc!r}")
    try:
        cleanup = await store.cleanup_legacy_media_artifacts()
        if cleanup.get("deleted_image_stickers") or cleanup.get("updated_samples"):
            logger.info(
                "ai companion legacy media cleanup: "
                f"deleted_image_stickers={cleanup.get('deleted_image_stickers', 0)}, "
                f"updated_samples={cleanup.get('updated_samples', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion legacy media cleanup skipped: {exc!r}")
    try:
        summary_cleanup = await store.normalize_sticker_summaries_from_protocol()
        if summary_cleanup.get("updated"):
            logger.info(f"ai companion sticker summary cleanup: updated={summary_cleanup.get('updated', 0)}")
    except Exception as exc:
        logger.warning(f"ai companion sticker summary cleanup skipped: {exc!r}")
    try:
        repo_cleanup = await cleanup_sticker_repo_manifest(plugin_config.ai_companion_sticker_repo_dir)
        if repo_cleanup.get("removed") or repo_cleanup.get("media_deleted") or repo_cleanup.get("summary_updated"):
            logger.info(
                "ai companion sticker repo cleanup: "
                f"removed={repo_cleanup.get('removed', 0)}, "
                f"media_deleted={repo_cleanup.get('media_deleted', 0)}, "
                f"summary_updated={repo_cleanup.get('summary_updated', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker repo cleanup skipped: {exc!r}")
    try:
        backfill = await store.backfill_stickers_from_raw_segments(plugin_config.ai_companion_sticker_segment_types)
        if backfill.get("samples_updated") or backfill.get("stickers_upserted"):
            logger.info(
                "ai companion sticker backfill: "
                f"samples_updated={backfill.get('samples_updated', 0)}, "
                f"stickers_upserted={backfill.get('stickers_upserted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker backfill skipped: {exc!r}")
    try:
        dedupe = await store.deduplicate_stickers_by_canonical_key(plugin_config.ai_companion_sticker_segment_types)
        if any(dedupe.values()):
            logger.info(
                "ai companion sticker dedupe: "
                f"merged_stickers={dedupe.get('merged_stickers', 0)}, "
                f"rekeyed_stickers={dedupe.get('rekeyed_stickers', 0)}, "
                f"rewritten_samples={dedupe.get('rewritten_samples', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker dedupe skipped: {exc!r}")
    try:
        admin_server.start()
    except Exception as exc:
        logger.exception(f"ai companion admin server failed to start: {exc!r}")
    if plugin_config.ai_companion_mlx_auto_load:
        try:
            load_result = await mlx_runtime.load()
            logger.info(
                f"ai companion mlx runtime load result: {load_result.get('status')}, model={load_result.get('model')}"
            )
        except Exception as exc:
            logger.exception(f"ai companion mlx runtime auto load failed: {exc!r}")
    _install_schedule()
    _start_memory_extraction_loop()


def _start_memory_extraction_loop() -> None:
    """后台批量增量抽取记忆：每 check_seconds 扫一次，够条数/够时长的会话才跑。"""
    global _memory_loop_started
    if _memory_loop_started:
        return
    _memory_loop_started = True
    asyncio.create_task(_memory_extraction_loop())


async def _memory_extraction_loop() -> None:
    from .conversation.summary import run_summary_round
    from .memory.extractor import run_memory_extraction_round

    while True:
        try:
            await asyncio.sleep(max(60, int(plugin_config.ai_companion_memory_extract_check_seconds or 300)))
            # 找人信箱 TTL：超时未投递的记录连同附件滚动清理，
            # 避免长期不上线的人攒下陈年积压、上线时被一次性补发。
            if bool(getattr(plugin_config, "ai_companion_ping_enabled", True)):
                try:
                    ttl_hours = float(getattr(plugin_config, "ai_companion_ping_ttl_hours", 24.0) or 0.0)
                    if ttl_hours > 0:
                        purged = await ping_service.purge_expired(ttl_seconds=ttl_hours * 3600)
                        if purged:
                            logger.info(f"ai companion ping inbox purged {purged} expired entries")
                except Exception as exc:  # noqa: BLE001 — 清理失败不影响记忆循环
                    logger.warning(f"ai companion ping purge failed: {exc!r}")
            if plugin_config.ai_companion_summary_enabled:
                try:
                    summaries = await run_summary_round(
                        conversation=store.conversation,
                        names_repo=store.member_names,
                        repo=store.group_summaries,
                        chat=_data_cleaning_chat,
                        min_new_turns=int(plugin_config.ai_companion_summary_min_turns),
                        interval_seconds=float(plugin_config.ai_companion_summary_interval_seconds),
                        disabled_groups={str(item) for item in plugin_config.ai_companion_memory_disabled_groups},
                    )
                    for item in summaries:
                        if item.get("error"):
                            logger.warning(f"ai companion summary skipped {item.get('group_id')}: {item['error']}")
                except Exception as exc:  # noqa: BLE001 — 摘要失败不影响记忆抽取
                    logger.warning(f"ai companion summary round failed: {exc!r}")
            if not plugin_config.ai_companion_memory_enabled:
                continue
            results = await run_memory_extraction_round(
                conversation=store.conversation,
                memories=store.persona_memory,
                names_repo=store.member_names,
                data_cleaning_chat=_data_cleaning_chat,
                min_new_turns=int(plugin_config.ai_companion_memory_extract_min_turns),
                interval_seconds=float(plugin_config.ai_companion_memory_extract_interval_seconds),
                disabled_groups={str(item) for item in plugin_config.ai_companion_memory_disabled_groups},
            )
            for item in results:
                if item.get("error"):
                    logger.warning(f"ai companion memory extraction skipped {item.get('session_key')}: {item['error']}")
        except asyncio.CancelledError:  # pragma: no cover - 关闭时退出
            raise
        except Exception as exc:  # noqa: BLE001 — 循环必须活下去
            logger.warning(f"ai companion memory extraction loop error: {exc!r}")


def _install_schedule() -> None:
    global _schedule_installed
    if _schedule_installed:
        return
    _schedule_installed = install_training_schedule(plugin_config, trainer)


@driver.on_shutdown
async def _shutdown() -> None:
    await chat_service.shutdown()
    await mlx_runtime.close()
    admin_server.stop()


# ---------------------------------------------------------------------------
# matcher 路由
# ---------------------------------------------------------------------------
ping_service = PingService(
    config=plugin_config,
    env=env,
    store=store,
    message_logs=store.message_logs,
)

milestone_service = MilestoneService(
    config=plugin_config,
    env=env,
    store=store,
    message_logs=store.message_logs,
)

persona_learn = on_message(priority=plugin_config.ai_companion_learning_priority, block=False)


@persona_learn.handle()
async def _learn_message(bot: Bot, event: Event) -> None:
    await learning_service.handle_learn(bot, event)


persona_chat = on_message(priority=plugin_config.ai_companion_reply_priority, block=False)


@persona_chat.handle()
async def _chat(bot: Bot, event: Event, matcher: Matcher) -> None:
    await chat_service.handle_chat(bot, event, matcher)


persona_ping = on_message(priority=plugin_config.ai_companion_learning_priority + 1, block=False)


@persona_ping.handle()
async def _ping(bot: Bot, event: Event, matcher: Matcher) -> None:
    try:
        await ping_service.handle_message(bot, event, matcher)
    except Exception as exc:  # noqa: BLE001 — 提醒是附加服务，任何异常都不影响其他 matcher
        logger.debug(f"ping service skipped: {exc!r}")


persona_milestone = on_message(priority=plugin_config.ai_companion_learning_priority + 2, block=False)


@persona_milestone.handle()
async def _milestone(bot: Bot, event: Event, matcher: Matcher) -> None:
    try:
        await milestone_service.handle_message(bot, event, matcher)
    except Exception as exc:  # noqa: BLE001 — 成就卡是附加服务，任何异常不影响其他 matcher
        logger.debug(f"milestone service skipped: {exc!r}")


persona_poke = on_notice(priority=plugin_config.ai_companion_reply_priority - 1, block=False)


@persona_poke.handle()
async def _counter_poke(bot: Bot, event: Event, matcher: Matcher) -> None:
    if not is_poke_at_self(event, env.bot_self_id(bot)):
        return
    try:
        sent = await send_counter_poke(bot, event)
    except Exception as exc:
        logger.debug(f"ai companion counter poke failed: {exc!r}")
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
async def _manual_train(args: Any = CommandArg()) -> None:  # noqa: B008
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


persona_trace = on_command(
    "trace",
    aliases={"溯源", "消息溯源"},
    priority=5,
    block=True,
)


@persona_trace.handle()
async def _trace_reply(event: Event) -> None:
    """引用 Bot 的某条回复并发送 /trace：按 reply_message_id 反查溯源（文档十三节）。"""
    try:
        quoted_id = quoted_message_id(event)
        if not quoted_id:
            await persona_trace.finish("请先引用一条机器人回复，再发送 /trace。")
        record = await store.message_logs.find_by_reply_message_id(quoted_id)
        if record is None:
            await persona_trace.finish(
                "没找到这条回复的溯源记录。\n"
                "注意：只有开启记录之后机器人发出的回复才能溯源"
                "（更早的回复、以及重启前未落库的回复不在记录范围内）。"
            )
        await persona_trace.finish(format_trace(record))
    except Exception as exc:  # noqa: BLE001 — 命令异常必须可见，不能静默吞掉
        logger.exception(f"ai companion /trace failed: {exc!r}")
        await persona_trace.finish(f"溯源查询出错：{type(exc).__name__}: {exc}")


persona_sticker_add = on_command(
    "添加表情",
    aliases={"收录表情"},
    priority=5,
    block=True,
)


@persona_sticker_add.handle()
async def _add_quoted_sticker(event: Event, args: Any = CommandArg()) -> None:  # noqa: B008
    """引用一条含图/表情的消息并发送 /添加表情 [描述]：纳入表情仓库（文档十三节）。"""
    reply_obj = getattr(event, "reply", None)
    quoted_id = quoted_message_id(event)
    if not quoted_id:
        await persona_sticker_add.finish("请引用一条包含图片或表情的消息后再发送 /添加表情。")
    samples = await store.samples.recent_samples(group_id=None, limit=2000)
    sample = next((item for item in samples if str(item.get("message_id") or "") == str(quoted_id)), None)
    if sample is None:
        # 引用消息可能未入库（非学习范围）——尝试直接读取被引用消息本体
        try:
            reply_message = getattr(reply_obj, "message", None) if reply_obj is not None else None
        except Exception:
            reply_message = None
        if reply_message is None:
            await persona_sticker_add.finish("没找到被引用消息的记录（该消息未被学习，且无法读取原文）。")
        from nonebot.adapters.onebot.v11 import Message as ObMessage

        segments = list(reply_message) if not isinstance(reply_message, ObMessage) else list(reply_message)
        records = []
        for segment in segments:
            record = sticker_to_record(
                {"type": getattr(segment, "type", ""), "data": dict(getattr(segment, "data", {}) or {})},
                set(plugin_config.ai_companion_sticker_segment_types),
            )
            if record is not None:
                records.append(record)
        if not records:
            await persona_sticker_add.finish("被引用消息里没有可收录的图片/表情。")
        label = (args.extract_plain_text().strip() if args else "")[:80]
        upserted = 0
        for record in records:
            if label:
                record["summary"] = label
            result = await store.stickers.record_sticker_asset(
                sticker=record,
                group_id=getattr(event, "group_id", None),
                user_id=getattr(event, "user_id", None),
                summary=label or str(record.get("summary") or ""),
                increment_count=True,
            )
            upserted += result.get("stickers_upserted", 0) or 0
        await persona_sticker_add.finish(f"已收录 {len(records)} 个表情（来自被引用消息）。")
    # 引用消息已入库：从 raw_segments 恢复表情
    import json as _json

    try:
        raw_segments = _json.loads(str(sample.get("raw_segments") or "[]"))
    except _json.JSONDecodeError:
        raw_segments = []
    enabled_types = set(plugin_config.ai_companion_sticker_segment_types)
    records = []
    for segment in raw_segments if isinstance(raw_segments, list) else []:
        if not isinstance(segment, dict):
            continue
        record = sticker_to_record(segment, enabled_types)
        if record is not None:
            records.append(record)
    if not records:
        await persona_sticker_add.finish("被引用消息里没有可收录的图片/表情。")
    label = (args.extract_plain_text().strip() if args else "")[:80]
    for record in records:
        if label:
            record["summary"] = label
        await store.stickers.record_sticker_asset(
            sticker=record,
            group_id=str(sample.get("group_id") or "") or None,
            user_id=str(sample.get("user_id") or "") or None,
            summary=label or str(record.get("summary") or ""),
            increment_count=True,
        )
    await persona_sticker_add.finish(f"已收录 {len(records)} 个表情（来自历史消息）。")


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
