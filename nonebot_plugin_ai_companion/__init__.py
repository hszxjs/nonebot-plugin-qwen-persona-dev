"""nonebot_plugin_ai_companion：AI 对话伙伴插件入口。

matcher 层只做路由 + 依赖装配；业务编排在 chat_service / learning_service，
领域逻辑在各领域子包（见 docs/architecture.md）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nonebot import get_driver, logger, on_command, on_message, on_notice
from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

from .api.server import AdminServer
from .anime_captioner import AnimeCaptioner
from .bot_env import BotEnvironment
from .chat_service import ChatService
from .config import QwenPersonaConfig, load_plugin_config, normalize_matcher_priorities
from .context.nicknames import NicknameResolver
from .decision.repeat import ConversationPatternTracker, RepeatTracker
from .delivery.onebot_v11 import is_poke_at_self, send_counter_poke
from .finetune.schedule import install_training_schedule
from .finetune.trainer import PersonaTrainer
from .learning_service import LearningService
from .providers.client import QwenClient
from .providers.mlx_runtime import MlxVLMRuntime
from .runtime_config import (
    apply_config_values,
    config_to_dict,
    load_config_overrides,
    save_config_overrides,
)
from .sticker.repo_files import cleanup_sticker_repo_manifest
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
        f"learning={plugin_config.qwen_persona_learning_priority}, "
        f"reply={plugin_config.qwen_persona_reply_priority}"
    )

env = BotEnvironment(driver.config, plugin_config)
store = PersonaStore(Path(plugin_config.qwen_persona_data_dir))
mlx_runtime = MlxVLMRuntime(plugin_config)
client = QwenClient(plugin_config, runtime=mlx_runtime)
anime_captioner = AnimeCaptioner(plugin_config)
nickname_resolver = NicknameResolver()
repeat_tracker = RepeatTracker()
conversation_pattern_tracker = ConversationPatternTracker()
tool_registry = ToolRegistry()
trainer = PersonaTrainer(plugin_config, store)

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
)

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
_schedule_installed = False


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
@driver.on_startup
async def _startup() -> None:
    await store.init()
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
            logger.info(
                "ai companion sticker summary cleanup: "
                f"updated={summary_cleanup.get('updated', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker summary cleanup skipped: {exc!r}")
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
                "ai companion sticker repo cleanup: "
                f"removed={repo_cleanup.get('removed', 0)}, "
                f"media_deleted={repo_cleanup.get('media_deleted', 0)}, "
                f"summary_updated={repo_cleanup.get('summary_updated', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker repo cleanup skipped: {exc!r}")
    try:
        backfill = await store.backfill_stickers_from_raw_segments(
            plugin_config.qwen_persona_sticker_segment_types
        )
        if backfill.get("samples_updated") or backfill.get("stickers_upserted"):
            logger.info(
                "ai companion sticker backfill: "
                f"samples_updated={backfill.get('samples_updated', 0)}, "
                f"stickers_upserted={backfill.get('stickers_upserted', 0)}"
            )
    except Exception as exc:
        logger.warning(f"ai companion sticker backfill skipped: {exc!r}")
    try:
        dedupe = await store.deduplicate_stickers_by_canonical_key(
            plugin_config.qwen_persona_sticker_segment_types
        )
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
    if plugin_config.qwen_persona_mlx_auto_load:
        try:
            load_result = await mlx_runtime.load()
            logger.info(
                "ai companion mlx runtime load result: "
                f"{load_result.get('status')}, model={load_result.get('model')}"
            )
        except Exception as exc:
            logger.exception(f"ai companion mlx runtime auto load failed: {exc!r}")
    _install_schedule()


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
persona_learn = on_message(priority=plugin_config.qwen_persona_learning_priority, block=False)


@persona_learn.handle()
async def _learn_message(bot: Bot, event: Event) -> None:
    await learning_service.handle_learn(bot, event)


persona_chat = on_message(priority=plugin_config.qwen_persona_reply_priority, block=False)


@persona_chat.handle()
async def _chat(bot: Bot, event: Event, matcher: Matcher) -> None:
    await chat_service.handle_chat(bot, event, matcher)


persona_poke = on_notice(priority=plugin_config.qwen_persona_reply_priority - 1, block=False)


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
