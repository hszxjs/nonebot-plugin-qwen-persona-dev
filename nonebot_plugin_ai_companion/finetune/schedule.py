"""微调域：定时训练任务安装（nonebot-plugin-apscheduler）。"""

from __future__ import annotations

import logging

from nonebot import require

from ..config import AiCompanionConfig
from .trainer import PersonaTrainer, parse_time_slot

logger = logging.getLogger(__name__)


async def run_scheduled_training(trainer: PersonaTrainer) -> None:
    try:
        result = await trainer.run_training(reason="scheduled")
        logger.info(
            "ai companion scheduled training finished: "
            f"{result.status}, samples={result.sample_count}"
        )
    except Exception as exc:
        logger.exception(f"ai companion scheduled training failed: {exc!r}")


def install_training_schedule(
    config: AiCompanionConfig,
    trainer: PersonaTrainer,
) -> bool:
    """按 train_times 注册 cron 任务；返回是否成功安装。幂等由调用方保证。"""
    if not config.ai_companion_enable_training:
        return False

    try:
        require("nonebot_plugin_apscheduler")
        from nonebot_plugin_apscheduler import scheduler
    except Exception as exc:
        logger.warning(f"ai companion training schedule disabled: {exc!r}")
        return False

    async def _job() -> None:
        await run_scheduled_training(trainer)

    for slot in config.ai_companion_train_times:
        try:
            hour, minute = parse_time_slot(slot)
        except ValueError as exc:
            logger.warning(f"Invalid ai_companion_train_times item {slot!r}: {exc}")
            continue

        scheduler.add_job(
            _job,
            "cron",
            hour=hour,
            minute=minute,
            id=f"ai_companion_lora_{hour:02d}_{minute:02d}",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info(f"ai companion LoRA training scheduled at {hour:02d}:{minute:02d}")
    return True


def schedule_job_ids(config: AiCompanionConfig) -> list[str]:
    ids: list[str] = []
    for slot in config.ai_companion_train_times:
        try:
            hour, minute = parse_time_slot(slot)
        except ValueError:
            continue
        ids.append(f"ai_companion_lora_{hour:02d}_{minute:02d}")
    return ids
