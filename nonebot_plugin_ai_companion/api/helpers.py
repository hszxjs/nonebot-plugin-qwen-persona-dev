"""admin API 共享助手：异步解包、运行时摘要、训练摘要、日志文件读取。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import normalized_lora_modules
from ..model_manager.presets import MLX_MODEL_PRESETS


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def runtime_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime": status.get("runtime", "mlx_vlm"),
        "state": status.get("state", "unknown"),
        "status": status.get("status", ""),
        "healthy": bool(status.get("healthy")),
        "model": status.get("model", ""),
        "loaded_model": status.get("loaded_model", ""),
        "adapter_path": status.get("adapter_path", ""),
        "loaded_adapter_path": status.get("loaded_adapter_path", ""),
        "last_error": status.get("last_error", ""),
        "operation": status.get("operation"),
        "generation": status.get("generation", {"active": 0, "queued": 0, "current": None}),
        "control": status.get("control", {}),
        "metrics": status.get("metrics", {"last": None, "history": []}),
        "model_presets": MLX_MODEL_PRESETS,
    }


def training_operation(trainer: Any | None) -> dict[str, Any]:
    progress = (
        trainer.progress_snapshot()
        if trainer is not None and hasattr(trainer, "progress_snapshot")
        else None
    )
    return {
        "running": bool(progress and progress.get("status") == "running"),
        "available": trainer is not None,
        "progress": progress,
    }


def training_running(trainer: Any | None) -> bool:
    if trainer is None or not hasattr(trainer, "progress_snapshot"):
        return False
    progress = trainer.progress_snapshot()
    return bool(progress and progress.get("status") == "running")


def training_config(config: Any) -> dict[str, Any]:
    return {
        "target_configured": True,
        "helper_configured": bool(
            str(getattr(config, "ai_companion_lora_helper_api_url", "") or "").strip()
        ),
        "min_train_samples": int(getattr(config, "ai_companion_min_train_samples", 0) or 0),
        "dataset_limit": int(getattr(config, "ai_companion_dataset_limit", 0) or 0),
        "external_corpus_enabled": bool(
            getattr(config, "ai_companion_external_corpus_enabled", False)
        ),
        "external_corpus_limit": int(
            getattr(config, "ai_companion_external_corpus_limit", 0) or 0
        ),
        "multimodal_training_enabled": bool(
            getattr(config, "ai_companion_multimodal_training_enabled", False)
        ),
        "mlx_lora_modules": normalized_lora_modules(config),
    }


def log_tail_entries(log_tail: str, *, stream: str = "train") -> list[dict[str, str]]:
    lines = [line.strip() for line in str(log_tail or "").splitlines() if line.strip()]
    return [{"time": "", "stream": stream, "message": line} for line in lines[-300:]]


def plugin_log_path(config: Any) -> Path:
    data_dir = Path(
        str(getattr(config, "ai_companion_data_dir", "data/qwen_persona") or "data/qwen_persona")
    )
    return data_dir / "bot_runtime.log"


def file_log_entries(path: Path, *, tail: int, stream: str) -> list[dict[str, str]]:
    limit = max(0, min(int(tail), 1000))
    if limit <= 0 or not path.is_file():
        return []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return [{"time": "", "stream": stream, "message": line} for line in lines[-limit:]]
