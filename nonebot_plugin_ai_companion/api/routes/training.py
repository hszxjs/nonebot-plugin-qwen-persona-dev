"""训练路由：日志尾部读取与后台训练触发。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..helpers import log_tail_entries, training_operation, training_running

logger = logging.getLogger(__name__)


def build_router(*, config: Any, auth, trainer: Any | None) -> APIRouter:
    router = APIRouter(prefix="/api/training", dependencies=[auth])

    @router.get("/logs")
    async def training_logs(tail: int = 300) -> dict[str, Any]:
        progress = (
            trainer.progress_snapshot()
            if trainer is not None and hasattr(trainer, "progress_snapshot")
            else None
        )
        logs = log_tail_entries(str((progress or {}).get("log_tail", "")), stream="train")
        limit = max(0, min(int(tail), 1000))
        return {"logs": logs[-limit:]}

    @router.post("/run")
    async def training_run(request: Request) -> dict[str, Any]:
        if trainer is None or not hasattr(trainer, "run_training"):
            raise HTTPException(status_code=503, detail="trainer unavailable")
        if training_running(trainer):
            raise HTTPException(status_code=409, detail="training already running")
        try:
            payload = await request.json()
        except Exception as exc:
            logger.warning("ai companion admin training request body ignored: %r", exc)
            payload = {}
        if not isinstance(payload, dict):
            logger.warning("ai companion admin training request body ignored: expected object")
            payload = {}
        include_external_corpus = payload.get("include_external_corpus")
        if include_external_corpus is not None:
            include_external_corpus = bool(include_external_corpus)
        full_external_corpus = bool(payload.get("full_external_corpus"))
        reason = "admin_full_external" if full_external_corpus else "admin"

        async def run_background() -> None:
            try:
                await trainer.run_training(
                    reason=reason,
                    include_external_corpus=include_external_corpus,
                    full_external_corpus=full_external_corpus,
                )
            except Exception:
                logger.exception("ai companion admin training failed")

        asyncio.create_task(run_background())
        await asyncio.sleep(0)
        action = "run_full_external" if full_external_corpus else "run"
        return {
            "action": action,
            "message": (
                "start full external corpus training"
                if full_external_corpus
                else "start training"
            ),
            "training": training_operation(trainer),
        }

    return router
