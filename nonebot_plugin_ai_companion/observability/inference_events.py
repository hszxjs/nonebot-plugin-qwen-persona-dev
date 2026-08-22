from __future__ import annotations

import time
from typing import Any


def _event_id(ids: dict[str, str | None], *, created_at: float) -> str:
    message_id = ids.get("message_id") or "no-message"
    group_id = ids.get("group_id") or "private"
    user_id = ids.get("user_id") or "unknown"
    return f"{group_id}:{user_id}:{message_id}:{created_at:.6f}"


async def record_inference_event(
    *,
    store: Any,
    ids: dict[str, str | None],
    trigger_reason: str,
    model: str,
    adapter: str,
    started_at: float,
    finished_at: float | None = None,
    input_text: str = "",
    output_text: str = "",
    media_count: int = 0,
    status: str = "success",
    error_summary: str = "",
    tps: float | None = None,
) -> None:
    if not store or not hasattr(store, "record_inference_event"):
        return
    done_at = float(finished_at if finished_at is not None else time.perf_counter())
    latency = max(0.0, done_at - float(started_at))
    await store.record_inference_event(
        event_id=_event_id(ids, created_at=time.time()),
        status=status,
        trigger_reason=trigger_reason,
        model=model,
        adapter=adapter,
        latency_seconds=latency,
        tps=tps,
        media_count=max(0, int(media_count or 0)),
        response_chars=len(output_text or ""),
        error_summary=error_summary,
        input_text=input_text,
        output_text=output_text,
        group_id=ids.get("group_id"),
        user_id=ids.get("user_id"),
    )
