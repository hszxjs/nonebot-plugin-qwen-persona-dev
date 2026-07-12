from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

_UNSET = object()


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RuntimeOperation:
    type: str
    phase: str
    message: str = ""
    state: str = "running"
    id: str = field(default_factory=lambda: uuid4().hex)
    percent: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    started_at: str = field(default_factory=utc_timestamp)
    finished_at: str | None = None
    error: str = ""
    target_model: str = ""
    local_path: str = ""
    previous_model: str = ""

    def update(
        self,
        *,
        phase: str | None = None,
        message: str | None = None,
        percent: float | None | object = _UNSET,
        downloaded_bytes: int | None = None,
        total_bytes: int | None = None,
        local_path: str | None = None,
    ) -> None:
        if phase is not None:
            self.phase = phase
        if message is not None:
            self.message = message
        if downloaded_bytes is not None:
            self.downloaded_bytes = int(downloaded_bytes)
        if total_bytes is not None:
            self.total_bytes = int(total_bytes)
        if percent is not _UNSET:
            self.percent = (
                None
                if percent is None
                else round(max(0.0, min(float(percent), 100.0)), 2)
            )
        elif self.downloaded_bytes is not None and self.total_bytes:
            self.percent = round(
                max(0.0, min((self.downloaded_bytes / self.total_bytes) * 100.0, 100.0)),
                2,
            )
        if local_path is not None:
            self.local_path = local_path

    def finish(self, *, state: str, phase: str, message: str = "", error: str = "") -> None:
        self.state = state
        self.phase = phase
        self.finished_at = utc_timestamp()
        if message:
            self.message = message
        if error:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationProgress:
    def __init__(self, update: Any) -> None:
        self._update = update

    def update(self, **kwargs: Any) -> None:
        self._update(**kwargs)


@dataclass(frozen=True)
class MlxRuntimeLog:
    time: str
    stream: str
    message: str
    level: str = "info"
    event: str = ""
    source: str = "mlx_runtime"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceMetric:
    time: str
    model: str
    latency_seconds: float
    completion_tokens: int | None
    response_chars: int
    tps: float
    source: str
