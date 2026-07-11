from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import time
from typing import Any

import httpx

from .config import QwenPersonaConfig, effective_api_base, effective_chat_model
from .prompt_budget import effective_max_tokens


@dataclass(frozen=True)
class InferenceMetric:
    time: str
    model: str
    latency_seconds: float
    completion_tokens: int | None
    response_chars: int
    tps: float
    source: str


class QwenClient:
    def __init__(self, config: QwenPersonaConfig) -> None:
        self.config = config
        self._metrics: deque[InferenceMetric] = deque(maxlen=120)

    async def chat(self, messages: list[dict[str, str]]) -> str:
        base = effective_api_base(self.config).rstrip("/")
        headers = {"Content-Type": "application/json"}
        api_key = self.config.qwen_persona_api_key.strip()
        if api_key and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": effective_chat_model(self.config),
            "messages": messages,
            "temperature": self.config.qwen_persona_temperature,
            "top_p": self.config.qwen_persona_top_p,
            "max_tokens": effective_max_tokens(self.config),
            "stream": False,
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.config.qwen_persona_request_timeout) as client:
            response = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = str(data["choices"][0]["message"]["content"]).strip()
        self._record_metric(data, content, time.perf_counter() - started)
        return content

    def metrics_snapshot(self) -> dict[str, Any]:
        history = [asdict(item) for item in self._metrics]
        return {
            "last": history[-1] if history else None,
            "history": history,
        }

    def _record_metric(self, data: dict[str, Any], content: str, elapsed: float) -> None:
        if elapsed <= 0:
            return

        usage = data.get("usage") if isinstance(data, dict) else None
        completion_tokens = self._completion_tokens(usage)
        source = "usage"
        if completion_tokens is None:
            completion_tokens = self._estimate_tokens(content)
            source = "estimated"
        if completion_tokens is None or completion_tokens <= 0:
            return

        self._metrics.append(
            InferenceMetric(
                time=datetime.now(UTC).isoformat(timespec="seconds"),
                model=effective_chat_model(self.config),
                latency_seconds=round(elapsed, 3),
                completion_tokens=completion_tokens if source == "usage" else None,
                response_chars=len(content),
                tps=round(completion_tokens / elapsed, 2),
                source=source,
            )
        )

    @staticmethod
    def _completion_tokens(usage: Any) -> int | None:
        if not isinstance(usage, dict):
            return None
        value = usage.get("completion_tokens")
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            return None
        return tokens if tokens > 0 else None

    @staticmethod
    def _estimate_tokens(content: str) -> int | None:
        content = str(content or "").strip()
        if not content:
            return None
        return max(1, round(len(content) / 2))
