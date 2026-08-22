"""OpenAI 兼容 HTTP 后端基类：Ollama / llama.cpp / 任意兼容端点共用。

与 GlmBackend 同构（chat / chat_with_tools / status_snapshot / metrics），
但面向"用户可自建多个实例"的场景（InferenceBackendConfig 持久化配置，
见 providers/instances.py），构造参数来自实例设置而非插件全局配置。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import asdict
from typing import Any

import httpx

from ..contracts import InferenceMetric, utc_timestamp
from .base import ProviderCapabilities
from .glm_backend import _build_openai_messages

logger = logging.getLogger(__name__)

_MEDIA_PART_TYPES = {"image", "mface", "emoji", "sticker"}
_RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatBackend:
    """base_url 需指向 OpenAI 兼容根（如 http://127.0.0.1:11434/v1）。"""

    provider_name = "openai_compat"

    def __init__(
        self,
        *,
        provider_name: str,
        display_name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        capabilities: ProviderCapabilities | None = None,
        default_models: list[dict[str, str]] | None = None,
        docs_url: str = "",
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.display_name = display_name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._capabilities = capabilities or ProviderCapabilities()
        self._default_models = default_models or []
        self.docs_url = docs_url
        self._timeout = timeout
        self._extra_body = dict(extra_body or {})
        self._metrics: deque[InferenceMetric] = deque(maxlen=120)
        self._healthy: bool = True
        self._last_error: str = ""

    # ---- 配置 ----
    def update_settings(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if base_url:
            self._base_url = base_url.rstrip("/")
        if model:
            self._model = model
        if api_key is not None:
            self._api_key = api_key
        if extra_body is not None:
            self._extra_body = dict(extra_body)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ---- BaseProvider 接口 ----
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def default_models(self) -> list[dict[str, str]]:
        return list(self._default_models)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self._base_url}/models", headers=self._headers())
            self._healthy = response.status_code == 200
            if self._healthy:
                self._last_error = ""
        except httpx.HTTPError as exc:
            self._healthy = False
            self._last_error = f"health check failed: {exc!r}"
        return self._healthy

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: str = "",
    ) -> str:
        content, _tool_calls = await self._post_chat(messages, media_parts=media_parts)
        return content

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> tuple[str, list[dict[str, Any]]]:
        return await self._post_chat(messages, tools=tools, tool_choice=tool_choice)

    def metrics_snapshot(self) -> dict[str, Any]:
        history = [asdict(item) for item in self._metrics]
        return {"last": history[-1] if history else None, "history": history}

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "runtime": f"{self.provider_name}_http",
            "status": "loaded",
            "state": "loaded",
            "healthy": self._healthy,
            "model": self._model,
            "loaded_model": self._model,
            "loaded_model_path": self._base_url,
            "adapter_path": "",
            "loaded_adapter_path": "",
            "last_error": self._last_error,
            "operation": None,
            "generation": None,
            "control": {"can_load": False, "can_unload": False, "can_reload": False},
            "metrics": self.metrics_snapshot(),
        }

    # ---- 内部 ----
    async def _post_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str, list[dict[str, Any]]]:
        if self._capabilities.vision and media_parts:
            payload_messages: list[dict[str, Any]] = _build_openai_messages(messages, media_parts)
        else:
            if media_parts:
                dropped = sum(
                    1
                    for part in media_parts
                    if isinstance(part, dict) and str(part.get("type") or "") in _MEDIA_PART_TYPES
                )
                if dropped:
                    logger.info("%s: %d media part(s) dropped (vision disabled)", self.provider_name, dropped)
            payload_messages = messages

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            **self._extra_body,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        started_monotonic = asyncio.get_event_loop().time()
        data = await self._post_with_retry(payload)
        message = (data.get("choices") or [{}])[0].get("message", {}) or {}
        content = str(message.get("content") or "")
        elapsed = asyncio.get_event_loop().time() - started_monotonic
        usage = data.get("usage") or {}
        self._record_metric(
            content=content,
            latency=elapsed,
            completion_tokens=usage.get("completion_tokens"),
            purpose=str(message.get("purpose") or ""),
        )
        tool_calls = [dict(call) for call in message.get("tool_calls") or [] if isinstance(call, dict)]
        return content, tool_calls

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        last_error = ""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                    response = await client.post(url, headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._healthy = False
            else:
                if response.status_code == 200:
                    self._healthy = True
                    self._last_error = ""
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                self._healthy = response.status_code < 500
                if response.status_code not in _RETRY_STATUS:
                    break
            await asyncio.sleep(min(2.0**attempt, 4.0))
        self._last_error = last_error
        raise RuntimeError(f"{self.provider_name} chat failed: {last_error}")

    def _record_metric(
        self,
        *,
        content: str,
        latency: float,
        completion_tokens: int | None,
        purpose: str,
    ) -> None:
        self._metrics.append(
            InferenceMetric(
                time=utc_timestamp(),
                model=self._model,
                latency_seconds=round(max(0.0, latency), 3),
                completion_tokens=completion_tokens,
                response_chars=len(content or ""),
                tps=round(len(content or "") / latency, 1) if latency > 0 else 0.0,
                source=self.provider_name,
                purpose=purpose,
            )
        )
