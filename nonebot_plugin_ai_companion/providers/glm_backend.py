"""GLM 在线推理 backend（OpenAI 兼容）。

当 ai_companion_inference_backend == "glm" 时，QwenClient 把对话推理分发到这里。
接口对齐 MlxVLMRuntime 的 chat / metrics_snapshot / status_snapshot / reconfigure，
使上层（admin_server、inference_events）无需感知后端差异。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import re
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import AiCompanionConfig
from ..contracts import InferenceMetric, utc_timestamp
from .base import ProviderCapabilities
from .registry import register

logger = logging.getLogger(__name__)

# 匹配 GLM 等模型常用 ```json ... ``` 代码块围栏，用于容错解析。
_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```", re.DOTALL)

# 仅对这些状态码做指数退避重试；4xx（鉴权/参数错误）直接抛出。
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3

# 支持多模态的 OneBot 段落类型。
_MEDIA_PART_TYPES = {"image", "mface", "emoji", "sticker"}


def _normalize_models_url(chat_url: str) -> str:
    """从 chat/completions 端点派生 models 端点。

    https://api.z.ai/api/coding/paas/v4/chat/completions
        -> https://api.z.ai/api/coding/paas/v4/models
    """
    url = (chat_url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    path = parts.path or ""
    # 去掉尾部 /chat/completions
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    elif path.endswith("/chat/completions/"):
        path = path[: -len("/chat/completions/")]
    # /v1 结尾的补成 /v1/models；否则直接 /models
    if path.endswith("/v1") or path.endswith("/v4"):
        path = path + "/models"
    elif not path.endswith("/models"):
        path = path.rstrip("/") + "/models"
    return urlunsplit(parts._replace(path=path))


def _extract_chat_content(data: dict[str, Any]) -> str:
    """兼容 OpenAI 标准与部分 GLM 中转的字段差异，提取回复文本。"""
    try:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("empty choices in chat response")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, dict):
                    parts.append(str(seg.get("text") or seg.get("content") or ""))
                else:
                    parts.append(str(seg))
            content = "".join(parts)
        if not content:
            raise ValueError("empty content in chat response")
        return str(content)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unrecognized chat response shape: {exc!r}") from exc


def _uri_to_data_url(uri: str) -> str | None:
    """把图片 URI 规范化成 GLM 可接受的 data URL 或 http(s) URL。

    - http(s) URL：直接返回（GLM 可访问公网图床）。
    - 本地文件路径：读文件转 base64 data URL。
    - 其它（file_id、base64://、无法读取）：返回 None，由调用方跳过。
    """
    uri = (uri or "").strip()
    if not uri:
        return None
    if uri.startswith(("http://", "https://")):
        return uri
    if uri.startswith("base64://"):
        # MLX 路径会跳过这种，但 GLM 需要它；补上 data URL 前缀需要 MIME，缺失则按 jpeg。
        raw = uri[len("base64://") :]
        return f"data:image/jpeg;base64,{raw}"
    # 本地路径
    candidate = uri[7:] if uri.startswith("file://") else uri
    try:
        path = Path(candidate)
        if not path.is_file():
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        data = path.read_bytes()
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except (OSError, ValueError) as exc:
        logger.debug("glm_backend: skip unreadable media uri=%r err=%r", uri, exc)
        return None


def _media_part_uri(part: dict[str, Any]) -> str:
    """从 OneBot 段落里提取图片 URI（url/file/file_id/uri 任一）。"""
    if not isinstance(part, dict):
        return ""
    part_type = str(part.get("type") or "")
    if part_type not in _MEDIA_PART_TYPES:
        return ""
    data = part.get("data")
    if not isinstance(data, dict):
        return ""
    for field in ("url", "file", "file_id", "uri"):
        value = str(data.get(field) or "").strip()
        if value:
            return value
    return ""


def _build_openai_messages(
    messages: list[dict[str, Any]],
    media_parts: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """把插件 messages + media_parts 组装成 OpenAI 多模态 messages。

    多模态图片附加到最后一条 user 消息的 content 里（转成 content 数组）。
    无法转换的图片跳过并记日志，不阻塞回复。
    """
    # 收集所有可用的图片 data URL
    image_urls: list[str] = []
    if media_parts:
        for part in media_parts:
            uri = _media_part_uri(part)
            if not uri:
                continue
            data_url = _uri_to_data_url(uri)
            if data_url:
                image_urls.append(data_url)
            else:
                logger.info("glm_backend: skip media part uri=%r (unreadable)", uri)

    if not image_urls:
        return messages

    # 找最后一条 user 消息，把图片附加进去
    out = [dict(m) for m in messages]
    target_idx = None
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].get("role") == "user":
            target_idx = idx
            break
    if target_idx is None:
        # 没有 user 消息，新建一条承载图片
        out.append({"role": "user", "content": []})
        target_idx = len(out) - 1

    original_content = out[target_idx].get("content")
    content_list: list[dict[str, Any]] = []
    if isinstance(original_content, str):
        if original_content:
            content_list.append({"type": "text", "text": original_content})
    elif isinstance(original_content, list):
        content_list.extend(original_content)
    for url in image_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})
    out[target_idx]["content"] = content_list
    return out


@register(
    "glm",
    provider_type="remote",
    display_name="GLM 在线 API",
    docs_url="https://open.bigmodel.cn/dev/api",
)
class GlmBackend:
    """GLM 在线推理 backend。无本地状态，可并发，无 load/unload 概念。

    多模态说明：z.ai 的 Coding 订阅端点（coding/paas/v4）只支持纯文本，
    带图消息会触发 400。因此默认 multimodal=False：遇到图片时丢弃图片、
    仅发送文字部分（并记日志）。若你的端点支持多模态（如普通 paas/v4
    + API 资源包），可在 online 配置里设 multimodal=true 开启图片转 base64。
    """

    provider_name = "glm"

    def __init__(self, config: AiCompanionConfig) -> None:
        self.config = config
        self._metrics: deque[InferenceMetric] = deque(maxlen=120)
        self._last_error: str = ""
        self._healthy: bool = True

    def reconfigure(self, config: AiCompanionConfig) -> None:
        self.config = config

    @property
    def _api_url(self) -> str:
        return (self.config.ai_companion_online_api_url or "").strip()

    @property
    def _model(self) -> str:
        return (self.config.ai_companion_online_model or "glm-5.2").strip() or "glm-5.2"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = (self.config.ai_companion_online_api_key or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: str = "",
    ) -> str:
        url = self._api_url
        if not url:
            self._last_error = "ai_companion_online_api_url 未配置"
            self._healthy = False
            raise RuntimeError(self._last_error)

        # Coding 订阅端点只支持纯文本：multimodal=False 时丢弃图片仅发文字。
        multimodal = bool(getattr(self.config, "ai_companion_online_multimodal", False))
        if multimodal:
            openai_messages = _build_openai_messages(messages, media_parts)
        else:
            # 纯文本模式：剥离图片，但记录有图片被丢弃（便于排查）。
            if media_parts:
                dropped = sum(
                    1 for p in media_parts if isinstance(p, dict) and str(p.get("type") or "") in _MEDIA_PART_TYPES
                )
                if dropped:
                    logger.info("glm_backend: %d media part(s) dropped (multimodal disabled)", dropped)
            openai_messages = messages
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": 0.7,
        }
        timeout = float(self.config.ai_companion_online_timeout or 90.0)
        headers = self._headers()

        started = utc_timestamp()
        elapsed_wall = asyncio.get_event_loop().time()
        data = await self._post_with_retry(url, headers, payload, timeout)
        content = _extract_chat_content(data)
        elapsed = asyncio.get_event_loop().time() - elapsed_wall
        self._record_metric(
            content=content,
            elapsed=elapsed,
            usage=data.get("usage"),
            finish_reason=(data.get("choices") or [{}])[0].get("finish_reason", ""),
            purpose=purpose,
            started=started,
        )
        self._healthy = True
        self._last_error = ""
        return content

    async def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code in _RETRYABLE_STATUS:
                        raise httpx.HTTPStatusError(
                            f"glm status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    # 4xx 等不可重试错误：立刻抛出。
                    if exc.response is None or exc.response.status_code not in _RETRYABLE_STATUS:
                        self._last_error = f"HTTP {exc.response.status_code if exc.response else '?'}"
                        self._healthy = False
                        raise
                    last_exc = exc
                    if attempt >= _MAX_RETRIES:
                        self._last_error = f"HTTP {exc.response.status_code} after retries"
                        self._healthy = False
                        raise
                    await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
                except httpx.TransportError as exc:
                    last_exc = exc
                    if attempt >= _MAX_RETRIES:
                        self._last_error = f"transport error: {type(exc).__name__}"
                        self._healthy = False
                        raise
                    await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
        # 理论上不可达
        raise last_exc  # type: ignore[misc]

    def _record_metric(
        self,
        *,
        content: str,
        elapsed: float,
        usage: Any | None,
        finish_reason: str,
        purpose: str,
        started: str,
    ) -> None:
        if elapsed <= 0:
            return
        tokens = None
        if isinstance(usage, dict):
            try:
                tokens = int(usage.get("completion_tokens") or 0) or None
            except (TypeError, ValueError):
                tokens = None
        source = "glm_api" if tokens is not None else "estimated"
        if tokens is None:
            # 粗估：中文约 1.5 字符/token，英文约 4 字符/token，取折中。
            tokens = max(1, int(len(content) / 2.5))
        self._metrics.append(
            InferenceMetric(
                time=started,
                model=self._model,
                latency_seconds=round(elapsed, 3),
                completion_tokens=tokens if source == "glm_api" else None,
                response_chars=len(content),
                tps=round(tokens / elapsed, 2),
                source=source,
                finish_reason=finish_reason,
                purpose=purpose,
            )
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        history = [asdict(item) for item in self._metrics]
        return {"last": history[-1] if history else None, "history": history}

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            vision=bool(getattr(self.config, "ai_companion_online_multimodal", False)),
            function_calling=False,
            streaming=False,
            lora_hotload=False,
        )

    def default_models(self) -> list[dict[str, str]]:
        return [{"value": name, "label": name, "description": "GLM 在线 API 模型"} for name in _FALLBACK_MODELS]

    async def health_check(self) -> bool:
        # 无独立探活端点：以最近一次调用的健康状态为准（进程启动后默认健康）。
        return bool(self.status_snapshot().get("healthy"))

    def status_snapshot(self) -> dict[str, Any]:
        configured = bool(self._api_url)
        return {
            "runtime": "glm_online",
            "status": "loaded" if configured else "not_configured",
            "state": "loaded" if configured else "not_configured",
            "healthy": bool(configured and self._healthy),
            "model": self._model,
            "loaded_model": self._model if configured else "",
            "loaded_model_path": "",
            "adapter_path": "",
            "loaded_adapter_path": "",
            "last_error": self._last_error,
            "operation": None,
            "generation": None,
            "control": {
                "can_load": False,
                "can_unload": False,
                "can_reload": False,
            },
            "metrics": self.metrics_snapshot(),
        }


async def fetch_online_models(
    chat_url: str,
    api_key: str,
    *,
    timeout: float = 8.0,
) -> tuple[list[str], str]:
    """从 /models 端点拉取可用模型列表。返回 (models, source)。

    source 为 "api" 表示真实拉取，"fallback" 表示用兜底列表。
    任何异常都走兜底，不抛出。
    """
    models_url = _normalize_models_url(chat_url)
    if not models_url:
        return list(_FALLBACK_MODELS), "fallback"
    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(models_url, headers=headers)
            response.raise_for_status()
            data = response.json()
        # OpenAI 标准是 {"data": [{"id": ...}, ...]}，部分网关返回 {"models": [...]}
        items: list[Any] = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
        elif isinstance(data, list):
            items = data
        names: list[str] = []
        for item in items:
            name = ""
            if isinstance(item, dict):
                name = str(item.get("id") or item.get("model") or item.get("name") or "")
            elif isinstance(item, str):
                name = item
            name = name.strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return list(_FALLBACK_MODELS), "fallback"
        return names, "api"
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.info("fetch_online_models fallback: %r", exc)
        return list(_FALLBACK_MODELS), "fallback"


# 动态拉取失败时的兜底模型列表（z.ai GLM 常见可用模型）。
_FALLBACK_MODELS: tuple[str, ...] = (
    "glm-5.2",
    "glm-4.5-flash",
    "glm-4.6",
    "glm-4.5",
)
