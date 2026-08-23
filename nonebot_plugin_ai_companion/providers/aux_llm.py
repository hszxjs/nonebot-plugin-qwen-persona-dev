"""辅助 LLM 调用：复用在线推理配置做轻量判断/分析（图文相关性、风格分析等）。

与主推理通道（providers/client.py）分离：辅助调用固定走 HTTP、可用更便宜的
flash 模型，失败时由调用方决定保守策略。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..config import AiCompanionConfig

logger = logging.getLogger(__name__)


class AuxLlmError(RuntimeError):
    """辅助 LLM 调用失败（未配置/网络错误/非 200/空响应）。"""


def resolve_aux_endpoint(config: AiCompanionConfig) -> tuple[str, str] | None:
    """解析辅助调用端点：优先 online 配置，降级 lora_helper。返回 (url, key) 或 None。"""
    url = (config.ai_companion_online_api_url or "").strip()
    key = (config.ai_companion_online_api_key or "").strip()
    if not url:
        url = (config.ai_companion_lora_helper_api_url or "").strip()
        key = (config.ai_companion_lora_helper_api_key or "").strip()
    if not url or not key:
        return None
    return url, key


async def aux_chat(
    config: AiCompanionConfig,
    *,
    prompt: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: float = 10.0,
    disable_thinking: bool = True,
) -> str:
    """一次辅助对话调用，返回文本内容；失败抛 AuxLlmError。

    默认关闭思考（thinking: disabled）：辅助任务是轻量判断/结构化提取，
    思考型模型会把 max_tokens 耗在 reasoning 上导致 content 为空。
    端点不认识 thinking 字段（400）时自动去参重试一次。
    """
    endpoint = resolve_aux_endpoint(config)
    if endpoint is None:
        raise AuxLlmError("aux llm endpoint not configured")
    url, key = endpoint
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}

    async def post(body: dict[str, Any]):
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            return await client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
            )

    response = None
    last_transport_error: Exception | None = None
    for attempt in range(3):  # 传输层瞬时错误（连接抖动等）轻量重试
        try:
            response = await post(payload)
            if disable_thinking and response.status_code == 400:
                # 端点可能不支持 thinking 字段（如部分 OpenAI 兼容后端）
                response = await post({k: v for k, v in payload.items() if k != "thinking"})
            break
        except httpx.HTTPError as exc:
            last_transport_error = exc
            if attempt < 2:
                await asyncio.sleep(1.0 + attempt)
    if response is None:
        raise AuxLlmError(f"aux llm request failed: {last_transport_error!r}") from last_transport_error
    if response.status_code != 200:
        raise AuxLlmError(f"aux llm status {response.status_code}: {response.text[:200]}")
    try:
        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = str(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "")
        has_reasoning = bool(message.get("reasoning_content"))
    except ValueError as exc:
        raise AuxLlmError(f"aux llm invalid json: {exc!r}") from exc
    if not content.strip():
        hint = (
            "reasoning 耗尽 max_tokens（调大 max_tokens 或关闭思考）"
            if (has_reasoning or finish_reason == "length")
            else "模型未返回内容"
        )
        raise AuxLlmError(f"aux llm empty content (model={model}, finish={finish_reason}; {hint})")
    return content.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """容错解析模型输出中的 JSON 对象（含 ```json 围栏与前后噪声）。"""
    import json
    import re

    body = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```", body, re.DOTALL)
    if fence:
        body = fence.group("body").strip()
    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        body = body[start : end + 1]
    last_error: Exception | None = None
    for candidate in _json_repair_candidates(body):
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = AuxLlmError("aux llm json is not an object")
    raise AuxLlmError(f"aux llm json parse failed: {last_error!r}")


def _json_repair_candidates(body: str) -> list[str]:
    """按宽松度递增给出可尝试的 JSON 文本（原文 → 去尾逗号）。"""
    import re

    candidates = [body]
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", body)
    if without_trailing_commas != body:
        candidates.append(without_trailing_commas)
    return candidates


async def aux_chat_json(
    config: AiCompanionConfig,
    *,
    prompt: str,
    model: str,
    max_tokens: int = 800,
    temperature: float = 0.0,
    timeout_seconds: float = 30.0,
    attempts: int = 2,
) -> dict[str, Any]:
    """辅助调用并解析 JSON 对象；偶发的格式瑕疵自动重试一次。"""
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        reply = await aux_chat(
            config,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        try:
            return parse_json_object(reply)
        except AuxLlmError as exc:
            last_error = exc
            if attempt < max(1, int(attempts)) - 1:
                logger.warning("aux llm json parse failed (%r), retrying", exc)
    assert last_error is not None
    raise last_error
