"""辅助 LLM 调用：复用在线推理配置做轻量判断/分析（图文相关性、风格分析等）。

与主推理通道（providers/client.py）分离：辅助调用固定走 HTTP、可用更便宜的
flash 模型，失败时由调用方决定保守策略。
"""

from __future__ import annotations

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
) -> str:
    """一次辅助对话调用，返回文本内容；失败抛 AuxLlmError。"""
    endpoint = resolve_aux_endpoint(config)
    if endpoint is None:
        raise AuxLlmError("aux llm endpoint not configured")
    url, key = endpoint
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
    except httpx.HTTPError as exc:
        raise AuxLlmError(f"aux llm request failed: {exc!r}") from exc
    if response.status_code != 200:
        raise AuxLlmError(f"aux llm status {response.status_code}: {response.text[:200]}")
    try:
        content = str((response.json().get("choices") or [{}])[0].get("message", {}).get("content", ""))
    except ValueError as exc:
        raise AuxLlmError(f"aux llm invalid json: {exc!r}") from exc
    if not content.strip():
        raise AuxLlmError("aux llm empty content")
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
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AuxLlmError(f"aux llm json parse failed: {exc!r}") from exc
    if not isinstance(data, dict):
        raise AuxLlmError("aux llm json is not an object")
    return data
