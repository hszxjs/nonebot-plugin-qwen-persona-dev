"""图文相关性判断：图片描述与对话文字无关时丢弃图片描述（保留文字）。"""

from __future__ import annotations

import logging

import httpx

from ..config import AiCompanionConfig

logger = logging.getLogger(__name__)

# 判断模型：online 模型通常是 glm-5.2（贵），判断用 flash 更省
JUDGE_MODEL = "glm-4.5-flash"


async def check_media_relevance(
    config: AiCompanionConfig,
    media_context_text: str,
    conversation_text: str,
) -> bool:
    """判断图片描述与文字内容是否相关。返回 True=相关保留，False=无关丢弃。

    用 glm-4.5-flash 做轻量判断。纯图消息（无文字）跳过判断直接返回 True；
    任何失败均保守返回 True（宁可多保留描述，不误删）。
    """
    if not media_context_text or not conversation_text:
        return True
    try:
        # 复用在线推理配置；优先 online 端点，降级用 lora_helper 端点。
        url = (config.ai_companion_online_api_url or "").strip()
        if not url:
            url = (config.ai_companion_lora_helper_api_url or "").strip()
        key = (config.ai_companion_online_api_key or "").strip()
        if not key:
            key = (config.ai_companion_lora_helper_api_key or "").strip()
        if not url or not key:
            return True
        prompt = (
            "判断下面【图片描述】和【对话文字】是否讨论同一话题/角色/场景。\n"
            f"【图片描述】{media_context_text[:200]}\n"
            f"【对话文字】{conversation_text[:300]}\n"
            "只回复一个字：相关 或 无关"
        )
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0,
                },
            )
        if r.status_code != 200:
            return True
        content = str((r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        relevant = "无关" not in content
        logger.info("ai companion media relevance: judge=%r relevant=%s", content[:20], relevant)
        return relevant
    except Exception as exc:  # noqa: BLE001 — 判断失败保守保留，属预期内可吞异常
        logger.debug("media relevance check skipped: %r", exc)
        return True
