"""图文相关性判断：图片描述与对话文字无关时丢弃图片描述（保留文字）。"""

from __future__ import annotations

import logging

from ..config import AiCompanionConfig
from ..providers.aux_llm import AuxLlmError, aux_chat

logger = logging.getLogger(__name__)


async def check_media_relevance(
    config: AiCompanionConfig,
    media_context_text: str,
    conversation_text: str,
) -> bool:
    """判断图片描述与文字内容是否相关。返回 True=相关保留，False=无关丢弃。

    经辅助 LLM 通道（thinking 关闭）做轻量判断。纯图消息（无文字）跳过判断
    直接返回 True；任何失败均保守返回 True（宁可多保留描述，不误删）。
    """
    if not media_context_text or not conversation_text:
        return True
    try:
        prompt = (
            "判断下面【图片描述】和【对话文字】是否讨论同一话题/角色/场景。\n"
            f"【图片描述】{media_context_text[:200]}\n"
            f"【对话文字】{conversation_text[:300]}\n"
            "只回复一个字：相关 或 无关"
        )
        model = str(getattr(config, "ai_companion_member_mimic_model", "") or "") or str(
            getattr(config, "ai_companion_online_model", "") or "deepseek-flash"
        )
        content = await aux_chat(
            config,
            prompt=prompt,
            model=model,
            max_tokens=512,
            temperature=0.0,
            timeout_seconds=10.0,
        )
        relevant = "无关" not in content
        logger.info("ai companion media relevance: judge=%r relevant=%s", content[:20], relevant)
        return relevant
    except AuxLlmError as exc:
        logger.debug("media relevance check skipped: %s", exc)
        return True  # 判断失败保守保留
