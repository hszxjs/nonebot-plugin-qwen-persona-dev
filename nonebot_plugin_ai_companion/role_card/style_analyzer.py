"""member_mimic 风格分析：用辅助 LLM 从该群友近期对话增量提炼语言风格画像。

不做每条消息全量重算：仅在"生成/刷新"动作触发时跑一次（文档五节）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AiCompanionConfig
from ..memory.samples import ChatSampleRepo
from ..prompt.media_text import context_text_from_sample
from ..providers.aux_llm import AuxLlmError, aux_chat, parse_json_object

logger = logging.getLogger(__name__)

_STYLE_PROMPT = """下面是群友 {user_id} 的近期群聊消息（每行一条，时间正序）。

{messages}

请基于以上真实语料，输出这位群友的语言风格画像，严格输出 JSON（不要多余文字）：
{{
  "summary": "一句话概括这个人说话给人的整体印象",
  "style_tags": ["3-6 个性格/语言风格短标签"],
  "catchphrases": ["高频口头禅或标志性表达，最多 8 个，没有则空数组"],
  "tone_examples": ["2-4 条最能代表其语气的原句（可轻度压缩）"],
  "topic_preferences": ["常聊话题，最多 6 个"],
  "emotion_baseline": "情绪基调一句话（如：日常淡定偶尔兴奋）",
  "relation_profile": "与群友的关系互动方式一句话"
}}"""


@dataclass
class StyleProfile:
    system_prompt: str = ""
    style_tags: list[str] = field(default_factory=list)
    catchphrases: list[str] = field(default_factory=list)
    tone_examples: list[str] = field(default_factory=list)
    topic_preferences: list[str] = field(default_factory=list)
    emotion_baseline: str = ""
    relation_profile: str = ""


async def analyze_user_style(
    config: AiCompanionConfig,
    samples_repo: ChatSampleRepo,
    *,
    user_id: str,
    group_id: str | None = None,
) -> StyleProfile:
    """分析指定群友近期消息，产出风格画像；样本不足或调用失败抛 AuxLlmError。"""
    limit = int(getattr(config, "ai_companion_member_mimic_sample_limit", 120) or 120)
    samples = await samples_repo.recent_samples_by_user(user_id=user_id, limit=limit, group_id=group_id)
    lines: list[str] = []
    for sample in reversed(samples):  # 时间正序展示
        text = context_text_from_sample(sample)
        if not text:
            continue
        lines.append(text[:160])
    if len(lines) < 10:
        raise AuxLlmError(f"not enough samples for user {user_id} (got {len(lines)} lines)")

    model = str(getattr(config, "ai_companion_member_mimic_model", "glm-4.5-flash") or "glm-4.5-flash")
    reply = await aux_chat(
        config,
        prompt=_STYLE_PROMPT.format(user_id=user_id, messages="\n".join(lines[-200:])),
        model=model,
        max_tokens=900,
        temperature=0.2,
        timeout_seconds=30.0,
    )
    data = parse_json_object(reply)

    def _str_list(key: str, cap: int) -> list[str]:
        raw = data.get(key)
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()][:cap]

    summary = str(data.get("summary") or "").strip()
    return StyleProfile(
        system_prompt=(
            f"你现在以群友 {user_id} 的口吻在群里说话。整体印象：{summary}"
            if summary
            else f"你现在以群友 {user_id} 的口吻在群里说话。"
        ),
        style_tags=_str_list("style_tags", 8),
        catchphrases=_str_list("catchphrases", 8),
        tone_examples=_str_list("tone_examples", 4),
        topic_preferences=_str_list("topic_preferences", 6),
        emotion_baseline=str(data.get("emotion_baseline") or "").strip()[:120],
        relation_profile=str(data.get("relation_profile") or "").strip()[:160],
    )


def profile_to_patch_kwargs(profile: StyleProfile) -> dict[str, Any]:
    return {
        "system_prompt": profile.system_prompt,
        "style_tags": profile.style_tags,
        "catchphrases": profile.catchphrases,
        "tone_examples": profile.tone_examples,
        "topic_preferences": profile.topic_preferences,
        "emotion_baseline": profile.emotion_baseline,
        "relation_profile": profile.relation_profile,
    }


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M")
