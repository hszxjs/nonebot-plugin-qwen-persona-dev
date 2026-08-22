"""角色卡域：Bot 人格与群友模仿的统一数据模型。

`bot_persona`（管理员手工维护的人格）与 `member_mimic`（按 QQ 号自动生成的
群友模仿画像）共享同一字段结构，区别只在 `role_type` 与风格字段的来源。
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

TEMPLATE_VARIABLES: tuple[str, ...] = ("persona_name", "user_nickname", "time", "memory_summary")


class RoleType(StrEnum):
    BOT_PERSONA = "bot_persona"
    MEMBER_MIMIC = "member_mimic"


class RoleCard(BaseModel):
    id: str
    role_type: RoleType
    name: str
    system_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    topic_preferences: list[str] = Field(default_factory=list)
    emotion_baseline: str = ""
    relation_profile: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    avatar: str = ""
    enabled: bool = True
    target_user_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def render_system_prompt(
        self,
        *,
        persona_name: str | None = None,
        user_nickname: str | None = None,
        time_text: str | None = None,
        memory_summary: str | None = None,
    ) -> str:
        """渲染系统提示词模板；仅替换已知变量，未知占位符原样保留。"""
        rendered = self.system_prompt or ""
        values = {
            "persona_name": persona_name if persona_name is not None else self.name,
            "user_nickname": user_nickname or "群友",
            "time": time_text or "",
            "memory_summary": memory_summary or "",
        }
        for variable, value in values.items():
            rendered = rendered.replace("{" + variable + "}", str(value))
        return rendered

    def style_summary_lines(self) -> list[str]:
        """member_mimic 风格字段 → 提示词身份段补充行。"""
        lines: list[str] = []
        if self.style_tags:
            lines.append("性格/语言风格标签：" + "、".join(self.style_tags[:10]))
        if self.catchphrases:
            lines.append("口头禅：" + "、".join(self.catchphrases[:10]))
        if self.tone_examples:
            lines.append("语气示例：" + "；".join(self.tone_examples[:5]))
        if self.topic_preferences:
            lines.append("话题偏好：" + "、".join(self.topic_preferences[:10]))
        if self.emotion_baseline:
            lines.append(f"情绪基线：{self.emotion_baseline}")
        if self.relation_profile:
            lines.append(f"关系画像：{self.relation_profile}")
        return lines


class RoleCardCreate(BaseModel):
    """创建/更新载荷（API 与服务层共用）。"""

    name: str
    role_type: RoleType = RoleType.BOT_PERSONA
    system_prompt: str = ""
    style_tags: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    avatar: str = ""
    enabled: bool = True
    target_user_id: str | None = None


class SessionRoleBinding(BaseModel):
    session_key: str
    role_card_id: str
    bound_at: float


def session_key_for(*, group_id: str | None, user_id: str | None) -> str:
    """会话键：群聊以 group 为主，私聊以 private:user 为主（与上下文域一致）。"""
    if group_id:
        return f"group:{group_id}"
    if user_id:
        return f"private:{user_id}"
    return "private:unknown"


def binding_row_to_model(row: Any) -> SessionRoleBinding:
    return SessionRoleBinding(
        session_key=row["session_key"],
        role_card_id=row["role_card_id"],
        bound_at=float(row["bound_at"]),
    )
