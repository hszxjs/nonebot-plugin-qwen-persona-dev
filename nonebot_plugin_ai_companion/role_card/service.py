"""角色卡域服务：CRUD / 克隆 / 启停 / 会话绑定 / member_mimic 生成与刷新。

会话绑定语义（文档五节）：切换角色卡后，该会话的短期上下文从绑定时刻重新计
（`binding.bound_at` 之后的样本才进 style memory），长期记忆检索不受影响。
"""

from __future__ import annotations

import logging

from ..config import AiCompanionConfig
from ..memory.samples import ChatSampleRepo
from ..providers.aux_llm import AuxLlmError
from .model import RoleCard, RoleCardCreate, RoleType, SessionRoleBinding, session_key_for
from .repo import RoleCardRepo, SessionRoleBindingRepo
from .style_analyzer import analyze_user_style, profile_to_patch_kwargs

logger = logging.getLogger(__name__)


class RoleCardService:
    def __init__(
        self,
        *,
        config: AiCompanionConfig,
        cards: RoleCardRepo,
        bindings: SessionRoleBindingRepo,
        samples: ChatSampleRepo,
    ) -> None:
        self.config = config
        self.cards = cards
        self.bindings = bindings
        self.samples = samples

    # ---- 查询 ----
    async def list_cards(self, *, role_type: RoleType | None = None) -> list[RoleCard]:
        return await self.cards.list_cards(role_type=role_type)

    async def get_card(self, card_id: str) -> RoleCard | None:
        return await self.cards.get_card(card_id)

    async def get_effective_card(
        self, *, group_id: str | None, user_id: str | None
    ) -> tuple[RoleCard | None, SessionRoleBinding | None]:
        """会话当前生效的角色卡：有绑定且卡存在且启用才生效。"""
        session_key = session_key_for(group_id=group_id, user_id=user_id)
        binding = await self.bindings.get_binding(session_key)
        if binding is None:
            return None, None
        card = await self.cards.get_card(binding.role_card_id)
        if card is None or not card.enabled:
            return None, binding
        return card, binding

    async def list_bindings(self) -> list[SessionRoleBinding]:
        return await self.bindings.list_bindings()

    # ---- bot_persona 维护 ----
    async def create_card(self, payload: RoleCardCreate) -> RoleCard:
        return await self.cards.create_card(payload)

    async def update_card(self, card_id: str, payload: RoleCardCreate) -> RoleCard | None:
        return await self.cards.update_card(card_id, payload)

    async def set_enabled(self, card_id: str, *, enabled: bool) -> RoleCard | None:
        return await self.cards.set_enabled(card_id, enabled=enabled)

    async def delete_card(self, card_id: str) -> bool:
        return await self.cards.delete_card(card_id)

    async def clone_card(self, card_id: str, *, name: str) -> RoleCard | None:
        return await self.cards.clone_card(card_id, name=name)

    # ---- 会话绑定 ----
    async def bind_session(
        self, *, role_card_id: str, group_id: str | None, user_id: str | None
    ) -> SessionRoleBinding | None:
        card = await self.cards.get_card(role_card_id)
        if card is None:
            return None
        session_key = session_key_for(group_id=group_id, user_id=user_id)
        return await self.bindings.bind(session_key, role_card_id=role_card_id)

    async def unbind_session(self, *, group_id: str | None, user_id: str | None) -> bool:
        session_key = session_key_for(group_id=group_id, user_id=user_id)
        return await self.bindings.unbind(session_key)

    # ---- member_mimic ----
    async def generate_member_mimic(
        self,
        *,
        user_id: str,
        group_id: str | None = None,
    ) -> RoleCard:
        """为指定 QQ 号生成（或更新）member_mimic 角色卡。样本不足/LLM 失败抛 AuxLlmError。"""
        profile = await analyze_user_style(self.config, self.samples, user_id=user_id, group_id=group_id)
        existing = await self.cards.get_member_mimic_by_user(user_id)
        if existing is not None:
            updated = await self.cards.patch_style_fields(existing.id, **profile_to_patch_kwargs(profile))
            assert updated is not None
            return updated
        created = await self.cards.create_card(
            RoleCardCreate(
                name=f"群友 {user_id}",
                role_type=RoleType.MEMBER_MIMIC,
                system_prompt=profile.system_prompt,
                style_tags=profile.style_tags,
                tone_examples=profile.tone_examples,
                allowed_tools=[],
                enabled=True,
                target_user_id=str(user_id),
            )
        )
        # RoleCardCreate 不含 member_mimic 专属风格字段，创建后补齐完整画像
        created = await self.cards.patch_style_fields(created.id, **profile_to_patch_kwargs(profile))
        assert created is not None
        return created

    async def refresh_member_mimic_style(self, card_id: str) -> RoleCard:
        """刷新 member_mimic 卡的语言风格（不动身份与工具白名单）。"""
        card = await self.cards.get_card(card_id)
        if card is None:
            raise KeyError(f"role card not found: {card_id}")
        if card.role_type != RoleType.MEMBER_MIMIC or not card.target_user_id:
            raise ValueError(f"role card {card_id} is not a member_mimic card")
        try:
            profile = await analyze_user_style(self.config, self.samples, user_id=card.target_user_id)
        except AuxLlmError:
            # 刷新失败保留旧画像并向上抛，由 API 层转换为可读错误。
            raise
        updated = await self.cards.patch_style_fields(card_id, **profile_to_patch_kwargs(profile))
        assert updated is not None
        return updated

    # ---- 提示词侧 ----
    def identity_prompt(
        self,
        card: RoleCard,
        *,
        user_nickname: str | None,
        memory_summary: str | None = None,
    ) -> str:
        """渲染角色卡身份段：模板插值 + 风格补充行。"""
        from .style_analyzer import now_text

        rendered = card.render_system_prompt(
            persona_name=card.name,
            user_nickname=user_nickname,
            time_text=now_text(),
            memory_summary=memory_summary,
        )
        lines = [rendered] if rendered else []
        lines.extend(card.style_summary_lines())
        return "\n".join(line for line in lines if line)
