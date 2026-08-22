"""角色卡域测试：模型/仓库/服务/模板渲染/member_mimic 生成（mock 辅助 LLM）。"""

from __future__ import annotations

import time

import pytest

from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.memory.samples import ChatSampleRepo
from nonebot_plugin_ai_companion.role_card.model import RoleCardCreate, RoleType, session_key_for
from nonebot_plugin_ai_companion.role_card.repo import RoleCardRepo, SessionRoleBindingRepo
from nonebot_plugin_ai_companion.role_card.service import RoleCardService
from nonebot_plugin_ai_companion.role_card.style_analyzer import (
    StyleProfile,
    analyze_user_style,
    profile_to_patch_kwargs,
)


@pytest.fixture
def service(persona_store) -> RoleCardService:
    return RoleCardService(
        config=AiCompanionConfig(),
        cards=RoleCardRepo(persona_store.db),
        bindings=SessionRoleBindingRepo(persona_store.db),
        samples=persona_store.samples,
    )


async def test_role_card_crud_roundtrip(service: RoleCardService) -> None:
    card = await service.create_card(
        RoleCardCreate(
            name="测试人格",
            system_prompt="你是{name}，正在和{user_nickname}聊天",
            style_tags=["傲娇", "简短"],
            allowed_tools=["memory_query"],
        )
    )
    assert card.role_type == RoleType.BOT_PERSONA
    assert card.enabled

    fetched = await service.get_card(card.id)
    assert fetched is not None
    assert fetched.style_tags == ["傲娇", "简短"]

    updated = await service.update_card(
        card.id, RoleCardCreate(name="测试人格2", system_prompt="改了", enabled=False)
    )
    assert updated is not None
    assert updated.name == "测试人格2"
    assert not updated.enabled

    clone = await service.clone_card(card.id, name="克隆人格")
    assert clone is not None
    assert clone.id != card.id
    assert clone.name == "克隆人格"
    assert not clone.enabled  # 克隆默认停用
    assert clone.system_prompt == "改了"

    assert await service.delete_card(card.id) is True
    assert await service.get_card(card.id) is None


async def test_template_rendering_known_and_unknown_vars(service: RoleCardService) -> None:
    card = await service.create_card(
        RoleCardCreate(
            name="小滴",
            system_prompt="{persona_name}在{time}对{user_nickname}说：{memory_summary}{unknown_var}",
        )
    )
    rendered = card.render_system_prompt(
        persona_name="小滴",
        user_nickname="阿明",
        time_text="2026-08-23 03:00",
        memory_summary="喜欢打游戏",
    )
    assert "小滴" in rendered
    assert "阿明" in rendered
    assert "2026-08-23 03:00" in rendered
    assert "喜欢打游戏" in rendered
    assert "{unknown_var}" in rendered  # 未知占位符原样保留


async def test_binding_effective_card_and_cascade(service: RoleCardService) -> None:
    card = await service.create_card(RoleCardCreate(name="人格A"))
    binding = await service.bind_session(role_card_id=card.id, group_id="12345", user_id=None)
    assert binding.session_key == "group:12345"

    effective, bound = await service.get_effective_card(group_id="12345", user_id=None)
    assert effective is not None and effective.id == card.id
    assert bound is not None

    # 停用后不再生效
    await service.set_enabled(card.id, enabled=False)
    effective, bound = await service.get_effective_card(group_id="12345", user_id=None)
    assert effective is None and bound is not None

    # 删除卡 → 绑定级联清理
    await service.delete_card(card.id)
    assert await service.bindings.get_binding("group:12345") is None
    effective, _ = await service.get_effective_card(group_id="12345", user_id=None)
    assert effective is None


async def test_session_key_rules() -> None:
    assert session_key_for(group_id="g1", user_id="u1") == "group:g1"
    assert session_key_for(group_id=None, user_id="u1") == "private:u1"
    assert session_key_for(group_id=None, user_id=None) == "private:unknown"


async def test_generate_and_refresh_member_mimic(service: RoleCardService, monkeypatch) -> None:
    async def fake_analyze(config, samples_repo, *, user_id, group_id=None):
        assert isinstance(samples_repo, ChatSampleRepo)
        return StyleProfile(
            system_prompt="你现在以群友 42 的口吻说话",
            style_tags=["话痨"],
            catchphrases=["蚌埠住了"],
            tone_examples=["这波啊，这波是肉蛋葱鸡"],
            topic_preferences=["游戏"],
            emotion_baseline="日常high",
            relation_profile="群里活跃分子",
        )

    monkeypatch.setattr("nonebot_plugin_ai_companion.role_card.service.analyze_user_style", fake_analyze)

    card = await service.generate_member_mimic(user_id="42")
    assert card.role_type == RoleType.MEMBER_MIMIC
    assert card.target_user_id == "42"
    assert card.catchphrases == ["蚌埠住了"]

    # 再次生成 → 更新既有卡（不新建）
    card2 = await service.generate_member_mimic(user_id="42")
    assert card2.id == card.id
    assert len(await service.list_cards(role_type=RoleType.MEMBER_MIMIC)) == 1

    refreshed = await service.refresh_member_mimic_style(card.id)
    assert refreshed.style_tags == ["话痨"]

    with pytest.raises(ValueError, match="not a member_mimic"):
        await service.refresh_member_mimic_style((await service.create_card(RoleCardCreate(name="普通"))).id)


async def test_analyze_user_style_requires_enough_samples(
    service: RoleCardService, monkeypatch
) -> None:
    # 无样本 → AuxLlmError
    from nonebot_plugin_ai_companion.providers.aux_llm import AuxLlmError

    with pytest.raises(AuxLlmError, match="not enough samples"):
        await analyze_user_style(AiCompanionConfig(), service.samples, user_id="nobody")


async def test_identity_prompt_includes_style_lines(service: RoleCardService) -> None:
    card = await service.create_card(
        RoleCardCreate(
            name="模仿卡",
            system_prompt="base",
            style_tags=["短句"],
        )
    )
    # 手工补风格字段（bot_persona 通常不填，这里验证渲染逻辑）
    await service.cards.patch_style_fields(
        card.id, catchphrases=["绝了"], emotion_baseline="平静"
    )
    refreshed = await service.get_card(card.id)
    assert refreshed is not None
    text = service.identity_prompt(refreshed, user_nickname="阿明")
    assert "base" in text
    assert "口头禅：绝了" in text
    assert "情绪基线：平静" in text


async def test_profile_patch_kwargs_mapping() -> None:
    profile = StyleProfile(system_prompt="s", style_tags=["a"], catchphrases=["b"])
    kwargs = profile_to_patch_kwargs(profile)
    assert set(kwargs) == {
        "system_prompt",
        "style_tags",
        "catchphrases",
        "tone_examples",
        "topic_preferences",
        "emotion_baseline",
        "relation_profile",
    }
    assert time.time() > 0  # noqa: PT018 - 占位避免 unused import
