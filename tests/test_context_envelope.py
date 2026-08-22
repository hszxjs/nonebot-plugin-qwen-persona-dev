from __future__ import annotations

from types import SimpleNamespace


def test_at_bot_forces_reply_even_when_text_looks_command_like() -> None:
    from nonebot_plugin_ai_companion.decision.gates import should_attempt_reply

    decision = should_attempt_reply(
        enable_reply=True,
        group_enabled=True,
        self_event=False,
        blacklisted=False,
        command_like=True,
        force_reply=True,
        has_text=True,
        reply_probability=0.0,
        follow_reply_probability=0.0,
        random_value=1.0,
    )

    assert decision.allowed
    assert decision.reason == "force_reply"


def test_plain_command_like_message_is_filtered_before_random_reply() -> None:
    from nonebot_plugin_ai_companion.decision.gates import should_attempt_reply

    decision = should_attempt_reply(
        enable_reply=True,
        group_enabled=True,
        self_event=False,
        blacklisted=False,
        command_like=True,
        force_reply=False,
        has_text=True,
        reply_probability=1.0,
        follow_reply_probability=1.0,
        random_value=0.0,
    )

    assert not decision.allowed
    assert decision.reason == "command_like"


def test_question_prompt_prioritizes_answering_over_persona_imitation() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent, build_reply_plan

    plan = build_reply_plan(
        system_prompt="你正在扮演群友。",
        conversation_lines=["bob: 前面说的是一张游戏详情页"],
        style_memory_lines=["alice: 哈哈这波有点抽象"],
        current_text="这游戏为什么下载量这么高？",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
    )

    assert plan.intent == ReplyIntent.QUESTION_ANSWER
    user_content = plan.messages[1]["content"]
    assert "当前对话上下文" in user_content
    assert "语气参考" in user_content
    assert "先直接回答问题" in user_content
    assert "最近群聊样本" not in user_content
    assert "自然接一句" not in user_content


def test_prompt_declares_intent_and_requires_meme_gate() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import build_reply_plan

    gated = build_reply_plan(
        system_prompt="system",
        conversation_lines=["相关历史：影之诗这游戏有2000张卡"],
        style_memory_lines=[],
        current_text="影之诗卡多吗？",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
        active_memes=[
            {
                "label": "2000张卡",
                "hint": "只有在聊影之诗卡牌数量时才顺手使用",
                "recent_count": 6,
            }
        ],
    )
    ungated = build_reply_plan(
        system_prompt="system",
        conversation_lines=[],
        style_memory_lines=[],
        current_text="兰州拉面怎么做？",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
        active_memes=[],
    )

    gated_content = gated.messages[1]["content"]
    ungated_content = ungated.messages[1]["content"]
    assert "回复意图：question_answer" in gated_content
    assert "可用梗" in gated_content
    assert "2000张卡" in gated_content
    assert "只有在聊影之诗卡牌数量时才顺手使用" in gated_content
    assert "可用梗" not in ungated_content
    assert "口头禅" not in ungated_content


def test_style_memory_is_marked_as_non_factual_reference() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import build_reply_plan

    plan = build_reply_plan(
        system_prompt="你正在扮演群友。",
        conversation_lines=[],
        style_memory_lines=["alice: 这波可以"],
        current_text="你觉得呢",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
    )

    user_content = plan.messages[1]["content"]
    assert "不能把语气参考当作当前事实" in user_content
    assert "不要输出用户 id" in user_content


def test_extracts_quoted_and_forwarded_text_context() -> None:
    from nonebot_plugin_ai_companion.context.envelope import build_message_context

    message = [
        {"type": "text", "data": {"text": "这是什么"}},
        {"type": "node", "data": {"content": [{"type": "text", "data": {"text": "合并里的前文"}}]}},
    ]
    event = SimpleNamespace(
        reply=SimpleNamespace(
            message=[{"type": "text", "data": {"text": "被引用的消息"}}],
        )
    )

    envelope = build_message_context(
        message=message,
        event=event,
        current_text="这是什么",
        media_parts=[],
        sticker_keys=[],
        forced_by_at=True,
    )

    assert envelope.current_text == "这是什么"
    assert envelope.quoted_texts == ["被引用的消息"]
    assert envelope.forwarded_texts == ["合并里的前文"]
    assert "quoted" in envelope.context_sources
    assert "forwarded" in envelope.context_sources


async def test_background_media_analysis_decision_filters_plain_commands() -> None:
    from nonebot_plugin_ai_companion.sticker.analysis import should_analyze_background_media

    command = should_analyze_background_media(
        enable_learning=True,
        group_enabled=True,
        self_event=False,
        blacklisted=False,
        command_like=True,
        force_reply=False,
        has_media=True,
    )
    forced = should_analyze_background_media(
        enable_learning=True,
        group_enabled=True,
        self_event=False,
        blacklisted=False,
        command_like=True,
        force_reply=True,
        has_media=True,
    )

    assert not command.allowed
    assert command.reason == "command_like"
    assert forced.allowed
    assert forced.reason == "force_reply"


async def test_background_media_analysis_decision_rejects_when_background_busy() -> None:
    from nonebot_plugin_ai_companion.sticker.analysis import should_analyze_background_media

    decision = should_analyze_background_media(
        enable_learning=True,
        group_enabled=True,
        self_event=False,
        blacklisted=False,
        command_like=False,
        force_reply=False,
        has_media=True,
        background_busy=True,
    )

    assert not decision.allowed
    assert decision.reason == "background_busy"


async def test_search_context_samples_returns_matching_recent_text(persona_store) -> None:
    await persona_store.record_sample(
        platform="qq",
        user_id="u1",
        group_id="g1",
        message_id="m1",
        plain_text="影之诗这游戏有很多卡",
        training_text="影之诗这游戏有很多卡",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    await persona_store.record_sample(
        platform="qq",
        user_id="u2",
        group_id="g1",
        message_id="m2",
        plain_text="完全无关的话",
        training_text="完全无关的话",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )

    results = await persona_store.search_context_samples(group_id="g1", query="影之诗 卡", limit=3)

    assert [item["plain_text"] for item in results] == ["影之诗这游戏有很多卡"]


async def test_search_media_assets_returns_matching_cached_vlm_result(persona_store) -> None:
    await persona_store.record_media_asset(
        source="sticker",
        source_key="s1",
        uri="https://example.test/sticker.webp",
        prompt="请描述图片",
        answer="影之诗表情包，角色震惊地看着卡牌",
        metadata={"group_id": "g1"},
    )
    await persona_store.record_media_asset(
        source="sticker",
        source_key="s2",
        uri="https://example.test/other.webp",
        prompt="请描述图片",
        answer="完全无关的表情",
        metadata={"group_id": "g1"},
    )

    results = await persona_store.search_media_assets(group_id="g1", query="影之诗 卡牌", limit=3)

    assert [item["answer"] for item in results] == ["影之诗表情包，角色震惊地看着卡牌"]


async def test_background_media_analysis_records_only_sticker_like_images(persona_store) -> None:
    from nonebot_plugin_ai_companion.sticker.analysis import analyze_media_parts

    class Captioner:
        async def caption_media(self, part: dict[str, object]) -> str:
            url = str((part.get("data") or {}).get("url") or "")
            if "sticker" in url:
                return "一个二次元表情包，角色震惊地张大嘴"
            return "一张网页截图，显示应用详情和下载按钮"

    result = await analyze_media_parts(
        store=persona_store,
        captioner=Captioner(),
        parts=[
            {"type": "image", "data": {"url": "https://example.test/sticker.webp"}},
            {"type": "image", "data": {"url": "https://example.test/screenshot.png"}},
        ],
        group_id="g1",
        user_id="u1",
    )
    stats = await persona_store.stats()

    assert result.accepted == 1
    assert result.rejected == 1
    assert stats["media_assets"] == 1


async def test_background_media_analysis_does_not_double_count_known_image_sticker(persona_store) -> None:
    from nonebot_plugin_ai_companion.sticker.analysis import analyze_media_parts
    from nonebot_plugin_ai_companion.sticker.classifier import classify_sticker_segment
    from nonebot_plugin_ai_companion.sticker.model import sticker_to_record

    class Captioner:
        async def caption_media(self, _part: dict[str, object]) -> str:
            return "二次元表情包，角色震惊"

    part = {
        "type": "image",
        "data": {
            "url": "https://example.test/custom.webp",
            "emoji_id": "10001",
            "emoji_package_id": "20002",
        },
    }
    record = sticker_to_record(part, {"image"}, classification=classify_sticker_segment(part))
    assert record is not None

    await persona_store.record_sticker_asset(
        sticker=record,
        group_id="g1",
        user_id="u1",
        summary="已入库表情",
    )
    await analyze_media_parts(
        store=persona_store,
        captioner=Captioner(),
        parts=[part],
        group_id="g1",
        user_id="u1",
    )
    sticker = await persona_store.get_sticker(record["key"])

    assert sticker is not None
    assert sticker["count"] == 1


def test_tool_registry_defaults_to_no_available_tools() -> None:
    from nonebot_plugin_ai_companion.tools.registry import ToolRegistry

    registry = ToolRegistry()

    assert not registry.has_tools
    assert registry.describe_for_prompt() == ""


def test_default_system_prompt_prioritizes_context_answering_over_roleplay() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig

    prompt = AiCompanionConfig().ai_companion_system_prompt

    assert "优先理解上下文" in prompt
    assert "回答问题" in prompt
    assert "梗" not in prompt
    assert "口头禅" not in prompt
    assert "扮演" not in prompt
    assert "模仿" not in prompt
    assert "学习群友说话习惯" not in prompt
