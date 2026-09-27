"""会话时间线域：写入/窗口/昵称/回填/引用/上下文预算。"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


def _fake_message_classes() -> tuple[object, object]:
    class Message:
        def __init__(self) -> None:
            self.parts: list[object] = []

        def __iadd__(self, value: object) -> Message:
            self.parts.append(value)
            return self

    class MessageSegment:
        @staticmethod
        def text(value: str) -> object:
            return SimpleNamespace(type="text", data={"text": value})

        @staticmethod
        def reply(value: str) -> object:
            return SimpleNamespace(type="reply", data={"id": value})

        @staticmethod
        def image(value: str) -> object:
            return SimpleNamespace(type="image", data={"file": value})

    return Message, MessageSegment


# ---- 时间线读写 ----


async def test_conversation_records_and_reads_in_order(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_BOT, SPEAKER_MEMBER

    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind=SPEAKER_MEMBER,
        text="今天打本吗",
        group_id="100",
        user_id="1",
        speaker_name="阿强",
        created_at=100.0,
    )
    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind=SPEAKER_BOT,
        text="来，等我五分钟",
        group_id="100",
        sticker_keys=["face:smile"],
        trigger_reason="force_reply",
        created_at=101.0,
    )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)

    assert [turn.speaker_kind for turn in turns] == [SPEAKER_MEMBER, SPEAKER_BOT]
    assert turns[0].speaker_name == "阿强"
    assert turns[1].sticker_keys == ["face:smile"]
    assert turns[1].trigger_reason == "force_reply"


async def test_conversation_window_is_session_scoped(persona_store) -> None:
    """私聊与各自的群互不串台——这是"私聊拿别的群当语气参考"的修复点。"""
    for session_key, group_id, user_id, text in [
        ("group:100", "100", "1", "群100的话"),
        ("group:200", "200", "2", "群200的话"),
        ("private:1", None, "1", "私聊的话"),
    ]:
        await persona_store.conversation.record_turn(
            session_key=session_key,
            speaker_kind="member",
            text=text,
            group_id=group_id,
            user_id=user_id,
            created_at=100.0,
        )

    group_100 = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    private_1 = await persona_store.conversation.recent_turns(session_key="private:1", limit=10)

    assert [turn.text for turn in group_100] == ["群100的话"]
    assert [turn.text for turn in private_1] == ["私聊的话"]


async def test_conversation_since_filter_drops_pre_binding_turns(persona_store) -> None:
    for created_at, text in [(100.0, "绑定前"), (200.0, "绑定后")]:
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind="member",
            text=text,
            group_id="100",
            user_id="1",
            created_at=created_at,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10, since=150.0)
    assert [turn.text for turn in turns] == ["绑定后"]


# ---- 昵称 ----


async def test_member_names_upsert_and_batch_lookup(persona_store) -> None:
    await persona_store.member_names.upsert_name(group_id="100", user_id="1", name="阿强")
    await persona_store.member_names.upsert_name(group_id="100", user_id="2", name="小红")
    await persona_store.member_names.upsert_name(group_id="100", user_id="1", name="阿强（改名）")

    names = await persona_store.member_names.get_names(group_id="100", user_ids=["1", "2", "3"])
    assert names == {"1": "阿强（改名）", "2": "小红"}
    # 私聊昵称独立存储（group_id 为空串）
    await persona_store.member_names.upsert_name(group_id=None, user_id="1", name="私聊里的大名")
    assert await persona_store.member_names.get_names(group_id="100", user_ids=["1"]) == {"1": "阿强（改名）"}


# ---- 渲染 ----


async def test_history_rendering_has_speakers_and_never_leaks_qq(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_BOT
    from nonebot_plugin_ai_companion.conversation.window import collect_user_ids, render_history_lines

    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind="member",
        text="这波怎么说",
        group_id="100",
        user_id="1234567890",
        speaker_name="阿强",
        created_at=100.0,
    )
    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind="member",
        text="我先开一把",
        group_id="100",
        user_id="987654321",
        speaker_name="",
        created_at=101.0,
    )
    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind=SPEAKER_BOT,
        text="等我两分钟",
        group_id="100",
        created_at=102.0,
    )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    lines = render_history_lines(turns)

    assert lines[0] == "阿强：这波怎么说"
    assert lines[1] == "群友4321：我先开一把"  # 无昵称时用「群友+尾号」，绝不落 QQ 全号
    assert lines[2] == "你：等我两分钟"
    assert all("1234567890" not in line and "987654321" not in line for line in lines)
    assert collect_user_ids(turns) == ["1234567890", "987654321"]


async def test_history_rendering_excludes_current_message_and_caps_lines(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.window import render_history_lines

    for index in range(5):
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind="member",
            text=f"第{index}条",
            group_id="100",
            user_id="1",
            speaker_name="阿强",
            source_message_id=str(1000 + index),
            created_at=100.0 + index,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    lines = render_history_lines(turns, exclude_message_ids=["1004"], max_lines=2)
    assert lines == ["阿强：第2条", "阿强：第3条"]


# ---- 回填 ----


async def test_backfill_from_samples_is_idempotent(persona_store) -> None:
    await persona_store.samples.record_sample(
        platform="qq",
        user_id="1",
        group_id="100",
        message_id="9001",
        plain_text="历史消息一",
        training_text="历史消息一",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    await persona_store.samples.record_sample(
        platform="qq",
        user_id="2",
        group_id="100",
        message_id="9002",
        plain_text="历史消息二",
        training_text="历史消息二",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )

    first = await persona_store.conversation.backfill_from_samples()
    second = await persona_store.conversation.backfill_from_samples()

    assert first["inserted"] == 2 and first["skipped"] == 0
    assert second["inserted"] == 0 and second["skipped"] == 2
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    assert [turn.text for turn in turns] == ["历史消息一", "历史消息二"]
    assert all(turn.source_message_id for turn in turns)


# ---- 引用回复 ----


def test_reply_segment_is_prepended_when_quoting() -> None:
    from nonebot_plugin_ai_companion.sticker.model import build_onebot_v11_sticker_message

    Message, MessageSegment = _fake_message_classes()
    reply = build_onebot_v11_sticker_message(
        Message,
        MessageSegment,
        "接你这句话",
        [],
        reply_to_message_id="12345",
    )

    assert isinstance(reply, Message)
    assert [(part.type, part.data) for part in reply.parts] == [
        ("reply", {"id": "12345"}),
        ("text", {"text": "接你这句话"}),
    ]


def test_without_quote_text_only_reply_stays_plain_string() -> None:
    from nonebot_plugin_ai_companion.sticker.model import build_onebot_v11_sticker_message

    Message, MessageSegment = _fake_message_classes()
    assert build_onebot_v11_sticker_message(Message, MessageSegment, "只有文本", []) == "只有文本"


# ---- 上下文预算 ----


def test_context_budget_is_backend_aware() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.prompt.budget import (
        effective_context_message_limit,
        trim_history_lines_for_context,
    )

    config = AiCompanionConfig(ai_companion_context_messages=40, ai_companion_mlx_lora_max_seq_length=2048)
    assert effective_context_message_limit(config, online=True) == 40
    assert effective_context_message_limit(config) == 8  # 本地按 2048//256 压

    long_line = "昵称：" + "字数很多" * 40
    online_kept = trim_history_lines_for_context([long_line] * 60, config, online=True)
    local_kept = trim_history_lines_for_context([long_line] * 60, config)
    assert len(online_kept) > len(local_kept)


# ---- 提示词 ----


def test_planner_renders_multi_speaker_history_section() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import build_reply_plan

    plan = build_reply_plan(
        system_prompt="你是群友",
        history_lines=["阿强：今天打本吗", "你：等我五分钟", "小红：我也来"],
        current_text="几点开",
        media_parts=[],
        sticker_reply_allowed=True,
        incoming_has_sticker=False,
    )
    user_content = plan.messages[1]["content"]

    assert "最近的会话记录" in user_content
    assert "「你：」是你自己之前说过的话" in user_content
    assert "阿强：今天打本吗" in user_content
    assert "你：等我五分钟" in user_content


def test_reply_quote_config_defaults_on() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig

    assert AiCompanionConfig().ai_companion_reply_quote is True


@pytest.mark.parametrize("name", ["", "   ", "阿强: 备注", "超长" * 30])
def test_safe_speaker_name_never_returns_raw_qq(name: str) -> None:
    from nonebot_plugin_ai_companion.conversation.window import safe_speaker_name

    result = safe_speaker_name(name, user_id="123456789")
    assert "123456789" not in result
    assert "：" not in result and ":" not in result
    assert len(result) <= 16


# ---- 昵称回填与 bot 时间线写入 ----


def test_names_from_group_member_list_prefers_card() -> None:
    from nonebot_plugin_ai_companion.api.routes.conversation import names_from_group_member_list

    names = names_from_group_member_list(
        [
            {"user_id": 111, "card": "阿强", "nickname": "qiang"},
            {"user_id": "222", "card": "", "nickname": "小红"},
            {"user_id": "", "nickname": "无 id 应丢弃"},
            {"user_id": "333", "card": "", "nickname": ""},
            "not-a-dict",
        ]
    )
    assert names == {"111": "阿强", "222": "小红"}


async def test_bot_turn_writer_puts_own_reply_on_timeline(persona_store) -> None:
    """bot 自己的回复必须进时间线：下一轮它才看得见自己说过什么。"""
    from types import SimpleNamespace

    from nonebot_plugin_ai_companion.chat_service import ChatService

    fake_self = SimpleNamespace(store=persona_store)
    await ChatService._record_bot_turn(
        fake_self,
        ids={"group_id": "100", "user_id": "1", "message_id": "5001"},
        text="来，等我五分钟",
        sticker_keys=["face:smile"],
        trigger_reason="random_reply",
        reply_to_message_id="5001",
    )

    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=5)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.is_bot and turn.text == "来，等我五分钟"
    assert turn.sticker_keys == ["face:smile"]
    assert turn.trigger_reason == "random_reply"
    assert turn.reply_to_message_id == "5001"
