from __future__ import annotations

import logging
from types import SimpleNamespace


def test_media_question_routes_to_image_explain_without_sticker_token_prompt() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import (
        ReplyIntent,
        build_reply_plan,
    )

    plan = build_reply_plan(
        system_prompt="你正在扮演群友。",
        history_lines=["alice: 这游戏有2000张卡"],
        current_text="这是什么 图片：这是一款名为《Shadowverse》的手机游戏应用详情页面",
        media_parts=[{"type": "image", "data": {"url": "https://example.invalid/card.png"}}],
        sticker_reply_allowed=True,
        incoming_has_sticker=False,
    )

    assert plan.intent == ReplyIntent.IMAGE_EXPLAIN
    assert plan.allow_sticker
    assert plan.messages[0]["role"] == "system"
    user_content = plan.messages[1]["content"]
    assert "回答用户对图片或表情的疑问" in user_content
    assert "只选择并输出一个最贴合当前消息的可用表情 token" not in user_content
    assert "/no_think" not in user_content


def test_plain_text_routes_to_persona_reply() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import (
        ReplyIntent,
        build_reply_plan,
    )

    plan = build_reply_plan(
        system_prompt="你正在扮演群友。",
        history_lines=[],
        current_text="卧槽，这游戏有2000张卡，爽吧！",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
    )

    assert plan.intent == ReplyIntent.PERSONA_TEXT
    assert not plan.allow_sticker
    assert "先判断当前上下文意图" in plan.messages[1]["content"]
    assert "请自然接一句" not in plan.messages[1]["content"]


def test_explicit_sticker_request_routes_to_sticker_request() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import (
        ReplyIntent,
        build_reply_plan,
    )

    plan = build_reply_plan(
        system_prompt="你正在扮演群友。",
        history_lines=[],
        current_text="发个表情看看",
        media_parts=[],
        sticker_reply_allowed=False,
        incoming_has_sticker=False,
    )

    assert plan.intent == ReplyIntent.STICKER_REQUEST
    assert plan.allow_sticker
    assert "用户明确想要表情" in plan.messages[1]["content"]


async def test_prepare_message_learning_rejects_document_screenshot_as_sticker() -> None:
    from nonebot_plugin_ai_companion.memory.ingest import prepare_message_learning
    from nonebot_plugin_ai_companion.sticker.classifier import classify_sticker_segment

    class Captioner:
        async def caption_media(self, _part: dict[str, object]) -> str:
            return "一张 Axios 安全警报页面截图，包含漏洞列表和修复建议"

    part = {
        "type": "image",
        "data": {
            "url": "https://multimedia.nt.qq.com.cn/download?appid=1407",
            "file": "A" * 32 + ".jpg",
            "sub_type": "1",
            "summary": "[动画表情]",
        },
    }

    classification = classify_sticker_segment(
        part,
        caption="一张 Axios 安全警报页面截图，包含漏洞列表和修复建议",
    )
    learning = await prepare_message_learning(
        [part],
        sticker_segment_types={"image"},
        captioner=Captioner(),
        image_fallback_text="[图片]",
        sticker_fallback_text="[表情]",
    )

    assert not classification.sendable
    assert learning["stickers"] == []
    assert "图片：一张 Axios 安全警报页面截图" in learning["media_context_text"]
    assert learning["sticker_events"][0]["status"] == "rejected"


async def test_explicit_custom_image_metadata_is_sendable_sticker() -> None:
    from nonebot_plugin_ai_companion.memory.ingest import prepare_message_learning

    part = {
        "type": "image",
        "data": {
            "url": "https://example.test/custom.webp",
            "emoji_id": "10001",
            "emoji_package_id": "20002",
            "summary": "拍桌狂笑",
        },
    }

    learning = await prepare_message_learning(
        [part],
        sticker_segment_types={"image"},
        captioner=None,
        image_fallback_text="[图片]",
        sticker_fallback_text="[表情]",
    )

    assert len(learning["stickers"]) == 1
    sticker = learning["stickers"][0]
    assert sticker["data"]["_classification"]["sendable"]
    assert learning["sticker_events"][0]["status"] == "accepted"


async def test_prepare_message_learning_rejects_garbled_caption_summary() -> None:
    from nonebot_plugin_ai_companion.memory.ingest import prepare_message_learning

    class Captioner:
        async def caption_media(self, _part: dict[str, object]) -> str:
            return "mp窗 Turing - pleas nodeType <think> Gam Gam Gam"

    part = {
        "type": "image",
        "data": {
            "url": "https://example.test/custom.webp",
            "emoji_id": "10001",
            "emoji_package_id": "20002",
        },
    }

    learning = await prepare_message_learning(
        [part],
        sticker_segment_types={"image"},
        captioner=Captioner(),
        image_fallback_text="[图片]",
        sticker_fallback_text="[表情]",
    )

    assert len(learning["stickers"]) == 1
    assert not learning["stickers"][0].get("summary")
    assert "Turing" not in learning["training_text"]


async def test_selector_uses_short_candidate_id_and_maps_to_sticker_key(tmp_path) -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent
    from nonebot_plugin_ai_companion.providers.generation_profiles import GenerationPurpose
    from nonebot_plugin_ai_companion.sticker.selector import select_reply_sticker

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def chat(
            self,
            messages: list[dict[str, object]],
            *,
            media_parts: list[dict[str, object]] | None = None,
            purpose: object,
        ) -> str:
            self.calls.append({"messages": messages, "media_parts": media_parts or [], "purpose": purpose})
            return '{"id": "S1", "reason": "fits"}'

    runtime = Runtime()
    long_key = "image:" + "very-long-sticker-key-" * 10
    local_file = tmp_path / "good.gif"
    local_file.write_bytes(b"GIF89a")
    candidates = [
        {
            "key": long_key,
            "type": "image",
            "summary": "震惊",
            "uri": str(local_file),
            "data": {"url": str(local_file)},
            "count": 10,
        },
        {
            "key": "face:bad",
            "type": "face",
            "summary": "困",
            "uri": "123",
            "data": {"id": "123"},
            "count": 5,
        },
        {
            # 远程-only 候选：QQ http URL 会过期（rkey 轮换），不得作为附图发给后端。
            "key": "image:remote-only",
            "type": "image",
            "summary": "远端",
            "uri": "https://example.invalid/expired.gif",
            "data": {"url": "https://example.invalid/expired.gif"},
            "count": 8,
        },
    ]

    selection = await select_reply_sticker(
        runtime,
        candidates=candidates,
        trigger_text="这是什么",
        reply_text="这是影之诗",
        intent=ReplyIntent.IMAGE_EXPLAIN,
    )

    assert selection.key == long_key
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["purpose"] == GenerationPurpose.STICKER_SELECTOR
    media_parts = runtime.calls[0]["media_parts"]
    assert len(media_parts) == 1
    assert media_parts[0]["data"]["url"] == str(local_file)
    prompt = runtime.calls[0]["messages"][1]["content"]
    assert "S1" in prompt
    assert long_key not in prompt
    assert "只返回 JSON" in prompt and "候选 id" in prompt


async def test_selector_rejects_non_exact_key_with_warning(caplog) -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent
    from nonebot_plugin_ai_companion.sticker.selector import select_reply_sticker

    class Runtime:
        async def chat(
            self,
            _messages: list[dict[str, object]],
            *,
            media_parts: list[dict[str, object]] | None = None,
            purpose: object | None = None,
        ) -> str:
            return '{"key": "good"}'

    caplog.set_level(logging.WARNING, logger="nonebot_plugin_ai_companion.sticker.selector")
    selection = await select_reply_sticker(
        Runtime(),
        candidates=[{"key": "face:good", "type": "face", "summary": "震惊", "data": {"id": "1"}, "count": 5}],
        trigger_text="这是什么",
        reply_text="这是影之诗",
        intent=ReplyIntent.IMAGE_EXPLAIN,
    )

    assert selection.key is None
    assert any("invalid sticker selection" in record.getMessage() for record in caplog.records)


async def test_selector_excludes_non_sendable_plain_image_candidates() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent
    from nonebot_plugin_ai_companion.sticker.selector import select_reply_sticker

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def chat(
            self,
            messages: list[dict[str, object]],
            *,
            media_parts: list[dict[str, object]] | None = None,
            purpose: object | None = None,
        ) -> str:
            self.calls.append({"messages": messages, "media_parts": media_parts or []})
            return '{"key": "face:smile", "reason": "requested sticker"}'

    runtime = Runtime()
    selection = await select_reply_sticker(
        runtime,
        candidates=[
            {
                "key": "image:screenshot",
                "type": "image",
                "summary": "Axios 安全警报页面截图",
                "uri": "https://example.test/screenshot.png",
                "data": {
                    "url": "https://example.test/screenshot.png",
                    "_classification": {
                        "label": "screenshot",
                        "reason": "document screenshot",
                        "sendable": False,
                    },
                },
                "count": 30,
            },
            {
                "key": "face:smile",
                "type": "face",
                "summary": "龇牙",
                "uri": "123",
                "data": {"id": "123"},
                "count": 5,
            },
        ],
        trigger_text="发个表情看看",
        reply_text="",
        intent=ReplyIntent.STICKER_REQUEST,
    )

    assert selection.key == "face:smile"
    prompt = runtime.calls[0]["messages"][1]["content"]
    assert "S1" in prompt
    assert "image:screenshot" not in prompt
    assert runtime.calls[0]["media_parts"] == []


async def test_explicit_sticker_request_falls_back_to_sendable_candidate_on_invalid_output() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent
    from nonebot_plugin_ai_companion.sticker.selector import select_reply_sticker

    class Runtime:
        async def chat(
            self,
            _messages: list[dict[str, object]],
            *,
            media_parts: list[dict[str, object]] | None = None,
            purpose: object | None = None,
        ) -> str:
            return "🙂"

    selection = await select_reply_sticker(
        Runtime(),
        candidates=[
            {
                "key": "face:smile",
                "type": "face",
                "summary": "龇牙",
                "uri": "123",
                "data": {"id": "123"},
                "count": 5,
            }
        ],
        trigger_text="发个表情看看",
        reply_text="",
        intent=ReplyIntent.STICKER_REQUEST,
    )

    assert selection.key == "face:smile"
    assert selection.fallback
    assert selection.raw_text == "🙂"


async def test_selector_filters_low_frequency_candidates_and_falls_back_on_empty_output() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent
    from nonebot_plugin_ai_companion.sticker.selector import select_reply_sticker

    class Runtime:
        async def chat(
            self,
            _messages: list[dict[str, object]],
            *,
            media_parts: list[dict[str, object]] | None = None,
            purpose: object | None = None,
        ) -> str:
            return ""

    selection = await select_reply_sticker(
        Runtime(),
        candidates=[
            {
                "key": "face:low",
                "type": "face",
                "summary": "低频",
                "uri": "11",
                "data": {"id": "11"},
                "count": 4,
            },
            {
                "key": "face:high",
                "type": "face",
                "summary": "高频",
                "uri": "22",
                "data": {"id": "22"},
                "count": 5,
            },
        ],
        trigger_text="随便说点啥",
        reply_text="短回复",
        intent=ReplyIntent.PERSONA_TEXT,
        min_count=5,
    )

    assert selection.key == "face:high"
    assert selection.fallback
    assert selection.usable_count == 1


def test_onebot_sticker_message_preserves_text_and_sticker_in_single_message() -> None:
    from nonebot_plugin_ai_companion.sticker.model import build_onebot_v11_sticker_message

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
        def face(value: int) -> object:
            return SimpleNamespace(type="face", data={"id": value})

        @staticmethod
        def image(value: str) -> object:
            return SimpleNamespace(type="image", data={"file": value})

    reply = build_onebot_v11_sticker_message(
        Message,
        MessageSegment,
        "这是影之诗",
        [{"key": "face:smile", "type": "face", "data": {"id": "123"}}],
    )

    assert isinstance(reply, Message)
    assert [part.type for part in reply.parts] == ["text", "face"]
    assert reply.parts[0].data["text"] == "这是影之诗"


def test_onebot_custom_image_sticker_message_is_sendable() -> None:
    from nonebot_plugin_ai_companion.sticker.model import build_onebot_v11_sticker_message

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
        def image(value: str) -> object:
            return SimpleNamespace(type="image", data={"file": value})

    reply = build_onebot_v11_sticker_message(
        Message,
        MessageSegment,
        "",
        [
            {
                "key": "image:custom",
                "type": "image",
                "uri": "file:///tmp/custom.webp",
                "data": {
                    "url": "file:///tmp/custom.webp",
                    "_classification": {
                        "label": "custom_sticker",
                        "reason": "explicit custom metadata",
                        "sendable": True,
                    },
                },
            }
        ],
    )

    assert isinstance(reply, Message)
    assert [part.type for part in reply.parts] == ["image"]


def test_sticker_segment_prefers_existing_local_file(tmp_path) -> None:
    """远程 URL 会过期（NapCat 下载 400）：存在本地副本时优先用本地文件。"""
    from types import SimpleNamespace

    from nonebot_plugin_ai_companion.sticker.model import build_onebot_v11_sticker_segment

    local = tmp_path / "sticker.webp"
    local.write_bytes(b"local-bytes")

    used: list = []
    segment_ns = SimpleNamespace(image=lambda value: used.append(value) or value)

    record = {
        "key": "image:x",
        "type": "image",
        "uri": "https://multimedia.nt.qq.com.cn/download?fileid=expired",
        "data": {
            "url": "https://multimedia.nt.qq.com.cn/download?fileid=expired",
            "_classification": {"label": "sticker_like_image", "sendable": True},
        },
    }
    build_onebot_v11_sticker_segment(segment_ns, record)
    assert used == ["https://multimedia.nt.qq.com.cn/download?fileid=expired"]  # 无本地副本 → 用远程

    record["uri"] = local.as_uri()
    used.clear()
    build_onebot_v11_sticker_segment(segment_ns, record)
    assert used and used[0] == local.as_uri()  # 有本地副本 → file://（不会过期）
