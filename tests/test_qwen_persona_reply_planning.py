from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class ReplyPlanningTests(unittest.TestCase):
    def test_media_question_routes_to_image_explain_without_sticker_token_prompt(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import (
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

        self.assertEqual(plan.intent, ReplyIntent.IMAGE_EXPLAIN)
        self.assertTrue(plan.allow_sticker)
        self.assertEqual(plan.messages[0]["role"], "system")
        user_content = plan.messages[1]["content"]
        self.assertIn("回答用户对图片或表情的疑问", user_content)
        self.assertNotIn("只选择并输出一个最贴合当前消息的可用表情 token", user_content)
        self.assertNotIn("/no_think", user_content)

    def test_plain_text_routes_to_persona_reply(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import (
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

        self.assertEqual(plan.intent, ReplyIntent.PERSONA_TEXT)
        self.assertFalse(plan.allow_sticker)
        self.assertIn("自然接一句", plan.messages[1]["content"])

    def test_explicit_sticker_request_routes_to_sticker_request(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import (
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

        self.assertEqual(plan.intent, ReplyIntent.STICKER_REQUEST)
        self.assertTrue(plan.allow_sticker)
        self.assertIn("用户明确想要表情", plan.messages[1]["content"])


class StickerClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_message_learning_rejects_document_screenshot_as_sticker(self) -> None:
        from nonebot_plugin_qwen_persona.media import prepare_message_learning
        from nonebot_plugin_qwen_persona.sticker_classifier import classify_sticker_segment

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

        self.assertFalse(classification.sendable)
        self.assertEqual(learning["stickers"], [])
        self.assertIn("图片：一张 Axios 安全警报页面截图", learning["media_context_text"])
        self.assertEqual(learning["sticker_events"][0]["status"], "rejected")

    async def test_explicit_custom_image_metadata_is_sendable_sticker(self) -> None:
        from nonebot_plugin_qwen_persona.media import prepare_message_learning

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

        self.assertEqual(len(learning["stickers"]), 1)
        sticker = learning["stickers"][0]
        self.assertTrue(sticker["data"]["_classification"]["sendable"])
        self.assertEqual(learning["sticker_events"][0]["status"], "accepted")

    async def test_prepare_message_learning_rejects_garbled_caption_summary(self) -> None:
        from nonebot_plugin_qwen_persona.media import prepare_message_learning

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

        self.assertEqual(len(learning["stickers"]), 1)
        self.assertFalse(learning["stickers"][0].get("summary"))
        self.assertNotIn("Turing", learning["training_text"])


class StickerSelectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_selector_uses_short_candidate_id_and_maps_to_sticker_key(self) -> None:
        from nonebot_plugin_qwen_persona.generation_profiles import GenerationPurpose
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

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
        candidates = [
            {
                "key": long_key,
                "type": "image",
                "summary": "震惊",
                "uri": "https://example.invalid/good.gif",
                "data": {"url": "https://example.invalid/good.gif"},
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
        ]

        selection = await select_reply_sticker(
            runtime,
            candidates=candidates,
            trigger_text="这是什么",
            reply_text="这是影之诗",
            intent=ReplyIntent.IMAGE_EXPLAIN,
        )

        self.assertEqual(selection.key, long_key)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.calls[0]["purpose"], GenerationPurpose.STICKER_SELECTOR)
        self.assertEqual(len(runtime.calls[0]["media_parts"]), 1)
        prompt = runtime.calls[0]["messages"][1]["content"]
        self.assertIn("S1", prompt)
        self.assertNotIn(long_key, prompt)
        self.assertIn("只返回候选 id", prompt)

    async def test_selector_rejects_non_exact_key_with_warning(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

        class Runtime:
            async def chat(
                self,
                _messages: list[dict[str, object]],
                *,
                media_parts: list[dict[str, object]] | None = None,
                purpose: object | None = None,
            ) -> str:
                return '{"key": "good"}'

        with self.assertLogs("nonebot_plugin_qwen_persona.sticker_selector", level="WARNING") as logs:
            selection = await select_reply_sticker(
                Runtime(),
                candidates=[{"key": "face:good", "type": "face", "summary": "震惊", "data": {"id": "1"}, "count": 5}],
                trigger_text="这是什么",
                reply_text="这是影之诗",
                intent=ReplyIntent.IMAGE_EXPLAIN,
            )

        self.assertIsNone(selection.key)
        self.assertTrue(any("invalid sticker selection" in item for item in logs.output))

    async def test_selector_excludes_non_sendable_plain_image_candidates(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

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

        self.assertEqual(selection.key, "face:smile")
        prompt = runtime.calls[0]["messages"][1]["content"]
        self.assertIn("S1", prompt)
        self.assertNotIn("image:screenshot", prompt)
        self.assertEqual(runtime.calls[0]["media_parts"], [])

    async def test_explicit_sticker_request_falls_back_to_sendable_candidate_on_invalid_output(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

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

        self.assertEqual(selection.key, "face:smile")
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.raw_text, "🙂")

    async def test_selector_filters_low_frequency_candidates_and_falls_back_on_empty_output(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

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

        self.assertEqual(selection.key, "face:high")
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.usable_count, 1)


class StickerMessageTests(unittest.TestCase):
    def test_onebot_sticker_message_preserves_text_and_sticker_in_single_message(self) -> None:
        from nonebot_plugin_qwen_persona.sticker import build_onebot_v11_sticker_message

        class Message:
            def __init__(self) -> None:
                self.parts: list[object] = []

            def __iadd__(self, value: object) -> "Message":
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

        self.assertIsInstance(reply, Message)
        self.assertEqual([part.type for part in reply.parts], ["text", "face"])
        self.assertEqual(reply.parts[0].data["text"], "这是影之诗")

    def test_onebot_custom_image_sticker_message_is_sendable(self) -> None:
        from nonebot_plugin_qwen_persona.sticker import build_onebot_v11_sticker_message

        class Message:
            def __init__(self) -> None:
                self.parts: list[object] = []

            def __iadd__(self, value: object) -> "Message":
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

        self.assertIsInstance(reply, Message)
        self.assertEqual([part.type for part in reply.parts], ["image"])


if __name__ == "__main__":
    unittest.main()
