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


class StickerSelectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_selector_uses_mlx_vlm_to_pick_exact_candidate_key(self) -> None:
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
            ) -> str:
                self.calls.append({"messages": messages, "media_parts": media_parts or []})
                return '{"key": "image:good", "reason": "fits"}'

        runtime = Runtime()
        candidates = [
            {
                "key": "image:good",
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
                "count": 3,
            },
        ]

        selection = await select_reply_sticker(
            runtime,
            candidates=candidates,
            trigger_text="这是什么",
            reply_text="这是影之诗",
            intent=ReplyIntent.IMAGE_EXPLAIN,
        )

        self.assertEqual(selection.key, "image:good")
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(runtime.calls[0]["media_parts"]), 1)
        prompt = runtime.calls[0]["messages"][1]["content"]
        self.assertIn("image:good", prompt)
        self.assertIn("只能选择候选 key", prompt)

    async def test_selector_rejects_non_exact_key_with_warning(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent
        from nonebot_plugin_qwen_persona.sticker_selector import select_reply_sticker

        class Runtime:
            async def chat(
                self,
                _messages: list[dict[str, object]],
                *,
                media_parts: list[dict[str, object]] | None = None,
            ) -> str:
                return '{"key": "good"}'

        with self.assertLogs("nonebot_plugin_qwen_persona.sticker_selector", level="WARNING") as logs:
            selection = await select_reply_sticker(
                Runtime(),
                candidates=[{"key": "image:good", "type": "image", "summary": "震惊", "count": 1}],
                trigger_text="这是什么",
                reply_text="这是影之诗",
                intent=ReplyIntent.IMAGE_EXPLAIN,
            )

        self.assertIsNone(selection.key)
        self.assertTrue(any("invalid sticker selection" in item for item in logs.output))


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

        reply = build_onebot_v11_sticker_message(
            Message,
            MessageSegment,
            "这是影之诗",
            [{"key": "face:smile", "type": "face", "data": {"id": "123"}}],
        )

        self.assertIsInstance(reply, Message)
        self.assertEqual([part.type for part in reply.parts], ["text", "face"])
        self.assertEqual(reply.parts[0].data["text"], "这是影之诗")


if __name__ == "__main__":
    unittest.main()
