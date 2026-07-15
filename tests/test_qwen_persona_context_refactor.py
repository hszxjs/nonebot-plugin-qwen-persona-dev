from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class ReplyDecisionTests(unittest.TestCase):
    def test_at_bot_forces_reply_even_when_text_looks_command_like(self) -> None:
        from nonebot_plugin_qwen_persona.reply_decision import should_attempt_reply

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

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "force_reply")

    def test_plain_command_like_message_is_filtered_before_random_reply(self) -> None:
        from nonebot_plugin_qwen_persona.reply_decision import should_attempt_reply

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

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "command_like")


class ReplyPlannerContextTests(unittest.TestCase):
    def test_question_prompt_prioritizes_answering_over_persona_imitation(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import ReplyIntent, build_reply_plan

        plan = build_reply_plan(
            system_prompt="你正在扮演群友。",
            conversation_lines=["bob: 前面说的是一张游戏详情页"],
            style_memory_lines=["alice: 哈哈这波有点抽象"],
            current_text="这游戏为什么下载量这么高？",
            media_parts=[],
            sticker_reply_allowed=False,
            incoming_has_sticker=False,
        )

        self.assertEqual(plan.intent, ReplyIntent.QUESTION_ANSWER)
        user_content = plan.messages[1]["content"]
        self.assertIn("当前对话上下文", user_content)
        self.assertIn("语气参考", user_content)
        self.assertIn("先直接回答问题", user_content)
        self.assertNotIn("最近群聊样本", user_content)
        self.assertNotIn("自然接一句", user_content)

    def test_style_memory_is_marked_as_non_factual_reference(self) -> None:
        from nonebot_plugin_qwen_persona.reply_planner import build_reply_plan

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
        self.assertIn("不能把语气参考当作当前事实", user_content)
        self.assertIn("不要输出用户 id", user_content)


class MessageContextEnvelopeTests(unittest.TestCase):
    def test_extracts_quoted_and_forwarded_text_context(self) -> None:
        from nonebot_plugin_qwen_persona.message_context import build_message_context

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

        self.assertEqual(envelope.current_text, "这是什么")
        self.assertEqual(envelope.quoted_texts, ["被引用的消息"])
        self.assertEqual(envelope.forwarded_texts, ["合并里的前文"])
        self.assertIn("quoted", envelope.context_sources)
        self.assertIn("forwarded", envelope.context_sources)


class RetrievalAndMediaAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_context_samples_returns_matching_recent_text(self) -> None:
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.record_sample(
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
            await store.record_sample(
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

            results = await store.search_context_samples(group_id="g1", query="影之诗 卡", limit=3)

        self.assertEqual([item["plain_text"] for item in results], ["影之诗这游戏有很多卡"])

    async def test_search_media_assets_returns_matching_cached_vlm_result(self) -> None:
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.record_media_asset(
                source="sticker",
                source_key="s1",
                uri="https://example.test/sticker.webp",
                prompt="请描述图片",
                answer="影之诗表情包，角色震惊地看着卡牌",
                metadata={"group_id": "g1"},
            )
            await store.record_media_asset(
                source="sticker",
                source_key="s2",
                uri="https://example.test/other.webp",
                prompt="请描述图片",
                answer="完全无关的表情",
                metadata={"group_id": "g1"},
            )

            results = await store.search_media_assets(group_id="g1", query="影之诗 卡牌", limit=3)

        self.assertEqual([item["answer"] for item in results], ["影之诗表情包，角色震惊地看着卡牌"])

    async def test_background_media_analysis_records_only_sticker_like_images(self) -> None:
        from nonebot_plugin_qwen_persona.media_analysis import analyze_media_parts
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        class Captioner:
            async def caption_media(self, part: dict[str, object]) -> str:
                url = str((part.get("data") or {}).get("url") or "")
                if "sticker" in url:
                    return "一个二次元表情包，角色震惊地张大嘴"
                return "一张网页截图，显示应用详情和下载按钮"

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            result = await analyze_media_parts(
                store=store,
                captioner=Captioner(),
                parts=[
                    {"type": "image", "data": {"url": "https://example.test/sticker.webp"}},
                    {"type": "image", "data": {"url": "https://example.test/screenshot.png"}},
                ],
                group_id="g1",
                user_id="u1",
            )
            stats = await store.stats()

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(stats["media_assets"], 1)

    async def test_background_media_analysis_does_not_double_count_known_image_sticker(self) -> None:
        from nonebot_plugin_qwen_persona.media_analysis import analyze_media_parts
        from nonebot_plugin_qwen_persona.sticker import sticker_to_record
        from nonebot_plugin_qwen_persona.sticker_classifier import classify_sticker_segment
        from nonebot_plugin_qwen_persona.storage import PersonaStore

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
        self.assertIsNotNone(record)

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.record_sticker_asset(
                sticker=record,
                group_id="g1",
                user_id="u1",
                summary="已入库表情",
            )
            await analyze_media_parts(
                store=store,
                captioner=Captioner(),
                parts=[part],
                group_id="g1",
                user_id="u1",
            )
            sticker = await store.get_sticker(record["key"])

        self.assertIsNotNone(sticker)
        self.assertEqual(sticker["count"], 1)


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_defaults_to_no_available_tools(self) -> None:
        from nonebot_plugin_qwen_persona.tool_registry import ToolRegistry

        registry = ToolRegistry()

        self.assertFalse(registry.has_tools)
        self.assertEqual(registry.describe_for_prompt(), "")
