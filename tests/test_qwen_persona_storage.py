from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class PersonaStoreDatasetTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_sft_dataset_filters_self_and_generated_chat_log_outputs(self) -> None:
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.record_sample(
                platform="qq",
                user_id="10001",
                group_id="g1",
                message_id="m1",
                plain_text="前面聊拉面",
                training_text="前面聊拉面",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            await store.record_sample(
                platform="qq",
                user_id="bot",
                group_id="g1",
                message_id="m2",
                plain_text="我是 bot 输出",
                training_text="我是 bot 输出",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            await store.record_sample(
                platform="qq",
                user_id="10002",
                group_id="g1",
                message_id="m3",
                plain_text="3305167706: 先煮面再调汤吧",
                training_text="3305167706: 先煮面再调汤吧",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            await store.record_sample(
                platform="qq",
                user_id="10003",
                group_id="g1",
                message_id="m4",
                plain_text="兰州拉面一般先熬汤再煮面",
                training_text="兰州拉面一般先熬汤再煮面",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            output_path = Path(temp_root) / "dataset.jsonl"

            count = await store.export_sft_dataset(
                output_path=output_path,
                limit=0,
                history_window=4,
                system_prompt="system",
                excluded_user_ids={"bot"},
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        assistant_texts = [row["messages"][-1]["content"] for row in rows]
        user_prompts = [row["messages"][1]["content"] for row in rows]
        self.assertEqual(count, 2)
        self.assertEqual(assistant_texts, ["前面聊拉面", "兰州拉面一般先熬汤再煮面"])
        self.assertTrue(all("3305167706:" not in prompt for prompt in user_prompts))
        self.assertTrue(all("先煮面再调汤" not in prompt for prompt in user_prompts))
        self.assertTrue(all("10001:" not in prompt for prompt in user_prompts))

    async def test_export_sft_dataset_skips_low_quality_media_asset_answers(self) -> None:
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.record_media_asset(
                source="sticker",
                source_key="bad",
                uri="file:///tmp/bad.webp",
                prompt="请描述这个表情。",
                answer="mp窗 Turing - pleas nodeType <think> Gam Gam Gam",
            )
            await store.record_media_asset(
                source="sticker",
                source_key="good",
                uri="file:///tmp/good.webp",
                prompt="请描述这个表情。",
                answer="一个动漫角色惊讶地张大嘴巴。",
            )
            output_path = Path(temp_root) / "dataset.jsonl"

            count = await store.export_sft_dataset(
                output_path=output_path,
                limit=0,
                history_window=0,
                system_prompt="system",
                include_multimodal_assets=True,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metadata"]["source_key"], "good")
        self.assertEqual(rows[0]["messages"][-1]["content"], "一个动漫角色惊讶地张大嘴巴。")

    async def test_export_sft_dataset_filters_external_and_media_chat_log_prefixes(self) -> None:
        from nonebot_plugin_qwen_persona.storage import PersonaStore

        with tempfile.TemporaryDirectory() as temp_root:
            store = PersonaStore(Path(temp_root))
            await store.init()
            await store.import_external_corpus_samples(
                source="external",
                records=[
                    {
                        "source_id": "bad-external",
                        "text": "3305167706: 这是模型生成污染",
                    },
                    {
                        "source_id": "good-external",
                        "text": "自然外部语料",
                        "metadata": {"user_prompt": "3305167706: 外部上下文"},
                    },
                ],
                min_chars=1,
                max_chars=200,
            )
            await store.record_media_asset(
                source="sticker",
                source_key="bad-media",
                uri="file:///tmp/bad.webp",
                prompt="请描述图片",
                answer="3305167706: 这是污染图片描述",
            )
            await store.record_media_asset(
                source="sticker",
                source_key="good-media",
                uri="file:///tmp/good.webp",
                prompt="3305167706: 请描述图片",
                answer="一个动漫角色惊讶地张大嘴巴。",
            )
            output_path = Path(temp_root) / "dataset.jsonl"

            count = await store.export_sft_dataset(
                output_path=output_path,
                limit=0,
                history_window=0,
                system_prompt="system",
                external_limit=0,
                include_multimodal_assets=True,
            )

            content = output_path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in content.splitlines()]

        self.assertEqual(count, 2)
        self.assertEqual({row["metadata"]["source"] for row in rows}, {"external", "sticker"})
        self.assertNotIn("3305167706:", content)
        self.assertNotIn("模型生成污染", content)
        self.assertNotIn("污染图片描述", content)
