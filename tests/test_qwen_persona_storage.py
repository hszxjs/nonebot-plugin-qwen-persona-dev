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
