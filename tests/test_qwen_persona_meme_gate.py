from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class MemeGateTests(unittest.TestCase):
    def test_candidate_builder_requires_repeated_recent_samples_for_heat(self) -> None:
        from nonebot_plugin_qwen_persona.meme_gate import (
            build_meme_candidates_from_samples,
            select_active_memes,
        )

        single_candidates = build_meme_candidates_from_samples(
            samples=[
                {
                    "plain_text": "快变回人形了",
                    "training_text": "快变回人形了",
                }
            ],
            media_assets=[],
        )
        repeated_candidates = build_meme_candidates_from_samples(
            samples=[
                {"plain_text": "快变回人形了", "training_text": "快变回人形了"},
                {"plain_text": "快变回人形了", "training_text": "快变回人形了"},
            ],
            media_assets=[],
        )

        self.assertEqual(
            select_active_memes(
                single_candidates,
                context_text="快变回人形是什么意思",
                min_recent_count=2,
            ),
            [],
        )
        selected = select_active_memes(
            repeated_candidates,
            context_text="快变回人形是什么意思",
            min_recent_count=2,
        )
        self.assertEqual([item["label"] for item in selected], ["快变回人形了"])

    def test_high_heat_meme_is_rejected_without_context_match(self) -> None:
        from nonebot_plugin_qwen_persona.meme_gate import select_active_memes

        selected = select_active_memes(
            [
                {
                    "label": "快变回人形",
                    "keywords": ["变回人形", "人形"],
                    "recent_count": 20,
                }
            ],
            context_text="兰州拉面怎么做？",
            min_recent_count=5,
        )

        self.assertEqual(selected, [])

    def test_context_match_and_recent_heat_enables_limited_memes(self) -> None:
        from nonebot_plugin_qwen_persona.meme_gate import select_active_memes

        selected = select_active_memes(
            [
                {"label": "低频影之诗梗", "keywords": ["影之诗"], "recent_count": 1},
                {"label": "2000张卡", "keywords": ["影之诗", "卡"], "recent_count": 7},
                {"label": "爽吧", "keywords": ["爽"], "recent_count": 6},
                {"label": "卡组地狱", "keywords": ["卡组"], "recent_count": 9},
                {"label": "退环境", "keywords": ["退环境"], "recent_count": 8},
            ],
            context_text="影之诗这游戏卡多吗，卡组会不会很复杂？",
            min_recent_count=5,
            max_items=3,
        )

        self.assertEqual([item["label"] for item in selected], ["卡组地狱", "2000张卡"])
        self.assertLessEqual(len(selected), 3)
