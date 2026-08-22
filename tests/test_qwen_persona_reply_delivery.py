from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_ai_companion"
package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


class ReplyDeliveryTests(unittest.TestCase):
    def test_short_reply_uses_normal_message_delivery(self) -> None:
        from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

        decision = plan_reply_delivery("短回复", [], long_text_threshold=20)

        self.assertFalse(decision.use_forward)
        self.assertEqual(decision.forward_chunks, [])
        self.assertEqual(decision.trailing_sticker_keys, [])

    def test_long_or_code_reply_uses_forward_chunks(self) -> None:
        from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

        text = "```asm\n" + "\n".join(f"line {index}" for index in range(12)) + "\n```"
        decision = plan_reply_delivery(text, [], long_text_threshold=400, max_chunk_chars=60)

        self.assertTrue(decision.use_forward)
        self.assertGreater(len(decision.forward_chunks), 1)
        self.assertEqual("".join(decision.forward_chunks), text)
        self.assertTrue(all(len(chunk) <= 60 for chunk in decision.forward_chunks))

    def test_long_reply_keeps_stickers_for_after_forward_delivery(self) -> None:
        from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

        decision = plan_reply_delivery("很长" * 200, ["face:smile"], long_text_threshold=30)

        self.assertTrue(decision.use_forward)
        self.assertEqual(decision.trailing_sticker_keys, ["face:smile"])

