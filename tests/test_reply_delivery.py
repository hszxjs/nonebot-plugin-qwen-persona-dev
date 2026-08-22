from __future__ import annotations


def test_short_reply_uses_normal_message_delivery() -> None:
    from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

    decision = plan_reply_delivery("短回复", [], long_text_threshold=20)

    assert not decision.use_forward
    assert decision.forward_chunks == []
    assert decision.trailing_sticker_keys == []


def test_long_or_code_reply_uses_forward_chunks() -> None:
    from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

    text = "```asm\n" + "\n".join(f"line {index}" for index in range(12)) + "\n```"
    decision = plan_reply_delivery(text, [], long_text_threshold=400, max_chunk_chars=60)

    assert decision.use_forward
    assert len(decision.forward_chunks) > 1
    assert "".join(decision.forward_chunks) == text
    assert all(len(chunk) <= 60 for chunk in decision.forward_chunks)


def test_long_reply_keeps_stickers_for_after_forward_delivery() -> None:
    from nonebot_plugin_ai_companion.delivery.plan import plan_reply_delivery

    decision = plan_reply_delivery("很长" * 200, ["face:smile"], long_text_threshold=30)

    assert decision.use_forward
    assert decision.trailing_sticker_keys == ["face:smile"]
