"""「有人找你」提醒：触发判定、在场判定、风控、卡片 HTML。"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


def _envelope(*, at_targets=None, reply_sender=None):
    from nonebot_plugin_ai_companion.context.envelope import MessageContextEnvelope

    return MessageContextEnvelope(
        current_text="在吗",
        quoted_texts=["上次说的那个东西"] if reply_sender else [],
        forwarded_texts=[],
        media_parts=[],
        sticker_keys=[],
        forced_by_at=False,
        context_sources=[],
        at_targets=list(at_targets or []),
        reply_sender=reply_sender,
    )


# ---- 触发判定 ----


def test_at_targets_produce_candidates() -> None:
    from nonebot_plugin_ai_companion.notify.detector import detect_ping_candidates

    decision = detect_ping_candidates(
        envelope=_envelope(at_targets=[{"qq": "42", "name": "阿强"}]),
        sender_user_id="1",
        bot_self_id="999",
    )
    assert decision.skip_reason == ""
    assert [(c.target_user_id, c.reason) for c in decision.candidates] == [("42", "at")]


def test_quote_produces_candidate_when_enabled() -> None:
    from nonebot_plugin_ai_companion.notify.detector import detect_ping_candidates

    decision = detect_ping_candidates(
        envelope=_envelope(reply_sender={"user_id": "42", "nickname": "阿强"}),
        sender_user_id="1",
        bot_self_id="999",
    )
    assert [(c.target_user_id, c.reason) for c in decision.candidates] == [("42", "quote")]


def test_bot_targeted_means_no_ping() -> None:
    """bot 自己被 @ / 被引用：走正常回复路径，不找人。"""
    from nonebot_plugin_ai_companion.notify.detector import detect_ping_candidates

    at_decision = detect_ping_candidates(
        envelope=_envelope(at_targets=[{"qq": "999", "name": "bot"}, {"qq": "42", "name": "阿强"}]),
        sender_user_id="1",
        bot_self_id="999",
    )
    assert at_decision.skip_reason == "bot_targeted" and not at_decision.candidates

    quote_decision = detect_ping_candidates(
        envelope=_envelope(reply_sender={"user_id": "999", "nickname": "bot"}),
        sender_user_id="1",
        bot_self_id="999",
    )
    assert quote_decision.skip_reason == "bot_targeted"


def test_self_ping_at_all_and_duplicates_excluded() -> None:
    from nonebot_plugin_ai_companion.notify.detector import detect_ping_candidates

    decision = detect_ping_candidates(
        envelope=_envelope(
            at_targets=[
                {"qq": "1", "name": "自己"},
                {"qq": "all", "name": "全体"},
                {"qq": "42", "name": "阿强"},
                {"qq": "42", "name": "阿强"},
            ]
        ),
        sender_user_id="1",
        bot_self_id="999",
    )
    assert [c.target_user_id for c in decision.candidates] == ["42"]


def test_quote_can_be_disabled() -> None:
    from nonebot_plugin_ai_companion.notify.detector import detect_ping_candidates

    decision = detect_ping_candidates(
        envelope=_envelope(reply_sender={"user_id": "42", "nickname": "阿强"}),
        sender_user_id="1",
        bot_self_id="999",
        quote_enabled=False,
    )
    assert decision.candidates == ()


# ---- 在场判定 ----


def test_target_last_spoke_age() -> None:
    from nonebot_plugin_ai_companion.notify.detector import target_last_spoke_age

    turns = [
        {"user_id": "42", "speaker_kind": "member", "created_at": 1000.0},
        {"user_id": "1", "speaker_kind": "member", "created_at": 1100.0},
        {"user_id": "999", "speaker_kind": "bot", "created_at": 1150.0},  # bot 的发言不算
    ]
    assert target_last_spoke_age(turns, target_user_id="42", now=1300.0) == 300.0
    assert target_last_spoke_age(turns, target_user_id="77", now=1300.0) is None  # 没说过话


# ---- 卡片 HTML（不泄露 QQ、图文齐全）----


def _card():
    from nonebot_plugin_ai_companion.notify.model import FinderEntry, PingCard

    card = PingCard(group_id="100", target_user_id="42", target_name="阿强")
    card.finders.append(FinderEntry(user_id="3305167706", name="小红", text="@阿强 你看这个", reason="at"))
    return card


def test_card_html_contains_content_and_never_qq_ids() -> None:
    from nonebot_plugin_ai_companion.notify.render import build_card_html

    page = build_card_html(_card())
    assert "阿强同学，有 1 人找你哦" in page
    assert "小红" in page and "你看这个" in page
    assert "3305167706" not in page  # QQ 号绝不进卡片


def test_card_title_pluralizes() -> None:
    from nonebot_plugin_ai_companion.notify.model import FinderEntry, PingCard

    card = PingCard(group_id="100", target_user_id="42", target_name="阿强")
    card.finders.append(FinderEntry(user_id="1", name="小红", text="a"))
    card.finders.append(FinderEntry(user_id="2", name="小刚", text="b"))
    assert card.title == "阿强同学们，有 2 人找你哦"


def test_card_html_escapes_hostile_text() -> None:
    from nonebot_plugin_ai_companion.notify.render import build_card_html

    card = _card()
    card.finders[0].text = "<script>alert(1)</script>"
    page = build_card_html(card)
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_card_image_embedding_downscales() -> None:
    import io

    from PIL import Image

    from nonebot_plugin_ai_companion.notify.render import build_card_html

    buffer = io.BytesIO()
    Image.new("RGB", (1200, 800), (200, 60, 60)).save(buffer, format="JPEG")
    page = build_card_html(_card(), image_bytes_list=[[buffer.getvalue()]])
    assert "data:image/jpeg;base64," in page


# ---- 风控 ----


def _ping_service(store, config):
    from nonebot_plugin_ai_companion.notify.service import PingService

    return PingService(
        config=config,
        env=SimpleNamespace(),
        store=store,
        message_logs=store.message_logs,
    )


async def test_target_cooldown_blocks_second_ping(persona_store) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig

    config = AiCompanionConfig(ai_companion_ping_target_cooldown_seconds=300)
    service = _ping_service(persona_store, config)
    assert service._target_blocked("100", "42", now=1000.0) == ""
    service._mark_sent("100", "42", 1000.0)
    assert service._target_blocked("100", "42", now=1100.0) == "target_cooldown"
    assert service._target_blocked("100", "42", now=1400.0) == ""  # 冷却过了
    assert service._target_blocked("100", "77", now=1100.0) == ""  # 别的目标不受影响


async def test_group_hourly_limit(persona_store) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig

    config = AiCompanionConfig(ai_companion_ping_hourly_limit=2, ai_companion_ping_target_cooldown_seconds=0)
    service = _ping_service(persona_store, config)
    service._mark_sent("100", "42", 1000.0)
    service._mark_sent("100", "77", 1010.0)
    assert service._target_blocked("100", "88", now=1020.0) == "group_hourly_limit"
    assert service._target_blocked("200", "88", now=1020.0) == ""  # 别的群不受影响
    assert service._target_blocked("100", "88", now=4700.0) == ""  # 一小时后窗口滑出


async def test_ping_records_leave_a_trace(persona_store) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.notify.model import PingCandidate

    config = AiCompanionConfig()
    service = _ping_service(persona_store, config)
    candidate = PingCandidate(target_user_id="42", reason="at", finder_user_id="1")
    await service._record("100", {"platform": "qq", "message_id": "m1"}, candidate, status="skipped_target_active")
    rows = await persona_store.message_logs.list_logs(kind="ping", limit=5)
    items = rows.get("items") if isinstance(rows, dict) else rows
    assert any(
        row["reply_kind"] == "ping" and row["status"] == "skipped_target_active" and row["user_id"] == "42"
        for row in items
    )
