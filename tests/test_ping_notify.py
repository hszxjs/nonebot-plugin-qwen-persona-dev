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


# ---- 宽限期判定（@ 后 10 分钟没说话才算错过）----


def test_split_pending_by_grace() -> None:
    from nonebot_plugin_ai_companion.notify.service import split_pending

    entries = [
        SimpleNamespace(created_at=1000.0, id=1),  # 12 分钟前 → 错过
        SimpleNamespace(created_at=1360.0, id=2),  # 4 分钟前 → 在场看到，作废
    ]
    missed, fresh = split_pending(entries, now=1720.0, grace_seconds=600.0)
    assert [entry.id for entry in missed] == [1]
    assert [entry.id for entry in fresh] == [2]

    # 宽限期 = 0：全部视为错过（不看时间）
    missed, fresh = split_pending(entries, now=1720.0, grace_seconds=0)
    assert len(missed) == 2 and fresh == []

    # 恰好 10 分钟：算错过（用户口径是「10 分钟以上」，边界归入错过）
    boundary = [SimpleNamespace(created_at=1120.0, id=3)]
    missed, fresh = split_pending(boundary, now=1720.0, grace_seconds=600.0)
    assert [entry.id for entry in missed] == [3]


# ---- 区间合并（±3 相连合并）----


def _turns(*texts: str) -> list[dict[str, object]]:
    return [
        {"source_message_id": f"m{index}", "created_at": 1000.0 + index, "turn": None, "thread_id": 0}
        for index, text in enumerate(texts)
    ]


def test_merge_overlapping_and_adjacent_windows() -> None:
    from nonebot_plugin_ai_companion.notify.digest import merge_context_blocks

    turns = _turns(*[f"第{i}条" for i in range(12)])
    # 两个锚点相距 4 条：窗口 [a-3,a+3] 与 [b-3,b+3] 重叠 → 一个连续块
    blocks, dropped = merge_context_blocks(turns, [("m2", 1002.0), ("m6", 1006.0)], context_lines=3, max_lines=40)
    assert len(blocks) == 1 and dropped == 0
    covered = [str(t["source_message_id"]) for t in blocks[0].turns]
    assert covered == [f"m{i}" for i in range(0, 10)]


def test_merge_keeps_separate_blocks_when_gapped() -> None:
    from nonebot_plugin_ai_companion.notify.digest import merge_context_blocks

    turns = _turns(*[f"第{i}条" for i in range(14)])
    # 相距 7 条：窗口之间有空隙 → 两个独立块
    blocks, dropped = merge_context_blocks(turns, [("m1", 1001.0), ("m10", 1010.0)], context_lines=3, max_lines=40)
    assert len(blocks) == 2 and dropped == 0
    assert str(blocks[1].turns[0]["source_message_id"]) == "m7"


def test_merge_caps_total_lines_dropping_oldest() -> None:
    from nonebot_plugin_ai_companion.notify.digest import merge_context_blocks

    turns = _turns(*[f"第{i}条" for i in range(20)])
    blocks, dropped = merge_context_blocks(turns, [("m3", 1003.0), ("m15", 1015.0)], context_lines=3, max_lines=10)
    total = sum(len(block.turns) for block in blocks)
    assert total <= 10 and dropped > 0
    # 最老的行先被截掉，锚点 m15 的窗口保留
    last_block_ids = [str(t["source_message_id"]) for t in blocks[-1].turns]
    assert "m15" in last_block_ids


def test_merge_falls_back_to_time_anchor_for_unknown_message() -> None:
    """触发消息没进时间线（如命令式 @）：按时间退化锚定，窗口仍成立。"""
    from nonebot_plugin_ai_companion.notify.digest import merge_context_blocks

    turns = _turns(*[f"第{i}条" for i in range(10)])
    blocks, dropped = merge_context_blocks(turns, [("missing", 1005.5)], context_lines=2, max_lines=40)
    assert len(blocks) == 1 and dropped == 0
    ids = [str(t["source_message_id"]) for t in blocks[0].turns]
    assert ids == ["m3", "m4", "m5", "m6", "m7"]


# ---- 信箱仓库（持久化 + TTL）----


async def test_inbox_add_list_delete_and_purge(persona_store) -> None:
    repo = persona_store.db and _inbox(persona_store)
    first = await repo.add(
        session_key="group:100",
        group_id="100",
        target_user_id="42",
        finder_user_id="1",
        reason="at",
        source_message_id="m1",
        text="在吗",
    )
    await repo.add(
        session_key="group:100",
        group_id="100",
        target_user_id="42",
        finder_user_id="2",
        reason="quote",
        source_message_id="m2",
        text="看看这个",
    )
    pending = await repo.list_for_target(session_key="group:100", target_user_id="42")
    assert [entry.finder_user_id for entry in pending] == ["1", "2"]
    assert pending[0].text == "在吗"

    await repo.delete_entries([first])
    assert len(await repo.list_for_target(session_key="group:100", target_user_id="42")) == 1

    # TTL 清理：把两条都改成 25 小时前 → 24h TTL 全清
    import time as _time

    with persona_store.db.connection() as conn:
        conn.execute("UPDATE ping_inbox SET created_at = ?", (_time.time() - 25 * 3600,))
    expired = await repo.purge_expired(ttl_seconds=24 * 3600)
    assert len(expired) == 1
    assert await repo.list_for_target(session_key="group:100", target_user_id="42") == []


def _inbox(store):
    from nonebot_plugin_ai_companion.notify.inbox import PingInboxRepo

    return PingInboxRepo(store.db)


# ---- 摘要卡 HTML ----


def _blocks():
    from nonebot_plugin_ai_companion.conversation.model import ConversationTurn
    from nonebot_plugin_ai_companion.notify.digest import DigestBlock

    def turn(user_id, name, text, message_id, *, bot=False):
        return {
            "turn": ConversationTurn(
                session_key="group:100",
                speaker_kind="bot" if bot else "member",
                text=text,
                group_id="100",
                user_id=user_id,
                speaker_name=name,
                source_message_id=message_id,
            ),
            "source_message_id": message_id,
            "created_at": 1000.0,
            "thread_id": 0,
        }

    block = DigestBlock(trigger_message_ids={"m2"})
    block.turns = [turn("1", "小红", "在吗", "m1"), turn("42", "", "谁叫我", "m2")]
    return [block]


def test_digest_html_renders_trigger_badge_and_never_qq() -> None:
    from nonebot_plugin_ai_companion.notify.render import build_digest_html

    page = build_digest_html(
        title="阿强同学，有人找过你",
        subtitle="以下是你不在时被找的原话和上下文：",
        blocks=_blocks(),
        names={},
    )
    assert "阿强同学，有人找过你" in page
    assert "在吗" in page and "谁叫我" in page
    assert "找你" in page  # 触发行徽标
    assert 'class="content trigger"' in page
    assert "3305167706" not in page  # QQ 号绝不出现


def test_digest_html_escapes_and_truncation_note() -> None:
    from nonebot_plugin_ai_companion.notify.digest import DigestBlock
    from nonebot_plugin_ai_companion.notify.render import build_digest_html

    block = DigestBlock(trigger_message_ids=set())
    block.turns = [
        {
            "turn": SimpleNamespace(  # 最小 duck 类型即可
                is_bot=False,
                user_id="1",
                speaker_name="小红",
                text="<b>粗体</b>",
                sticker_keys=[],
                source_message_id="m1",
                reply_to_turn_id=0,
            ),
            "source_message_id": "m1",
            "created_at": 1000.0,
            "thread_id": 0,
        }
    ]
    page = build_digest_html(
        title="t",
        subtitle="s",
        blocks=[block],
        dropped_lines=7,
    )
    assert "&lt;b&gt;" in page and "<b>" not in page
    assert "更早的 7 条已省略" in page


def test_digest_html_embeds_attachments() -> None:
    import io

    from PIL import Image

    from nonebot_plugin_ai_companion.notify.render import build_digest_html

    buffer = io.BytesIO()
    Image.new("RGB", (900, 600), (60, 120, 200)).save(buffer, format="JPEG")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(buffer.getvalue())
        path = handle.name
    blocks = _blocks()
    page = build_digest_html(
        title="t",
        subtitle="s",
        blocks=blocks,
        attachments_by_message_id={"m2": [path]},
    )
    assert "data:image/jpeg;base64," in page
    Path(path).unlink(missing_ok=True)


async def test_service_purge_expired_clears_inbox(persona_store) -> None:
    """TTL 清理接线：超时未投递的信箱条目被清掉（后台循环滚动调用）。"""
    import time as _time

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.notify.service import PingService

    service = PingService(
        config=AiCompanionConfig(),
        env=object(),
        store=persona_store,
        message_logs=persona_store.message_logs,
    )
    await service.inbox.add(
        session_key="group:100",
        group_id="100",
        target_user_id="42",
        source_message_id="m1",
        text="在吗",
    )
    await service.inbox.add(
        session_key="group:100",
        group_id="100",
        target_user_id="42",
        source_message_id="m2",
        text="刚说的",
    )
    with persona_store.db.connection() as conn:
        conn.execute("UPDATE ping_inbox SET created_at = ? WHERE source_message_id = 'm1'", (_time.time() - 25 * 3600,))

    assert await service.purge_expired(ttl_seconds=24 * 3600) == 1
    remaining = await service.inbox.list_for_target(session_key="group:100", target_user_id="42")
    assert [entry.source_message_id for entry in remaining] == ["m2"]
