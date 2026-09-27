"""话题线程：推断规则、箭头渲染、当前话题聚焦、摘要多线。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


def _rows(*specs: tuple[int, str, float, int]) -> list[dict[str, object]]:
    """(id, user_id, created_at, thread_id) → previous_turns 结构。"""
    return [
        {
            "id": item[0],
            "user_id": item[1],
            "created_at": item[2],
            "thread_id": item[3],
            "source_message_id": str(item[0]),
        }
        for item in specs
    ]


def test_infer_thread_explicit_reply_wins() -> None:
    from nonebot_plugin_ai_companion.conversation.threads import infer_thread

    previous = _rows((1, "A", 100.0, 1), (2, "B", 101.0, 1))
    assignment = infer_thread(
        previous,
        new_turn_id=3,
        user_id="C",
        created_at=102.0,
        reply_to_message_id="1",  # 显式引用第一条
    )
    assert (assignment.thread_id, assignment.reply_to_turn_id, assignment.reason) == (1, 1, "explicit")


def test_infer_thread_follows_previous_within_window() -> None:
    from nonebot_plugin_ai_companion.conversation.threads import infer_thread

    previous = _rows((1, "A", 100.0, 1))
    # 别人 2 分钟后接话 → 并入对方线程
    joined = infer_thread(previous, new_turn_id=2, user_id="B", created_at=220.0)
    assert (joined.thread_id, joined.reply_to_turn_id, joined.reason) == (1, 1, "reply_to_previous")
    # 自己接着说 → 仍是同一条线，但不算"回复谁"
    own = infer_thread(previous, new_turn_id=3, user_id="A", created_at=230.0)
    assert (own.thread_id, own.reply_to_turn_id, own.reason) == (1, 1, "same_speaker")


def test_infer_thread_opens_new_thread_after_gap() -> None:
    from nonebot_plugin_ai_companion.conversation.threads import infer_thread

    previous = _rows((1, "A", 100.0, 1))
    fresh = infer_thread(previous, new_turn_id=2, user_id="B", created_at=100.0 + 601.0, window_seconds=300)
    assert (fresh.thread_id, fresh.reply_to_turn_id, fresh.reason) == (2, 0, "new_thread")


def test_infer_thread_without_history_starts_thread() -> None:
    from nonebot_plugin_ai_companion.conversation.threads import infer_thread

    assignment = infer_thread([], new_turn_id=9, user_id="A", created_at=100.0)
    assert (assignment.thread_id, assignment.reply_to_turn_id, assignment.reason) == (9, 0, "new_thread")


def test_infer_thread_unresolved_reply_falls_back_to_previous() -> None:
    """引用的是窗口外/找不到的消息 → 退回"上一条"推断，不丢线程。"""
    from nonebot_plugin_ai_companion.conversation.threads import infer_thread

    previous = _rows((5, "A", 500.0, 5))
    assignment = infer_thread(previous, new_turn_id=6, user_id="B", created_at=510.0, reply_to_message_id="999")
    assert assignment.reason == "reply_to_previous" and assignment.thread_id == 5


async def test_turn_writer_assigns_threads(persona_store) -> None:
    """真实写入路径：A 发言 → B 接话（并入 A 的线）→ 5 分钟后新话题（新线）。"""
    for user_id, text, created_at in [("A", "今天打本吗", 100.0), ("B", "来", 140.0), ("A", "八点", 180.0)]:
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind="member",
            text=text,
            group_id="100",
            user_id=user_id,
            created_at=created_at,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    assert [turn.thread_id for turn in turns] == [turns[0].id] * 3
    assert turns[1].reply_to_turn_id == turns[0].id  # B 接 A
    assert turns[2].reply_to_turn_id == turns[1].id  # A 再接 B

    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind="member",
        text="完全无关的新话题",
        group_id="100",
        user_id="C",
        created_at=100.0 + 4000.0,
    )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    assert turns[-1].thread_id == turns[-1].id  # 新线自成一脉
    assert turns[-1].reply_to_turn_id == 0


async def test_thread_lines_render_arrows_and_thread_lookup(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.threads import current_thread_digest
    from nonebot_plugin_ai_companion.conversation.window import render_history_lines

    for user_id, name, text, created_at in [
        ("A", "阿强", "今天打本吗", 100.0),
        ("B", "小红", "来", 140.0),
        ("C", "小刚", "我不打", 150.0),
    ]:
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind="member",
            text=text,
            group_id="100",
            user_id=user_id,
            speaker_name=name,
            created_at=created_at,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    lines = render_history_lines(turns)
    assert lines[0] == "阿强：今天打本吗"
    assert lines[1] == "小红 → 阿强：来"
    assert lines[2] == "小刚 → 小红：我不打"

    # 当前话题聚焦：线程内的人与话，供提示词"只回这一条线"
    thread_id = int(turns[0].thread_id or 0)
    thread_turns = await persona_store.conversation.recent_thread_turns(
        session_key="group:100", thread_id=thread_id, limit=10
    )
    digest = current_thread_digest(
        [{"thread_id": int(turn.thread_id or 0), "turn": turn} for turn in thread_turns],
        thread_id=thread_id,
        names={},
    )
    assert digest.participants == ["阿强", "小红", "小刚"]
    assert digest.lines[1] == "小红 → 阿强：来"


def test_planner_renders_current_thread_block() -> None:
    from nonebot_plugin_ai_companion.prompt.planner import build_reply_plan

    plan = build_reply_plan(
        system_prompt="你是群友",
        history_lines=["阿强：今天打本吗", "小红 → 阿强：来"],
        current_thread_lines=["阿强：今天打本吗", "小红 → 阿强：来"],
        current_thread_participants=["阿强", "小红"],
        current_text="八点开",
        media_parts=[],
        sticker_reply_allowed=True,
        incoming_has_sticker=False,
    )
    user_content = plan.messages[1]["content"]
    assert "你正准备接的这条话题（参与者：阿强、小红）" in user_content
    assert "只回这一条线，不要同时回其他人" in user_content


async def test_backfill_threads_is_idempotent(persona_store) -> None:
    for user_id, text, created_at in [("A", "一", 100.0), ("B", "二", 130.0)]:
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind="member",
            text=text,
            group_id="100",
            user_id=user_id,
            created_at=created_at,
        )
    # 模拟迁移前的老数据：线程字段清零后再回填
    with persona_store.db.connection() as conn:
        conn.execute("UPDATE conversation_turns SET thread_id = 0, reply_to_turn_id = 0")
    first = await persona_store.conversation.backfill_threads()
    second = await persona_store.conversation.backfill_threads()
    assert first["sessions"] == 1 and first["rows"] == 2
    assert second["sessions"] == 0 and second["rows"] == 0  # 已算过就跳过
