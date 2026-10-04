"""群聊成就：跨线判定、独立计数仓库、统计查询、服务接线、卡片 HTML。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


def _counter(store):
    from nonebot_plugin_ai_companion.notify.counters import MilestoneCounterRepo

    return MilestoneCounterRepo(store.db)


async def _record_turn(store, *, user_id: str, message_id: str, sticker: bool = False, group_id="100") -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER

    await store.conversation.record_turn(
        session_key=f"group:{group_id}",
        speaker_kind=SPEAKER_MEMBER,
        text=f"消息 {message_id}",
        group_id=group_id,
        user_id=user_id,
        sticker_keys=["face:x"] if sticker else [],
        source_message_id=message_id,
        created_at=100.0,
    )


# ---- 阈值跨线判定（纯函数）----


def test_threshold_set_is_configured_plus_round_ten_thousands() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import MILESTONE_DEFAULTS, milestone_thresholds

    assert milestone_thresholds([50, 100, 100], up_to=25000) == [50, 100, 10000, 20000]
    assert milestone_thresholds([], up_to=9999) == []
    # 默认表包含用户指定的趣味号
    assert {50, 250, 325, 666, 888, 1145, 1314} <= set(MILESTONE_DEFAULTS)


def test_crossed_milestone_fires_once_per_threshold() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import crossed_milestone

    assert crossed_milestone(49, 0, [50, 100]) is None
    assert crossed_milestone(50, 0, [50, 100]) == 50
    # 已播报过的阈值不再触发（计数停在阈值上也不会重复发卡）
    assert crossed_milestone(50, 50, [50, 100]) is None
    assert crossed_milestone(99, 50, [50, 100]) is None
    assert crossed_milestone(100, 50, [50, 100]) == 100
    # 一次跨多档：只播最大的那档
    assert crossed_milestone(120, 0, [50, 100]) == 100
    # 整万数自动档
    assert crossed_milestone(20000, 10000, []) == 20000
    assert crossed_milestone(15000, 10000, []) is None


def test_milestone_baseline_swallows_history() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import milestone_baseline

    assert milestone_baseline(0, [50, 100]) == 0
    assert milestone_baseline(99, [50, 100]) == 50
    assert milestone_baseline(100, [50, 100]) == 100
    assert milestone_baseline(15000, []) == 10000


# ---- 独立计数仓库 ----


async def test_counter_claims_each_threshold_once(persona_store) -> None:
    repo = _counter(persona_store)
    claimed = []
    bump = None
    for _ in range(51):
        bump = await repo.bump(group_id="100", user_id="42", configured=[50, 100])
        if bump.claimed is not None:
            claimed.append(bump.claimed)
    assert claimed == [50]
    assert bump is not None and bump.total == 51
    assert bump.announced == 50


async def test_counter_counts_messages_that_never_reach_timeline(persona_store) -> None:
    """黑名单/命令消息不进时间线（学习管线拦掉），计数照样 +1。"""
    repo = _counter(persona_store)
    for index in range(50):
        bump = await repo.bump(group_id="100", user_id="42", configured=[50], source_message_id=f"m{index}")
    assert bump.claimed == 50
    with persona_store.db.connection() as conn:
        turns = conn.execute("SELECT COUNT(*) AS c FROM conversation_turns WHERE group_id = '100'").fetchone()
        row = conn.execute("SELECT message_count FROM member_message_counts WHERE group_id = '100'").fetchone()
    assert int(turns["c"]) == 0
    assert int(row["message_count"]) == 50


async def test_seed_backfills_history_without_backfill_cards(persona_store) -> None:
    """老群上线：历史发言回填进基线，不补发一摞成就卡。"""
    for index in range(600):
        await _record_turn(persona_store, user_id="42", message_id=f"old{index}")
    repo = _counter(persona_store)

    bump = await repo.bump(group_id="100", user_id="42", configured=[50, 100, 500, 666], source_message_id="now1")
    assert bump.total == 601  # 600 条历史 + 这一条
    assert bump.claimed is None
    assert bump.announced == 500


async def test_seed_excludes_current_message_already_in_timeline(persona_store) -> None:
    """学习管线在前：当前这条已进时间线，回填必须排除它，免得 +1 变 +2。"""
    for index in range(99):
        await _record_turn(persona_store, user_id="42", message_id=f"old{index}")
    await _record_turn(persona_store, user_id="42", message_id="now1")
    repo = _counter(persona_store)

    bump = await repo.bump(group_id="100", user_id="42", configured=[50, 100], source_message_id="now1")
    assert bump.total == 100  # 99 + 1，不能是 101
    assert bump.claimed == 100  # 正是这条跨线 → 照常播报


async def test_counter_is_per_group_and_per_member(persona_store) -> None:
    repo = _counter(persona_store)
    await repo.bump(group_id="100", user_id="42", configured=[1000])
    await repo.bump(group_id="100", user_id="42", configured=[1000])
    await repo.bump(group_id="100", user_id="77", configured=[1000])
    other_group = await repo.bump(group_id="200", user_id="42", configured=[1000])
    assert other_group.total == 1  # 别的群不串号
    with persona_store.db.connection() as conn:
        rows = conn.execute(
            "SELECT group_id, user_id, message_count FROM member_message_counts ORDER BY group_id, user_id"
        ).fetchall()
    assert [(str(r["group_id"]), str(r["user_id"]), int(r["message_count"])) for r in rows] == [
        ("100", "42", 2),
        ("100", "77", 1),
        ("200", "42", 1),
    ]


# ---- 统计查询（排名/表情包）----


async def test_member_stats_use_counter_and_timeline(persona_store) -> None:
    repo = _counter(persona_store)
    for _ in range(6):
        await repo.bump(group_id="100", user_id="42", configured=[1000])
    for _ in range(8):
        await repo.bump(group_id="100", user_id="77", configured=[1000])
    await repo.bump(group_id="200", user_id="42", configured=[1000])

    # 表情包数仍来自时间线；私聊消息、bot 消息不参与群统计
    for index, sticker in enumerate((True, True, False, False)):
        await _record_turn(persona_store, user_id="42", message_id=f"m{index}", sticker=sticker)
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER

    await persona_store.conversation.record_turn(
        session_key="private:9", speaker_kind=SPEAKER_MEMBER, text="私聊不算", group_id=None, user_id="42"
    )
    await persona_store.conversation.record_turn(
        session_key="group:100", speaker_kind="bot", text="bot 的话", group_id="100"
    )

    from nonebot_plugin_ai_companion.notify.milestone import fetch_member_stats

    stats = await fetch_member_stats(persona_store, group_id="100", user_id="42")
    assert stats is not None
    assert stats["total"] == 6
    assert stats["sticker_count"] == 2
    assert stats["rank"] == 2  # 77 有 8 条排第一
    assert stats["total_members"] == 2
    assert await fetch_member_stats(persona_store, group_id="100", user_id="999") is None


# ---- 服务接线 ----


class _FakeEnv:
    def __init__(self, *, blocked: tuple[str, ...] = ()) -> None:
        self.blocked = {str(item) for item in blocked}

    def dump_event_ids(self, event):
        return dict(event)

    def group_enabled(self, group_id):
        return True

    def is_self_event(self, bot, ids):
        return False

    def is_blacklisted(self, user_id, group_id, text):
        return str(user_id) in self.blocked


class _FakeMatcher:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, message):
        self.sent.append(message)


def _service(store, *, env, thresholds):
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.notify.milestone_service import MilestoneService

    return MilestoneService(
        config=AiCompanionConfig(ai_companion_milestone_list=list(thresholds)),
        env=env,
        store=store,
        message_logs=store.message_logs,
    )


def _event(user_id: str, index: int) -> dict[str, str]:
    return {"group_id": "100", "user_id": user_id, "message_id": f"m{index}", "platform": "qq"}


async def test_blocked_member_counts_but_never_gets_a_card(persona_store) -> None:
    """黑名单成员的消息计入总数，但不在群里给他庆祝（阈值照常推进）。"""
    service = _service(persona_store, env=_FakeEnv(blocked=("66",)), thresholds=[3])
    matcher = _FakeMatcher()
    for index in range(3):
        await service.handle_message(object(), _event("66", index), matcher)

    assert matcher.sent == []
    with persona_store.db.connection() as conn:
        row = conn.execute("SELECT message_count FROM member_message_counts WHERE user_id = '66'").fetchone()
        logged = conn.execute("SELECT status FROM message_logs WHERE reply_kind = 'milestone'").fetchall()
    assert int(row["message_count"]) == 3
    assert [str(item["status"]) for item in logged] == ["skipped_blocked"]


async def test_crossing_sends_once_and_falls_back_to_text(persona_store, monkeypatch) -> None:
    """跨线发卡；第 4 条不重复触发。渲染在测试环境不可用 → 退化成纯文本 @。"""
    from nonebot_plugin_ai_companion.notify import milestone_service as module

    async def _no_avatar(qq: str):
        return None

    async def _no_render(**kwargs):
        raise RuntimeError("no renderer in tests")

    monkeypatch.setattr(module, "fetch_avatar", _no_avatar)
    monkeypatch.setattr(module, "render_milestone_png", _no_render)

    service = _service(persona_store, env=_FakeEnv(), thresholds=[3])
    matcher = _FakeMatcher()
    for index in range(4):
        await service.handle_message(object(), _event("42", index), matcher)

    assert len(matcher.sent) == 1
    text = str(matcher.sent[0])
    assert "42" in text and "总发言突破 3" in text


# ---- 卡片 HTML ----


def test_milestone_html_matches_layout_and_never_leaks_qq() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import build_milestone_html

    page = build_milestone_html(
        badge="群聊成就",
        name="奶茄",
        count=6666,
        rank=7,
        total_members=933,
        sticker_count=3347,
        avatar=None,
        footer_left="< 某群[1094210961] >",
    )
    assert "群聊成就" in page and "奶茄" in page
    assert "总发言破<span class='num'>6666</span>" in page
    assert "#7" in page and "99.25%" in page  # (933-7)/933 = 99.25%
    assert "3347" in page and "50.2%" in page
    assert "milestone card :)" in page
    assert "1094210961" in page  # 群号在脚注是设计需要；成员 QQ 号才是不许出现的


def test_milestone_html_escapes_hostile_name() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import build_milestone_html

    page = build_milestone_html(
        badge="群聊成就",
        name="<script>彩蛋</script>",
        count=100,
        rank=1,
        total_members=10,
        sticker_count=10,
    )
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_milestone_avatar_fallback_initial() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import build_milestone_html

    page = build_milestone_html(
        badge="群聊成就", name="奶茄", count=100, rank=1, total_members=5, sticker_count=0, avatar=None
    )
    assert "initial" in page and ">奶<" in page
