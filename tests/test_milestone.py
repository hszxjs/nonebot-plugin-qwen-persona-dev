"""群聊成就：阈值判定、统计查询、卡片 HTML。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


def test_is_milestone_matches_configured_and_round_numbers() -> None:
    from nonebot_plugin_ai_companion.notify.milestone import MILESTONE_DEFAULTS, is_milestone

    assert is_milestone(100, MILESTONE_DEFAULTS)
    assert is_milestone(6666, MILESTONE_DEFAULTS)
    assert not is_milestone(6665, MILESTONE_DEFAULTS)
    assert not is_milestone(150, MILESTONE_DEFAULTS)
    assert is_milestone(20000)  # >=10000 的整万自动触发
    assert is_milestone(10000)
    assert not is_milestone(15000)
    assert not is_milestone(0)
    assert not is_milestone(-5)


async def test_member_stats_counts_rank_and_stickers(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER

    async def add_turn(user_id: str, text: str, *, sticker: bool = False, created_at: float = 100.0) -> None:
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind=SPEAKER_MEMBER,
            text=text,
            group_id="100",
            user_id=user_id,
            sticker_keys=["face:x"] if sticker else [],
            created_at=created_at,
        )
        await persona_store.conversation.record_turn(
            session_key="private:9",
            speaker_kind=SPEAKER_MEMBER,
            text="私聊不该算进群计数",
            group_id=None,
            user_id=user_id,
            created_at=created_at,
        )

    # 目标用户 6 条（2 条表情包）；另一人 8 条；bot 的话不算
    for index in range(4):
        await add_turn("42", f"普通消息{index}", created_at=100.0 + index)
    await add_turn("42", "[表情]", sticker=True, created_at=110.0)
    await add_turn("42", "[表情]", sticker=True, created_at=111.0)
    for index in range(8):
        await add_turn("77", f"别人{index}", created_at=100.0 + index)
    await persona_store.conversation.record_turn(
        session_key="group:100", speaker_kind="bot", text="bot 的话", group_id="100", created_at=120.0
    )

    from nonebot_plugin_ai_companion.notify.milestone import fetch_member_stats

    stats = await fetch_member_stats(persona_store, group_id="100", user_id="42")
    assert stats is not None
    assert stats["total"] == 6
    assert stats["sticker_count"] == 2
    assert stats["rank"] == 2  # 77 有 8 条排第一
    assert stats["total_members"] == 2


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
