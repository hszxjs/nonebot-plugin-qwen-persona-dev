"""群记忆 / 成员记忆：解析、去重、敏感度分级注入、衰减、批量抽取调度。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


# ---- 解析 ----


def test_parse_memories_accepts_valid_and_fenced_json() -> None:
    from nonebot_plugin_ai_companion.memory.extractor import parse_memories

    raw = """好的，结果如下：
```json
{"memories": [
  {"scope": "group", "kind": "梗", "content": "打本指一起刷副本", "importance": 4, "sensitivity": "neutral"},
  {"scope": "member", "user_id": "42", "kind": "偏好", "content": "喜欢周末开黑", "importance": 2, "sensitivity": "relation"}
]}
```"""
    items = parse_memories(raw)
    assert [item["scope"] for item in items] == ["group", "member"]
    assert items[0]["content"] == "打本指一起刷副本"
    assert items[1]["user_id"] == "42"
    assert items[1]["sensitivity"] == "relation"


def test_parse_memories_drops_invalid_and_hallucinated_entries() -> None:
    from nonebot_plugin_ai_companion.memory.extractor import parse_memories

    raw = json.dumps(
        {
            "memories": [
                {"scope": "member", "kind": "偏好", "content": "没有 user_id 的成员条目"},  # 丢弃
                {"scope": "universe", "kind": "事实", "content": "未知作用域"},  # 丢弃
                {"scope": "group", "kind": "不存在的类别", "content": "类别回退", "importance": 99},
                {"scope": "group", "content": ""},  # 空内容丢弃
                "not-a-dict",
            ]
        }
    )
    items = parse_memories(raw)
    assert len(items) == 1
    assert items[0]["kind"] == "事实"  # 未知类别回退
    assert items[0]["importance"] == 5.0  # 夹到上限


def test_parse_memories_tolerates_garbage() -> None:
    from nonebot_plugin_ai_companion.memory.extractor import parse_memories

    assert parse_memories("") == []
    assert parse_memories("模型今天不想输出 JSON") == []
    assert parse_memories('{"memories": "not-a-list"}') == []


# ---- 仓库：写入 / 去重 / 分级注入 ----


async def test_memories_dedup_merges_importance(persona_store) -> None:
    repo = persona_store.persona_memory
    assert await repo.add_group_memory(group_id="100", kind="梗", content="打本指一起刷副本", importance=2) is True
    # 近义内容 + 同类别 → 合并：重要度取大、不新增行
    assert await repo.add_group_memory(group_id="100", kind="梗", content="打本指一起刷副本啦", importance=5) is False
    items = await repo.list_memories(scope="group", group_id="100")
    assert len(items) == 1
    assert items[0].importance == 5.0

    # 不同群互不影响
    assert await repo.add_group_memory(group_id="200", kind="梗", content="打本指一起刷副本", importance=1) is True
    assert len(await repo.list_memories(scope="group", group_id="200")) == 1


async def test_member_memory_requires_user_and_group(persona_store) -> None:
    repo = persona_store.persona_memory
    assert await repo.add_member_memory(group_id="100", user_id="", kind="偏好", content="喜欢开黑") is False
    assert await repo.add_group_memory(group_id="", kind="事实", content="没有群号的群记忆") is False
    assert await repo.add_member_memory(group_id="100", user_id="42", kind="偏好", content="喜欢开黑") is True
    # 空群号 = 私聊里形成的成员记忆（不属于任何群），允许写入
    assert await repo.add_member_memory(group_id="", user_id="42", kind="偏好", content="半夜爱点外卖") is True


async def test_group_injection_hides_sensitive_memories(persona_store) -> None:
    repo = persona_store.persona_memory
    await repo.add_group_memory(group_id="100", kind="事实", content="本群周末打本", importance=4, sensitivity="neutral")
    await repo.add_group_memory(group_id="100", kind="事件", content="群主上周失恋了", importance=5, sensitivity="emotion")
    await repo.add_group_memory(group_id="100", kind="关系", content="阿强和小红常互怼", importance=3, sensitivity="relation")
    await repo.add_member_memory(
        group_id="100", user_id="42", kind="偏好", content="喜欢开黑", importance=3, sensitivity="neutral"
    )
    await repo.add_member_memory(
        group_id="100", user_id="42", kind="事件", content="最近在闹分手", importance=5, sensitivity="private"
    )
    await persona_store.member_names.upsert_name(group_id="100", user_id="42", name="阿强")

    group_lines = await repo.prompt_lines_for_session(group_id="100", user_ids=["42"], include_sensitive=False)
    assert any("本群周末打本" in line for line in group_lines)
    assert any("阿强和小红常互怼" in line for line in group_lines)  # relation 允许进群聊
    assert all("失恋" not in line and "闹分手" not in line for line in group_lines)
    assert any("阿强的偏好" in line for line in group_lines)

    private_lines = await repo.prompt_lines_for_session(group_id="100", user_ids=["42"], include_sensitive=True)
    assert any("失恋" in line for line in private_lines)
    assert any("闹分手" in line for line in private_lines)


async def test_memory_edit_delete_and_purge(persona_store) -> None:
    repo = persona_store.persona_memory
    await repo.add_member_memory(group_id="100", user_id="42", kind="偏好", content="喜欢开黑", importance=2)
    await repo.add_member_memory(group_id="100", user_id="42", kind="称呼", content="外号叫强哥", importance=1)

    items = await repo.list_memories(scope="member", group_id="100", user_id="42")
    first = items[0]
    assert await repo.update_memory(scope="member", memory_id=first.id, content="喜欢通宵开黑", sensitivity="private")
    refreshed = await repo.list_memories(scope="member", group_id="100", user_id="42")
    assert any(item.content == "喜欢通宵开黑" and item.sensitivity == "private" for item in refreshed)

    assert await repo.delete_memory(scope="member", memory_id=first.id) is True
    assert await repo.purge_member(group_id="100", user_id="42") == 1
    assert await repo.list_memories(scope="member", group_id="100", user_id="42") == []


async def test_decay_marks_old_low_importance_memories(persona_store) -> None:
    import time

    repo = persona_store.persona_memory
    await repo.add_group_memory(group_id="100", kind="事实", content="很久以前的小事", importance=1, sensitivity="neutral")
    # 手动改老时间：半衰期 1 天 + 重要度 1 → 200 天后必然低于阈值
    with persona_store.db.connection() as conn:
        old = time.time() - 200 * 86400
        conn.execute("UPDATE group_memories SET created_at = ?, last_seen_at = ?", (old, old))

    result = await repo.decay(half_life_days=1.0, floor=0.4)
    assert result["decayed"] == 1
    assert await repo.list_memories(scope="group", group_id="100") == []
    assert len(await repo.list_memories(scope="group", group_id="100", include_decayed=True)) == 1


# ---- 抽取 ----


async def test_extract_session_saves_and_guards_hallucinated_members(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER
    from nonebot_plugin_ai_companion.memory.extractor import extract_session_memories

    for index, (user_id, name, text) in enumerate(
        [("42", "阿强", "周六晚上一起打本吗"), ("77", "小红", "来，我带新号")]
    ):
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind=SPEAKER_MEMBER,
            text=text,
            group_id="100",
            user_id=user_id,
            speaker_name=name,
            created_at=100.0 + index,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)

    async def fake_chat(_prompt: str) -> str:
        fake_chat.prompt = _prompt  # type: ignore[attr-defined]
        return json.dumps(
            {
                "memories": [
                    {"scope": "group", "kind": "约定", "content": "周六晚打本", "importance": 4},
                    {"scope": "member", "user_id": "M1", "kind": "偏好", "content": "喜欢打本", "importance": 3},
                    {"scope": "member", "user_id": "M9", "kind": "偏好", "content": "幻觉成员"},  # 批里没有 M9
                    {"scope": "member", "user_id": "999999", "kind": "偏好", "content": "编造 QQ 号"},  # 非编号也丢弃
                ]
            },
            ensure_ascii=False,
        )

    result = await extract_session_memories(
        session_key="group:100",
        group_id="100",
        turns=turns,
        names={"42": "阿强", "77": "小红"},
        memories=persona_store.persona_memory,
        data_cleaning_chat=fake_chat,
    )
    assert result == {"group_saved": 1, "member_saved": 1, "skipped": 2}
    assert len(await persona_store.persona_memory.list_memories(scope="group", group_id="100")) == 1
    member_items = await persona_store.persona_memory.list_memories(scope="member", group_id="100")
    # M1 映射回真实 QQ，且提示词里只出现编号
    assert [item.user_id for item in member_items] == ["42"]
    assert "M1（阿强）" in fake_chat.prompt
    assert "3305167706" not in fake_chat.prompt


async def test_extraction_round_respects_thresholds_and_disabled_groups(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER
    from nonebot_plugin_ai_companion.memory.extractor import run_memory_extraction_round

    for group_id in ("100", "200"):
        await persona_store.conversation.record_turn(
            session_key=f"group:{group_id}",
            speaker_kind=SPEAKER_MEMBER,
            text="随便聊聊",
            group_id=group_id,
            user_id="42",
            speaker_name="阿强",
            created_at=100.0,
        )

    calls: list[str] = []

    async def fake_chat(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"memories": [{"scope": "group", "kind": "事实", "content": "聊过天"}]})

    # 阈值没到（需要 30 条新 turn，且间隔未到 1800s）→ 不跑
    skipped = await run_memory_extraction_round(
        conversation=persona_store.conversation,
        memories=persona_store.persona_memory,
        names_repo=persona_store.member_names,
        data_cleaning_chat=fake_chat,
        min_new_turns=30,
        interval_seconds=1800.0,
        now=1000.0,
    )
    assert skipped == [] and calls == []

    # force（阈值=1、间隔=0）→ 跑，但跳过被禁用的群
    results = await run_memory_extraction_round(
        conversation=persona_store.conversation,
        memories=persona_store.persona_memory,
        names_repo=persona_store.member_names,
        data_cleaning_chat=fake_chat,
        min_new_turns=1,
        interval_seconds=0.0,
        now=1000.0,
        disabled_groups={"200"},
    )
    assert [item["session_key"] for item in results] == ["group:100"]
    assert results[0]["group_saved"] == 1
    assert len(calls) == 1

    # 再跑一次：group:100 进度已推进（不重复抽取）；group:200 上轮被禁用，这次补跑
    again = await run_memory_extraction_round(
        conversation=persona_store.conversation,
        memories=persona_store.persona_memory,
        names_repo=persona_store.member_names,
        data_cleaning_chat=fake_chat,
        min_new_turns=1,
        interval_seconds=0.0,
        now=2000.0,
    )
    assert [item["session_key"] for item in again] == ["group:200"]
    assert len(calls) == 2


async def test_turns_after_reads_incrementally(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER

    ids: list[int] = []
    for index in range(3):
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind=SPEAKER_MEMBER,
            text=f"第{index}条",
            group_id="100",
            user_id="42",
            created_at=100.0 + index,
        )
    turns = await persona_store.conversation.recent_turns(session_key="group:100", limit=10)
    ids = [int(turn.id) for turn in turns if turn.id is not None]

    incremental = await persona_store.conversation.turns_after(session_key="group:100", after_id=ids[1], limit=10)
    assert [turn.text for turn in incremental] == ["第2条"]


@pytest.mark.parametrize("sensitivity", ["neutral", "relation", "emotion", "private"])
async def test_memory_keeps_valid_sensitivity(persona_store, sensitivity: str) -> None:
    repo = persona_store.persona_memory
    await repo.add_group_memory(
        group_id="100", kind="事实", content=f"内容-{sensitivity}", importance=3, sensitivity=sensitivity
    )
    items = await repo.list_memories(scope="group", group_id="100")
    assert items[0].sensitivity == sensitivity


# ---- P0 回归：私聊记忆与检索作用域 ----


async def test_private_chat_memories_aggregate_across_groups(persona_store) -> None:
    repo = persona_store.persona_memory
    await repo.add_member_memory(group_id="100", user_id="42", kind="偏好", content="喜欢开黑", sensitivity="neutral")
    await repo.add_member_memory(group_id="200", user_id="42", kind="关系", content="和小红是同学", sensitivity="relation")
    await repo.add_member_memory(group_id="100", user_id="42", kind="事件", content="最近在闹分手", sensitivity="private")
    # 私聊作用域（group_id 空串）的条目
    await repo.add_member_memory(group_id="", user_id="42", kind="偏好", content="半夜爱点外卖", sensitivity="neutral")
    # 别人的记忆不该混进来
    await repo.add_member_memory(group_id="100", user_id="77", kind="偏好", content="不关他的事", sensitivity="neutral")

    lines = await repo.prompt_lines_for_private_member(user_id="42", include_sensitive=True)
    joined = "\n".join(lines)
    assert "你在群 100 认识的" in joined and "你在群 200 认识的" in joined
    assert "你们私聊里聊过的" in joined
    assert "最近在闹分手" in joined  # 私聊允许敏感类
    assert "不关他的事" not in joined

    # 不过滤敏感度时只有中性/关系
    conservative = "\n".join(await repo.prompt_lines_for_private_member(user_id="42", include_sensitive=False))
    assert "最近在闹分手" not in conservative


async def test_sample_search_is_session_scoped(persona_store) -> None:
    """群内只搜本群、私聊只搜自己的私聊，不再出现"私聊里冒出别的群的话"。"""
    for group_id, text in [("100", "群100的话"), ("200", "群200的话")]:
        await persona_store.samples.record_sample(
            platform="qq",
            user_id="42",
            group_id=group_id,
            message_id=f"{group_id}-1",
            plain_text=text,
            training_text=text,
            raw_segments=[],
            stickers=[],
            learn_reason="accepted",
        )
    await persona_store.samples.record_sample(
        platform="qq",
        user_id="42",
        group_id=None,
        message_id="dm-1",
        plain_text="私聊的话",
        training_text="私聊的话",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )

    in_group = await persona_store.search_context_samples(group_id="100", query="的话", limit=10)
    assert [item["plain_text"] for item in in_group] == ["群100的话"]

    in_private = await persona_store.search_context_samples(group_id=None, user_id="42", query="的话", limit=10)
    assert [item["plain_text"] for item in in_private] == ["私聊的话"]

    global_hits = await persona_store.search_context_samples(
        group_id=None, user_id="42", query="的话", limit=10, global_scope=True
    )
    assert len(global_hits) == 3  # 运维显式要求跨会话时才允许
