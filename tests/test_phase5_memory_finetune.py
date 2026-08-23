"""第五期测试：LoraAdapter 版本管理、长期记忆（提取/衰减/去重/注入）、ToolCallLog、KPI。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nonebot_plugin_ai_companion.api.server import create_admin_app
from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.finetune.adapters import LoraAdapterRepo, detect_language
from nonebot_plugin_ai_companion.memory.longterm import LongTermMemoryRepo, extract_memories
from nonebot_plugin_ai_companion.observability.tool_calls import ToolCallLogRepo


# ---------------------------------------------------------------------------
# 语言检测 / LoraAdapter
# ---------------------------------------------------------------------------
def test_detect_language() -> None:
    assert detect_language("今天天气不错") == "zh"
    assert detect_language("what a nice day today") == "en"
    assert detect_language("这个 feature 真的 nice") == "zh-mixed-en"
    assert detect_language("") == "unknown"


async def test_lora_adapter_versioning_and_rollback(persona_store) -> None:
    repo = LoraAdapterRepo(persona_store.db)
    v1 = await repo.register_adapter(role_card_id="card1", source_language="zh", path="/runs/v1", note="first")
    v2 = await repo.register_adapter(
        role_card_id="card1", source_language="zh", path="/runs/v2", note="second", activate=True
    )
    # 同卡不同语种独立版本序列
    v_en = await repo.register_adapter(role_card_id="card1", source_language="en", path="/runs/en-v1")
    assert (v1.version, v2.version, v_en.version) == (1, 2, 1)

    active = await repo.find_active(role_card_id="card1", source_language="zh")
    assert active is not None and active.id == v2.id

    # 回滚 → v1 激活
    rolled = await repo.rollback(role_card_id="card1", source_language="zh")
    assert rolled is not None and rolled.id == v1.id
    assert await repo.find_active(role_card_id="card1", source_language="en") is None

    # en 只有一版 → 无可回滚
    assert await repo.rollback(role_card_id="card1", source_language="en") is None

    # 删除
    assert await repo.delete_adapter(v2.id) is True
    assert len(await repo.list_adapters(role_card_id="card1")) == 2


# ---------------------------------------------------------------------------
# 长期记忆
# ---------------------------------------------------------------------------
async def test_memory_upsert_merge_and_prompt_lines(persona_store) -> None:
    repo = LongTermMemoryRepo(persona_store.db)
    first = await repo.upsert(user_id="42", kind="偏好", content="喜欢玩怪物猎人", importance=3.0)
    assert first["merged"] is False
    # 高度相似 → 合并（不新增），重要度取高
    merged = await repo.upsert(user_id="42", kind="偏好", content="喜欢玩怪物猎人这款游戏", importance=4.0)
    assert merged["merged"] is True
    items = await repo.list_memories(user_id="42")
    assert len(items) == 1 and items[0]["importance"] == 4.0

    await repo.upsert(user_id="42", kind="称呼", content="大家叫他老王", importance=2.0)
    lines = await repo.prompt_lines(user_id="42")
    assert any("偏好｜喜欢玩怪物猎人" in line for line in lines)
    assert any("称呼｜大家叫他老王" in line for line in lines)


async def test_memory_decay_marks_old_low_importance(persona_store) -> None:
    repo = LongTermMemoryRepo(persona_store.db)
    now = time.time()
    await repo.upsert(user_id="u1", kind="事实", content="很旧的琐碎事实", importance=1.0, created_at=now - 90 * 86400)
    await repo.upsert(user_id="u1", kind="重要事件", content="重要的近期事件", importance=5.0, created_at=now - 86400)
    result = await repo.decay(half_life_days=30.0)
    assert result["decayed"] == 1
    alive = await repo.list_memories(user_id="u1")
    assert len(alive) == 1 and "重要" in alive[0]["content"]


async def test_extract_memories_via_data_cleaning(persona_store, monkeypatch) -> None:
    payload = [{"kind": "偏好", "content": "喜欢二次元", "importance": 3}]

    async def fake_chat(prompt: str) -> str:
        return json.dumps(payload, ensure_ascii=False)

    for index in range(6):
        await persona_store.samples.record_sample(
            platform="qq",
            user_id="77",
            group_id="g",
            message_id=f"m{index}",
            plain_text=f"样本消息 {index}",
            training_text=f"样本消息 {index}",
            raw_segments=[],
            stickers=[],
            learn_reason="accepted",
        )
    written = await extract_memories(
        AiCompanionConfig(),
        persona_store.samples,
        user_id="77",
        memories=persona_store.long_term_memories,
        data_cleaning_chat=fake_chat,
    )
    assert len(written) == 1 and written[0]["content"] == "喜欢二次元"
    stored = await persona_store.long_term_memories.list_memories(user_id="77")
    assert stored[0]["kind"] == "偏好"


async def test_extract_memories_requires_samples(persona_store) -> None:
    from nonebot_plugin_ai_companion.providers.aux_llm import AuxLlmError

    with pytest.raises(AuxLlmError, match="not enough samples"):
        await extract_memories(
            AiCompanionConfig(),
            persona_store.samples,
            user_id="nobody",
            memories=persona_store.long_term_memories,
        )


# ---------------------------------------------------------------------------
# ToolCallLog
# ---------------------------------------------------------------------------
async def test_tool_call_log_stats(persona_store) -> None:
    repo = ToolCallLogRepo(persona_store.db)
    await repo.record(name="web_search", ok=True, latency_seconds=0.5, arguments={"query": "x"}, result=[{"t": 1}])
    await repo.record(name="web_search", ok=False, latency_seconds=1.5, arguments={"query": "y"}, result=None)
    await repo.record(name="memory_query", ok=True, latency_seconds=0.2)
    stats = await repo.success_stats(days=7)
    assert stats["total_calls"] == 3
    assert stats["overall_success_rate"] == pytest.approx(2 / 3, abs=0.01)
    by_name = {item["name"]: item for item in stats["items"]}
    assert by_name["web_search"]["ok"] == 1 and by_name["web_search"]["total"] == 2


# ---------------------------------------------------------------------------
# KPI / 会话 / 记忆 / adapter API
# ---------------------------------------------------------------------------
@pytest.fixture
def phase5_client(persona_store):
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        static_dir=Path("/nonexistent"),
        proactive_sender=None,
    )
    return TestClient(app)


async def test_kpi_dashboard_aggregates(phase5_client, persona_store) -> None:
    await persona_store.message_logs.record(
        platform="qq",
        group_id="g1",
        user_id="u1",
        trigger_message_id="t1",
        reply_message_id="r1",
        trigger_reason="force_reply",
        latency_seconds=1.0,
    )
    await persona_store.tool_call_logs.record(name="web_search", ok=True, latency_seconds=0.3)
    resp = phase5_client.get("/api/dashboard/kpi")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_replies"] >= 1
    assert {"reason": "force_reply", "count": 1} in data["reply_reasons"]
    assert data["tools"]["total_calls"] >= 1
    assert data["active_sessions"] >= 1
    assert "backend" in data


async def test_sessions_list_aggregates_by_group(phase5_client, persona_store) -> None:
    await persona_store.message_logs.record(
        platform="qq",
        group_id="g1",
        user_id="u1",
        trigger_message_id="t1",
        reply_message_id="r1",
        trigger_reason="random_reply",
    )
    await persona_store.message_logs.record(
        platform="qq",
        group_id="g1",
        user_id="u2",
        trigger_message_id="t2",
        reply_message_id="r2",
        trigger_reason="random_reply",
    )
    resp = phase5_client.get("/api/sessions")
    assert resp.status_code == 200
    items = {item["session_key"]: item for item in resp.json()["items"]}
    assert items["g1"]["count"] == 2


def test_sessions_poke_requires_sender(phase5_client) -> None:
    resp = phase5_client.post("/api/sessions/poke", json={"group_id": "123"})
    assert resp.status_code == 503


async def test_memories_and_adapters_api(phase5_client, persona_store) -> None:
    # 记忆
    created = phase5_client.post("/api/memories/extract", json={"user_id": "404"})
    assert created.status_code == 502  # 样本不足

    await persona_store.long_term_memories.upsert(user_id="9", kind="偏好", content="喜欢猫", importance=3.0)
    listing = phase5_client.get("/api/memories", params={"user_id": "9"})
    assert listing.status_code == 200 and listing.json()["total"] == 1
    memory_id = listing.json()["items"][0]["id"]

    decayed = phase5_client.post("/api/memories/decay")
    assert decayed.status_code == 200

    deleted = phase5_client.delete(f"/api/memories/{memory_id}")
    assert deleted.status_code == 200
    assert phase5_client.delete(f"/api/memories/{memory_id}").status_code == 404

    # adapter
    created = phase5_client.post(
        "/api/lora-adapters",
        json={"role_card_id": "c1", "source_language": "zh", "path": "/runs/a", "activate": True},
    )
    assert created.status_code == 200
    adapter_id = created.json()["item"]["id"]
    assert created.json()["item"]["version"] == 1

    phase5_client.post(
        "/api/lora-adapters",
        json={"role_card_id": "c1", "source_language": "zh", "path": "/runs/b", "activate": True},
    )
    rolled = phase5_client.post("/api/lora-adapters/rollback", json={"role_card_id": "c1", "source_language": "zh"})
    assert rolled.status_code == 200 and rolled.json()["item"]["id"] == adapter_id

    listing = phase5_client.get("/api/lora-adapters", params={"role_card_id": "c1"})
    assert listing.json()["total"] == 2
