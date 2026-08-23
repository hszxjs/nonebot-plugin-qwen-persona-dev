"""第四期测试：MessageLog 溯源、引用命令助手、事件总线、统一日志 API、黑名单测试端点。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nonebot_plugin_ai_companion.api.server import create_admin_app
from nonebot_plugin_ai_companion.bot_env import BotEnvironment
from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.observability.bus import EventBus, _Subscriber
from nonebot_plugin_ai_companion.observability.message_log import (
    MessageLogRepo,
    format_trace,
    quoted_message_id,
)


# ---------------------------------------------------------------------------
# MessageLogRepo
# ---------------------------------------------------------------------------
async def test_message_log_record_and_lookup_by_both_ids(persona_store) -> None:
    repo = MessageLogRepo(persona_store.db)
    item = await repo.record(
        platform="qq",
        group_id="123",
        user_id="42",
        trigger_message_id="T-100",
        reply_message_id="R-200",
        reply_kind="chat",
        trigger_reason="force_reply",
        decision_detail={"reason": "force_reply", "score": 0.9},
        role_card_id="card1",
        role_card_name="傲娇人格",
        provider="glm",
        model="glm-5.2",
        latency_seconds=1.25,
        sticker_keys=["face:abc"],
        input_text="在吗",
        output_text="干嘛",
    )
    assert item["trigger_message_id"] == "T-100"

    by_reply = await repo.find_by_reply_message_id("R-200")
    assert by_reply is not None
    assert by_reply["role_card_name"] == "傲娇人格"
    assert by_reply["decision_detail"]["score"] == 0.9
    assert by_reply["sticker_keys"] == ["face:abc"]
    assert by_reply["time"]

    by_trigger = await repo.find_by_trigger_message_id("T-100")
    assert by_trigger is not None and by_trigger["reply_message_id"] == "R-200"

    assert await repo.find_by_reply_message_id("missing") is None


async def test_message_log_ring_buffer_eviction(persona_store) -> None:
    repo = MessageLogRepo(persona_store.db, max_records=100)
    repo.set_max_records(3)
    for index in range(6):
        await repo.record(
            platform=None,
            group_id="g",
            user_id=None,
            trigger_message_id=f"T{index}",
            reply_message_id=f"R{index}",
        )
    payload = await repo.list_logs(group_id="g")
    assert payload["total"] == 3
    ids = [item["trigger_message_id"] for item in payload["items"]]
    assert ids == ["T5", "T4", "T3"]


async def test_message_log_list_kind_filter(persona_store) -> None:
    repo = MessageLogRepo(persona_store.db)
    await repo.record(
        platform=None,
        group_id=None,
        user_id=None,
        trigger_message_id="t1",
        reply_message_id="r1",
        reply_kind="chat",
    )
    await repo.record(
        platform=None,
        group_id=None,
        user_id=None,
        trigger_message_id="t2",
        reply_message_id="r2",
        reply_kind="repeat",
    )
    chats = await repo.list_logs(kind="chat")
    assert [item["reply_kind"] for item in chats["items"]] == ["chat"]
    everything = await repo.list_logs(kind="all")
    assert everything["total"] == 2


def test_format_trace_renders_fields() -> None:
    text = format_trace(
        {
            "trigger_reason": "mention",
            "reply_kind": "chat",
            "role_card_name": "模仿卡",
            "provider": "instance:abc",
            "model": "qwen3:8b",
            "latency_seconds": 2.4,
            "sticker_keys": ["face:x"],
            "decision_detail": {"reason": "mention"},
            "output_preview": "哈哈",
        }
    )
    assert "mention" in text and "模仿卡" in text and "qwen3:8b" in text
    assert "2.4s" in text and "哈哈" in text


# ---------------------------------------------------------------------------
# 引用消息 ID 提取（/trace、/添加表情 共用）
# ---------------------------------------------------------------------------
class _FakeSegment:
    def __init__(self, type: str, data: dict) -> None:
        self.type = type
        self.data = data


class _FakeEvent:
    def __init__(self, reply=None, segments=()) -> None:
        self.reply = reply
        self._segments = list(segments)

    def get_message(self):
        return self._segments


def test_quoted_message_id_from_reply_object() -> None:
    event = _FakeEvent(reply=SimpleNamespace(message_id=10086))
    assert quoted_message_id(event) == "10086"


def test_quoted_message_id_from_reply_segment() -> None:
    event = _FakeEvent(
        reply=None, segments=[_FakeSegment("text", {"text": "/trace"}), _FakeSegment("reply", {"id": "555"})]
    )
    assert quoted_message_id(event) == "555"


def test_quoted_message_id_absent() -> None:
    assert quoted_message_id(_FakeEvent()) is None
    broken = SimpleNamespace(get_message=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert quoted_message_id(broken) is None


# ---------------------------------------------------------------------------
# 事件总线
# ---------------------------------------------------------------------------
async def test_event_bus_pubsub() -> None:
    bus = EventBus()
    async with bus.subscribe() as queue:
        bus.publish({"type": "message", "item": {"id": 1}})
        assert await asyncio.wait_for(queue.get(), timeout=1.0) == {"type": "message", "item": {"id": 1}}
    bus.publish({"type": "message"})  # 退订后发布不报错
    assert not bus._subscribers


def test_event_bus_drops_when_full() -> None:
    bus = EventBus()
    bus._subscribers.add(_Subscriber(asyncio.Queue(maxsize=1), asyncio.new_event_loop()))  # 满队列
    bus.publish({"a": 1})
    bus.publish({"a": 2})  # 不阻塞不抛错
    assert True


# ---------------------------------------------------------------------------
# API：统一日志 + SSE 503 + 黑名单测试
# ---------------------------------------------------------------------------
@pytest.fixture
def logs_client(persona_store):
    bus = EventBus()
    persona_store.attach_event_bus(bus)
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        event_bus=bus,
        env=BotEnvironment(SimpleNamespace(command_start=set(), nickname=set()), AiCompanionConfig()),
        static_dir=Path("/nonexistent"),
    )
    return TestClient(app)


async def test_unified_logs_merges_sources(logs_client, persona_store) -> None:
    await persona_store.message_logs.record(
        platform=None,
        group_id="g1",
        user_id=None,
        trigger_message_id="t1",
        reply_message_id="r1",
        reply_kind="chat",
    )
    await persona_store.record_sticker_event(event_type="classifier", status="accepted", group_id="g1", message="ok")
    resp = logs_client.get("/api/logs")
    assert resp.status_code == 200
    sources = {item["source"] for item in resp.json()["items"]}
    assert sources >= {"message", "sticker"}

    filtered = logs_client.get("/api/logs", params={"kinds": "message", "group_id": "g1"})
    assert {item["source"] for item in filtered.json()["items"]} == {"message"}

    other = logs_client.get("/api/logs", params={"kinds": "message", "group_id": "nope"})
    assert other.json()["total"] == 0


def test_logs_stream_requires_bus(persona_store) -> None:
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        event_bus=None,
        env=None,
        static_dir=Path("/nonexistent"),
    )
    client = TestClient(app)
    assert client.get("/api/logs/stream").status_code == 503


async def test_logs_stream_pushes_bus_events(persona_store) -> None:
    import httpx

    bus = EventBus()
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        event_bus=bus,
        env=None,
        static_dir=Path("/nonexistent"),
    )

    async def publish_soon() -> None:
        await asyncio.sleep(0.2)
        bus.publish({"type": "message", "item": {"reply_message_id": "R9"}})

    task = asyncio.create_task(publish_soon())
    frames: list[str] = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        async with ac.stream("GET", "/api/logs/stream", params={"max_seconds": 1.5}) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    frames.append(line)  # 收满有限流的全部 data 帧后流自动结束
    await task
    assert frames
    assert json.loads(frames[0].removeprefix("data: "))["item"]["reply_message_id"] == "R9"


def test_blacklist_test_endpoint(logs_client) -> None:
    resp = logs_client.post("/api/blacklist/test", json={"text": "这个 产品经理 真的离谱"})
    assert resp.status_code == 200
    # 默认配置无该词 → 未命中
    assert resp.json()["matched"] is False

    resp2 = logs_client.post("/api/blacklist/test", json={"text": "sfw安全词检测"})
    assert resp2.status_code == 200
    assert isinstance(resp2.json()["matches"], list)


def test_blacklist_matches_breakdown() -> None:
    env = BotEnvironment(
        SimpleNamespace(command_start=set(), nickname=set()),
        AiCompanionConfig(
            ai_companion_blacklist_words=["广告", "加微信"],
            ai_companion_blacklist_regexes=[r"折扣券?\d+"],
        ),
    )
    matches = env.blacklist_matches("来看折扣券99的广告")
    kinds = {(match["kind"], match["value"]) for match in matches}
    assert ("word", "广告") in kinds
    assert ("regex", r"折扣券?\d+") in kinds
    assert env.blacklist_matches("正常聊天") == []
