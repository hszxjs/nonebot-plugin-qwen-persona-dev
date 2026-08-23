"""辅助 LLM 通道加固测试：thinking 参数、400 回退、传输重试、容错 JSON 解析。"""

from __future__ import annotations

import json

import httpx
import pytest

from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.providers.aux_llm import (
    AuxLlmError,
    aux_chat,
    aux_chat_json,
    parse_json_object,
)


class _Response:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "json": json})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok_response(content: str, finish: str = "stop", reasoning: str = "") -> _Response:
    message: dict = {"content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return _Response(payload={"choices": [{"message": message, "finish_reason": finish}]})


def _config() -> AiCompanionConfig:
    return AiCompanionConfig(
        ai_companion_online_api_url="https://aux.test/v1/chat/completions",
        ai_companion_online_api_key="k",
    )


async def test_aux_chat_sends_thinking_disabled_and_parses(monkeypatch) -> None:
    client = _FakeClient([_ok_response("好的")])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    reply = await aux_chat(_config(), prompt="p", model="m", max_tokens=8)
    assert reply == "好的"
    body = client.requests[0]["json"]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 8


async def test_aux_chat_empty_content_reports_reasoning_hint(monkeypatch) -> None:
    client = _FakeClient([_ok_response("", finish="length", reasoning="x" * 10)])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    with pytest.raises(AuxLlmError, match="reasoning 耗尽"):
        await aux_chat(_config(), prompt="p", model="m")


async def test_aux_chat_400_without_thinking_support_falls_back(monkeypatch) -> None:
    client = _FakeClient([_Response(status_code=400), _ok_response("ok")])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    reply = await aux_chat(_config(), prompt="p", model="m")
    assert reply == "ok"
    assert "thinking" in client.requests[0]["json"]
    assert "thinking" not in client.requests[1]["json"]


async def test_aux_chat_retries_transport_errors(monkeypatch) -> None:
    client = _FakeClient([httpx.ConnectError("boom"), _ok_response("recovered")])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    reply = await aux_chat(_config(), prompt="p", model="m")
    assert reply == "recovered"


async def test_aux_chat_transport_failure_raises_after_retries(monkeypatch) -> None:
    client = _FakeClient([httpx.ConnectError("down")] * 3)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    with pytest.raises(AuxLlmError, match="request failed"):
        await aux_chat(_config(), prompt="p", model="m")


async def test_aux_chat_json_retries_on_malformed_json(monkeypatch) -> None:
    bad = _ok_response('{"summary": "a" "b"}')  # 缺逗号
    client = _FakeClient([bad, _ok_response('{"summary": "good"}')])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    data = await aux_chat_json(_config(), prompt="p", model="m", attempts=2)
    assert data == {"summary": "good"}


def test_parse_json_object_variants() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_object('前置噪声 {"a": 3} 后置噪声') == {"a": 3}
    # 尾逗号容错
    assert parse_json_object('{"a": [1, 2,],}') == {"a": [1, 2]}
    with pytest.raises(AuxLlmError, match="json parse failed"):
        parse_json_object("完全不是 JSON")


# ---------------------------------------------------------------------------
# 伪工具调用文本提取（glm 在 FC 关闭时的"表演"输出）
# ---------------------------------------------------------------------------
def test_extract_pseudo_tool_reply_full_performance() -> None:
    from nonebot_plugin_ai_companion.prompt.text import extract_pseudo_tool_reply

    performed = (
        '**调用工具：memory_query**（query: "异环", limit: 3） '
        "工具返回：无相关记忆片段。 "
        "生成最终回复：异环就是NTE嘛，你咋突然想搜这个？"
    )
    assert extract_pseudo_tool_reply(performed) == "异环就是NTE嘛，你咋突然想搜这个？"


def test_extract_pseudo_tool_reply_passthrough_and_degenerate() -> None:
    from nonebot_plugin_ai_companion.prompt.text import extract_pseudo_tool_reply

    assert extract_pseudo_tool_reply("普通回复，无表演") == "普通回复，无表演"
    degenerate = "调用工具：web_search（query: x） 工具返回：y"  # 没有最终回复标记
    result = extract_pseudo_tool_reply(degenerate)
    assert isinstance(result, str)


def test_tools_prompt_segment_forbids_roleplay() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.prompt.builder import PromptContext, build_system_prompt
    from nonebot_plugin_ai_companion.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register("memory_query", description="查记忆")
    async def _tool(_arguments):
        return []

    prompt = build_system_prompt(PromptContext(config=AiCompanionConfig(), tool_registry=registry, command_prefixes=[]))
    assert "由系统自动执行" in prompt
    assert "不要输出「调用工具」" in prompt
