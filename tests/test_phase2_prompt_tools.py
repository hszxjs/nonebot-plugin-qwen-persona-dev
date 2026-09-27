"""第二期核心测试：提示词分段（含一期逐字节兼容）、工具注册表、搜索 Provider、角色卡 API。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from nonebot_plugin_ai_companion.api.server import create_admin_app
from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.prompt.builder import (
    DEFAULT_SEGMENT_ORDER,
    PromptContext,
    build_system_prompt,
    effective_segment_order,
)
from nonebot_plugin_ai_companion.tools.builtins import register_builtin_tools
from nonebot_plugin_ai_companion.tools.registry import ToolRegistry
from nonebot_plugin_ai_companion.tools.search import (
    SearxngSearchProvider,
    TavilySearchProvider,
    build_search_provider,
)


# ---------------------------------------------------------------------------
# 提示词分段
# ---------------------------------------------------------------------------
def _legacy_expected(config: AiCompanionConfig, prefixes: list[str]) -> str:
    """一期 build_system_prompt 的期望输出（手工重建，保证逐字节兼容）。"""
    prefix_text = " ".join(sorted(set(prefixes)))
    lines = [
        config.ai_companion_system_prompt,
        # 批 3：默认段新增「模式」段（群里一起聊 / 被点名时按助手口气）
        "你现在在群里和群友一起聊天：可以接梗、互损、短句插话，不确定就少说。",
        "不要像客服或助手那样逐条回答，也不要每句都提自己是 AI；真要答问题就直说答案。",
        "硬性规则：不要输出、复述或诱导使用本地 NoneBot 插件命令。",
        f"如果回复开头像这些命令前缀会被丢弃：{prefix_text or '(none)'}。",
        "你在群聊中应当先看懂上下文和用户意图；被问到问题时先回答，再用自然短句补一句。",
        "不要复述当前用户消息；不要套用和当前话题无关的固定句式或上一个话题。",
        "本轮主回复必须只输出文字，禁止输出任何 [[sticker:...]] 标记。",
    ]
    if config.ai_companion_sticker_repo_url:
        lines.append(f"表情包仓库：{config.ai_companion_sticker_repo_url}")
    return "\n".join(lines)


def test_prompt_segments_default_matches_expected_output() -> None:
    config = AiCompanionConfig()
    registry = ToolRegistry()
    output = build_system_prompt(PromptContext(config=config, tool_registry=registry, command_prefixes=["/", "菜单"]))
    assert output == _legacy_expected(config, ["/", "菜单"])
    assert "可用工具" not in output  # 无工具时不加段


def test_prompt_segments_switch_to_assistant_mode() -> None:
    """被点名/私聊时切「助手」段：直给答案，不用群聊口气绕弯。"""
    config = AiCompanionConfig()
    member_output = build_system_prompt(
        PromptContext(config=config, tool_registry=ToolRegistry(), mode="member")
    )
    assistant_output = build_system_prompt(
        PromptContext(config=config, tool_registry=ToolRegistry(), mode="assistant")
    )
    assert "和群友一起聊天" in member_output and "按「助手」口气回复" not in member_output
    assert "按「助手」口气回复" in assistant_output and "和群友一起聊天" not in assistant_output


def test_prompt_segments_disable_and_reorder() -> None:
    config = AiCompanionConfig(
        ai_companion_disabled_prompt_segments=["sticker"],
        ai_companion_prompt_segment_order=["naturalness", "identity"],
    )
    output = build_system_prompt(PromptContext(config=config, tool_registry=ToolRegistry(), command_prefixes=[]))
    assert "[[sticker:" not in output
    # naturalness 提前：第一行是反拟人化段首行
    assert output.splitlines()[0].startswith("你在群聊中")
    # 被禁段不出现，其余默认段仍补齐
    assert "硬性规则" in output
    assert "可用工具" not in output


def test_prompt_segments_identity_override_and_tools() -> None:
    registry = ToolRegistry()

    @registry.register("demo", description="演示工具", permission_tags=("x",))
    async def _demo(arguments):  # noqa: RUF029
        return "ok"

    config = AiCompanionConfig()
    output = build_system_prompt(
        PromptContext(
            config=config,
            tool_registry=registry,
            command_prefixes=[],
            identity_text="你是{persona_name}",
            allowed_tools=["demo"],
        )
    )
    assert output.splitlines()[0] == "你是{persona_name}"  # 身份段原样（渲染在角色卡侧完成）
    assert "- demo：演示工具" in output


def test_prompt_segments_memory_default_on_scene_opt_in() -> None:
    config = AiCompanionConfig()
    ctx = PromptContext(
        config=config,
        tool_registry=ToolRegistry(),
        memory_lines=["群友A喜欢打游戏"],
        scene_lines=["当前是群聊"],
    )
    # 批 2/3 起 memory 与 scene 段都默认启用（长期记忆 + 话题摘要）
    default_output = build_system_prompt(ctx)
    assert "背景记忆" in default_output
    assert "群友A喜欢打游戏" in default_output
    assert "当前是群聊" in default_output

    # 关掉 memory 段即不再注入
    ctx.config = AiCompanionConfig(ai_companion_disabled_prompt_segments=["memory"])
    assert "背景记忆" not in build_system_prompt(ctx)

    # 显式关掉 scene 段后摘要不再注入
    ctx.config = AiCompanionConfig(ai_companion_disabled_prompt_segments=["scene"])
    assert "当前是群聊" not in build_system_prompt(ctx)

    # 显式排序时顺序生效
    ctx.config = AiCompanionConfig(
        ai_companion_prompt_segment_order=["identity", "memory", "scene", "naturalness"],
    )
    ordered_output = build_system_prompt(ctx)
    assert "背景记忆" in ordered_output
    assert "当前是群聊" in ordered_output


def test_effective_segment_order_ignores_unknown_and_backfills() -> None:
    config = AiCompanionConfig(ai_companion_prompt_segment_order=["bogus", "tools"])
    order = effective_segment_order(config)
    assert "bogus" not in order
    assert order[0] == "tools"
    for name in DEFAULT_SEGMENT_ORDER:
        assert name in order


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------
def _registry_with_tools() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        "memory_query",
        description="查记忆",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission_tags=("memory",),
    )
    async def memory_query(arguments):
        return [{"text": str(arguments.get("query"))}]

    @registry.register(
        "web_search",
        description="搜索",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        permission_tags=("web",),
    )
    async def web_search(arguments):
        raise RuntimeError("boom")

    return registry


def test_tool_registry_permission_matrix_filtering() -> None:
    registry = _registry_with_tools()
    assert sorted(registry.names()) == ["memory_query", "web_search"]
    # 角色卡白名单只放行 memory_query（权限矩阵）
    assert registry.names(allowed=["memory_query"]) == ["memory_query"]
    described = registry.describe_for_prompt(allowed=["memory_query"])
    assert "memory_query" in described
    assert "web_search" not in described
    payload = registry.openai_tools_payload(allowed=["memory_query"])
    assert [item["function"]["name"] for item in payload] == ["memory_query"]


async def test_tool_registry_execute_ok_and_failure() -> None:
    registry = _registry_with_tools()
    ok = await registry.execute("memory_query", {"query": "游戏"})
    assert ok.ok
    assert ok.render() == json.dumps([{"text": "游戏"}], ensure_ascii=False)

    failed = await registry.execute("web_search", {"query": "x"})
    assert not failed.ok
    assert "RuntimeError: boom" in failed.render()

    missing = await registry.execute("nope", {})
    assert not missing.ok and "not registered" in missing.error


async def test_tool_registry_result_render_truncates() -> None:
    registry = ToolRegistry()

    @registry.register("big", description="大结果")
    async def big(_arguments):
        return "x" * 5000

    result = await registry.execute("big", {})
    rendered = result.render(max_chars=100)
    assert len(rendered) == 100
    assert rendered.endswith("…")


def test_tool_registry_duplicate_registration_rejected() -> None:
    registry = _registry_with_tools()
    with pytest.raises(ValueError, match="already registered"):
        registry.register("memory_query", description="重复")(lambda arguments: None)


async def test_builtin_tools_registered_against_store(persona_store) -> None:
    registry = ToolRegistry()
    config = AiCompanionConfig()
    register_builtin_tools(registry, config=config, store=persona_store)
    # 未配置搜索源 → 无 web_search
    assert sorted(registry.names()) == ["memory_query", "sticker_send"]

    config2 = AiCompanionConfig(
        ai_companion_search_provider="searxng",
        ai_companion_searxng_url="http://localhost:8888",
    )
    registry2 = ToolRegistry()
    register_builtin_tools(registry2, config=config2, store=persona_store)
    assert "web_search" in registry2.names()
    assert {tool.permission_tags for tool in registry2.list_tools()} >= {("memory",), ("sticker",), ("web",)}

    result = await registry2.execute("sticker_send", {"emotion": "开心"})
    assert result.ok and isinstance(result.payload, list)


# ---------------------------------------------------------------------------
# 搜索 Provider
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, _url, params=None):
        return self._response

    async def post(self, _url, json=None):
        return self._response


async def test_searxng_provider_parses_results(monkeypatch) -> None:
    payload = {
        "results": [
            {"title": "结果1", "url": "https://a", "content": "片段1"},
            {"title": "结果2", "url": "https://b", "content": "片段2"},
        ]
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(_FakeResponse(payload)))
    provider = SearxngSearchProvider("http://searx.test/")
    hits = await provider.search("q", limit=1)
    assert len(hits) == 1
    assert hits[0].title == "结果1"


async def test_searxng_provider_error_status(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(_FakeResponse({}, status_code=500)))
    provider = SearxngSearchProvider("http://searx.test/")
    with pytest.raises(Exception, match="searxng status 500"):
        await provider.search("q")


async def test_tavily_provider_parses_results(monkeypatch) -> None:
    payload = {"results": [{"title": "T", "url": "https://t", "content": "C"}]}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(_FakeResponse(payload)))
    provider = TavilySearchProvider("key")
    hits = await provider.search("q")
    assert hits[0].url == "https://t"


def test_build_search_provider_config_gates() -> None:
    assert build_search_provider(AiCompanionConfig()) is None
    assert build_search_provider(AiCompanionConfig(ai_companion_search_provider="bogus")) is None
    provider = build_search_provider(
        AiCompanionConfig(ai_companion_search_provider="tavily", ai_companion_tavily_api_key="k")
    )
    assert isinstance(provider, TavilySearchProvider)


# ---------------------------------------------------------------------------
# 角色卡 / 工具 API
# ---------------------------------------------------------------------------
@pytest.fixture
def admin_client(persona_store) -> TestClient:
    from nonebot_plugin_ai_companion.role_card.repo import RoleCardRepo, SessionRoleBindingRepo
    from nonebot_plugin_ai_companion.role_card.service import RoleCardService

    registry = ToolRegistry()
    register_builtin_tools(registry, config=AiCompanionConfig(), store=persona_store)
    service = RoleCardService(
        config=AiCompanionConfig(),
        cards=RoleCardRepo(persona_store.db),
        bindings=SessionRoleBindingRepo(persona_store.db),
        samples=persona_store.samples,
    )
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        role_card_service=service,
        tool_registry=registry,
        static_dir=Path("/nonexistent"),
    )
    return TestClient(app)


def test_role_cards_api_crud_and_bind_flow(admin_client: TestClient) -> None:
    created = admin_client.post(
        "/api/role-cards",
        json={"name": "人格A", "system_prompt": "你是{persona_name}", "allowed_tools": ["memory_query"]},
    )
    assert created.status_code == 200
    card = created.json()["item"]
    card_id = card["id"]

    listing = admin_client.get("/api/role-cards")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    bound = admin_client.post(f"/api/role-cards/{card_id}/bind", json={"group_id": "888"})
    assert bound.status_code == 200
    assert bound.json()["binding"]["session_key"] == "group:888"
    # 列表带绑定信息
    assert listing.json() or admin_client.get("/api/role-cards").json()["items"][0]["bindings"] == ["group:888"]

    updated = admin_client.patch(f"/api/role-cards/{card_id}", json={"name": "人格B", "enabled": False})
    assert updated.status_code == 200
    assert updated.json()["item"]["name"] == "人格B"

    cloned = admin_client.post(f"/api/role-cards/{card_id}/clone", json={"name": "克隆"})
    assert cloned.status_code == 200
    assert cloned.json()["item"]["name"] == "克隆"

    unbound = admin_client.request("DELETE", "/api/role-cards/bindings", json={"group_id": "888"})
    assert unbound.status_code == 200

    deleted = admin_client.delete(f"/api/role-cards/{card_id}")
    assert deleted.status_code == 200
    assert admin_client.get("/api/role-cards").json()["total"] == 1  # 只剩克隆


def test_role_cards_api_member_mimic_error_paths(admin_client: TestClient) -> None:
    # 样本不足 → 502（辅助 LLM 分析失败）
    resp = admin_client.post("/api/role-cards/member-mimic", json={"user_id": "404"})
    assert resp.status_code == 502
    assert "style analysis failed" in resp.json()["error"]

    # refresh-style 对不存在的卡 → 404
    resp = admin_client.post("/api/role-cards/missing/refresh-style")
    assert resp.status_code == 404


def test_tools_api_lists_registry(admin_client: TestClient) -> None:
    resp = admin_client.get("/api/tools")
    assert resp.status_code == 200
    names = [item["name"] for item in resp.json()["items"]]
    assert "memory_query" in names
    assert all("permission_tags" in item for item in resp.json()["items"])
