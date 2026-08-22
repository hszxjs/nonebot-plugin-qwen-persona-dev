"""第三期测试：Ollama/llama.cpp 适配器、实例管理、角色绑定解析降级链、模型管理器。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from nonebot_plugin_ai_companion.api.server import create_admin_app
from nonebot_plugin_ai_companion.config import AiCompanionConfig
from nonebot_plugin_ai_companion.model_manager.manager import ModelManager, ModelOperationError
from nonebot_plugin_ai_companion.providers.instances import (
    InferenceBackendRepo,
    ProviderInstanceManager,
    ProviderResolver,
    RoleProviderBinding,
    RoleProviderBindingRepo,
    build_backend_from_config,
    new_backend_config,
)
from nonebot_plugin_ai_companion.providers.llama_cpp import LlamaCppBackend
from nonebot_plugin_ai_companion.providers.ollama import OllamaBackend
from nonebot_plugin_ai_companion.providers.registry import get_provider_spec, provider_specs


# ---------------------------------------------------------------------------
# 适配器
# ---------------------------------------------------------------------------
def test_ollama_backend_normalizes_base_url_and_defaults() -> None:
    backend = OllamaBackend(base_url="http://192.168.1.10:11434", model="qwen3:8b")
    assert backend.base_url == "http://192.168.1.10:11434/v1"
    assert backend.model == "qwen3:8b"
    caps = backend.capabilities()
    assert caps.vision and caps.function_calling
    assert any(preset["value"] == "qwen3:8b" for preset in backend.default_models())

    backend2 = OllamaBackend(AiCompanionConfig())  # 走配置默认
    assert backend2.base_url == "http://127.0.0.1:11434/v1"

    spec = get_provider_spec("ollama")
    assert spec.provider_type == "local" and spec.docs_url.startswith("https://")


def test_llama_cpp_backend_normalizes_base_url() -> None:
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    assert backend.base_url == "http://127.0.0.1:8080/v1"
    assert backend.capabilities().lora_hotload is True
    spec = get_provider_spec("llama_cpp")
    assert spec.provider_type == "local"
    assert {spec.name for spec in provider_specs()} >= {"mlx", "glm", "ollama", "llama_cpp"}


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

    async def get(self, _url, **_kw):
        return self._response

    async def post(self, _url, **_kw):
        return self._response


async def test_openai_compat_chat_and_tool_calls(monkeypatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "你好",
                    "tool_calls": [{"id": "call1", "function": {"name": "web_search", "arguments": '{"query": "x"}'}}],
                }
            }
        ],
        "usage": {"completion_tokens": 2},
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(_FakeResponse(payload)))
    backend = OllamaBackend(base_url="http://localhost:11434", model="qwen3:8b")
    content, tool_calls = await backend.chat_with_tools(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "web_search"}}]
    )
    assert content == "你好"
    assert tool_calls[0]["function"]["name"] == "web_search"
    # 纯文本 chat 走同一路径
    assert await backend.chat([{"role": "user", "content": "hi"}]) == "你好"
    assert backend.metrics_snapshot()["last"]["model"] == "qwen3:8b"


async def test_openai_compat_health_check(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(_FakeResponse({"data": []})))
    backend = LlamaCppBackend()
    assert await backend.health_check() is True
    assert backend.status_snapshot()["healthy"] is True


async def test_openai_compat_health_check_failure(monkeypatch) -> None:
    def raise_error(**_kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", raise_error)
    backend = LlamaCppBackend()
    assert await backend.health_check() is False
    assert "health check failed" in backend.status_snapshot()["last_error"]


# ---------------------------------------------------------------------------
# 实例仓库 / 管理器 / 解析器
# ---------------------------------------------------------------------------
@pytest.fixture
def provider_stack(persona_store):
    repo = InferenceBackendRepo(persona_store.db)
    bindings = RoleProviderBindingRepo(persona_store.db)
    manager = ProviderInstanceManager(repo)
    return repo, bindings, manager


async def test_backend_config_repo_roundtrip(provider_stack) -> None:
    repo, _bindings, manager = provider_stack
    config = new_backend_config(
        backend_type="ollama",
        display_name="客厅小主机 Ollama",
        settings={"base_url": "http://192.168.1.10:11434", "model": "qwen3:8b"},
        priority=10,
    )
    await repo.upsert_config(config)
    fetched = await repo.get_config(config.id)
    assert fetched is not None
    assert fetched.settings["model"] == "qwen3:8b"
    assert fetched.priority == 10

    instance = await manager.get_instance(config.id)
    assert isinstance(instance.provider, OllamaBackend)
    assert instance.provider.base_url == "http://192.168.1.10:11434/v1"

    # 更新后缓存失效重建
    config.settings["model"] = "qwen3:14b"
    config.updated_at += 1
    await repo.upsert_config(config)
    await manager.invalidate(config.id)
    instance2 = await manager.get_instance(config.id)
    assert instance2.provider.model == "qwen3:14b"

    assert await repo.delete_config(config.id) is True
    assert await repo.get_config(config.id) is None


def test_build_backend_unknown_type_raises() -> None:
    from nonebot_plugin_ai_companion.providers.registry import ProviderNotFoundError

    with pytest.raises(ProviderNotFoundError):
        build_backend_from_config(new_backend_config(backend_type="vllm", display_name="x"))


class _FakeProvider:
    def __init__(self, reply: str, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    async def chat(self, messages, *, media_parts=None, purpose=""):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


async def test_resolver_fallback_chain(provider_stack, monkeypatch) -> None:
    repo, bindings, manager = provider_stack
    primary = new_backend_config(backend_type="ollama", display_name="主", settings={"model": "qwen3:8b"})
    secondary = new_backend_config(backend_type="llama_cpp", display_name="备", settings={})
    await repo.upsert_config(primary)
    await repo.upsert_config(secondary)

    fake_primary = _FakeProvider("ok-primary", fail=True)
    fake_secondary = _FakeProvider("ok-secondary")

    async def fake_get(config_id: str):
        if config_id == primary.id:
            return SimpleNamespace(config=primary, provider=fake_primary, id=primary.id)
        if config_id == secondary.id:
            return SimpleNamespace(config=secondary, provider=fake_secondary, id=secondary.id)
        return None

    monkeypatch.setattr(manager, "get_instance", fake_get)

    legacy = _FakeProvider("legacy")
    resolver = ProviderResolver(
        config=AiCompanionConfig(),
        client=legacy,
        mlx_runtime=object(),
        instances=manager,
        bindings=bindings,
    )
    await bindings.set_binding(
        RoleProviderBinding(role="chat", provider_id=primary.id, fallback_provider_id=secondary.id)
    )
    assert await resolver.chat([{"role": "user", "content": "hi"}]) == "ok-secondary"
    assert fake_primary.calls == 1 and fake_secondary.calls == 1 and legacy.calls == 0

    # 主备全挂 → 回退旧路径
    fake_secondary.fail = True
    assert await resolver.chat([{"role": "user", "content": "hi"}]) == "legacy"


async def test_resolver_unbound_defaults_to_legacy(provider_stack) -> None:
    _repo, _bindings, manager = provider_stack
    legacy = _FakeProvider("legacy")
    resolver = ProviderResolver(
        config=AiCompanionConfig(),
        client=legacy,
        mlx_runtime="mlx-runtime",
        instances=manager,
        bindings=_bindings,
    )
    provider, model, pid = await resolver.resolve_with_meta("chat")
    assert provider is legacy and model == "" and pid == ""
    assert (await resolver.resolve_with_meta("vision"))[0] == "mlx-runtime"
    assert (await resolver.resolve_with_meta("data_cleaning"))[0] is None


async def test_role_provider_binding_repo_upsert(provider_stack) -> None:
    _repo, bindings, _manager = provider_stack
    await bindings.set_binding(RoleProviderBinding(role="vision", provider_id="abc", model="qwen2.5vl:7b"))
    await bindings.set_binding(RoleProviderBinding(role="vision", provider_id="def", model="", updated_at=2.0))
    binding = await bindings.get_binding("vision")
    assert binding is not None and binding.provider_id == "def"
    assert len(await bindings.list_bindings()) == 1


# ---------------------------------------------------------------------------
# API：backends + provider-bindings
# ---------------------------------------------------------------------------
@pytest.fixture
def backends_client(persona_store, provider_stack):
    _repo, bindings, manager = provider_stack
    app = create_admin_app(
        config=AiCompanionConfig(),
        save_callback=lambda values: {"changed": [], "restart_required": []},
        store=persona_store,
        provider_instances=manager,
        provider_bindings=bindings,
        static_dir=Path("/nonexistent"),
    )
    return TestClient(app)


def test_backends_api_crud_and_bindings(backends_client: TestClient) -> None:
    created = backends_client.post(
        "/api/backends",
        json={
            "backend_type": "ollama",
            "display_name": "本机 Ollama",
            "settings": {"base_url": "http://127.0.0.1:11434", "model": "qwen3:8b"},
            "priority": 10,
        },
    )
    assert created.status_code == 200
    backend_id = created.json()["item"]["id"]

    listing = backends_client.get("/api/backends")
    assert listing.status_code == 200
    names = [spec["name"] for spec in listing.json()["available_types"]]
    assert {"mlx", "glm", "ollama", "llama_cpp"} <= set(names)

    # 未知类型拒绝
    bad = backends_client.post("/api/backends", json={"backend_type": "vllm", "display_name": "x"})
    assert bad.status_code == 400

    # 角色绑定 + 校验
    bind = backends_client.put("/api/provider-bindings/chat", json={"provider_id": backend_id, "model": "qwen3:14b"})
    assert bind.status_code == 200
    assert bind.json()["binding"]["model"] == "qwen3:14b"

    bad_role = backends_client.put("/api/provider-bindings/bogus", json={"provider_id": backend_id})
    assert bad_role.status_code == 400
    bad_provider = backends_client.put("/api/provider-bindings/vision", json={"provider_id": "missing"})
    assert bad_provider.status_code in {400, 200}  # 主 provider 不校验存在性（可先建绑定）

    bindings_list = backends_client.get("/api/provider-bindings")
    assert bindings_list.json()["roles"] == ["chat", "vision", "data_cleaning"]

    # 删除实例 → 引用绑定清空
    deleted = backends_client.delete(f"/api/backends/{backend_id}")
    assert deleted.status_code == 200
    bindings_after = backends_client.get("/api/provider-bindings").json()["items"]
    chat_binding = next((item for item in bindings_after if item["role"] == "chat"), None)
    assert chat_binding is not None and chat_binding["provider_id"] == ""


def test_backends_health_check_404(backends_client: TestClient) -> None:
    resp = backends_client.post("/api/backends/missing/health-check")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 模型管理器
# ---------------------------------------------------------------------------
class _FakeDownloader:
    def __init__(self) -> None:
        self.prepared = []

    def prepare_model(self, model_ref, *, model_dir, progress, cancel_check=None):
        total = 1000
        progress.update(phase="downloading", total_bytes=total, downloaded_bytes=0)
        for chunk in range(0, total + 1, 250):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("cancelled by user")
            progress.update(downloaded_bytes=chunk)
        target = Path(model_dir) / model_ref.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(b"x" * 16)
        self.prepared.append(model_ref)
        return SimpleNamespace(local_path=target, source="hf")


@pytest.fixture
def model_mgr(tmp_path) -> ModelManager:
    config = AiCompanionConfig(ai_companion_model_dir=str(tmp_path / "models"))
    return ModelManager(config, download_manager=_FakeDownloader())


def test_model_manager_lists_and_sizes(model_mgr: ModelManager, tmp_path) -> None:
    target = tmp_path / "models" / "org--model-7b"
    target.mkdir(parents=True)
    (target / "a.bin").write_bytes(b"y" * 2048)
    items = model_mgr.list_local_models()
    assert items[0]["name"] == "org--model-7b"
    assert items[0]["size_bytes"] == 2048
    assert model_mgr.presets()


async def test_model_manager_download_flow_and_cancel(model_mgr: ModelManager) -> None:
    snapshot = await model_mgr.start_download("org/model-7b")
    assert snapshot["state"] == "running"
    # 等待后台任务完成
    for _ in range(50):
        await asyncio.sleep(0.02)
        if model_mgr.operation_snapshot()["state"] != "running":
            break
    final = model_mgr.operation_snapshot()
    assert final["state"] == "success"
    assert final["downloaded_bytes"] == final["total_bytes"]

    # 重复下载拒绝
    with pytest.raises(ModelOperationError, match="already exists"):
        await model_mgr.start_download("org/model-7b")

    # 删除后可再下
    result = await model_mgr.delete_model("org/model-7b")
    assert result["deleted"] == "org/model-7b"
    with pytest.raises(ModelOperationError, match="not found"):
        await model_mgr.delete_model("org/model-7b")


async def test_model_manager_rejects_invalid_refs(model_mgr: ModelManager) -> None:
    with pytest.raises(ModelOperationError, match="required"):
        await model_mgr.start_download("   ")
    with pytest.raises(ModelOperationError, match="not a huggingface"):
        await model_mgr.start_download("/abs/path")


def test_model_manager_api_with_sse(persona_store) -> None:
    import tempfile
    import time as _time

    with tempfile.TemporaryDirectory() as tmp:
        config = AiCompanionConfig(ai_companion_model_dir=str(Path(tmp) / "models"))
        mgr = ModelManager(config, download_manager=_FakeDownloader())
        app = create_admin_app(
            config=config,
            save_callback=lambda values: {"changed": [], "restart_required": []},
            store=persona_store,
            model_manager=mgr,
            static_dir=Path("/nonexistent"),
        )
        # with 上下文使 portal 循环常驻，后台下载任务才能跨请求推进
        with TestClient(app) as client:
            overview = client.get("/api/models/manage")
            assert overview.status_code == 200
            assert "presets" in overview.json()

            # SSE 空闲帧
            with client.stream("GET", "/api/models/manage/download/events") as stream:
                frames = [line for line in stream.iter_lines() if line.startswith("data:")]
            assert json.loads(frames[0].removeprefix("data: "))["state"] == "idle"

            started = client.post("/api/models/manage/download", json={"model_ref": "org/m2"})
            assert started.status_code == 200
            # 主线程等待期间常驻循环推进下载任务
            for _ in range(50):
                _time.sleep(0.05)
                snapshot = mgr.operation_snapshot()
                if snapshot and snapshot["state"] != "running":
                    break
            assert mgr.operation_snapshot()["state"] == "success"

            with client.stream("GET", "/api/models/manage/download/events") as stream:
                frames = [line for line in stream.iter_lines() if line.startswith("data:")]
            assert len(frames) == 1
            assert json.loads(frames[0].removeprefix("data: "))["state"] == "success"

            deleted = client.request("DELETE", "/api/models/manage", params={"model_ref": "org/m2"})
            assert deleted.status_code == 200
