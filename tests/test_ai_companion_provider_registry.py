"""backend_registry 注册/查找 + Provider 能力声明测试（架构文档十七.6）。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)

from nonebot_plugin_ai_companion.config import AiCompanionConfig  # noqa: E402
from nonebot_plugin_ai_companion.providers.registry import (  # noqa: E402
    ProviderNotFoundError,
    ProviderRegistrationError,
    create_provider,
    get_provider_spec,
    normalize_provider_name,
    provider_specs,
    register,
)


def test_mlx_and_online_llm_registered_with_types() -> None:
    mlx = get_provider_spec("mlx")
    llm = get_provider_spec("llm")
    assert mlx.provider_type == "local"
    assert llm.provider_type == "remote"
    assert {spec.name for spec in provider_specs()} >= {"mlx", "llm"}
    assert all(spec.docs_url == "" or spec.docs_url.startswith("http") for spec in provider_specs())


def test_glm_alias_resolves_to_llm() -> None:
    """旧配置/旧实例行里的 "glm" 必须继续解析到同一个后端。"""
    assert get_provider_spec("glm").name == "llm"
    assert create_provider("glm", AiCompanionConfig()).provider_name == "llm"
    assert normalize_provider_name("GLM") == "llm"
    assert normalize_provider_name("mlx") == "mlx"


def test_provider_specs_filter_by_type() -> None:
    locals_only = provider_specs(provider_type="local")
    remotes_only = provider_specs(provider_type="remote")
    assert [spec.name for spec in locals_only] == ["llama_cpp", "mlx", "ollama"]
    assert [spec.name for spec in remotes_only] == ["llm"]


def test_create_provider_builds_backends() -> None:
    config = AiCompanionConfig()
    mlx = create_provider("mlx", config)
    llm = create_provider("llm", config)
    assert mlx.provider_name == "mlx"
    assert llm.provider_name == "llm"


def test_unknown_provider_raises_with_available_hint() -> None:
    with pytest.raises(ProviderNotFoundError, match="vllm"):
        create_provider("vllm", AiCompanionConfig())


def test_duplicate_registration_rejected() -> None:
    with pytest.raises(ProviderRegistrationError, match="already registered"):

        @register("llm", provider_type="remote")
        class FakeOnlineLlm:
            pass


def test_invalid_provider_type_rejected() -> None:
    with pytest.raises(ProviderRegistrationError, match="provider_type"):
        register("weird", provider_type="hybrid")


def test_backend_capabilities_and_default_models() -> None:
    config = AiCompanionConfig()
    mlx = create_provider("mlx", config)
    llm = create_provider("llm", config)
    mlx_caps = mlx.capabilities()
    llm_caps = llm.capabilities()
    assert (mlx_caps.vision, mlx_caps.lora_hotload) == (True, True)
    assert llm_caps.lora_hotload is False
    mlx_models = mlx.default_models()
    llm_models = llm.default_models()
    assert mlx_models and all({"value", "label", "description"} <= set(m) for m in mlx_models)
    assert llm_models and all("value" in m for m in llm_models)


@pytest.mark.asyncio
async def test_backend_health_check_shape() -> None:
    config = AiCompanionConfig()
    mlx = create_provider("mlx", config)
    llm = create_provider("llm", config)
    assert isinstance(await mlx.health_check(), bool)
    assert isinstance(await llm.health_check(), bool)


def test_client_routes_via_registry() -> None:
    from nonebot_plugin_ai_companion.providers.client import QwenClient

    client = QwenClient(AiCompanionConfig())
    assert client.active_backend() == "mlx"
    spec = client.active_backend_spec()
    assert spec["name"] == "mlx" and spec["provider_type"] == "local"
    client.config = AiCompanionConfig(ai_companion_inference_backend="llm")
    assert client.active_backend_spec()["name"] == "llm"
    # 旧值 "glm"（历史配置/旧数据库行）仍应路由到在线后端
    client.config = AiCompanionConfig(ai_companion_inference_backend="glm")
    assert client.active_backend() == "llm"
    assert client.active_backend_spec()["name"] == "llm"
