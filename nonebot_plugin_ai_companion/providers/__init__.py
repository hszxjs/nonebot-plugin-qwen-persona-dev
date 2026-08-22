"""推理来源域：本地/在线推理后端、采样档位、输出安全与 token 预算。"""

from .base import BaseProvider, ProviderCapabilities
from .client import QwenClient
from .generation_profiles import GenerationProfile, GenerationPurpose, profile_for_purpose
from .glm_backend import GlmBackend
from .llama_cpp import LlamaCppBackend
from .mlx_backend import DefaultMlxBackend, MlxBackend
from .mlx_runtime import MlxVLMRuntime
from .ollama import OllamaBackend
from .registry import (
    ProviderNotFoundError,
    ProviderRegistrationError,
    ProviderSpec,
    create_provider,
    get_provider_spec,
    provider_specs,
    register,
)

__all__ = [
    "QwenClient",
    "GlmBackend",
    "MlxVLMRuntime",
    "OllamaBackend",
    "LlamaCppBackend",
    "DefaultMlxBackend",
    "MlxBackend",
    "BaseProvider",
    "ProviderCapabilities",
    "GenerationProfile",
    "GenerationPurpose",
    "profile_for_purpose",
    "ProviderNotFoundError",
    "ProviderRegistrationError",
    "ProviderSpec",
    "create_provider",
    "get_provider_spec",
    "provider_specs",
    "register",
]
