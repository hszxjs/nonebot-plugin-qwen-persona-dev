from __future__ import annotations

import importlib
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

BackendWarningSink = Callable[[str, str, dict[str, Any]], None]

_QWEN35_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_QWEN35_SANITIZE_HOTFIX_ATTR = "_qwen_persona_qwen35_norm_sanitize_hotfixed"


class MlxBackend(Protocol):
    def load(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        trust_remote_code: bool,
    ) -> tuple[Any, Any]:
        ...

    def apply_chat_template(
        self,
        processor: Any,
        config: Any,
        prompt: str,
        *,
        num_images: int,
    ) -> str:
        ...

    def generate(
        self,
        model: Any,
        processor: Any,
        prompt: str,
        images: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Any:
        ...


class DefaultMlxBackend:
    def __init__(self, *, warning_sink: BackendWarningSink | None = None) -> None:
        self._warning_sink = warning_sink

    def load(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        trust_remote_code: bool,
    ) -> tuple[Any, Any]:
        from mlx_vlm import load

        patched_modules = _apply_qwen35_norm_sanitize_hotfix()
        if patched_modules:
            self._warn(
                "mlx_qwen35_norm_sanitize_hotfix_applied",
                "applied mlx-vlm Qwen3.5 norm sanitize hotfix",
                {"modules": patched_modules},
            )

        kwargs: dict[str, Any] = {}
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        try:
            return load(model_path, **kwargs)
        except TypeError:
            if "trust_remote_code" in kwargs:
                self._warn(
                    "mlx_load_trust_remote_code_unsupported",
                    "mlx-vlm load does not accept trust_remote_code; retrying without it",
                    {"model_path": model_path},
                )
            kwargs.pop("trust_remote_code", None)
            return load(model_path, **kwargs)

    def apply_chat_template(
        self,
        processor: Any,
        config: Any,
        prompt: str,
        *,
        num_images: int,
    ) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template

        return str(
            apply_chat_template(
                processor,
                config,
                prompt,
                num_images=num_images,
            )
        )

    def generate(
        self,
        model: Any,
        processor: Any,
        prompt: str,
        images: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Any:
        from mlx_vlm import generate

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "skip_special_tokens": True,
            "verbose": False,
        }
        if images:
            kwargs["image"] = images
        try:
            return generate(model, processor, prompt, **kwargs)
        except TypeError:
            self._warn(
                "mlx_generate_top_p_unsupported",
                "mlx-vlm generate does not accept top_p; retrying without it",
                {"model": getattr(getattr(model, "config", None), "model_type", "")},
            )
            kwargs.pop("top_p", None)
            return generate(model, processor, prompt, **kwargs)

    def _warn(self, event: str, message: str, context: dict[str, Any]) -> None:
        logger.warning("%s: %s", event, message)
        if self._warning_sink is not None:
            self._warning_sink(event, message, context)


def _apply_qwen35_norm_sanitize_hotfix() -> list[str]:
    try:
        from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key
    except Exception as exc:
        logger.debug("mlx_qwen35_norm_sanitize_hotfix_unavailable: %s", exc)
        return []

    patched_modules: list[str] = []
    for module_name in (
        "mlx_vlm.models.qwen3_5.qwen3_5",
        "mlx_vlm.models.qwen3_5_moe.qwen3_5_moe",
    ):
        try:
            module = importlib.import_module(module_name)
            if _patch_qwen35_norm_sanitize_module(module, sanitize_key):
                patched_modules.append(module_name)
        except Exception as exc:
            logger.warning(
                "mlx_qwen35_norm_sanitize_hotfix_failed: module=%s error=%s",
                module_name,
                exc,
            )
    return patched_modules


def _patch_qwen35_norm_sanitize_module(model_module: Any, sanitize_key: Callable[[str], str]) -> bool:
    model_cls = getattr(model_module, "Model", None)
    sanitize = getattr(model_cls, "sanitize", None)
    if sanitize is None or getattr(sanitize, _QWEN35_SANITIZE_HOTFIX_ATTR, False):
        return False
    if not _qwen35_norm_sanitize_needs_hotfix(sanitize):
        return False

    @wraps(sanitize)
    def patched_sanitize(self: Any, weights: dict[str, Any]) -> dict[str, Any]:
        mlx_norm_keys = {
            sanitize_key(key)
            for key, value in weights.items()
            if key.startswith("language_model.")
            and _is_qwen35_norm_key(sanitize_key(key))
            and getattr(value, "ndim", None) == 1
        }
        sanitized = sanitize(self, weights)
        for key in mlx_norm_keys:
            value = sanitized.get(key)
            if getattr(value, "ndim", None) == 1:
                sanitized[key] = value - 1.0
        return sanitized

    setattr(patched_sanitize, _QWEN35_SANITIZE_HOTFIX_ATTR, True)
    setattr(model_cls, "sanitize", patched_sanitize)
    return True


def _qwen35_norm_sanitize_needs_hotfix(sanitize: Callable[..., Any]) -> bool:
    try:
        source = inspect.getsource(sanitize)
    except (OSError, TypeError):
        return False
    return "value += 1.0" in source and "original_key" not in source


def _is_qwen35_norm_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _QWEN35_NORM_SUFFIXES)
