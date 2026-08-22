from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_ai_companion"
package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


class FakeArray:
    ndim = 1

    def __init__(self, value: float) -> None:
        self.value = value

    def __add__(self, other: float) -> "FakeArray":
        return FakeArray(self.value + other)

    def __sub__(self, other: float) -> "FakeArray":
        return FakeArray(self.value - other)


def identity_sanitize_key(key: str) -> str:
    return key


class MlxBackendHotfixTests(unittest.TestCase):
    def test_default_backend_forwards_repetition_controls_to_mlx_generate(self) -> None:
        from nonebot_plugin_ai_companion.mlx_backend import DefaultMlxBackend

        calls: list[dict[str, object]] = []

        def fake_generate(
            _model: object,
            _processor: object,
            _prompt: str,
            **kwargs: object,
        ) -> object:
            calls.append(kwargs)
            return SimpleNamespace(text="ok")

        fake_mlx_vlm = types.ModuleType("mlx_vlm")
        fake_mlx_vlm.generate = fake_generate  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"mlx_vlm": fake_mlx_vlm}):
            result = DefaultMlxBackend().generate(
                SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")),
                object(),
                "prompt",
                [],
                max_tokens=32,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.08,
                repetition_context_size=64,
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(calls[0]["repetition_penalty"], 1.08)
        self.assertEqual(calls[0]["repetition_context_size"], 64)

    def test_qwen35_hotfix_preserves_converted_mlx_norm_weights(self) -> None:
        from nonebot_plugin_ai_companion.mlx_backend import _patch_qwen35_norm_sanitize_module

        class BrokenModel:
            def sanitize(self, weights: dict[str, FakeArray]) -> dict[str, FakeArray]:
                sanitized = {}
                for key, value in weights.items():
                    if key.endswith(".input_layernorm.weight") and value.ndim == 1:
                        value += 1.0
                    sanitized[key] = value
                return sanitized

        module = types.SimpleNamespace(Model=BrokenModel)

        self.assertTrue(_patch_qwen35_norm_sanitize_module(module, identity_sanitize_key))
        out = BrokenModel().sanitize(
            {"language_model.model.layers.0.input_layernorm.weight": FakeArray(0.0)}
        )

        self.assertEqual(out["language_model.model.layers.0.input_layernorm.weight"].value, 0.0)

    def test_qwen35_hotfix_preserves_hf_norm_shift(self) -> None:
        from nonebot_plugin_ai_companion.mlx_backend import _patch_qwen35_norm_sanitize_module

        class BrokenModel:
            def sanitize(self, weights: dict[str, FakeArray]) -> dict[str, FakeArray]:
                sanitized = {}
                for key, value in weights.items():
                    if key.endswith(".input_layernorm.weight") and value.ndim == 1:
                        value += 1.0
                    sanitized[key] = value
                return sanitized

        module = types.SimpleNamespace(Model=BrokenModel)

        self.assertTrue(_patch_qwen35_norm_sanitize_module(module, identity_sanitize_key))
        out = BrokenModel().sanitize({"model.language_model.layers.0.input_layernorm.weight": FakeArray(0.0)})

        self.assertEqual(out["model.language_model.layers.0.input_layernorm.weight"].value, 1.0)

    def test_qwen35_hotfix_preserves_hf_text_only_norm_shift(self) -> None:
        from nonebot_plugin_ai_companion.mlx_backend import _patch_qwen35_norm_sanitize_module

        class BrokenModel:
            def sanitize(self, weights: dict[str, FakeArray]) -> dict[str, FakeArray]:
                sanitized = {}
                for key, value in weights.items():
                    if key.endswith(".input_layernorm.weight") and value.ndim == 1:
                        value += 1.0
                    sanitized[key] = value
                return sanitized

        module = types.SimpleNamespace(Model=BrokenModel)

        self.assertTrue(_patch_qwen35_norm_sanitize_module(module, identity_sanitize_key))
        out = BrokenModel().sanitize({"model.layers.0.input_layernorm.weight": FakeArray(0.0)})

        self.assertEqual(out["model.layers.0.input_layernorm.weight"].value, 1.0)

    def test_qwen35_hotfix_is_idempotent(self) -> None:
        from nonebot_plugin_ai_companion.mlx_backend import _patch_qwen35_norm_sanitize_module

        class BrokenModel:
            def sanitize(self, weights: dict[str, FakeArray]) -> dict[str, FakeArray]:
                sanitized = {}
                for key, value in weights.items():
                    if key.endswith(".input_layernorm.weight") and value.ndim == 1:
                        value += 1.0
                    sanitized[key] = value
                return sanitized

        module = types.SimpleNamespace(Model=BrokenModel)

        self.assertTrue(_patch_qwen35_norm_sanitize_module(module, identity_sanitize_key))
        self.assertFalse(_patch_qwen35_norm_sanitize_module(module, identity_sanitize_key))


if __name__ == "__main__":
    unittest.main()
