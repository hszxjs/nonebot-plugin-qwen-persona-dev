from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


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
    def test_qwen35_hotfix_preserves_converted_mlx_norm_weights(self) -> None:
        from nonebot_plugin_qwen_persona.mlx_backend import _patch_qwen35_norm_sanitize_module

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
        from nonebot_plugin_qwen_persona.mlx_backend import _patch_qwen35_norm_sanitize_module

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
        from nonebot_plugin_qwen_persona.mlx_backend import _patch_qwen35_norm_sanitize_module

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
        from nonebot_plugin_qwen_persona.mlx_backend import _patch_qwen35_norm_sanitize_module

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
