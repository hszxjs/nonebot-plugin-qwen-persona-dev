from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class DummyRuntime:
    def status_snapshot(self) -> dict[str, object]:
        return {
            "runtime": "mlx_vlm",
            "state": "loaded",
            "healthy": True,
            "model": "target-model",
            "loaded_model": "target-model",
            "adapter_path": "adapter",
            "loaded_adapter_path": "adapter",
            "last_error": "",
            "control": {"can_load": True, "can_unload": True, "can_reload": True},
            "metrics": {"last": None, "history": []},
        }

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        return [{"time": "now", "stream": "system", "message": f"tail={tail}"}]

    async def load(self) -> dict[str, object]:
        return self.status_snapshot() | {"status": "loaded"}

    async def reload(self) -> dict[str, object]:
        return self.status_snapshot() | {"status": "reloaded"}

    async def unload(self) -> dict[str, object]:
        return self.status_snapshot() | {"status": "unloaded", "state": "not_loaded"}


class DummyStore:
    async def dashboard_stats(self) -> dict[str, object]:
        return {
            "totals": {"learned_messages": 10, "trainable_samples": 8},
            "training": {"last_success": None, "last_run": None, "pending_since": None},
            "charts": {"learning_trend": [], "group_contribution": []},
        }

    async def list_inference_events(self, *, limit: int = 300, offset: int = 0) -> dict[str, object]:
        return {
            "items": [{"event_id": "evt-1", "status": "success"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def list_stickers(self, **_kwargs: object) -> dict[str, object]:
        return {"items": [], "total": 0, "limit": 80, "offset": 0, "query": "", "type": "", "type_counts": {}}

    async def update_sticker_summary(self, key: str, summary: str) -> dict[str, object]:
        return {"updated": 1, "item": {"key": key, "summary": summary}}

    async def delete_stickers(self, keys: list[str]) -> dict[str, int]:
        return {"requested": len(keys), "deleted": len(keys)}


class DummyTrainer:
    def progress_snapshot(self) -> dict[str, object]:
        return {
            "stage": "idle",
            "status": "",
            "reason": "",
            "sample_count": 0,
            "dataset_path": "",
            "output_dir": "",
            "log_tail": "training idle\nwaiting for samples",
            "started_at": None,
            "finished_at": None,
        }


class WebuiRebuildApiTests(unittest.TestCase):
    def make_client(
        self,
        *,
        static_dir: Path | None = None,
        saved: list[dict[str, object]] | None = None,
        **config_overrides: object,
    ) -> TestClient:
        from nonebot_plugin_qwen_persona.admin_server import create_admin_app
        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig

        config = QwenPersonaConfig(
            qwen_persona_admin_token="secret",
            **config_overrides,
        )

        def save_callback(values: dict[str, object]) -> dict[str, object]:
            if saved is not None:
                saved.append(values)
            return {"changed": list(values), "restart_required": list(values)}

        app = create_admin_app(
            config=config,
            save_callback=save_callback,
            mlx_runtime=DummyRuntime(),
            store=DummyStore(),
            trainer=DummyTrainer(),
            qwen_client=SimpleNamespace(metrics_snapshot=lambda: {"last": None, "history": []}),
            static_dir=static_dir,
        )
        return TestClient(app)

    def test_operations_overview_includes_recent_inference_events(self) -> None:
        response = self.make_client().get("/api/operations/overview?token=secret")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runtime"]["state"], "loaded")
        self.assertEqual(payload["inference_events"]["limit"], 300)
        self.assertEqual(payload["inference_events"]["items"][0]["event_id"], "evt-1")

    def test_reload_sensitive_setting_requires_confirmation(self) -> None:
        saved: list[dict[str, object]] = []
        client = self.make_client(saved=saved)

        response = client.patch(
            "/api/settings/fields/qwen_persona_mlx_model?token=secret",
            json={"value": "new-model"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["confirmation_required"], True)
        self.assertEqual(saved, [])

    def test_settings_payload_is_value_contract_without_ui_metadata(self) -> None:
        response = self.make_client().get("/api/settings?token=secret")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("values", payload)
        self.assertIn("override_file", payload)
        self.assertNotIn("sections", payload)
        self.assertNotIn("fields", payload)
        self.assertEqual(
            payload["values"]["qwen_persona_mlx_model"],
            "mlx-community/Qwen3.5-9B-MLX-4bit",
        )

    def test_setting_update_rejects_unknown_and_invalid_values(self) -> None:
        client = self.make_client()

        unknown = client.patch(
            "/api/settings/fields/not_a_real_field?token=secret",
            json={"value": "x"},
        )
        invalid = client.patch(
            "/api/settings/fields/qwen_persona_max_tokens?token=secret",
            json={"value": "not-a-number"},
        )

        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(invalid.status_code, 400)

    def test_sticker_update_delete_and_diagnostics(self) -> None:
        client = self.make_client()

        updated = client.patch("/api/stickers/sticker-1?token=secret", json={"summary": "[肿么这样]"})
        deleted = client.request("DELETE", "/api/stickers?token=secret", json={"keys": ["sticker-1"]})
        diagnostics = client.get("/api/diagnostics?token=secret")

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(deleted.json()["deleted"], 1)
        self.assertIn("static", diagnostics.json())

    def test_static_host_uses_browser_router_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp)
            (static_dir / "assets").mkdir()
            (static_dir / "index.html").write_text("<main>webui</main>", encoding="utf-8")
            (static_dir / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
            client = self.make_client(static_dir=static_dir)

            asset = client.get("/assets/app.js")
            routed = client.get("/settings")

        self.assertEqual(asset.status_code, 200)
        self.assertEqual(routed.status_code, 200)
        self.assertIn("webui", routed.text)

    def test_training_logs_are_available_as_history_payload(self) -> None:
        response = self.make_client().get("/api/training/logs?token=secret")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["logs"][0]["message"], "training idle")

    def test_plugin_logs_tail_runtime_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "bot_runtime.log").write_text(
                "older\n07-12 plugin ready\n07-12 handled event\n",
                encoding="utf-8",
            )
            response = self.make_client(qwen_persona_data_dir=str(data_dir)).get(
                "/api/plugin/logs?tail=2&token=secret"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["message"] for item in response.json()["logs"]],
            ["07-12 plugin ready", "07-12 handled event"],
        )

    def test_runtime_actions_return_visible_operation_message(self) -> None:
        client = self.make_client()

        response = client.post("/api/runtime/load?token=secret", json={})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action"], "load")
        self.assertEqual(
            payload["message"],
            "load model 'mlx-community/Qwen3.5-9B-MLX-4bit'",
        )
        self.assertEqual(payload["runtime"]["status"], "loaded")


if __name__ == "__main__":
    unittest.main()
