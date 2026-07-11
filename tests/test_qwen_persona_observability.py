from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

raise unittest.SkipTest(
    "Legacy llama.cpp/vision sidecar observability suite is retired by the MLX-VLM migration; "
    "see tests/test_qwen_persona_mlx_migration.py."
)

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)

from nonebot_plugin_qwen_persona.admin_web import ADMIN_HTML, AdminWebServer
from nonebot_plugin_qwen_persona.config import (
    LLAMA_MODEL_PRESETS,
    QwenPersonaConfig,
    effective_api_base,
    effective_chat_model,
)
from nonebot_plugin_qwen_persona.media import (
    context_text_from_training_text,
    prepare_message_learning,
    sanitize_model_reply_media_text,
)
from nonebot_plugin_qwen_persona.sticker import (
    build_onebot_v11_sticker_segment,
    sticker_to_record,
)
from nonebot_plugin_qwen_persona.model_manager import (
    ManagedLlamaProcess,
    ModelDeployResult,
    ModelManager,
)
from nonebot_plugin_qwen_persona.qwen_client import QwenClient
from nonebot_plugin_qwen_persona.storage import PersonaStore
from nonebot_plugin_qwen_persona.trainer import PersonaTrainer
from nonebot_plugin_qwen_persona.vision_manager import VisionManager


class DummyLogger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        pass

    def info(self, *_args: object, **_kwargs: object) -> None:
        pass

    def warning(self, *_args: object, **_kwargs: object) -> None:
        pass

    def exception(self, *_args: object, **_kwargs: object) -> None:
        pass


class ModelManagerObservabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_llama_model_presets_cover_small_medium_and_large_choices(self) -> None:
        values = {item["value"] for item in LLAMA_MODEL_PRESETS}

        self.assertGreaterEqual(len(LLAMA_MODEL_PRESETS), 10)
        self.assertIn("Qwen/Qwen3-0.6B-GGUF:Q8_0", values)
        self.assertIn("Qwen/Qwen3-4B-GGUF:Q4_K_M", values)
        self.assertIn("Qwen/Qwen3-14B-GGUF:Q4_K_M", values)
        self.assertIn("Qwen/Qwen3-32B-GGUF:Q4_K_M", values)

    async def test_external_llama_runtime_is_read_only_but_uses_llama_endpoint(self) -> None:
        config = QwenPersonaConfig(
            qwen_persona_runtime="external_llama",
            qwen_persona_llama_host="127.0.0.1",
            qwen_persona_llama_port=18080,
            qwen_persona_llama_model="Qwen/External-GGUF",
        )
        manager = ModelManager(config)

        async def healthy() -> bool:
            return True

        manager._service_healthy = healthy  # type: ignore[method-assign]

        snapshot = await manager.status_snapshot()
        result = await manager.ensure_ready(force=True)

        self.assertEqual(effective_api_base(config), "http://127.0.0.1:18080/v1")
        self.assertEqual(effective_chat_model(config), "Qwen/External-GGUF")
        self.assertEqual(snapshot["state"], "external_llama_ready")
        self.assertEqual(snapshot["control"]["can_stop"], False)
        self.assertEqual(snapshot["control"]["can_restart"], False)
        self.assertEqual(snapshot["control"]["can_reload"], False)
        self.assertEqual(snapshot["control"]["can_switch"], False)
        self.assertEqual(result.status, "external_llama_ready")

    async def test_status_exposes_read_only_external_service_controls(self) -> None:
        config = QwenPersonaConfig(qwen_persona_runtime="external_api")
        manager = ModelManager(config)

        snapshot = await manager.status_snapshot()

        self.assertEqual(snapshot["runtime"], "external_api")
        self.assertEqual(snapshot["control"]["can_start"], False)
        self.assertEqual(snapshot["control"]["can_stop"], False)
        self.assertIn("external_api", snapshot["state"])

    async def test_compatible_existing_llama_service_can_be_rechecked_from_ui(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_runtime="llama_cpp"))

        async def healthy() -> bool:
            return True

        manager._service_healthy = healthy  # type: ignore[method-assign]
        manager._port_accepts_tcp = lambda: False  # type: ignore[method-assign]

        snapshot = await manager.status_snapshot()

        self.assertEqual(snapshot["state"], "server_present")
        self.assertEqual(snapshot["control"]["can_start"], True)
        self.assertEqual(snapshot["control"]["can_stop"], False)
        self.assertEqual(snapshot["control"]["can_restart"], False)

    async def test_port_conflict_still_allows_start_attempt_for_diagnostics(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_runtime="llama_cpp"))

        async def unhealthy() -> bool:
            return False

        manager._service_healthy = unhealthy  # type: ignore[method-assign]
        manager._port_accepts_tcp = lambda: True  # type: ignore[method-assign]

        snapshot = await manager.status_snapshot()

        self.assertEqual(snapshot["state"], "port_in_use")
        self.assertEqual(snapshot["control"]["can_start"], True)
        self.assertEqual(snapshot["control"]["can_stop"], False)

    async def test_stop_does_not_stop_non_plugin_owned_service(self) -> None:
        manager = ModelManager(QwenPersonaConfig())
        manager._started_by_plugin = False
        manager._server_process = SimpleNamespace(returncode=None)

        result = await manager.stop()

        self.assertEqual(result.status, "server_not_owned")
        self.assertIsNotNone(manager._server_process)

    def test_logs_are_tail_limited_and_redact_api_keys(self) -> None:
        manager = ModelManager(
            QwenPersonaConfig(qwen_persona_api_key="super-secret-token")
        )

        manager._append_log("system", "llama --api-key super-secret-token --port 18080")
        manager._append_log("stderr", "ready")

        logs = manager.logs(tail=1)
        self.assertEqual(len(logs), 1)
        all_text = "\n".join(entry["message"] for entry in manager.logs(tail=20))
        self.assertNotIn("super-secret-token", all_text)
        self.assertIn("[redacted]", all_text)

    async def test_auto_install_decisions_are_written_to_logs(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_llama_auto_install=False))

        installed, reason = await manager._install_llama_cpp()

        self.assertEqual(installed, False)
        self.assertEqual(reason, "auto install disabled")
        log_text = "\n".join(entry["message"] for entry in manager.logs(tail=20))
        self.assertIn("auto install disabled", log_text)

    async def test_existing_service_logs_reported_model_ids(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_runtime="llama_cpp"))

        async def healthy() -> bool:
            return True

        async def service_models() -> dict[str, object]:
            return {"ok": True, "status_code": 200, "models": ["Qwen/Qwen3-0.6B-GGUF"]}

        manager._service_healthy = healthy  # type: ignore[method-assign]
        manager._service_models = service_models  # type: ignore[method-assign]

        result = await manager.ensure_ready(force=True)

        self.assertEqual(result.status, "server_present")
        log_text = "\n".join(entry["message"] for entry in manager.logs(tail=20))
        self.assertIn("models: Qwen/Qwen3-0.6B-GGUF", log_text)

    async def test_managed_orphan_is_claimable_from_status(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_runtime="llama_cpp"))

        async def healthy() -> bool:
            return True

        manager._service_healthy = healthy  # type: ignore[method-assign]
        manager._detect_managed_orphan = lambda: ManagedLlamaProcess(1234, "llama-server -hf model")  # type: ignore[method-assign]

        snapshot = await manager.status_snapshot()

        self.assertEqual(snapshot["state"], "server_orphaned")
        self.assertEqual(snapshot["pid"], 1234)
        self.assertEqual(snapshot["claimable_by_plugin"], True)
        self.assertEqual(snapshot["control"]["can_stop"], True)
        self.assertEqual(snapshot["control"]["can_restart"], True)
        self.assertEqual(snapshot["control"]["can_reload"], True)
        self.assertEqual(snapshot["control"]["can_switch"], True)

    async def test_stop_can_terminate_managed_orphan(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_runtime="llama_cpp"))
        stopped: list[int] = []
        manager._detect_managed_orphan = lambda: ManagedLlamaProcess(1234, "llama-server -hf model")  # type: ignore[method-assign]
        manager._terminate_pid = lambda pid: stopped.append(pid)  # type: ignore[method-assign]

        result = await manager.stop()

        self.assertEqual(result.status, "orphan_stopped")
        self.assertEqual(stopped, [1234])

    async def test_switch_model_updates_config_then_reloads(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_llama_model="old-model"))
        calls: list[str] = []

        async def reload_model() -> ModelDeployResult:
            calls.append(manager.config.qwen_persona_llama_model)
            return ModelDeployResult("llama_reloaded", str(manager.model_dir), repo_id=manager.config.qwen_persona_llama_model)

        manager.reload = reload_model  # type: ignore[method-assign]

        result = await manager.switch_model("new-model", "")

        self.assertEqual(manager.config.qwen_persona_llama_model, "new-model")
        self.assertEqual(manager.config.qwen_persona_llama_model_file, "")
        self.assertEqual(calls, ["new-model"])
        self.assertEqual(result.status, "llama_reloaded")

    def test_llama_log_parser_records_tps_metrics(self) -> None:
        manager = ModelManager(QwenPersonaConfig(qwen_persona_llama_model="Qwen/Test"))

        manager._append_log(
            "stderr",
            "eval time = 1200.00 ms / 48 runs (25.00 ms per token, 40.00 tokens per second)",
        )

        snapshot = manager.performance_snapshot()
        self.assertEqual(snapshot["last"]["source"], "llama_log")
        self.assertEqual(snapshot["last"]["tps"], 40.0)
        self.assertEqual(snapshot["last"]["model"], "Qwen/Test")


class QwenClientMetricsTests(unittest.TestCase):
    def test_usage_metrics_take_priority(self) -> None:
        client = QwenClient(QwenPersonaConfig(qwen_persona_runtime="external_api", qwen_persona_model="qwen-test"))

        client._record_metric({"usage": {"completion_tokens": 24}}, "hello", 2.0)

        snapshot = client.metrics_snapshot()
        self.assertEqual(snapshot["last"]["source"], "usage")
        self.assertEqual(snapshot["last"]["completion_tokens"], 24)
        self.assertEqual(snapshot["last"]["tps"], 12.0)
        self.assertEqual(snapshot["last"]["model"], "qwen-test")

    def test_metric_falls_back_to_estimated_tokens(self) -> None:
        client = QwenClient(QwenPersonaConfig())

        client._record_metric({}, "abcdefgh", 2.0)

        snapshot = client.metrics_snapshot()
        self.assertEqual(snapshot["last"]["source"], "estimated")
        self.assertEqual(snapshot["last"]["completion_tokens"], None)
        self.assertEqual(snapshot["last"]["tps"], 2.0)


class FakeCaptioner:
    def __init__(self, caption: str = "一只猫坐在桌上") -> None:
        self.caption = caption
        self.calls: list[dict[str, object]] = []

    async def caption_media(self, part: dict[str, object]) -> str:
        self.calls.append(part)
        return self.caption


class MediaPrecleaningTests(unittest.IsolatedAsyncioTestCase):
    async def test_onebot_cq_codes_are_classified_captioned_and_not_leaked_to_training_text(self) -> None:
        captioner = FakeCaptioner()

        payload = await prepare_message_learning(
            "看看这个 [CQ:image,file=secret.jpg,url=https://example.invalid/secret.jpg] [CQ:face,id=123]",
            sticker_segment_types={"face", "mface"},
            captioner=captioner,
            image_fallback_text="[图片]",
            sticker_fallback_text="[表情]",
        )

        self.assertEqual(payload["plain_text"], "看看这个")
        self.assertIn("图片：一只猫坐在桌上", payload["training_text"])
        self.assertIn("[[sticker:face:", payload["training_text"])
        self.assertNotIn("[CQ:", payload["training_text"])
        self.assertNotIn("secret.jpg", payload["training_text"])
        self.assertNotIn("https://example.invalid", payload["training_text"])
        self.assertEqual(payload["stickers"][0]["type"], "face")
        self.assertEqual(captioner.calls[0]["type"], "image")

    async def test_segment_objects_are_precleaned_and_images_do_not_enter_sticker_repo(self) -> None:
        payload = await prepare_message_learning(
            [
                {"type": "text", "data": {"text": "看看"}},
                {
                    "type": "image",
                    "data": {
                        "file": "photo.jpg",
                        "url": "https://example.invalid/photo.jpg",
                    },
                },
                {
                    "type": "mface",
                    "data": {
                        "emoji_id": "9",
                        "emoji_package_id": "1",
                        "summary": "笑",
                    },
                },
            ],
            sticker_segment_types={"face", "mface"},
            captioner=FakeCaptioner("有人在笑"),
            image_fallback_text="[图片]",
            sticker_fallback_text="[表情]",
        )

        self.assertEqual(payload["plain_text"], "看看")
        self.assertIn("图片：有人在笑", payload["training_text"])
        self.assertIn("表情：笑", payload["training_text"])
        self.assertNotIn("表情：有人在笑", payload["training_text"])
        self.assertEqual([item["type"] for item in payload["stickers"]], ["mface"])
        self.assertNotIn("photo.jpg", payload["training_text"])

    async def test_napcat_image_sticker_is_classified_as_sticker_not_plain_image(self) -> None:
        payload = await prepare_message_learning(
            [
                {
                    "type": "image",
                    "data": {
                        "summary": "[肿么这样]",
                        "file": "fd-fdcb5ee982505e0369399249cf5449f6.gif",
                        "url": "https://gxh.vip.qq.com/club/item/parcel/item/fd/fdcb/raw300.gif",
                        "key": "4bdca04b54",
                    },
                },
            ],
            sticker_segment_types={"face", "mface", "image"},
            captioner=FakeCaptioner("普通图片"),
            image_fallback_text="[图片]",
            sticker_fallback_text="[表情]",
        )

        self.assertEqual(payload["plain_text"], "")
        self.assertIn("表情：[肿么这样]", payload["training_text"])
        self.assertNotIn("图片：", payload["training_text"])
        self.assertEqual([item["type"] for item in payload["stickers"]], ["image"])

    def test_model_reply_media_codes_are_removed_but_sticker_tokens_survive(self) -> None:
        cleaned = sanitize_model_reply_media_text(
            "发这个 [CQ:image,file=base64://abcdef] [[image:https://example.invalid/a.png]] "
            "[[media:file:secret]] [[sticker:face:abc123]] 好了"
        )

        self.assertNotIn("[CQ:image", cleaned)
        self.assertNotIn("base64://", cleaned)
        self.assertNotIn("[[image:", cleaned)
        self.assertNotIn("[[media:", cleaned)
        self.assertIn("[[sticker:face:abc123]]", cleaned)
        self.assertIn("好了", cleaned)

    def test_prompt_context_strips_learning_only_media_descriptions(self) -> None:
        cleaned = context_text_from_training_text(
            "前面 图片：图片 表情：有人在笑 [[sticker:image:old123]] "
            "[[sticker:face:abc123]] 后面"
        )

        self.assertIn("前面", cleaned)
        self.assertIn("后面", cleaned)
        self.assertIn("[[sticker:face:abc123]]", cleaned)
        self.assertNotIn("图片：", cleaned)
        self.assertNotIn("表情：", cleaned)
        self.assertNotIn("[[sticker:image:", cleaned)

    def test_model_reply_media_description_artifacts_are_removed(self) -> None:
        self.assertEqual(sanitize_model_reply_media_text("图：图片（表情包）"), "")
        self.assertEqual(sanitize_model_reply_media_text("图片：图片"), "")
        self.assertEqual(sanitize_model_reply_media_text("表情：表情包"), "")
        self.assertEqual(
            sanitize_model_reply_media_text("这个图：图片（表情包）挺好笑"),
            "这个挺好笑",
        )


class StickerProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_regular_image_is_not_recorded_as_sticker_even_if_configured(self) -> None:
        record = sticker_to_record(
            {
                "type": "image",
                "data": {
                    "file": "normal-image.jpg",
                    "url": "https://example.invalid/image.jpg",
                },
            },
            {"image", "face", "mface"},
        )

        self.assertIsNone(record)

    def test_napcat_image_sticker_is_recorded_and_sent_as_onebot_image(self) -> None:
        from nonebot.adapters.onebot.v11 import MessageSegment

        record = sticker_to_record(
            {
                "type": "image",
                "data": {
                    "summary": "[肿么这样]",
                    "file": "fd-fdcb5ee982505e0369399249cf5449f6.gif",
                    "url": "https://gxh.vip.qq.com/club/item/parcel/item/fd/fdcb/raw300.gif",
                    "key": "4bdca04b54",
                },
            },
            {"image", "face", "mface"},
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["type"], "image")
        self.assertEqual(record["summary"], "[肿么这样]")
        segment = build_onebot_v11_sticker_segment(MessageSegment, record)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.type, "image")
        self.assertEqual(segment.data["file"], record["uri"])

    def test_onebot_segment_builder_skips_historical_images_but_keeps_face_and_mface(self) -> None:
        from nonebot.adapters.onebot.v11 import MessageSegment

        image = build_onebot_v11_sticker_segment(
            MessageSegment,
            {
                "type": "image",
                "data": {"file": "normal-image.jpg", "url": "https://example.invalid/image.jpg"},
            },
        )
        face = build_onebot_v11_sticker_segment(
            MessageSegment,
            {"type": "face", "data": {"id": "123"}},
        )
        mface = build_onebot_v11_sticker_segment(
            MessageSegment,
            {
                "type": "mface",
                "data": {
                    "emoji_id": "1",
                    "emoji_package_id": "2",
                    "key": "abc",
                    "summary": "ok",
                },
            },
        )

        self.assertIsNone(image)
        self.assertEqual(face.type, "face")
        self.assertEqual(face.data["id"], "123")
        self.assertEqual(mface.type, "mface")
        self.assertEqual(mface.data["emoji_id"], "1")


class MatcherArchitectureTests(unittest.TestCase):
    def test_learning_uses_late_matcher_instead_of_event_preprocessor(self) -> None:
        config = QwenPersonaConfig()
        source = (
            PLUGIN_DIR / "__init__.py"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(config.qwen_persona_learning_priority, 1000)
        self.assertGreaterEqual(config.qwen_persona_reply_priority, 1000)
        self.assertLess(
            config.qwen_persona_learning_priority,
            config.qwen_persona_reply_priority,
        )
        self.assertNotIn("event_preprocessor", source)
        self.assertIn("persona_learn = on_message", source)
        self.assertIn("qwen_persona_learning_priority", source)


class VisionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_vision_manager_reports_read_only_disabled_state(self) -> None:
        manager = VisionManager(QwenPersonaConfig(qwen_persona_vision_enabled=False))

        result = await manager.ensure_ready()
        snapshot = await manager.status_snapshot()

        self.assertEqual(result.status, "vision_disabled")
        self.assertEqual(snapshot["state"], "vision_disabled")
        self.assertEqual(snapshot["control"]["can_start"], False)

    async def test_existing_vision_service_is_used_without_starting_process(self) -> None:
        manager = VisionManager(
            QwenPersonaConfig(
                qwen_persona_vision_enabled=True,
                qwen_persona_vision_host="127.0.0.1",
                qwen_persona_vision_port=18081,
            )
        )

        async def healthy() -> bool:
            return True

        manager._service_healthy = healthy  # type: ignore[method-assign]

        result = await manager.ensure_ready()
        snapshot = await manager.status_snapshot()

        self.assertEqual(result.status, "vision_present")
        self.assertEqual(snapshot["state"], "vision_present")
        self.assertEqual(snapshot["control"]["can_stop"], False)

    def test_vision_manager_builds_builtin_service_command(self) -> None:
        manager = VisionManager(
            QwenPersonaConfig(
                qwen_persona_vision_enabled=True,
                qwen_persona_vision_python="/usr/bin/python3",
                qwen_persona_vision_host="127.0.0.1",
                qwen_persona_vision_port=18081,
                qwen_persona_vision_model="Salesforce/blip-image-captioning-base",
            )
        )

        command = manager._build_vision_command()

        self.assertEqual(command[0], "/usr/bin/python3")
        self.assertTrue(command[1].endswith("vision_service.py"))
        self.assertNotIn("-m", command)
        self.assertNotIn("nonebot_plugin_qwen_persona.vision_service", command)
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--port", command)
        self.assertIn("18081", command)
        self.assertIn("--model", command)
        self.assertIn("Salesforce/blip-image-captioning-base", command)

    async def test_caption_media_logs_request_source_and_caption(self) -> None:
        manager = VisionManager(
            QwenPersonaConfig(
                qwen_persona_vision_enabled=True,
                qwen_persona_vision_auto_start=False,
            )
        )

        async def post_caption(payload: dict[str, object]) -> dict[str, object]:
            return {"caption": "a surprised sticker", "source": "transformers"}

        manager._post_caption_payload = post_caption  # type: ignore[method-assign]

        caption = await manager.caption_media(
            {
                "type": "image",
                "data": {
                    "url": "https://example.invalid/sticker.gif",
                    "summary": "[肿么这样]",
                },
            }
        )
        messages = [item["message"] for item in manager.logs(tail=10)]

        self.assertEqual(caption, "a surprised sticker")
        self.assertTrue(any("caption request" in item for item in messages))
        self.assertTrue(any("source=transformers" in item for item in messages))
        self.assertTrue(any("a surprised sticker" in item for item in messages))


class VisionServiceTests(unittest.TestCase):
    def test_default_vision_model_is_image_to_text_not_clip_classifier(self) -> None:
        from nonebot_plugin_qwen_persona.vision_service import DEFAULT_VISION_MODEL

        self.assertNotIn("clip", DEFAULT_VISION_MODEL.lower())

    def test_caption_payload_fallback_never_leaks_media_identifiers(self) -> None:
        from nonebot_plugin_qwen_persona.vision_service import caption_payload

        result = caption_payload(
            {
                "type": "image",
                "url": "https://example.invalid/private-image.jpg",
                "data": {
                    "file": "private-image.jpg",
                    "url": "https://example.invalid/private-image.jpg",
                    "base64": "abcdef",
                },
            },
            model_name="",
            offline=True,
        )

        self.assertIn("caption", result)
        self.assertIn("图片", result["caption"])
        self.assertNotIn("private-image", result["caption"])

    def test_clip_model_uses_zero_shot_classification_caption(self) -> None:
        from nonebot_plugin_qwen_persona import vision_service

        calls: list[tuple[str, str]] = []
        original_pipeline = vision_service._load_pipeline
        original_image = vision_service._load_image
        original_pipeline_cache = vision_service._PIPELINE
        original_pipeline_model = vision_service._PIPELINE_MODEL
        original_pipeline_task = vision_service._PIPELINE_TASK

        def fake_load_pipeline(task: str, model_name: str):
            calls.append((task, model_name))

            def fake_pipeline(_image, *, candidate_labels=None):
                return [
                    {"label": candidate_labels[1], "score": 0.91},
                    {"label": candidate_labels[0], "score": 0.09},
                ]

            return fake_pipeline

        try:
            vision_service._PIPELINE = None
            vision_service._PIPELINE_MODEL = ""
            vision_service._PIPELINE_TASK = ""
            vision_service._load_pipeline = fake_load_pipeline  # type: ignore[assignment]
            vision_service._load_image = lambda _payload: object()  # type: ignore[assignment]

            result = vision_service.caption_payload(
                {"type": "image", "url": "https://example.invalid/a.png", "data": {}},
                model_name="openai/clip-vit-base-patch32",
            )
        finally:
            vision_service._load_pipeline = original_pipeline  # type: ignore[assignment]
            vision_service._load_image = original_image  # type: ignore[assignment]
            vision_service._PIPELINE = original_pipeline_cache
            vision_service._PIPELINE_MODEL = original_pipeline_model
            vision_service._PIPELINE_TASK = original_pipeline_task

        self.assertEqual(calls, [("zero-shot-image-classification", "openai/clip-vit-base-patch32")])
        self.assertEqual(result["source"], "clip")
        self.assertIn("表情包", result["caption"])
        self.assertNotIn("example.invalid", result["caption"])
        self.assertNotIn("abcdef", result["caption"])


class PersonaStoreDashboardStatsTests(unittest.TestCase):
    def test_dashboard_stats_counts_pending_samples_after_last_successful_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            now = time.time()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO training_runs (
                        id, status, reason, started_at, finished_at,
                        dataset_path, output_dir, sample_count, log
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("run-success", "success", "manual", now - 100, now - 90, "dataset", "out", 1, ""),
                )
                for idx, created_at in enumerate([now - 120, now - 40, now - 20], start=1):
                    conn.execute(
                        """
                        INSERT INTO chat_samples (
                            created_at, platform, user_id, group_id, message_id,
                            plain_text, training_text, raw_segments, sticker_keys, learn_reason
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            created_at,
                            "test",
                            f"user-{idx}",
                            "1234567890",
                            f"msg-{idx}",
                            "plain",
                            "training",
                            "[]",
                            "[]",
                            "ok",
                        ),
                    )

            payload = store._dashboard_stats_sync(days=2)

            self.assertEqual(payload["totals"]["learned_messages"], 3)
            self.assertEqual(payload["totals"]["trainable_samples"], 3)
            self.assertEqual(payload["totals"]["pending_training_samples"], 2)
            self.assertEqual(payload["totals"]["groups"], 1)
            self.assertEqual(payload["totals"]["users"], 3)
            self.assertEqual(payload["totals"]["last_24h_samples"], 3)
            self.assertIsNotNone(payload["totals"]["avg_training_chars"])
            self.assertEqual(payload["training"]["last_success"]["id"], "run-success")
            self.assertEqual(payload["charts"]["group_contribution"][0]["label"], "12...90")
            self.assertEqual(payload["charts"]["group_contribution"][0]["count"], 3)


class PersonaStoreStickerGalleryTests(unittest.TestCase):
    def test_sticker_gallery_lists_searches_and_exposes_type_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            now = time.time()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "face:smile",
                        "face",
                        "smile",
                        "123",
                        json.dumps({"id": "123"}, ensure_ascii=False),
                        "group-a",
                        "user-a",
                        now - 10,
                        now,
                        4,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "mface:wink",
                        "mface",
                        "wink",
                        "",
                        json.dumps({"emoji_id": "9", "summary": "wink"}, ensure_ascii=False),
                        "group-b",
                        "user-b",
                        now - 5,
                        now - 1,
                        2,
                    ),
                )

            payload = store._list_stickers_sync(limit=20, offset=0, query="smile", segment_type="")

            self.assertEqual(payload["total"], 1)
            self.assertEqual(payload["limit"], 20)
            self.assertEqual(payload["offset"], 0)
            self.assertEqual(payload["items"][0]["key"], "face:smile")
            self.assertEqual(payload["items"][0]["data"]["id"], "123")
            self.assertEqual(payload["type_counts"], {"face": 1, "mface": 1})

    def test_delete_stickers_removes_repository_records_without_rewriting_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            now = time.time()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO chat_samples (
                        created_at, platform, user_id, group_id, message_id,
                        plain_text, training_text, raw_segments, sticker_keys, learn_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        "test",
                        "user",
                        "group",
                        "msg",
                        "plain",
                        "plain [[sticker:face:smile]]",
                        "[]",
                        json.dumps(["face:smile"]),
                        "ok",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "face:smile",
                        "face",
                        "smile",
                        "123",
                        json.dumps({"id": "123"}, ensure_ascii=False),
                        "group",
                        "user",
                        now,
                        now,
                        1,
                    ),
                )

            result = store._delete_stickers_sync(["face:smile", "missing"])

            self.assertEqual(result, {"requested": 2, "deleted": 1})
            with store._connection() as conn:
                stickers = conn.execute("SELECT COUNT(*) FROM stickers").fetchone()[0]
                sample = conn.execute(
                    "SELECT training_text, sticker_keys FROM chat_samples"
                ).fetchone()
            self.assertEqual(stickers, 0)
            self.assertIn("[[sticker:face:smile]]", sample["training_text"])
            self.assertEqual(json.loads(sample["sticker_keys"]), ["face:smile"])


class PersonaStoreMediaCleanupTests(unittest.TestCase):
    def test_export_sft_dataset_uses_prompt_safe_chat_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            output_path = Path(tmp) / "dataset.jsonl"
            now = time.time()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO chat_samples (
                        created_at, platform, user_id, group_id, message_id,
                        plain_text, training_text, raw_segments, sticker_keys, learn_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        "test",
                        "user",
                        "group",
                        "msg",
                        "真实文本",
                        "真实文本 图片：图片 [[sticker:image:old]] [[sticker:face:smile]]",
                        "[]",
                        json.dumps(["image:old", "face:smile"]),
                        "ok",
                    ),
                )

            exported = store._export_sft_dataset_sync(
                output_path=output_path,
                limit=5,
                history_window=1,
                system_prompt="system",
            )
            payload = output_path.read_text(encoding="utf-8")

            self.assertEqual(exported, 1)
            self.assertIn("真实文本", payload)
            self.assertIn("[[sticker:face:smile]]", payload)
            self.assertNotIn("图片：", payload)
            self.assertNotIn("[[sticker:image:", payload)

    def test_export_sft_dataset_zero_limits_include_all_chat_and_external_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            output_path = Path(tmp) / "dataset.jsonl"
            now = time.time()
            with store._connection() as conn:
                for index in range(2):
                    conn.execute(
                        """
                        INSERT INTO chat_samples (
                            created_at, platform, user_id, group_id, message_id,
                            plain_text, training_text, raw_segments, sticker_keys, learn_reason
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now + index,
                            "test",
                            f"user-{index}",
                            "group",
                            f"msg-{index}",
                            f"群聊样本{index}",
                            f"群聊样本{index}",
                            "[]",
                            "[]",
                            "ok",
                        ),
                    )
                for index in range(3):
                    conn.execute(
                        """
                        INSERT INTO external_corpus_samples (
                            source, source_id, text, metadata_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            "hf:test",
                            f"row-{index}",
                            f"外部语料{index}",
                            json.dumps({"user_prompt": f"帖子{index}"}, ensure_ascii=False),
                            now + 10 + index,
                        ),
                    )

            exported = store._export_sft_dataset_sync(
                output_path=output_path,
                limit=0,
                history_window=1,
                system_prompt="system",
                external_limit=0,
            )
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(exported, 5)
            self.assertEqual(
                [record["metadata"]["source"] for record in records],
                ["chat_samples", "chat_samples", "hf:test", "hf:test", "hf:test"],
            )

    def test_cleanup_legacy_media_artifacts_removes_image_stickers_and_empty_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            now = time.time()
            with store._connection() as conn:
                rows = [
                    (
                        "msg-1",
                        "说话 [[sticker:image:old-image]] [[sticker:face:smile]]",
                        ["image:old-image", "face:smile"],
                    ),
                    ("msg-2", "图片：图片", []),
                ]
                for message_id, training_text, sticker_keys in rows:
                    conn.execute(
                        """
                        INSERT INTO chat_samples (
                            created_at, platform, user_id, group_id, message_id,
                            plain_text, training_text, raw_segments, sticker_keys, learn_reason
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now,
                            "test",
                            "user",
                            "group",
                            message_id,
                            "",
                            training_text,
                            "[]",
                            json.dumps(sticker_keys),
                            "ok",
                        ),
                    )
                for key, segment_type in [("image:old-image", "image"), ("face:smile", "face")]:
                    conn.execute(
                        """
                        INSERT INTO stickers (
                            key, type, summary, uri, data_json, group_id, user_id,
                            first_seen, last_seen, count
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            segment_type,
                            segment_type,
                            "",
                            "{}",
                            "group",
                            "user",
                            now,
                            now,
                            1,
                        ),
                    )

            result = store._cleanup_legacy_media_artifacts_sync()

            self.assertEqual(result["deleted_image_stickers"], 1)
            self.assertEqual(result["updated_samples"], 2)
            with store._connection() as conn:
                stickers = conn.execute("SELECT key, type FROM stickers ORDER BY key").fetchall()
                samples = conn.execute(
                    "SELECT message_id, training_text, sticker_keys FROM chat_samples ORDER BY message_id"
                ).fetchall()

            self.assertEqual([dict(row) for row in stickers], [{"key": "face:smile", "type": "face"}])
            self.assertEqual(samples[0]["training_text"], "说话 [[sticker:face:smile]]")
            self.assertEqual(json.loads(samples[0]["sticker_keys"]), ["face:smile"])
            self.assertEqual(samples[1]["training_text"], "")
            self.assertEqual(json.loads(samples[1]["sticker_keys"]), [])


class PersonaTrainerObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_skipped_no_train_target_is_recorded_as_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = QwenPersonaConfig(
                qwen_persona_data_dir=tmp,
                qwen_persona_min_train_samples=1,
                qwen_persona_lora_api_url="",
                qwen_persona_lora_train_command="",
                qwen_persona_lora_helper_api_url="",
            )
            store = PersonaStore(Path(tmp))
            await store.init()
            await store.record_sample(
                platform="test",
                user_id="user",
                group_id="group",
                message_id="msg",
                plain_text="plain",
                training_text="training",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            trainer = PersonaTrainer(config, store)

            result = await trainer.run_training(reason="admin_web")
            stats = await store.dashboard_stats()

            self.assertEqual(result.status, "skipped_no_train_target")
            self.assertEqual(stats["totals"]["training_runs"], 1)
            self.assertEqual(stats["training"]["last_run"]["status"], "skipped_no_train_target")
            self.assertEqual(stats["totals"]["pending_training_samples"], 1)

    async def test_training_progress_snapshot_tracks_running_stage_and_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = QwenPersonaConfig(
                qwen_persona_data_dir=tmp,
                qwen_persona_min_train_samples=1,
                qwen_persona_lora_api_url="",
                qwen_persona_lora_train_command="",
                qwen_persona_lora_helper_api_url="",
            )
            store = PersonaStore(Path(tmp))
            await store.init()
            await store.record_sample(
                platform="test",
                user_id="user",
                group_id="group",
                message_id="msg",
                plain_text="plain",
                training_text="training",
                raw_segments=[],
                stickers=[],
                learn_reason="accepted",
            )
            trainer = PersonaTrainer(config, store)

            result = await trainer.run_training(reason="admin_web")
            snapshot = trainer.progress_snapshot()

            self.assertEqual(result.status, "skipped_no_train_target")
            self.assertEqual(snapshot["stage"], "finished")
            self.assertEqual(snapshot["status"], "skipped_no_train_target")
            self.assertEqual(snapshot["sample_count"], 1)
            self.assertIn("qwen_persona_lora_api_url", snapshot["log_tail"])


class FakeModelManager:
    def __init__(self) -> None:
        self.actions: list[str] = []

    async def status_snapshot(self) -> dict[str, object]:
        self.actions.append("status")
        return {"state": "server_running", "control": {"can_stop": True}}

    async def stop(self) -> object:
        self.actions.append("stop")
        return SimpleNamespace(status="server_stopped", model_dir="data", repo_id="", log="")

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        self.actions.append(f"logs:{tail}")
        return [{"time": "now", "stream": "system", "message": "ready"}]

    async def start(self) -> object:
        self.actions.append("start")
        await asyncio.sleep(0.02)
        return SimpleNamespace(status="server_started", model_dir="data", repo_id="", log="")

    async def reload(self) -> object:
        self.actions.append("reload")
        await asyncio.sleep(0)
        return SimpleNamespace(status="llama_reloaded", model_dir="data", repo_id="", log="")

    async def switch_model(self, model: str, model_file: str = "") -> object:
        self.actions.append(f"switch:{model}:{model_file}")
        await asyncio.sleep(0)
        return SimpleNamespace(status="llama_switched", model_dir="data", repo_id=model, log="")


class FakeVisionManager:
    def __init__(self) -> None:
        self.actions: list[str] = []

    async def status_snapshot(self) -> dict[str, object]:
        self.actions.append("vision_status")
        return {"state": "vision_running", "healthy": True}

    async def start(self) -> object:
        self.actions.append("vision_start")
        return SimpleNamespace(status="vision_started", endpoint="http://127.0.0.1:18081", log="")

    async def stop(self) -> object:
        self.actions.append("vision_stop")
        return SimpleNamespace(status="vision_stopped", endpoint="http://127.0.0.1:18081", log="")

    async def restart(self) -> object:
        self.actions.append("vision_restart")
        return SimpleNamespace(status="vision_started", endpoint="http://127.0.0.1:18081", log="")

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        self.actions.append(f"vision_logs:{tail}")
        return [{"time": "now", "stream": "system", "message": "vision ready"}]


class FakeTrainer:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.actions: list[dict[str, object]] = []

    async def run_training(
        self,
        *,
        reason: str,
        include_external_corpus: bool | None = None,
        full_external_corpus: bool = False,
    ) -> object:
        self.actions.append(
            {
                "reason": reason,
                "include_external_corpus": include_external_corpus,
                "full_external_corpus": full_external_corpus,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(
            status="success",
            sample_count=3,
            dataset_path="dataset.jsonl",
            output_dir="output",
            log="done",
        )


class AdminWebObservabilityTests(unittest.TestCase):
    def test_admin_html_contains_dashboard_metrics_and_chart_controls(self) -> None:
        self.assertIn('id="metric-tiles"', ADMIN_HTML)
        self.assertIn('data-chart-mode="learning"', ADMIN_HTML)
        self.assertIn('data-chart-mode="performance"', ADMIN_HTML)
        self.assertIn("/api/dashboard/stats", ADMIN_HTML)
        self.assertIn('id="training-run"', ADMIN_HTML)
        self.assertIn('id="training-run-external"', ADMIN_HTML)
        self.assertIn("include_external_corpus", ADMIN_HTML)
        self.assertIn("full_external_corpus", ADMIN_HTML)
        self.assertIn("/api/training/run", ADMIN_HTML)
        self.assertIn("#settings-sections", ADMIN_HTML)
        self.assertIn("gap: 16px", ADMIN_HTML)
        self.assertIn("option {", ADMIN_HTML)
        self.assertIn("renderTrainingProgress", ADMIN_HTML)

    def test_admin_html_contains_sticker_gallery_controls(self) -> None:
        self.assertIn('href="/stickers"', ADMIN_HTML)
        self.assertIn('id="stickers-view"', ADMIN_HTML)
        self.assertIn('id="sticker-app"', ADMIN_HTML)
        self.assertIn('data-ant-design-component="StickerGalleryApp"', ADMIN_HTML)
        self.assertIn("mountStickerGalleryApp", ADMIN_HTML)
        self.assertIn("sticker-summary-save", ADMIN_HTML)
        self.assertIn("/api/stickers", ADMIN_HTML)
        self.assertIn("/api/stickers/delete", ADMIN_HTML)
        self.assertIn("/api/stickers/update", ADMIN_HTML)

    def test_sticker_gallery_payload_lists_and_deletes_repository_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            now = time.time()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO stickers (
                        key, type, summary, uri, data_json, group_id, user_id,
                        first_seen, last_seen, count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "face:smile",
                        "face",
                        "smile",
                        "123",
                        json.dumps({"id": "123"}, ensure_ascii=False),
                        "group",
                        "user",
                        now,
                        now,
                        3,
                    ),
                )
            server = AdminWebServer(
                config=QwenPersonaConfig(),
                save_callback=lambda _values: {},
                logger=DummyLogger(),
                store=store,
            )

            payload = server._sticker_gallery_payload(
                {"limit": ["12"], "offset": ["0"], "q": ["smile"], "type": ["face"]}
            )
            delete_result = server._delete_stickers_payload({"keys": ["face:smile", "missing"]})
            after = server._sticker_gallery_payload({"limit": ["12"], "offset": ["0"]})

            self.assertEqual(payload["total"], 1)
            self.assertEqual(payload["items"][0]["key"], "face:smile")
            self.assertEqual(payload["type_counts"], {"face": 1})
            self.assertEqual(delete_result["deleted"], 1)
            self.assertEqual(after["total"], 0)

    def test_vision_api_uses_async_vision_manager(self) -> None:
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        vision = FakeVisionManager()
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            vision_manager=vision,
            event_loop=loop,
        )

        async def run_calls() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
            status = await server._run_model_coro(vision.status_snapshot())
            started = server._vision_result_payload(await server._run_model_coro(vision.start()))
            logs = server._vision_logs_payload(tail=1)
            return status, started, logs

        status, started, logs = loop.run_until_complete(run_calls())

        self.assertEqual(status["state"], "vision_running")
        self.assertEqual(started["status"], "vision_started")
        self.assertEqual(json.loads(json.dumps(logs))["logs"][0]["message"], "vision ready")
        self.assertEqual(vision.actions, ["vision_status", "vision_start", "vision_logs:1"])

    def test_admin_html_contains_vision_service_controls(self) -> None:
        self.assertIn('id="vision-status"', ADMIN_HTML)
        self.assertIn('id="vision-logs"', ADMIN_HTML)
        self.assertIn('id="vision-start"', ADMIN_HTML)
        self.assertIn('id="vision-stop"', ADMIN_HTML)
        self.assertIn('id="vision-restart"', ADMIN_HTML)
        self.assertIn("/api/vision/status", ADMIN_HTML)
        self.assertIn("/api/vision/logs?tail=200", ADMIN_HTML)
        self.assertIn('controlVision("start")', ADMIN_HTML)

    def test_runtime_field_is_first_model_deploy_item_with_visibility_rules(self) -> None:
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
        )

        payload = server._config_payload()
        deploy_fields = [field for field in payload["fields"] if field["category"] == "模型部署"]
        runtime_field = deploy_fields[0]
        fields = {field["name"]: field for field in payload["fields"]}

        self.assertEqual(runtime_field["name"], "qwen_persona_runtime")
        self.assertIn(
            "external_llama",
            [option["value"] for option in runtime_field["options"]],
        )
        self.assertEqual(fields["qwen_persona_llama_server_path"]["visibleWhen"]["runtime"], ["llama_cpp"])
        self.assertEqual(
            fields["qwen_persona_llama_host"]["visibleWhen"]["runtime"],
            ["llama_cpp", "external_llama"],
        )
        self.assertEqual(
            fields["qwen_persona_api_base"]["visibleWhen"]["runtime"],
            ["external_api"],
        )

    def test_llama_api_uses_async_model_manager(self) -> None:
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        manager = FakeModelManager()
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            model_manager=manager,
            event_loop=loop,
        )

        async def run_calls() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
            status = await server._run_model_coro(manager.status_snapshot())
            stopped = server._model_result_payload(await server._run_model_coro(manager.stop()))
            logs = server._llama_logs_payload(tail=1)
            return status, stopped, logs

        status, stopped, logs = loop.run_until_complete(run_calls())

        self.assertEqual(status["state"], "server_running")
        self.assertEqual(stopped["status"], "server_stopped")
        self.assertEqual(json.loads(json.dumps(logs))["logs"][0]["message"], "ready")
        self.assertEqual(manager.actions, ["status", "stop", "logs:1"])

    def test_dashboard_payload_combines_store_client_and_llama_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp))
            store._init_sync()
            with store._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO chat_samples (
                        created_at, platform, user_id, group_id, message_id,
                        plain_text, training_text, raw_segments, sticker_keys, learn_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (time.time(), "test", "user", "987654321", "msg", "plain", "training", "[]", "[]", "ok"),
                )
            qwen_client = QwenClient(QwenPersonaConfig(qwen_persona_runtime="external_api", qwen_persona_model="qwen-test"))
            qwen_client._record_metric({"usage": {"completion_tokens": 10}}, "hello", 2.0)
            manager = ModelManager(QwenPersonaConfig(qwen_persona_llama_model="Qwen/Test"))
            manager._append_log("stderr", "eval time = 20.00 tokens per second")
            server = AdminWebServer(
                config=QwenPersonaConfig(),
                save_callback=lambda _values: {},
                logger=DummyLogger(),
                model_manager=manager,
                store=store,
                qwen_client=qwen_client,
            )

            payload = server._dashboard_stats_payload()

            self.assertEqual(payload["totals"]["learned_messages"], 1)
            self.assertEqual(payload["inference"]["last_tps"]["source"], "usage")
            self.assertEqual(payload["inference"]["last_tps"]["tps"], 5.0)
            self.assertEqual(payload["training_operation"]["available"], False)
            sources = {item["source"] for item in payload["charts"]["performance_trend"]}
            self.assertIn("usage", sources)
            self.assertIn("llama_log", sources)

    def test_llama_actions_are_scheduled_on_running_loop(self) -> None:
        loop = asyncio.new_event_loop()
        manager = FakeModelManager()
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            model_manager=manager,
            event_loop=loop,
        )
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.addCleanup(loop.close)

        try:
            payload = server._submit_model_action("start")
            self.assertEqual(payload["status"], "action_scheduled")
            self.assertEqual(payload["action"], "start")
            server._pending_action_future.result(timeout=1)
            self.assertIn("start", manager.actions)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)

    def test_training_action_is_scheduled_on_running_loop(self) -> None:
        loop = asyncio.new_event_loop()
        trainer = FakeTrainer()
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            trainer=trainer,
            event_loop=loop,
        )
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.addCleanup(loop.close)

        try:
            payload = server._submit_training_action({"reason": "admin_web"})
            self.assertEqual(payload["status"], "action_scheduled")
            self.assertEqual(payload["action"], "training")
            server._pending_training_future.result(timeout=1)
            self.assertEqual(
                trainer.actions,
                [
                    {
                        "reason": "admin_web",
                        "include_external_corpus": None,
                        "full_external_corpus": False,
                    }
                ],
            )
            stats = server._dashboard_stats_payload()
            self.assertEqual(stats["training_operation"]["last_result"]["status"], "success")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)

    def test_training_action_rejects_duplicate_submission(self) -> None:
        loop = asyncio.new_event_loop()
        trainer = FakeTrainer(delay=0.05)
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            trainer=trainer,
            event_loop=loop,
        )
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.addCleanup(loop.close)

        try:
            first = server._submit_training_action({"reason": "admin_web"})
            second = server._submit_training_action({"reason": "manual_retry"})
            self.assertEqual(first["status"], "action_scheduled")
            self.assertEqual(second["status"], "action_in_progress")
            self.assertEqual(second["reason"], "admin_web")
            server._pending_training_future.result(timeout=1)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)

    def test_training_action_can_request_full_external_corpus(self) -> None:
        loop = asyncio.new_event_loop()
        trainer = FakeTrainer()
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda _values: {},
            logger=DummyLogger(),
            trainer=trainer,
            event_loop=loop,
        )
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.addCleanup(loop.close)

        try:
            payload = server._submit_training_action(
                {
                    "reason": "external_corpus_full",
                    "include_external_corpus": True,
                    "full_external_corpus": True,
                }
            )
            self.assertEqual(payload["status"], "action_scheduled")
            server._pending_training_future.result(timeout=1)
            self.assertEqual(
                trainer.actions,
                [
                    {
                        "reason": "external_corpus_full",
                        "include_external_corpus": True,
                        "full_external_corpus": True,
                    }
                ],
            )
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)

    def test_switch_action_persists_model_before_scheduling(self) -> None:
        loop = asyncio.new_event_loop()
        manager = FakeModelManager()
        saved: list[dict[str, object]] = []
        server = AdminWebServer(
            config=QwenPersonaConfig(),
            save_callback=lambda values: saved.append(values) or {"changed": list(values)},
            logger=DummyLogger(),
            model_manager=manager,
            event_loop=loop,
        )
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.addCleanup(loop.close)

        try:
            payload = server._submit_model_action(
                "switch",
                {"model": "Qwen/New-GGUF:Q4_K_M", "model_file": ""},
            )
            self.assertEqual(payload["status"], "action_scheduled")
            self.assertEqual(saved[-1]["qwen_persona_llama_model"], "Qwen/New-GGUF:Q4_K_M")
            server._pending_action_future.result(timeout=1)
            self.assertIn("switch:Qwen/New-GGUF:Q4_K_M:", manager.actions)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
