from __future__ import annotations

import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class ReplacementTextBackend:
    def load(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
        return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "prompt"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="你好\ufffd呀")


class CorruptReplacementTextBackend:
    def load(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
        return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "prompt"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="\ufffd\ufffd\ufffd\ufffd线\ufffd\ufffd")


class SuccessfulTextBackend:
    def load(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
        return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "prompt"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="正常回复")


class RecordingBackend:
    def __init__(self) -> None:
        self.loaded_paths: list[str] = []

    def load(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        trust_remote_code: bool,
    ) -> tuple[object, object]:
        self.loaded_paths.append(model_path)
        return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "prompt"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="ok")


class StructuredRecordingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.templated_messages: object | None = None

    def apply_chat_template(
        self,
        _processor: object,
        _config: object,
        messages: object,
        *,
        num_images: int,
    ) -> str:
        self.templated_messages = messages
        return f"prompt images={num_images}"


class BlockingGenerateBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, *_args: object, **_kwargs: object) -> object:
        self.started.set()
        assert self.release.wait(timeout=5)
        return SimpleNamespace(text="ok")


class SamplingRecordingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, *_args: object, **kwargs: object) -> object:
        self.generate_kwargs = kwargs
        return SimpleNamespace(text="ok")


class RecordingDownloadManager:
    def __init__(self, local_path: Path) -> None:
        self.local_path = local_path
        self.calls: list[dict[str, object]] = []

    def prepare_model(
        self,
        model_ref: str,
        *,
        model_dir: Path,
        progress: object,
    ) -> object:
        self.calls.append({"model_ref": model_ref, "model_dir": model_dir})
        return SimpleNamespace(local_path=self.local_path, source="hf")


class MlxRuntimeTests(unittest.TestCase):
    def test_safe_bpe_detokenizer_drops_invalid_utf8_bytes(self) -> None:
        from nonebot_plugin_qwen_persona.mlx_runtime import _SafeBPEStreamingDetokenizer

        source = SimpleNamespace(
            trim_space=False,
            tokenmap=["°", " hello"],
            _byte_decoder={"°": 0xB0, " ": 32, "h": 104, "e": 101, "l": 108, "o": 111},
        )
        detokenizer = _SafeBPEStreamingDetokenizer(source)

        detokenizer.add_token(0)
        detokenizer.add_token(1)
        detokenizer.finalize()

        self.assertEqual(detokenizer.text, " hello")

    def test_runtime_patches_bpe_detokenizer_on_loaded_processor(self) -> None:
        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        config = QwenPersonaConfig()
        runtime = MlxVLMRuntime(config)
        bpe_type = type("BPEStreamingDetokenizer", (), {})
        detokenizer = bpe_type()
        detokenizer.trim_space = False
        detokenizer.tokenmap = ["a"]
        detokenizer._byte_decoder = {"a": 97}
        processor = SimpleNamespace(detokenizer=detokenizer)

        patched = runtime._with_safe_detokenizer(processor)

        self.assertIs(patched, processor)
        self.assertEqual(processor.detokenizer.__class__.__name__, "_SafeBPEStreamingDetokenizer")
        self.assertIn("patched BPE streaming detokenizer", [item["message"] for item in runtime.logs()])

    def test_chat_sanitizes_isolated_generated_replacement_characters(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        runtime = MlxVLMRuntime(QwenPersonaConfig(), backend=ReplacementTextBackend())

        reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(reply, "你好呀")
        self.assertTrue(
            any(item["event"] == "generated_text_sanitized" and item["level"] == "warn" for item in runtime.logs())
        )

    def test_chat_rejects_mostly_corrupt_generated_replacement_characters(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        runtime = MlxVLMRuntime(QwenPersonaConfig(), backend=CorruptReplacementTextBackend())

        with self.assertRaises(ValueError):
            asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

        error_logs = [item for item in runtime.logs() if item["event"] == "invalid_generated_text"]
        self.assertEqual(error_logs[-1]["level"], "error")
        self.assertEqual(error_logs[-1]["context"]["replacement_count"], 6)

    def test_chat_success_records_metrics_without_datetime_import_regression(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        runtime = MlxVLMRuntime(QwenPersonaConfig(), backend=SuccessfulTextBackend())

        reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(reply, "正常回复")
        self.assertEqual(runtime.metrics_snapshot()["last"]["response_chars"], 4)

    def test_chat_passes_structured_messages_to_backend_template(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        backend = StructuredRecordingBackend()
        runtime = MlxVLMRuntime(QwenPersonaConfig(), backend=backend)

        reply = asyncio.run(
            runtime.chat(
                [
                    {"role": "system", "content": "persona rules"},
                    {"role": "user", "content": "hello"},
                ],
                media_parts=[{"type": "image", "data": {"url": "https://example.invalid/a.png"}}],
            )
        )

        self.assertEqual(reply, "ok")
        self.assertIsInstance(backend.templated_messages, list)
        assert isinstance(backend.templated_messages, list)
        self.assertEqual(backend.templated_messages[0]["role"], "system")
        self.assertEqual(backend.templated_messages[1]["content"], "hello")

    def test_chat_passes_repetition_controls_to_backend_generate(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        backend = SamplingRecordingBackend()
        runtime = MlxVLMRuntime(
            QwenPersonaConfig(
                qwen_persona_repetition_penalty=1.08,
                qwen_persona_repetition_context_size=64,
            ),
            backend=backend,
        )

        reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

        self.assertEqual(reply, "ok")
        self.assertEqual(backend.generate_kwargs["repetition_penalty"], 1.08)
        self.assertEqual(backend.generate_kwargs["repetition_context_size"], 64)

    def test_generation_queue_is_visible_while_second_chat_waits(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        backend = BlockingGenerateBackend()
        runtime = MlxVLMRuntime(QwenPersonaConfig(), backend=backend)

        async def run_case() -> dict[str, object]:
            first = asyncio.create_task(runtime.chat([{"role": "user", "content": "first"}]))
            self.assertTrue(await asyncio.to_thread(backend.started.wait, 5))
            second = asyncio.create_task(runtime.chat([{"role": "user", "content": "second"}]))
            await asyncio.sleep(0.05)
            snapshot = runtime.status_snapshot()
            backend.release.set()
            await asyncio.gather(first, second)
            return snapshot

        snapshot = asyncio.run(run_case())

        self.assertEqual(snapshot["generation"]["active"], 1)
        self.assertEqual(snapshot["generation"]["queued"], 1)
        self.assertEqual(snapshot["generation"]["current"], "chat")

    def test_hf_repo_downloads_to_configured_dir_before_loading(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        with tempfile.TemporaryDirectory() as temp_root:
            local_path = Path(temp_root) / "models" / "org--model"
            local_path.mkdir(parents=True)
            backend = RecordingBackend()
            downloader = RecordingDownloadManager(local_path)
            config = QwenPersonaConfig(
                qwen_persona_mlx_model="org/model",
                qwen_persona_model_dir=str(Path(temp_root) / "controlled"),
            )
            runtime = MlxVLMRuntime(config, backend=backend, download_manager=downloader)

            result = asyncio.run(runtime.load())

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(backend.loaded_paths, [str(local_path)])
        self.assertEqual(
            downloader.calls,
            [{"model_ref": "org/model", "model_dir": Path(temp_root) / "controlled"}],
        )
        self.assertEqual(result["operation"]["type"], "load")
        self.assertEqual(result["operation"]["phase"], "loaded")

    def test_existing_local_model_path_skips_download_manager(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        with tempfile.TemporaryDirectory() as temp_root:
            local_path = Path(temp_root) / "local-model"
            local_path.mkdir()
            backend = RecordingBackend()
            downloader = RecordingDownloadManager(Path(temp_root) / "unused")
            runtime = MlxVLMRuntime(
                QwenPersonaConfig(qwen_persona_mlx_model=str(local_path)),
                backend=backend,
                download_manager=downloader,
            )

            result = asyncio.run(runtime.load())

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(backend.loaded_paths, [str(local_path)])
        self.assertEqual(downloader.calls, [])

    def test_download_progress_is_visible_while_load_is_running(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        started = threading.Event()
        release = threading.Event()

        class BlockingDownloadManager(RecordingDownloadManager):
            def prepare_model(self, model_ref: str, *, model_dir: Path, progress: object) -> object:
                progress.update(phase="downloading", downloaded_bytes=25, total_bytes=100, message="25/100")
                started.set()
                assert release.wait(timeout=5)
                return super().prepare_model(model_ref, model_dir=model_dir, progress=progress)

        async def run_case() -> dict[str, object]:
            with tempfile.TemporaryDirectory() as temp_root:
                local_path = Path(temp_root) / "prepared"
                local_path.mkdir()
                runtime = MlxVLMRuntime(
                    QwenPersonaConfig(
                        qwen_persona_mlx_model="org/model",
                        qwen_persona_model_dir=str(Path(temp_root) / "controlled"),
                    ),
                    backend=RecordingBackend(),
                    download_manager=BlockingDownloadManager(local_path),
                )
                task = asyncio.create_task(runtime.load())
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                snapshot = runtime.status_snapshot()
                release.set()
                await task
                return snapshot

        snapshot = asyncio.run(run_case())

        self.assertEqual(snapshot["state"], "downloading")
        self.assertEqual(snapshot["operation"]["phase"], "downloading")
        self.assertEqual(snapshot["operation"]["percent"], 25.0)
        self.assertEqual(snapshot["operation"]["downloaded_bytes"], 25)
        self.assertEqual(snapshot["operation"]["total_bytes"], 100)

    def test_mlx_load_phase_is_indeterminate_and_does_not_keep_old_model_on_failure(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        started = threading.Event()
        release = threading.Event()

        class BlockingFailingBackend(RecordingBackend):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next = False

            def load(self, model_path: str, *, adapter_path: str | None, trust_remote_code: bool) -> tuple[object, object]:
                self.loaded_paths.append(model_path)
                if self.fail_next:
                    started.set()
                    assert release.wait(timeout=5)
                    raise RuntimeError("boom")
                return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

        async def run_case() -> tuple[dict[str, object], dict[str, object]]:
            with tempfile.TemporaryDirectory() as temp_root:
                old_path = Path(temp_root) / "old"
                new_path = Path(temp_root) / "new"
                old_path.mkdir()
                new_path.mkdir()
                backend = BlockingFailingBackend()
                runtime = MlxVLMRuntime(
                    QwenPersonaConfig(qwen_persona_mlx_model=str(old_path)),
                    backend=backend,
                )
                await runtime.load()
                backend.fail_next = True
                runtime.reconfigure(QwenPersonaConfig(qwen_persona_mlx_model=str(new_path)))
                task = asyncio.create_task(runtime.reload())
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                loading_snapshot = runtime.status_snapshot()
                release.set()
                failed = await task
                return loading_snapshot, failed

        loading_snapshot, failed = asyncio.run(run_case())

        self.assertEqual(loading_snapshot["state"], "loading")
        self.assertEqual(loading_snapshot["operation"]["phase"], "loading_model")
        self.assertIsNone(loading_snapshot["operation"]["percent"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["loaded_model"], "")
        self.assertEqual(failed["operation"]["state"], "failed")

    def test_hf_download_percent_clears_when_entering_mlx_load_phase(self) -> None:
        import asyncio

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.mlx_runtime import MlxVLMRuntime

        started = threading.Event()
        release = threading.Event()

        class FinishedDownloadManager(RecordingDownloadManager):
            def prepare_model(self, model_ref: str, *, model_dir: Path, progress: object) -> object:
                progress.update(
                    phase="downloaded",
                    downloaded_bytes=100,
                    total_bytes=100,
                    percent=100.0,
                    message="downloaded",
                )
                return super().prepare_model(model_ref, model_dir=model_dir, progress=progress)

        class BlockingBackend(RecordingBackend):
            def load(self, model_path: str, *, adapter_path: str | None, trust_remote_code: bool) -> tuple[object, object]:
                self.loaded_paths.append(model_path)
                started.set()
                assert release.wait(timeout=5)
                return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_vl")), object()

        async def run_case() -> dict[str, object]:
            with tempfile.TemporaryDirectory() as temp_root:
                local_path = Path(temp_root) / "prepared"
                local_path.mkdir()
                runtime = MlxVLMRuntime(
                    QwenPersonaConfig(
                        qwen_persona_mlx_model="org/model",
                        qwen_persona_model_dir=str(Path(temp_root) / "controlled"),
                    ),
                    backend=BlockingBackend(),
                    download_manager=FinishedDownloadManager(local_path),
                )
                task = asyncio.create_task(runtime.load())
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                snapshot = runtime.status_snapshot()
                release.set()
                await task
                return snapshot

        snapshot = asyncio.run(run_case())

        self.assertEqual(snapshot["operation"]["phase"], "loading_model")
        self.assertIsNone(snapshot["operation"]["percent"])


if __name__ == "__main__":
    unittest.main()
