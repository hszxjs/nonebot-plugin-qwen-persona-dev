from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


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


class RepeatingTailBackend(SuccessfulTextBackend):
    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="咋了这是，刚想" + "润" * 80)


class RepeatingCaptionBackend(SuccessfulTextBackend):
    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="这张图片描绘了一位动漫少女，她手持一把带有“" + "润" * 80)


class TokenMetricsBackend(SuccessfulTextBackend):
    def generate(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(text="真实 token 指标", generation_tokens=7, finish_reason="length")


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


def test_safe_bpe_detokenizer_drops_invalid_utf8_bytes() -> None:
    from nonebot_plugin_ai_companion.providers.mlx_runtime import _SafeBPEStreamingDetokenizer

    source = SimpleNamespace(
        trim_space=False,
        tokenmap=["°", " hello"],
        _byte_decoder={"°": 0xB0, " ": 32, "h": 104, "e": 101, "l": 108, "o": 111},
    )
    detokenizer = _SafeBPEStreamingDetokenizer(source)

    detokenizer.add_token(0)
    detokenizer.add_token(1)
    detokenizer.finalize()

    assert detokenizer.text == " hello"


def test_runtime_patches_bpe_detokenizer_on_loaded_processor() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    config = AiCompanionConfig()
    runtime = MlxVLMRuntime(config)
    bpe_type = type("BPEStreamingDetokenizer", (), {})
    detokenizer = bpe_type()
    detokenizer.trim_space = False
    detokenizer.tokenmap = ["a"]
    detokenizer._byte_decoder = {"a": 97}
    processor = SimpleNamespace(detokenizer=detokenizer)

    patched = runtime._with_safe_detokenizer(processor)

    assert patched is processor
    assert processor.detokenizer.__class__.__name__ == "_SafeBPEStreamingDetokenizer"
    assert "patched BPE streaming detokenizer" in [item["message"] for item in runtime.logs()]


def test_chat_sanitizes_isolated_generated_replacement_characters() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=ReplacementTextBackend())

    reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    assert reply == "你好呀"
    assert any(
        item["event"] == "generated_text_sanitized" and item["level"] == "warn" for item in runtime.logs()
    )


def test_chat_rejects_mostly_corrupt_generated_replacement_characters() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=CorruptReplacementTextBackend())

    with pytest.raises(ValueError):
        asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    error_logs = [item for item in runtime.logs() if item["event"] == "invalid_generated_text"]
    assert error_logs[-1]["level"] == "error"
    assert error_logs[-1]["context"]["replacement_count"] == 6


def test_chat_truncates_repetitive_generated_tail() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=RepeatingTailBackend())

    reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    assert reply == "咋了这是，刚想润润"
    logs = [item for item in runtime.logs() if item["event"] == "generated_text_repetition_truncated"]
    assert logs[-1]["level"] == "warn"
    assert logs[-1]["context"]["repeated_unit"] == "润"


def test_chat_success_records_metrics_without_datetime_import_regression() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=SuccessfulTextBackend())

    reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    assert reply == "正常回复"
    assert runtime.metrics_snapshot()["last"]["response_chars"] == 4


def test_chat_records_backend_generation_metrics_when_available() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=TokenMetricsBackend())

    reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    assert reply == "真实 token 指标"
    metric = runtime.metrics_snapshot()["last"]
    assert metric["completion_tokens"] == 7
    assert metric["finish_reason"] == "length"
    assert metric["source"] == "mlx_vlm"


def test_chat_passes_structured_messages_to_backend_template() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    backend = StructuredRecordingBackend()
    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=backend)

    reply = asyncio.run(
        runtime.chat(
            [
                {"role": "system", "content": "persona rules"},
                {"role": "user", "content": "hello"},
            ],
            media_parts=[{"type": "image", "data": {"url": "https://example.invalid/a.png"}}],
        )
    )

    assert reply == "ok"
    assert isinstance(backend.templated_messages, list)
    assert isinstance(backend.templated_messages, list)
    assert backend.templated_messages[0]["role"] == "system"
    assert backend.templated_messages[1]["content"] == "hello"


def test_chat_passes_repetition_controls_to_backend_generate() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    backend = SamplingRecordingBackend()
    runtime = MlxVLMRuntime(
        AiCompanionConfig(
            ai_companion_repetition_penalty=1.08,
            ai_companion_repetition_context_size=64,
        ),
        backend=backend,
    )

    reply = asyncio.run(runtime.chat([{"role": "user", "content": "hello"}]))

    assert reply == "ok"
    assert backend.generate_kwargs["repetition_penalty"] == 1.08
    assert backend.generate_kwargs["repetition_context_size"] == 64


def test_caption_media_uses_short_generation_profile() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    backend = SamplingRecordingBackend()
    runtime = MlxVLMRuntime(
        AiCompanionConfig(
            ai_companion_max_tokens=2048,
            ai_companion_mlx_lora_max_seq_length=2048,
            ai_companion_temperature=0.85,
            ai_companion_top_p=0.9,
            ai_companion_repetition_penalty=1.08,
            ai_companion_repetition_context_size=64,
        ),
        backend=backend,
    )

    caption = asyncio.run(
        runtime.caption_media({"type": "image", "data": {"url": "https://example.invalid/a.png"}})
    )

    assert caption == "ok"
    assert backend.generate_kwargs["max_tokens"] == 64
    assert backend.generate_kwargs["temperature"] == 0.2
    assert backend.generate_kwargs["top_p"] == 0.8
    assert backend.generate_kwargs["repetition_penalty"] == 1.15
    assert backend.generate_kwargs["repetition_context_size"] == 128


def test_sticker_selector_generation_profile_is_not_caption_limited() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.generation_profiles import GenerationPurpose
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    backend = SamplingRecordingBackend()
    runtime = MlxVLMRuntime(
        AiCompanionConfig(
            ai_companion_max_tokens=2048,
            ai_companion_mlx_lora_max_seq_length=2048,
            ai_companion_temperature=0.85,
            ai_companion_top_p=0.9,
            ai_companion_repetition_penalty=1.08,
            ai_companion_repetition_context_size=64,
        ),
        backend=backend,
    )

    reply = asyncio.run(
        runtime.chat(
            [{"role": "user", "content": "select a sticker"}],
            purpose=GenerationPurpose.STICKER_SELECTOR,
        )
    )

    assert reply == "ok"
    assert backend.generate_kwargs["max_tokens"] >= 128
    assert backend.generate_kwargs["max_tokens"] != 64
    assert backend.generate_kwargs["temperature"] <= 0.3
    assert runtime.metrics_snapshot()["last"]["purpose"] == "sticker_selector"


def test_caption_media_rejects_repetitive_generated_caption() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=RepeatingCaptionBackend())

    caption = asyncio.run(
        runtime.caption_media({"type": "image", "data": {"url": "https://example.invalid/a.png"}})
    )

    assert caption == ""
    logs = [item for item in runtime.logs() if item["event"] == "generated_text_repetition_truncated"]
    assert logs[-1]["context"]["rejected"] is True
    assert logs[-1]["context"]["purpose"] == "media_caption"


def test_generation_queue_is_visible_while_second_chat_waits() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    backend = BlockingGenerateBackend()
    runtime = MlxVLMRuntime(AiCompanionConfig(), backend=backend)

    async def run_case() -> dict[str, object]:
        first = asyncio.create_task(runtime.chat([{"role": "user", "content": "first"}]))
        assert await asyncio.to_thread(backend.started.wait, 5)
        second = asyncio.create_task(runtime.chat([{"role": "user", "content": "second"}]))
        await asyncio.sleep(0.05)
        snapshot = runtime.status_snapshot()
        backend.release.set()
        await asyncio.gather(first, second)
        return snapshot

    snapshot = asyncio.run(run_case())

    assert snapshot["generation"]["active"] == 1
    assert snapshot["generation"]["queued"] == 1
    assert snapshot["generation"]["current"] == "persona_reply"


def test_hf_repo_downloads_to_configured_dir_before_loading(tmp_path: Path) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    local_path = tmp_path / "models" / "org--model"
    local_path.mkdir(parents=True)
    backend = RecordingBackend()
    downloader = RecordingDownloadManager(local_path)
    config = AiCompanionConfig(
        ai_companion_mlx_model="org/model",
        ai_companion_model_dir=str(tmp_path / "controlled"),
    )
    runtime = MlxVLMRuntime(config, backend=backend, download_manager=downloader)

    result = asyncio.run(runtime.load())

    assert result["status"] == "loaded"
    assert backend.loaded_paths == [str(local_path)]
    assert downloader.calls == [{"model_ref": "org/model", "model_dir": tmp_path / "controlled"}]
    assert result["operation"]["type"] == "load"
    assert result["operation"]["phase"] == "loaded"


def test_existing_local_model_path_skips_download_manager(tmp_path: Path) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    local_path = tmp_path / "local-model"
    local_path.mkdir()
    backend = RecordingBackend()
    downloader = RecordingDownloadManager(tmp_path / "unused")
    runtime = MlxVLMRuntime(
        AiCompanionConfig(ai_companion_mlx_model=str(local_path)),
        backend=backend,
        download_manager=downloader,
    )

    result = asyncio.run(runtime.load())

    assert result["status"] == "loaded"
    assert backend.loaded_paths == [str(local_path)]
    assert downloader.calls == []


def test_download_progress_is_visible_while_load_is_running(tmp_path: Path) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

    started = threading.Event()
    release = threading.Event()

    class BlockingDownloadManager(RecordingDownloadManager):
        def prepare_model(self, model_ref: str, *, model_dir: Path, progress: object) -> object:
            progress.update(phase="downloading", downloaded_bytes=25, total_bytes=100, message="25/100")
            started.set()
            assert release.wait(timeout=5)
            return super().prepare_model(model_ref, model_dir=model_dir, progress=progress)

    async def run_case() -> dict[str, object]:
        local_path = tmp_path / "prepared"
        local_path.mkdir()
        runtime = MlxVLMRuntime(
            AiCompanionConfig(
                ai_companion_mlx_model="org/model",
                ai_companion_model_dir=str(tmp_path / "controlled"),
            ),
            backend=RecordingBackend(),
            download_manager=BlockingDownloadManager(local_path),
        )
        task = asyncio.create_task(runtime.load())
        assert await asyncio.to_thread(started.wait, 5)
        snapshot = runtime.status_snapshot()
        release.set()
        await task
        return snapshot

    snapshot = asyncio.run(run_case())

    assert snapshot["state"] == "downloading"
    assert snapshot["operation"]["phase"] == "downloading"
    assert snapshot["operation"]["percent"] == 25.0
    assert snapshot["operation"]["downloaded_bytes"] == 25
    assert snapshot["operation"]["total_bytes"] == 100


def test_mlx_load_phase_is_indeterminate_and_does_not_keep_old_model_on_failure(tmp_path: Path) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

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
        old_path = tmp_path / "old"
        new_path = tmp_path / "new"
        old_path.mkdir()
        new_path.mkdir()
        backend = BlockingFailingBackend()
        runtime = MlxVLMRuntime(
            AiCompanionConfig(ai_companion_mlx_model=str(old_path)),
            backend=backend,
        )
        await runtime.load()
        backend.fail_next = True
        runtime.reconfigure(AiCompanionConfig(ai_companion_mlx_model=str(new_path)))
        task = asyncio.create_task(runtime.reload())
        assert await asyncio.to_thread(started.wait, 5)
        loading_snapshot = runtime.status_snapshot()
        release.set()
        failed = await task
        return loading_snapshot, failed

    loading_snapshot, failed = asyncio.run(run_case())

    assert loading_snapshot["state"] == "loading"
    assert loading_snapshot["operation"]["phase"] == "loading_model"
    assert loading_snapshot["operation"]["percent"] is None
    assert failed["status"] == "failed"
    assert failed["loaded_model"] == ""
    assert failed["operation"]["state"] == "failed"


def test_hf_download_percent_clears_when_entering_mlx_load_phase(tmp_path: Path) -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.providers.mlx_runtime import MlxVLMRuntime

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
        local_path = tmp_path / "prepared"
        local_path.mkdir()
        runtime = MlxVLMRuntime(
            AiCompanionConfig(
                ai_companion_mlx_model="org/model",
                ai_companion_model_dir=str(tmp_path / "controlled"),
            ),
            backend=BlockingBackend(),
            download_manager=FinishedDownloadManager(local_path),
        )
        task = asyncio.create_task(runtime.load())
        assert await asyncio.to_thread(started.wait, 5)
        snapshot = runtime.status_snapshot()
        release.set()
        await task
        return snapshot

    snapshot = asyncio.run(run_case())

    assert snapshot["operation"]["phase"] == "loading_model"
    assert snapshot["operation"]["percent"] is None
