from __future__ import annotations

import asyncio
import gc
import importlib.util
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import QwenPersonaConfig, effective_mlx_model
from .generation_profiles import GenerationProfile, GenerationPurpose, profile_for_purpose
from .generation_safety import trim_repetitive_tail
from .mlx_backend import DefaultMlxBackend, MlxBackend
from .model_download import HFDownloadManager, PreparedModel, is_existing_local_model, is_hf_repo_ref
from .runtime_state import InferenceMetric, MlxRuntimeLog, OperationProgress, RuntimeOperation, utc_timestamp


REPLACEMENT_CHAR = "\ufffd"


class MlxVLMRuntime:
    def __init__(
        self,
        config: QwenPersonaConfig,
        *,
        backend: MlxBackend | None = None,
        download_manager: Any | None = None,
    ) -> None:
        self.config = config
        self._state_lock = asyncio.Lock()
        self._generation_gate = asyncio.Semaphore(1)
        self._thread_lock = threading.RLock()
        provided_backend = backend
        self._backend = backend or DefaultMlxBackend(warning_sink=self._append_backend_warning)
        self._download_manager = download_manager or HFDownloadManager()
        self._managed_downloads = (
            download_manager is not None
            or provided_backend is None
            or isinstance(provided_backend, DefaultMlxBackend)
        )
        self._model: Any | None = None
        self._processor: Any | None = None
        self._state = "not_loaded"
        self._last_error = ""
        self._logs: deque[MlxRuntimeLog] = deque(maxlen=400)
        self._metrics: deque[InferenceMetric] = deque(maxlen=120)
        self._loaded_model = ""
        self._loaded_adapter = ""
        self._loaded_model_path = ""
        self._operation: RuntimeOperation | None = None
        self._generation_active = 0
        self._generation_queued = 0
        self._generation_current: str | None = None

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        self.config = config

    async def load(self, *, force: bool = False) -> dict[str, Any]:
        if force:
            async with self._generation_slot("reload"):
                async with self._state_lock:
                    return await self._load_without_reentrant_lock(force=True)
        async with self._state_lock:
            return await self._load_without_reentrant_lock(force=force)

    async def unload(self) -> dict[str, Any]:
        async with self._generation_slot("unload"):
            async with self._state_lock:
                self._start_operation("unload", phase="unloading", message="unloading model")
                self._state = "unloading"
                self._unload_locked()
                self._append_log("system", "unloaded")
                self._finish_operation(state="succeeded", phase="unloaded", message="unloaded")
                return self._status_payload(status="unloaded")

    async def reload(self) -> dict[str, Any]:
        return await self.load(force=True)

    async def close(self) -> None:
        await self.unload()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        media_parts: list[dict[str, Any]] | None = None,
        purpose: GenerationPurpose | str = GenerationPurpose.PERSONA_REPLY,
        generation_profile: GenerationProfile | None = None,
    ) -> str:
        profile = generation_profile or profile_for_purpose(self.config, purpose)
        async with self._generation_slot(profile.purpose.value):
            async with self._state_lock:
                structured_messages = self._structured_messages(messages)
                images = self._image_inputs_from_parts(media_parts or [])
                if self._model is None:
                    load_result = await self._load_without_reentrant_lock(force=False)
                    if load_result.get("status") not in {"loaded", "already_loaded"}:
                        raise RuntimeError(str(load_result.get("error") or load_result.get("status")))

                assert self._model is not None
                assert self._processor is not None
                model = self._model
                processor = self._processor
                model_config = getattr(model, "config", object())
                templated = self._backend.apply_chat_template(
                    processor,
                    model_config,
                    structured_messages,
                    num_images=len(images),
                )

            started = time.perf_counter()
            try:
                output = await asyncio.to_thread(
                    self._backend.generate,
                    model,
                    processor,
                    templated,
                    images,
                    max_tokens=profile.max_tokens,
                    temperature=profile.temperature,
                    top_p=profile.top_p,
                    repetition_penalty=profile.repetition_penalty,
                    repetition_context_size=profile.repetition_context_size,
                )
            except Exception as exc:
                self._append_log(
                    "stderr",
                    f"generate failed: {type(exc).__name__}: {exc}",
                    level="error",
                    event="generate_failed",
                )
                raise
            content = self._validate_generated_text(self._extract_text(output))
            content = self._truncate_repetitive_generated_text(
                content,
                reject=profile.reject_repetition,
                purpose=profile.purpose.value,
            )
            self._record_metric(
                content,
                time.perf_counter() - started,
                output=output,
                purpose=profile.purpose.value,
            )
            return content

    async def caption_media(self, part: dict[str, Any]) -> str:
        uri = self._image_input_from_part(part)
        if not uri:
            self._append_log(
                "system",
                "caption media skipped: no supported image uri",
                level="warn",
                event="caption_media_skipped",
                context={"type": str(part.get("type") or "")},
            )
            return ""
        prompt = "请用一句简短中文描述这张图片或表情的可见内容，不要复述文件名或链接。"
        return await self.chat(
            [{"role": "user", "content": prompt}],
            media_parts=[part],
            purpose=GenerationPurpose.MEDIA_CAPTION,
        )

    def status_snapshot(self) -> dict[str, Any]:
        payload = self._status_payload(status=self._state)
        payload["metrics"] = self.metrics_snapshot()
        payload["generation"] = self._generation_snapshot()
        return payload

    def logs(self, *, tail: int = 200) -> list[dict[str, Any]]:
        tail = max(0, min(int(tail), 1000))
        items = list(self._logs)[-tail:] if tail else []
        return [asdict(item) for item in items]

    def metrics_snapshot(self) -> dict[str, Any]:
        history = [asdict(item) for item in self._metrics]
        return {"last": history[-1] if history else None, "history": history}

    async def _load_without_reentrant_lock(self, *, force: bool) -> dict[str, Any]:
        if self._model is not None and not force:
            return self._status_payload(status="already_loaded")
        model_name = effective_mlx_model(self.config)
        adapter_path = self._adapter_path()
        operation_type = "reload" if force else "load"
        self._start_operation(
            operation_type,
            phase="preparing",
            message=f"preparing model={model_name}",
            target_model=model_name,
        )
        if force:
            self._unload_locked()
        if not self._backend_available():
            self._state = "missing_dependency"
            self._last_error = "mlx-vlm is not installed"
            self._append_log("system", self._last_error, level="error", event="missing_dependency")
            self._finish_operation(
                state="failed",
                phase="missing_dependency",
                error=self._last_error,
            )
            return self._status_payload(status="missing_dependency")

        try:
            prepared = await self._prepare_model(model_name)
        except Exception as exc:
            self._state = "failed"
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._append_log("stderr", self._last_error, level="error", event="prepare_failed")
            self._finish_operation(state="failed", phase="preparing", error=self._last_error)
            return self._status_payload(status="failed")

        self._state = "loading"
        self._last_error = ""
        self._update_operation(
            phase="loading_model",
            message=f"loading model={model_name}",
            percent=None,
            local_path=str(prepared.local_path),
        )
        self._append_log(
            "system",
            f"loading model={model_name} path={prepared.local_path} adapter={adapter_path or '-'}",
            event="load_start",
            context={"model": model_name, "local_path": str(prepared.local_path), "adapter": adapter_path or ""},
        )
        started = time.perf_counter()
        try:
            model, processor = await asyncio.to_thread(
                self._backend.load,
                str(prepared.local_path),
                adapter_path=adapter_path,
                trust_remote_code=bool(self.config.qwen_persona_mlx_trust_remote_code),
            )
        except Exception as exc:
            self._state = "failed"
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._append_log("stderr", self._last_error, level="error", event="load_failed")
            self._finish_operation(state="failed", phase="loading_model", error=self._last_error)
            return self._status_payload(status="failed")
        self._model = model
        self._processor = self._with_safe_detokenizer(processor)
        self._loaded_model = model_name
        self._loaded_adapter = adapter_path or ""
        self._loaded_model_path = str(prepared.local_path)
        self._state = "loaded"
        elapsed = time.perf_counter() - started
        self._append_log("system", f"loaded in {elapsed:.2f}s", event="load_succeeded")
        self._finish_operation(
            state="succeeded",
            phase="loaded",
            message=f"loaded in {elapsed:.2f}s",
        )
        return self._status_payload(status="loaded")

    def _unload_locked(self) -> None:
        self._model = None
        self._processor = None
        self._loaded_model = ""
        self._loaded_adapter = ""
        self._loaded_model_path = ""
        self._state = "not_loaded"
        gc.collect()
        try:
            import mlx.core as mx
        except ImportError:
            return
        try:
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
            else:
                mx.metal.clear_cache()
        except Exception as exc:
            self._append_log(
                "stderr",
                f"mlx metal cache clear failed: {type(exc).__name__}: {exc}",
                level="warn",
                event="clear_cache_failed",
            )

    def _status_payload(self, *, status: str) -> dict[str, Any]:
        return {
            "runtime": "mlx_vlm",
            "status": status,
            "state": self._state,
            "healthy": self._state == "loaded",
            "model": effective_mlx_model(self.config),
            "loaded_model": self._loaded_model,
            "loaded_model_path": self._loaded_model_path,
            "adapter_path": self._adapter_path() or "",
            "loaded_adapter_path": self._loaded_adapter,
            "last_error": self._last_error,
            "operation": self._operation_snapshot(),
            "generation": self._generation_snapshot(),
            "control": {
                "can_load": self._state not in {"preparing", "downloading", "loading", "unloading"},
                "can_unload": self._model is not None,
                "can_reload": self._state not in {"preparing", "downloading", "loading", "unloading"},
            },
        }

    def _backend_available(self) -> bool:
        if not isinstance(self._backend, DefaultMlxBackend):
            return True
        return importlib.util.find_spec("mlx_vlm") is not None

    def _adapter_path(self) -> str | None:
        value = str(getattr(self.config, "qwen_persona_mlx_adapter_path", "") or "").strip()
        return value or None

    async def _prepare_model(self, model_name: str) -> PreparedModel:
        if not self._managed_downloads:
            self._update_operation(phase="preparing", message=f"using direct model ref {model_name}")
            return PreparedModel(local_path=Path(model_name), source="direct")
        if is_existing_local_model(model_name):
            local_path = Path(model_name).expanduser()
            self._update_operation(
                phase="preparing",
                message=f"using local model {local_path}",
                local_path=str(local_path),
            )
            return PreparedModel(local_path=local_path, source="local")
        if not is_hf_repo_ref(model_name):
            self._append_log(
                "system",
                f"using direct model ref without managed download: {model_name}",
                level="warn",
                event="direct_model_ref",
                context={"model": model_name},
            )
            return PreparedModel(local_path=Path(model_name), source="direct")

        model_dir = Path(str(getattr(self.config, "qwen_persona_model_dir", "") or "data/qwen_persona/models"))
        progress = OperationProgress(self._update_download_progress)
        return await asyncio.to_thread(
            self._download_manager.prepare_model,
            model_name,
            model_dir=model_dir,
            progress=progress,
        )

    def _start_operation(
        self,
        operation_type: str,
        *,
        phase: str,
        message: str,
        target_model: str = "",
    ) -> None:
        with self._thread_lock:
            self._operation = RuntimeOperation(
                type=operation_type,
                phase=phase,
                message=message,
                target_model=target_model,
                previous_model=self._loaded_model,
            )

    def _update_operation(self, **kwargs: Any) -> None:
        with self._thread_lock:
            if self._operation is not None:
                self._operation.update(**kwargs)

    def _update_download_progress(self, **kwargs: Any) -> None:
        with self._thread_lock:
            phase = str(kwargs.get("phase") or "")
            if phase in {"preparing", "downloading"}:
                self._state = phase
            if self._operation is not None:
                self._operation.update(**kwargs)

    def _finish_operation(self, *, state: str, phase: str, message: str = "", error: str = "") -> None:
        with self._thread_lock:
            if self._operation is not None:
                self._operation.finish(state=state, phase=phase, message=message, error=error)

    def _operation_snapshot(self) -> dict[str, Any] | None:
        with self._thread_lock:
            return self._operation.to_dict() if self._operation is not None else None

    def _generation_snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            return {
                "active": self._generation_active,
                "queued": self._generation_queued,
                "current": self._generation_current,
            }

    @asynccontextmanager
    async def _generation_slot(self, operation: str) -> AsyncIterator[None]:
        acquired = False
        with self._thread_lock:
            self._generation_queued += 1
            queued = self._generation_queued
            active = self._generation_active
        if active or queued > 1:
            self._append_log(
                "system",
                f"generation queued: {operation}",
                event="generation_queued",
                context={"operation": operation, "active": active, "queued": queued},
            )
        try:
            await self._generation_gate.acquire()
            acquired = True
            with self._thread_lock:
                self._generation_queued = max(0, self._generation_queued - 1)
                self._generation_active += 1
                self._generation_current = operation
            self._append_log(
                "system",
                f"generation started: {operation}",
                event="generation_started",
                context={"operation": operation},
            )
            yield
        finally:
            if acquired:
                with self._thread_lock:
                    self._generation_active = max(0, self._generation_active - 1)
                    if self._generation_active == 0:
                        self._generation_current = None
                self._generation_gate.release()
                self._append_log(
                    "system",
                    f"generation finished: {operation}",
                    event="generation_finished",
                    context={"operation": operation},
                )
            else:
                with self._thread_lock:
                    self._generation_queued = max(0, self._generation_queued - 1)

    def _append_backend_warning(self, event: str, message: str, context: dict[str, Any]) -> None:
        self._append_log("stderr", message, level="warn", event=event, context=context)

    def _append_log(
        self,
        stream: str,
        message: str,
        *,
        level: str = "info",
        event: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        self._logs.append(
            MlxRuntimeLog(
                time=utc_timestamp(),
                stream=stream,
                message=self._redact_text(message),
                level=level,
                event=event,
                context=self._redact_context(context or {}),
            )
        )

    def _with_safe_detokenizer(self, processor: Any) -> Any:
        detokenizer = getattr(processor, "detokenizer", None)
        if detokenizer is None or detokenizer.__class__.__name__ != "BPEStreamingDetokenizer":
            return processor
        try:
            processor.detokenizer = _SafeBPEStreamingDetokenizer(detokenizer)
            self._append_log("system", "patched BPE streaming detokenizer")
        except Exception as exc:
            self._append_log("stderr", f"detokenizer patch skipped: {type(exc).__name__}: {exc}")
        return processor

    @staticmethod
    def _redact_text(text: str) -> str:
        redacted = str(text or "")
        for marker in ("api_key", "token", "secret", "password"):
            redacted = redacted.replace(marker + "=", marker + "=***")
        return redacted

    @classmethod
    def _redact_context(cls, context: dict[str, Any]) -> dict[str, Any]:
        redacted: dict[str, Any] = {}
        for key, value in context.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("api_key", "token", "secret", "password")):
                redacted[key_text] = "***"
            elif isinstance(value, str):
                redacted[key_text] = cls._redact_text(value)
            else:
                redacted[key_text] = value
        return redacted

    @classmethod
    def _structured_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        structured: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = item.get("content")
            if not cls._message_content_text(content).strip():
                continue
            structured.append({"role": role, "content": content})
        return structured or [{"role": "user", "content": "请自然接一句。"}]

    @staticmethod
    def _message_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        chunks.append(str(text))
                elif part:
                    chunks.append(str(part))
            return "\n".join(chunks)
        return str(content or "")

    def _image_inputs_from_parts(self, parts: list[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for part in parts:
            value = self._image_input_from_part(part)
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    def _image_input_from_part(self, part: dict[str, Any]) -> str:
        part_type = str(part.get("type") or "")
        if part_type not in {"image", "mface", "emoji", "sticker"}:
            if part_type:
                self._append_log(
                    "system",
                    f"media part ignored: unsupported type {part_type}",
                    level="warn",
                    event="media_part_ignored",
                    context={"type": part_type},
                )
            return ""
        data = part.get("data")
        if not isinstance(data, dict):
            self._append_log(
                "system",
                "media part ignored: missing data",
                level="warn",
                event="media_part_ignored",
                context={"type": part_type},
            )
            return ""
        for field in ("url", "file", "file_id", "uri"):
            value = str(data.get(field) or "").strip()
            if not value or value.startswith("base64://"):
                continue
            return value
        self._append_log(
            "system",
            "media part ignored: no supported uri field",
            level="warn",
            event="media_part_ignored",
            context={"type": part_type},
        )
        return ""

    @staticmethod
    def _extract_text(output: Any) -> str:
        for attr in ("text", "content", "response"):
            value = getattr(output, attr, None)
            if value is not None:
                return str(value).strip()
        if isinstance(output, dict):
            for key in ("text", "content", "response"):
                value = output.get(key)
                if value is not None:
                    return str(value).strip()
        return str(output or "").strip()

    def _validate_generated_text(self, content: str) -> str:
        if REPLACEMENT_CHAR not in content:
            return content
        replacement_count = content.count(REPLACEMENT_CHAR)
        response_chars = len(content)
        cleaned = content.replace(REPLACEMENT_CHAR, "").strip()
        replacement_ratio = replacement_count / max(1, response_chars)
        context = {
            "replacement_count": replacement_count,
            "response_chars": response_chars,
            "replacement_ratio": round(replacement_ratio, 3),
            "cleaned_chars": len(cleaned),
        }
        if not cleaned or self._mostly_replacement_text(
            replacement_count=replacement_count,
            response_chars=response_chars,
            cleaned_chars=len(cleaned),
        ):
            self._append_log(
                "stderr",
                "invalid generated text: contains replacement characters",
                level="error",
                event="invalid_generated_text",
                context=context,
            )
            raise ValueError("invalid generated text: contains replacement characters")
        self._append_log(
            "stderr",
            "generated text contained replacement characters; sanitized output",
            level="warn",
            event="generated_text_sanitized",
            context=context,
        )
        return cleaned

    @staticmethod
    def _mostly_replacement_text(*, replacement_count: int, response_chars: int, cleaned_chars: int) -> bool:
        if replacement_count >= 3 and replacement_count > cleaned_chars:
            return True
        return replacement_count >= 2 and (replacement_count / max(1, response_chars)) >= 0.5

    def _truncate_repetitive_generated_text(
        self,
        content: str,
        *,
        reject: bool = False,
        purpose: str = "",
    ) -> str:
        result = trim_repetitive_tail(content)
        if not result.changed:
            return content
        self._append_log(
            "stderr",
            "generated text ended with repeated tail; truncated output",
            level="warn",
            event="generated_text_repetition_truncated",
            context={
                "original_chars": result.original_chars,
                "truncated_chars": result.truncated_chars,
                "repeated_unit": result.repeated_unit,
                "repeated_count": result.repeated_count,
                "cleaned_chars": len(result.text),
                "rejected": bool(reject),
                "purpose": purpose,
            },
        )
        return "" if reject else result.text

    def _record_metric(
        self,
        content: str,
        elapsed: float,
        *,
        output: Any | None = None,
        purpose: str = "",
    ) -> None:
        if elapsed <= 0:
            return
        tokens = self._completion_tokens_from_output(output)
        source = "mlx_vlm" if tokens is not None else "estimated"
        if tokens is None:
            tokens = self._estimate_tokens(content)
        if not tokens:
            return
        self._metrics.append(
            InferenceMetric(
                time=utc_timestamp(),
                model=effective_mlx_model(self.config),
                latency_seconds=round(elapsed, 3),
                completion_tokens=tokens if source == "mlx_vlm" else None,
                response_chars=len(content),
                tps=round(tokens / elapsed, 2),
                source=source,
                finish_reason=self._finish_reason_from_output(output),
                purpose=purpose,
            )
        )

    @staticmethod
    def _completion_tokens_from_output(output: Any | None) -> int | None:
        value = _output_value(output, "generation_tokens")
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            return None
        return tokens if tokens > 0 else None

    @staticmethod
    def _finish_reason_from_output(output: Any | None) -> str:
        return str(_output_value(output, "finish_reason") or "")

    @staticmethod
    def _estimate_tokens(content: str) -> int | None:
        content = str(content or "").strip()
        if not content:
            return None
        return max(1, round(len(content) / 2))


def _output_value(output: Any | None, key: str) -> Any:
    if output is None:
        return None
    value = getattr(output, key, None)
    if value is not None:
        return value
    if isinstance(output, dict):
        return output.get(key)
    return None


class _SafeBPEStreamingDetokenizer:
    def __init__(self, source: Any) -> None:
        self.trim_space = bool(getattr(source, "trim_space", False))
        self.tokenmap = list(getattr(source, "tokenmap"))
        self._byte_decoder = dict(getattr(source, "_byte_decoder"))
        self.reset()

    def __copy__(self) -> "_SafeBPEStreamingDetokenizer":
        copied = object.__new__(type(self))
        copied.trim_space = self.trim_space
        copied.tokenmap = self.tokenmap
        copied._byte_decoder = self._byte_decoder
        copied.reset()
        return copied

    def reset(self) -> None:
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens: list[int] = []

    def add_token(self, token: int, skip_special_token_ids: list[int] | set[int] = []) -> None:
        if token in skip_special_token_ids:
            return
        token_id = int(token)
        if token_id < 0 or token_id >= len(self.tokenmap):
            return
        value = self.tokenmap[token_id]
        if not value:
            return
        if self._is_space_prefixed(value):
            current_text = self._decode_unflushed()
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_leading_space(current_text)
            self._unflushed = value
        else:
            self._unflushed += value
        self.tokens.append(token_id)

    def finalize(self) -> None:
        current_text = self._decode_unflushed()
        if self.text or not self.trim_space:
            self.text += current_text
        else:
            self.text += _remove_leading_space(current_text)
        self._unflushed = ""

    @property
    def last_segment(self) -> str:
        if self.text and self.text[-1] == REPLACEMENT_CHAR:
            return ""
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment

    def _is_space_prefixed(self, value: str) -> bool:
        try:
            return self._byte_decoder[value[0]] == 32
        except (KeyError, IndexError):
            return False

    def _decode_unflushed(self) -> str:
        if not self._unflushed:
            return ""
        try:
            raw = bytearray(self._byte_decoder[c] for c in self._unflushed)
        except KeyError:
            return self._unflushed
        return raw.decode("utf-8", errors="ignore")


def _remove_leading_space(value: str) -> str:
    return value[1:] if value.startswith(" ") else value
