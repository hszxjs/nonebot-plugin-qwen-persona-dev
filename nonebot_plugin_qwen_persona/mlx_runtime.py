from __future__ import annotations

import asyncio
import gc
import importlib.util
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import QwenPersonaConfig, effective_mlx_model
from .prompt_budget import effective_max_tokens


@dataclass(frozen=True)
class MlxRuntimeLog:
    time: str
    stream: str
    message: str


@dataclass(frozen=True)
class InferenceMetric:
    time: str
    model: str
    latency_seconds: float
    completion_tokens: int | None
    response_chars: int
    tps: float
    source: str


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
    def load(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        trust_remote_code: bool,
    ) -> tuple[Any, Any]:
        from mlx_vlm import load

        kwargs: dict[str, Any] = {}
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        try:
            return load(model_path, **kwargs)
        except TypeError:
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
            "verbose": False,
        }
        if images:
            kwargs["image"] = images
        try:
            return generate(model, processor, prompt, **kwargs)
        except TypeError:
            kwargs.pop("top_p", None)
            return generate(model, processor, prompt, **kwargs)


class MlxVLMRuntime:
    def __init__(
        self,
        config: QwenPersonaConfig,
        *,
        backend: MlxBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend or DefaultMlxBackend()
        self._lock = asyncio.Lock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._state = "not_loaded"
        self._last_error = ""
        self._logs: deque[MlxRuntimeLog] = deque(maxlen=400)
        self._metrics: deque[InferenceMetric] = deque(maxlen=120)
        self._loaded_model = ""
        self._loaded_adapter = ""

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        self.config = config

    async def load(self, *, force: bool = False) -> dict[str, Any]:
        async with self._lock:
            if self._model is not None and not force:
                return self._status_payload(status="already_loaded")
            if force:
                self._unload_locked()
            if not self._backend_available():
                self._state = "missing_dependency"
                self._last_error = "mlx-vlm is not installed"
                self._append_log("system", self._last_error)
                return self._status_payload(status="missing_dependency")

            model_name = effective_mlx_model(self.config)
            adapter_path = self._adapter_path()
            self._state = "loading"
            self._last_error = ""
            self._append_log("system", f"loading model={model_name} adapter={adapter_path or '-'}")
            started = time.perf_counter()
            try:
                model, processor = await asyncio.to_thread(
                    self._backend.load,
                    model_name,
                    adapter_path=adapter_path,
                    trust_remote_code=bool(self.config.qwen_persona_mlx_trust_remote_code),
                )
            except Exception as exc:
                self._state = "failed"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._append_log("stderr", self._last_error)
                return self._status_payload(status="failed")

            self._model = model
            self._processor = processor
            self._loaded_model = model_name
            self._loaded_adapter = adapter_path or ""
            self._state = "loaded"
            elapsed = time.perf_counter() - started
            self._append_log("system", f"loaded in {elapsed:.2f}s")
            return self._status_payload(status="loaded")

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            self._unload_locked()
            self._append_log("system", "unloaded")
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
    ) -> str:
        async with self._lock:
            if self._model is None:
                load_result = await self._load_without_reentrant_lock(force=False)
                if load_result.get("status") not in {"loaded", "already_loaded"}:
                    raise RuntimeError(str(load_result.get("error") or load_result.get("status")))

            assert self._model is not None
            assert self._processor is not None
            images = self._image_inputs_from_parts(media_parts or [])
            prompt = self._prompt_from_messages(messages)
            model_config = getattr(self._model, "config", object())
            templated = self._backend.apply_chat_template(
                self._processor,
                model_config,
                prompt,
                num_images=len(images),
            )

            started = time.perf_counter()
            output = await asyncio.to_thread(
                self._backend.generate,
                self._model,
                self._processor,
                templated,
                images,
                max_tokens=effective_max_tokens(self.config),
                temperature=float(self.config.qwen_persona_temperature),
                top_p=float(self.config.qwen_persona_top_p),
            )
            content = self._extract_text(output)
            self._record_metric(content, time.perf_counter() - started)
            return content

    async def caption_media(self, part: dict[str, Any]) -> str:
        uri = self._image_input_from_part(part)
        if not uri:
            return ""
        prompt = "请用一句简短中文描述这张图片或表情的可见内容，不要复述文件名或链接。"
        return await self.chat([{"role": "user", "content": prompt}], media_parts=[part])

    def status_snapshot(self) -> dict[str, Any]:
        payload = self._status_payload(status=self._state)
        payload["metrics"] = self.metrics_snapshot()
        return payload

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        tail = max(0, min(int(tail), 1000))
        items = list(self._logs)[-tail:] if tail else []
        return [asdict(item) for item in items]

    def metrics_snapshot(self) -> dict[str, Any]:
        history = [asdict(item) for item in self._metrics]
        return {"last": history[-1] if history else None, "history": history}

    async def _load_without_reentrant_lock(self, *, force: bool) -> dict[str, Any]:
        if self._model is not None and not force:
            return self._status_payload(status="already_loaded")
        if force:
            self._unload_locked()
        if not self._backend_available():
            self._state = "missing_dependency"
            self._last_error = "mlx-vlm is not installed"
            self._append_log("system", self._last_error)
            return self._status_payload(status="missing_dependency")
        model_name = effective_mlx_model(self.config)
        adapter_path = self._adapter_path()
        self._state = "loading"
        self._last_error = ""
        self._append_log("system", f"loading model={model_name} adapter={adapter_path or '-'}")
        try:
            model, processor = await asyncio.to_thread(
                self._backend.load,
                model_name,
                adapter_path=adapter_path,
                trust_remote_code=bool(self.config.qwen_persona_mlx_trust_remote_code),
            )
        except Exception as exc:
            self._state = "failed"
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._append_log("stderr", self._last_error)
            return self._status_payload(status="failed")
        self._model = model
        self._processor = processor
        self._loaded_model = model_name
        self._loaded_adapter = adapter_path or ""
        self._state = "loaded"
        self._append_log("system", "loaded")
        return self._status_payload(status="loaded")

    def _unload_locked(self) -> None:
        self._model = None
        self._processor = None
        self._loaded_model = ""
        self._loaded_adapter = ""
        self._state = "not_loaded"
        gc.collect()
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass

    def _status_payload(self, *, status: str) -> dict[str, Any]:
        return {
            "runtime": "mlx_vlm",
            "status": status,
            "state": self._state,
            "healthy": self._state == "loaded",
            "model": effective_mlx_model(self.config),
            "loaded_model": self._loaded_model,
            "adapter_path": self._adapter_path() or "",
            "loaded_adapter_path": self._loaded_adapter,
            "last_error": self._last_error,
            "control": {
                "can_load": self._state not in {"loading"},
                "can_unload": self._model is not None,
                "can_reload": self._state not in {"loading"},
            },
        }

    def _backend_available(self) -> bool:
        if not isinstance(self._backend, DefaultMlxBackend):
            return True
        return importlib.util.find_spec("mlx_vlm") is not None

    def _adapter_path(self) -> str | None:
        value = str(getattr(self.config, "qwen_persona_mlx_adapter_path", "") or "").strip()
        return value or None

    def _append_log(self, stream: str, message: str) -> None:
        self._logs.append(
            MlxRuntimeLog(
                time=datetime.now(UTC).isoformat(timespec="seconds"),
                stream=stream,
                message=self._redact_text(message),
            )
        )

    @staticmethod
    def _redact_text(text: str) -> str:
        redacted = str(text or "")
        for marker in ("api_key", "token", "secret", "password"):
            redacted = redacted.replace(marker + "=", marker + "=***")
        return redacted

    @staticmethod
    def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in messages:
            role = str(item.get("role") or "user").strip() or "user"
            content = MlxVLMRuntime._message_content_text(item.get("content"))
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines) or "user: 请自然接一句。"

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

    @classmethod
    def _image_inputs_from_parts(cls, parts: list[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for part in parts:
            value = cls._image_input_from_part(part)
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    @staticmethod
    def _image_input_from_part(part: dict[str, Any]) -> str:
        part_type = str(part.get("type") or "")
        if part_type not in {"image", "mface", "emoji", "sticker"}:
            return ""
        data = part.get("data")
        if not isinstance(data, dict):
            return ""
        for field in ("url", "file", "file_id", "uri"):
            value = str(data.get(field) or "").strip()
            if not value or value.startswith("base64://"):
                continue
            return value
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

    def _record_metric(self, content: str, elapsed: float) -> None:
        if elapsed <= 0:
            return
        tokens = self._estimate_tokens(content)
        if not tokens:
            return
        self._metrics.append(
            InferenceMetric(
                time=datetime.now(UTC).isoformat(timespec="seconds"),
                model=effective_mlx_model(self.config),
                latency_seconds=round(elapsed, 3),
                completion_tokens=None,
                response_chars=len(content),
                tps=round(tokens / elapsed, 2),
                source="estimated",
            )
        )

    @staticmethod
    def _estimate_tokens(content: str) -> int | None:
        content = str(content or "").strip()
        if not content:
            return None
        return max(1, round(len(content) / 2))
