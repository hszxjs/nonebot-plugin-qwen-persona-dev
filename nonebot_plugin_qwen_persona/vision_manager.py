from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import httpx

from .config import QwenPersonaConfig


@dataclass(frozen=True)
class VisionDeployResult:
    status: str
    endpoint: str
    log: str = ""


class VisionManager:
    def __init__(self, config: QwenPersonaConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._started_at: float | None = None
        self._started_by_plugin = False
        self._logs: deque[dict[str, str]] = deque(maxlen=1000)
        self._log_tasks: set[asyncio.Task[None]] = set()
        self._last_command: list[str] = []
        self.last_result: VisionDeployResult | None = None

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        self.config = config

    async def ensure_ready(self, *, force: bool = False) -> VisionDeployResult:
        async with self._lock:
            result = await self._ensure_ready_locked(force=force)
            self.last_result = result
            return result

    async def start(self) -> VisionDeployResult:
        return await self.ensure_ready(force=True)

    async def stop(self) -> VisionDeployResult:
        async with self._lock:
            result = await self._stop_locked()
            self.last_result = result
            return result

    async def restart(self) -> VisionDeployResult:
        async with self._lock:
            await self._stop_locked()
            result = await self._ensure_ready_locked(force=True)
            self.last_result = result
            return result

    async def close(self) -> None:
        await self.stop()

    async def caption_media(self, part: dict[str, Any]) -> str:
        if not self.config.qwen_persona_vision_enabled:
            return ""
        if self.config.qwen_persona_vision_auto_start:
            await self.ensure_ready()
        payload = self._caption_payload(part)
        try:
            self._append_log("caption", f"caption request type={payload.get('type', 'media')}")
            data = await self._post_caption_payload(payload)
            caption = str(data.get("caption") or "").strip()
            source = str(data.get("source") or "unknown")
            error = str(data.get("error") or "").strip()
            if source == "fallback":
                self._append_log(
                    "caption",
                    f"vision caption fallback: {error or 'no visual caption available'}",
                )
                return ""
            if caption:
                self._append_log("caption", f"source={source} caption={caption}")
            return caption
        except Exception as exc:
            self._append_log("system", f"vision caption failed: {type(exc).__name__}: {exc}")
            return ""

    async def _post_caption_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.qwen_persona_vision_timeout) as client:
            response = await client.post(f"{self.endpoint}/caption", json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def status_snapshot(self) -> dict[str, Any]:
        if not self.config.qwen_persona_vision_enabled:
            state = "vision_disabled"
            healthy = False
            can_start = can_stop = False
            reason = "vision captioning is disabled"
        else:
            process = self._process
            running = bool(process and process.returncode is None)
            healthy = await self._service_healthy()
            if running:
                state = "vision_running" if healthy else "vision_starting"
            elif healthy:
                state = "vision_present"
            elif process and process.returncode is not None:
                state = "vision_exited"
            else:
                state = "vision_stopped"
            can_start = bool(self.config.qwen_persona_vision_auto_start and not running)
            can_stop = bool(running and self._started_by_plugin)
            reason = "" if self.config.qwen_persona_vision_auto_start else "vision auto start is disabled"

        process = self._process
        return {
            "state": state,
            "healthy": healthy,
            "pid": getattr(process, "pid", None) if process else None,
            "returncode": getattr(process, "returncode", None) if process else None,
            "started_by_plugin": self._started_by_plugin,
            "uptime_seconds": (
                max(0, int(time.monotonic() - self._started_at))
                if self._started_at is not None and process and process.returncode is None
                else None
            ),
            "endpoint": self.endpoint,
            "model": self.config.qwen_persona_vision_model,
            "command": " ".join(shlex.quote(item) for item in self._last_command),
            "last_result": (
                {
                    "status": self.last_result.status,
                    "endpoint": self.last_result.endpoint,
                    "log": self.last_result.log,
                }
                if self.last_result
                else None
            ),
            "control": {
                "can_start": can_start,
                "can_stop": can_stop,
                "can_restart": can_start or can_stop,
                "reason": reason,
            },
        }

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        if tail <= 0:
            return []
        return list(self._logs)[-min(tail, self._logs.maxlen or tail) :]

    @property
    def endpoint(self) -> str:
        host = self.config.qwen_persona_vision_host.strip() or "127.0.0.1"
        return f"http://{host}:{int(self.config.qwen_persona_vision_port)}"

    async def _ensure_ready_locked(self, *, force: bool) -> VisionDeployResult:
        if not self.config.qwen_persona_vision_enabled:
            self._append_log("system", "vision captioning disabled")
            return VisionDeployResult("vision_disabled", self.endpoint, "vision captioning disabled")
        if self._process and self._process.returncode is None:
            return VisionDeployResult("vision_running", self.endpoint)
        if await self._service_healthy():
            self._append_log("system", f"compatible vision service present at {self.endpoint}")
            return VisionDeployResult("vision_present", self.endpoint)
        if not self.config.qwen_persona_vision_auto_start:
            self._append_log("system", "vision service unavailable and auto start disabled")
            return VisionDeployResult("vision_unavailable", self.endpoint, "auto start disabled")

        command = self._build_vision_command()
        self._last_command = command
        env = os.environ.copy()
        env["QWEN_PERSONA_VISION_MODEL"] = self.config.qwen_persona_vision_model
        self._append_log("system", "starting vision service: " + " ".join(shlex.quote(item) for item in command))
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._started_at = time.monotonic()
            self._started_by_plugin = True
            self._start_log_reader(self._process.stdout, "stdout")
            self._start_log_reader(self._process.stderr, "stderr")
            await self._wait_for_health()
        except Exception as exc:
            self._append_log("system", f"vision service start failed: {type(exc).__name__}: {exc}")
            await self._stop_locked()
            return VisionDeployResult("vision_start_failed", self.endpoint, f"{type(exc).__name__}: {exc}")
        return VisionDeployResult("vision_started", self.endpoint)

    async def _stop_locked(self) -> VisionDeployResult:
        process = self._process
        if not process:
            self._started_by_plugin = False
            self._started_at = None
            return VisionDeployResult("vision_not_running", self.endpoint)
        if process.returncode is not None:
            self._process = None
            self._started_by_plugin = False
            self._started_at = None
            await self._cancel_log_tasks()
            return VisionDeployResult("vision_exited", self.endpoint, f"process exited with {process.returncode}")
        if not self._started_by_plugin:
            return VisionDeployResult("vision_not_owned", self.endpoint, "vision service was not started by this plugin")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except TimeoutError:
            process.kill()
            await process.wait()
        finally:
            self._process = None
            self._started_by_plugin = False
            self._started_at = None
            await self._cancel_log_tasks()
        return VisionDeployResult("vision_stopped", self.endpoint)

    def _build_vision_command(self) -> list[str]:
        python = self.config.qwen_persona_vision_python.strip() or sys.executable
        service_script = Path(__file__).with_name("vision_service.py")
        return [
            python,
            str(service_script),
            "--host",
            self.config.qwen_persona_vision_host.strip() or "127.0.0.1",
            "--port",
            str(int(self.config.qwen_persona_vision_port)),
            "--model",
            self.config.qwen_persona_vision_model.strip() or "Salesforce/blip-image-captioning-base",
        ]

    async def _service_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.endpoint}/health")
            return response.status_code < 500
        except Exception:
            return False

    async def _wait_for_health(self) -> None:
        deadline = asyncio.get_running_loop().time() + 45
        while asyncio.get_running_loop().time() < deadline:
            if await self._service_healthy():
                self._append_log("system", "vision service health check passed")
                return
            if self._process and self._process.returncode is not None:
                raise RuntimeError(f"vision process exited with {self._process.returncode}")
            await asyncio.sleep(1)
        raise TimeoutError(f"vision service did not become healthy: {self.endpoint}/health")

    def _caption_payload(self, part: dict[str, Any]) -> dict[str, Any]:
        data = dict(part.get("data") or {})
        return {
            "type": str(part.get("type") or "media"),
            "url": data.get("url") or data.get("file") or data.get("file_id") or "",
            "data": data,
        }

    def _start_log_reader(
        self,
        stream: asyncio.StreamReader | None,
        stream_name: str,
    ) -> None:
        if stream is None:
            return
        task = asyncio.create_task(self._drain_stream(stream, stream_name))
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)

    async def _drain_stream(self, stream: asyncio.StreamReader, stream_name: str) -> None:
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._append_log(stream_name, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_log("system", f"vision log reader failed: {type(exc).__name__}: {exc}")

    async def _cancel_log_tasks(self) -> None:
        tasks = [task for task in self._log_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._log_tasks.clear()

    def _append_log(self, stream: str, message: str) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for line in str(message or "").splitlines() or [""]:
            self._logs.append({"time": now, "stream": stream, "message": line[:2000]})


class DisabledVisionCaptioner:
    async def caption_media(self, part: dict[str, Any]) -> str:
        return ""
