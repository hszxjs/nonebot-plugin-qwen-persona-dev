from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
import json
import os
import platform
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import QwenPersonaConfig, effective_api_base


@dataclass(frozen=True)
class ModelDeployResult:
    status: str
    model_dir: str
    repo_id: str = ""
    log: str = ""


@dataclass(frozen=True)
class ManagedLlamaProcess:
    pid: int
    command: str


class ModelManager:
    def __init__(self, config: QwenPersonaConfig) -> None:
        self.config = config
        self.model_dir = Path(config.qwen_persona_model_dir)
        self._lock = asyncio.Lock()
        self._server_process: asyncio.subprocess.Process | None = None
        self._started_by_plugin = False
        self._started_at: float | None = None
        self._last_executable = ""
        self._last_command: list[str] = []
        self._log_tasks: set[asyncio.Task[None]] = set()
        self._logs: deque[dict[str, str]] = deque(maxlen=1000)
        self._tps_history: deque[dict[str, Any]] = deque(maxlen=120)
        self._observed_orphan_pid: int | None = None
        self.last_result: ModelDeployResult | None = None

    def reconfigure(self, config: QwenPersonaConfig) -> None:
        self.config = config
        self.model_dir = Path(config.qwen_persona_model_dir)

    async def ensure_ready(self, *, force: bool = False) -> ModelDeployResult:
        async with self._lock:
            result = await self._ensure_ready_locked(force=force)
            self.last_result = result
            return result

    async def close(self) -> None:
        await self.stop()

    async def start(self) -> ModelDeployResult:
        return await self.ensure_ready(force=True)

    async def stop(self) -> ModelDeployResult:
        async with self._lock:
            result = await self._stop_locked()
            self.last_result = result
            return result

    async def restart(self) -> ModelDeployResult:
        if self.config.qwen_persona_runtime in {"external_api", "external_llama"}:
            result = ModelDeployResult(
                f"{self.config.qwen_persona_runtime}_read_only",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=f"{self.config.qwen_persona_runtime} is managed outside this plugin",
            )
            self._append_log("system", result.log)
            self.last_result = result
            return result
        async with self._lock:
            stopped = await self._stop_locked()
            if stopped.status == "server_not_owned":
                self.last_result = stopped
                return stopped
        return await self.ensure_ready(force=True)

    async def reload(self) -> ModelDeployResult:
        async with self._lock:
            if self.config.qwen_persona_runtime in {"external_api", "external_llama"}:
                result = ModelDeployResult(
                    f"{self.config.qwen_persona_runtime}_read_only",
                    str(self.model_dir),
                    repo_id=self.config.qwen_persona_model if self.config.qwen_persona_runtime == "external_api" else self._model_ref(),
                    log=f"{self.config.qwen_persona_runtime} runtime is managed outside this plugin",
                )
                self._append_log("system", result.log)
                self.last_result = result
                return result

            process = self._server_process
            if process and process.returncode is None:
                if not self._started_by_plugin:
                    result = ModelDeployResult(
                        "server_not_owned",
                        str(self.model_dir),
                        repo_id=self._model_ref(),
                        log="llama service was not started by this plugin; reload requires a plugin-managed process",
                    )
                    self._append_log("system", result.log)
                    self.last_result = result
                    return result
                self._append_log("system", f"reloading llama.cpp model: {self._model_ref()}")
                await self._stop_locked()
            elif orphan := self._detect_managed_orphan():
                self._append_log(
                    "system",
                    f"reclaiming legacy plugin-managed llama process before reload: pid={orphan.pid}",
                )
                await asyncio.to_thread(self._terminate_pid, orphan.pid)
                self._observed_orphan_pid = None
            elif await self._service_healthy():
                summary = await self._service_models_summary()
                result = ModelDeployResult(
                    "server_not_owned",
                    str(self.model_dir),
                    repo_id=self._model_ref(),
                    log=(
                        f"compatible service is already present at {effective_api_base(self.config)}; "
                        f"{summary}; reload/switch requires a plugin-managed process or a free port"
                    ),
                )
                self._append_log("system", result.log)
                self.last_result = result
                return result
            elif self._port_accepts_tcp():
                result = ModelDeployResult(
                    "port_in_use",
                    str(self.model_dir),
                    repo_id=self._model_ref(),
                    log=f"{self._host()}:{self._port()} is occupied; reload/switch cannot replace that process",
                )
                self._append_log("system", result.log)
                self.last_result = result
                return result
            elif process and process.returncode is not None:
                self._server_process = None
                self._started_by_plugin = False
                self._started_at = None
                await self._cancel_log_tasks()

        return await self.ensure_ready(force=True)

    async def switch_model(self, model: str, model_file: str = "") -> ModelDeployResult:
        model = str(model or "").strip()
        model_file = str(model_file or "").strip()
        if not model and not model_file:
            result = ModelDeployResult(
                "model_switch_invalid",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log="model or model_file is required",
            )
            self._append_log("system", result.log)
            self.last_result = result
            return result
        if model:
            self.config.qwen_persona_llama_model = model
        self.config.qwen_persona_llama_model_file = model_file
        self._append_log(
            "system",
            f"model switch requested: model={self._model_ref()}, model_file={self._model_file() or '-'}",
        )
        return await self.reload()

    async def status_snapshot(self) -> dict[str, Any]:
        process = self._server_process
        process_running = bool(process and process.returncode is None)
        health = await self._service_healthy()
        port_open = False if health else self._port_accepts_tcp()
        orphan = None if process_running else self._detect_managed_orphan()
        if orphan and orphan.pid != self._observed_orphan_pid:
            self._observed_orphan_pid = orphan.pid
            self._append_log(
                "system",
                f"legacy plugin-managed llama process detected: pid={orphan.pid}; restart/reload/switch can reclaim it",
            )

        if self.config.qwen_persona_runtime == "external_api":
            state = "external_api_ready" if health else "external_api_unchecked"
            can_start = can_stop = can_restart = False
            can_reload = can_switch = False
            control_reason = "external_api runtime is read-only from this panel"
        elif self.config.qwen_persona_runtime == "external_llama":
            state = "external_llama_ready" if health else "external_llama_unavailable"
            can_start = can_stop = can_restart = False
            can_reload = can_switch = False
            control_reason = "external llama service is managed outside this plugin"
        elif process_running:
            state = "server_running" if health else "server_starting"
            can_start = False
            can_stop = self._started_by_plugin
            can_restart = self._started_by_plugin
            can_reload = can_switch = self._started_by_plugin
            control_reason = "" if self._started_by_plugin else "service was not started by this plugin"
        elif orphan:
            state = "server_orphaned" if health else "server_orphaned_unhealthy"
            can_start = True
            can_stop = True
            can_restart = True
            can_reload = True
            can_switch = True
            control_reason = "legacy plugin-managed llama process found; restart/reload/switch will reclaim it"
        elif health:
            state = "server_present"
            can_start = True
            can_stop = False
            can_restart = False
            can_reload = can_switch = True
            control_reason = "compatible service is already present; start will re-check it"
        elif port_open:
            state = "port_in_use"
            can_start = True
            can_stop = False
            can_restart = False
            can_reload = can_switch = True
            control_reason = "port is occupied; start will report diagnostics without stopping it"
        elif process and process.returncode is not None:
            state = "server_exited"
            can_start = True
            can_stop = False
            can_restart = True
            can_reload = can_switch = True
            control_reason = ""
        else:
            state = "server_stopped"
            can_start = True
            can_stop = False
            can_restart = True
            can_reload = can_switch = True
            control_reason = ""

        last = self.last_result
        uptime = None
        if process_running and self._started_at is not None:
            uptime = max(0, int(time.monotonic() - self._started_at))

        return {
            "runtime": self.config.qwen_persona_runtime,
            "state": state,
            "healthy": health,
            "port_open": port_open,
            "pid": getattr(process, "pid", None) if process else (orphan.pid if orphan else None),
            "returncode": getattr(process, "returncode", None) if process else None,
            "started_by_plugin": self._started_by_plugin,
            "claimable_by_plugin": bool(orphan),
            "uptime_seconds": uptime,
            "host": self._host(),
            "port": self._port(),
            "api_base": effective_api_base(self.config),
            "model": self._model_ref(),
            "model_file": str(self._model_file() or ""),
            "model_dir": str(self.model_dir),
            "executable": self._last_executable,
            "command": (
                self._redact_text(orphan.command)
                if orphan
                else self._redact_text(" ".join(shlex.quote(item) for item in self._last_command))
            ),
            "last_result": (
                {
                    "status": last.status,
                    "model_dir": last.model_dir,
                    "repo_id": last.repo_id,
                    "log": self._redact_text(last.log),
                }
                if last
                else None
            ),
            "control": {
                "can_start": can_start,
                "can_stop": can_stop,
                "can_restart": can_restart,
                "can_reload": can_reload,
                "can_switch": can_switch,
                "reason": control_reason,
            },
        }

    def logs(self, *, tail: int = 200) -> list[dict[str, str]]:
        if tail <= 0:
            return []
        return list(self._logs)[-min(tail, self._logs.maxlen or tail) :]

    def performance_snapshot(self) -> dict[str, Any]:
        history = list(self._tps_history)
        return {
            "last": history[-1] if history else None,
            "history": history,
        }

    async def _stop_locked(self) -> ModelDeployResult:
        if self.config.qwen_persona_runtime in {"external_api", "external_llama"}:
            return ModelDeployResult(
                f"{self.config.qwen_persona_runtime}_read_only",
                str(self.model_dir),
                repo_id=self.config.qwen_persona_model if self.config.qwen_persona_runtime == "external_api" else self._model_ref(),
                log=f"{self.config.qwen_persona_runtime} is managed outside this plugin",
            )
        process = self._server_process
        if not process:
            orphan = self._detect_managed_orphan()
            if orphan:
                message = f"stopping legacy plugin-managed llama process pid={orphan.pid}"
                self._append_log("system", message)
                await asyncio.to_thread(self._terminate_pid, orphan.pid)
                self._observed_orphan_pid = None
                return ModelDeployResult(
                    "orphan_stopped",
                    str(self.model_dir),
                    repo_id=self._model_ref(),
                    log=message,
                )
            self._started_by_plugin = False
            self._started_at = None
            return ModelDeployResult("server_not_running", str(self.model_dir), repo_id=self._model_ref())
        if process.returncode is not None:
            self._server_process = None
            self._started_by_plugin = False
            self._started_at = None
            await self._cancel_log_tasks()
            return ModelDeployResult(
                "server_exited",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=f"process exited with {process.returncode}",
            )
        if not self._started_by_plugin:
            return ModelDeployResult(
                "server_not_owned",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log="llama service was not started by this plugin; refusing to stop it",
            )

        self._append_log("system", "stopping llama.cpp server")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except TimeoutError:
            self._append_log("system", "llama.cpp server did not exit after terminate; killing")
            process.kill()
            await process.wait()
        finally:
            self._server_process = None
            self._started_by_plugin = False
            self._started_at = None
            await self._cancel_log_tasks()
        return ModelDeployResult("server_stopped", str(self.model_dir), repo_id=self._model_ref())

    async def _ensure_ready_locked(self, *, force: bool) -> ModelDeployResult:
        if self.config.qwen_persona_runtime == "external_api":
            status = "external_api_ready" if await self._service_healthy() else "external_api_unchecked"
            self._append_log("system", f"runtime external_api: {status}")
            return ModelDeployResult(status, str(self.model_dir), repo_id=self.config.qwen_persona_model)

        if self.config.qwen_persona_runtime == "external_llama":
            healthy = await self._service_healthy()
            status = "external_llama_ready" if healthy else "external_llama_unavailable"
            summary = await self._service_models_summary()
            self._append_log(
                "system",
                f"runtime external_llama: {status} at {effective_api_base(self.config)}; {summary}",
            )
            return ModelDeployResult(status, str(self.model_dir), repo_id=self._model_ref())

        if self._server_process and self._server_process.returncode is None:
            self._append_log("system", "llama.cpp process is already running")
            return ModelDeployResult("server_running", str(self.model_dir), repo_id=self._model_ref())

        if orphan := self._detect_managed_orphan():
            message = (
                f"legacy plugin-managed llama process present at {effective_api_base(self.config)}; "
                f"pid={orphan.pid}; use restart/reload/switch to reclaim stdout/stderr logs"
            )
            self._append_log("system", message)
            return ModelDeployResult(
                "server_orphaned",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=message,
            )

        if await self._service_healthy():
            summary = await self._service_models_summary()
            self._append_log(
                "system",
                f"compatible llama service present at {effective_api_base(self.config)}; {summary}",
            )
            return ModelDeployResult("server_present", str(self.model_dir), repo_id=self._model_ref())

        if self._port_accepts_tcp():
            self._append_log("system", f"{self._host()}:{self._port()} is occupied by a non-compatible service")
            return ModelDeployResult(
                "port_in_use",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=f"{self._host()}:{self._port()} is occupied by a non-compatible service",
            )

        model_file = self._model_file()
        if model_file and not model_file.exists():
            self._append_log("system", f"model file missing: {model_file}")
            return ModelDeployResult(
                "model_file_missing",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=str(model_file),
            )

        executable = self._find_llama_executable()
        if not executable:
            self._append_log("system", "llama.cpp executable not found; checking auto install")
            installed, install_log = await self._install_llama_cpp()
            if installed:
                executable = self._find_llama_executable()
            elif self.config.qwen_persona_llama_auto_install:
                self._append_log("system", f"llama.cpp auto install failed: {install_log}")
                return ModelDeployResult(
                    "llama_install_failed",
                    str(self.model_dir),
                    repo_id=self._model_ref(),
                    log=install_log,
                )

        if not executable:
            self._append_log("system", "llama-server or llama was not found after install check")
            return ModelDeployResult(
                "llama_missing",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log="llama-server or llama was not found. Set qwen_persona_llama_server_path or add it to PATH.",
            )

        command = self._build_llama_command(executable)
        self._last_executable = executable
        self._last_command = command
        self._append_log("system", "starting llama.cpp: " + " ".join(shlex.quote(item) for item in command))
        if not self._model_file():
            self._append_log("system", f"llama.cpp will resolve/download model via -hf: {self._model_ref()}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "QWEN_PERSONA_RUNTIME": self.config.qwen_persona_runtime,
                "QWEN_PERSONA_MODEL_DIR": str(self.model_dir),
                "QWEN_PERSONA_API_BASE": effective_api_base(self.config),
            }
        )

        try:
            self._server_process = await asyncio.create_subprocess_exec(
                *command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._started_by_plugin = True
            self._started_at = time.monotonic()
            self._start_log_reader(self._server_process.stdout, "stdout")
            self._start_log_reader(self._server_process.stderr, "stderr")
            await self._wait_for_server_health()
        except PermissionError as exc:
            self._append_log("system", f"llama permission error: {executable}: {exc}")
            return ModelDeployResult(
                "llama_permission_error",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=f"{executable}: {exc}",
            )
        except Exception as exc:
            failure_log = self._startup_failure_log(exc)
            self._append_log("system", failure_log)
            await self._stop_locked()
            return ModelDeployResult(
                "llama_start_failed",
                str(self.model_dir),
                repo_id=self._model_ref(),
                log=failure_log,
            )

        command_text = " ".join(shlex.quote(item) for item in command)
        return ModelDeployResult(
            "llama_started",
            str(self.model_dir),
            repo_id=self._model_ref(),
            log=self._redact_text(command_text),
        )

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
        buffer = ""
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    if buffer.strip():
                        self._append_log(stream_name, buffer.rstrip())
                    break
                text = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
                parts = (buffer + text).split("\n")
                buffer = parts.pop()
                for part in parts:
                    if part.strip():
                        self._append_log(stream_name, part.rstrip())
                if len(buffer) > 4000:
                    self._append_log(stream_name, buffer.rstrip())
                    buffer = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_log("system", f"log reader failed for {stream_name}: {type(exc).__name__}: {exc}")

    async def _cancel_log_tasks(self) -> None:
        tasks = [task for task in self._log_tasks if not task.done()]
        if not tasks:
            self._log_tasks.clear()
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._log_tasks.clear()

    def _append_log(self, stream: str, message: str) -> None:
        clean_message = self._redact_text(message)
        lines = clean_message.splitlines() or [clean_message]
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for line in lines:
            self._logs.append({"time": now, "stream": stream, "message": line})
            self._record_tps_from_log(now, stream, line)

    def _record_tps_from_log(self, timestamp: str, stream: str, line: str) -> None:
        match = re.search(
            r"(?P<tps>\d+(?:\.\d+)?)\s*(?:tokens?\s+per\s+second|tok/s|t/s)\b",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            return
        try:
            tps = float(match.group("tps"))
        except (TypeError, ValueError):
            return
        if tps <= 0:
            return
        self._tps_history.append(
            {
                "time": timestamp,
                "model": self._model_ref(),
                "latency_seconds": None,
                "completion_tokens": None,
                "response_chars": None,
                "tps": round(tps, 2),
                "source": "llama_log",
                "stream": stream,
            }
        )

    def _startup_failure_log(self, exc: Exception) -> str:
        recent = [
            item["message"]
            for item in self.logs(tail=40)
            if item.get("stream") in {"stderr", "system"}
        ]
        detail = "\n".join(recent[-20:]).strip()
        prefix = f"llama start failed: {type(exc).__name__}: {exc}"
        if self._looks_like_hf_resolution_failure(detail):
            hint = (
                "Hugging Face model could not be resolved by llama.cpp. "
                f"model={self._model_ref()}. "
                "Check network access/HF_TOKEN, choose a cached model, or set "
                "qwen_persona_llama_model_file to a local GGUF file."
            )
            return "\n".join(item for item in [prefix, hint, detail] if item)
        return "\n".join(item for item in [prefix, detail] if item)

    @staticmethod
    def _looks_like_hf_resolution_failure(text: str) -> bool:
        lowered = text.lower()
        return (
            "failed to resolve commit" in lowered
            or "get_repo_files" in lowered
            or "hf-repo" in lowered and "failed" in lowered
        )

    def _redact_text(self, message: str) -> str:
        text = str(message)
        api_key = self.config.qwen_persona_api_key.strip()
        if api_key and api_key.upper() != "EMPTY":
            text = text.replace(api_key, "[redacted]")
        text = re.sub(r"(--api-key(?:=|\s+))\S+", r"\1[redacted]", text)
        text = re.sub(r"(authorization:\s*bearer\s+)\S+", r"\1[redacted]", text, flags=re.IGNORECASE)
        return text

    def model_present(self) -> bool:
        model_file = self._model_file()
        return bool(model_file and model_file.exists() and model_file.suffix.lower() == ".gguf")

    def _find_llama_executable(self) -> str | None:
        configured = self.config.qwen_persona_llama_server_path.strip()
        candidates: list[str] = []
        if configured:
            candidates.append(configured)

        candidates.extend(str(path) for path in self._managed_binary_candidates())
        for root in self._managed_binary_roots():
            if root.exists() and (found := self._find_executable_under(root)):
                candidates.append(str(found))
        names = ["llama-server", "llama"]
        if sys.platform == "win32":
            names = ["llama-server.exe", "llama.exe", *names]
        candidates.extend(path for name in names if (path := shutil.which(name)))

        for candidate in candidates:
            resolved = shutil.which(candidate) or candidate
            path = Path(resolved).expanduser()
            if path.exists() and path.is_file():
                return str(path)
        return None

    async def _install_llama_cpp(self) -> tuple[bool, str]:
        if not self.config.qwen_persona_llama_auto_install:
            self._append_log("system", "llama.cpp auto install disabled")
            return False, "auto install disabled"
        self._append_log("system", f"llama.cpp auto install requested for {self._platform_key()}")
        return await asyncio.to_thread(self._install_llama_cpp_sync)

    def _install_llama_cpp_sync(self) -> tuple[bool, str]:
        target_dir = Path(self.config.qwen_persona_data_dir) / "bin" / self._platform_key()
        target_dir.mkdir(parents=True, exist_ok=True)
        api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        try:
            self._append_log("system", f"fetching llama.cpp release metadata: {api_url}")
            with urllib.request.urlopen(api_url, timeout=15) as response:
                release = json.loads(response.read().decode("utf-8"))
            asset = self._select_release_asset(release.get("assets", []))
            if not asset:
                self._append_log("system", f"no llama.cpp release asset matched {self._platform_key()}")
                return False, f"no llama.cpp release asset matched {self._platform_key()}"
            asset_name = str(asset["name"])
            archive_path = target_dir / asset_name
            self._append_log("system", f"downloading llama.cpp asset: {asset_name}")
            with urllib.request.urlopen(str(asset["browser_download_url"]), timeout=60) as response:
                data = response.read()
                archive_path.write_bytes(data)
            self._append_log("system", f"downloaded llama.cpp asset: {asset_name} ({len(data)} bytes)")
            self._append_log("system", f"extracting llama.cpp asset to {target_dir}")
            self._extract_archive(archive_path, target_dir)
            executable = self._find_executable_under(target_dir)
            if executable:
                executable.chmod(executable.stat().st_mode | 0o111)
                self._append_log("system", f"llama.cpp installed: {executable}")
                return True, f"installed {asset_name} to {target_dir}"
            self._append_log("system", f"downloaded {asset_name}, but llama-server was not found")
            return False, f"downloaded {asset_name}, but llama-server was not found"
        except Exception as exc:
            self._append_log("system", f"llama.cpp auto install failed: {type(exc).__name__}: {exc}")
            return False, f"{type(exc).__name__}: {exc}"

    def _select_release_asset(self, assets: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored: list[tuple[int, dict[str, Any]]] = []
        system = platform.system().lower()
        machine = platform.machine().lower().replace("amd64", "x86_64")
        for asset in assets:
            name = str(asset.get("name", "")).lower()
            if not asset.get("browser_download_url"):
                continue
            score = 0
            if "bin" not in name:
                score -= 5
            if system == "darwin":
                if not any(token in name for token in ("macos", "darwin")):
                    continue
                score += 8
            elif system == "linux":
                if not any(token in name for token in ("linux", "ubuntu")):
                    continue
                score += 8
            elif system == "windows":
                if "win" not in name:
                    continue
                score += 8
                if "cpu" in name:
                    score += 4
            else:
                continue

            if machine in {"arm64", "aarch64"}:
                if any(token in name for token in ("arm64", "aarch64")):
                    score += 8
                else:
                    continue
            elif machine in {"x86_64", "x64"}:
                if any(token in name for token in ("x64", "x86_64")):
                    score += 8
                else:
                    continue

            if any(token in name for token in ("cuda", "hip", "kompute")):
                score -= 6
            if any(token in name for token in ("server", "cpu", "metal")):
                score += 1
            scored.append((score, asset))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (target_dir / member.filename).resolve()
                    if not self._is_relative_to(target, target_dir.resolve()):
                        raise ValueError(f"unsafe archive path: {member.filename}")
                    archive.extract(member, target_dir)
            return
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    target = (target_dir / member.name).resolve()
                    if not self._is_relative_to(target, target_dir.resolve()):
                        raise ValueError(f"unsafe archive path: {member.name}")
                archive.extractall(target_dir)
            return
        if archive_path.name.startswith("llama"):
            archive_path.chmod(archive_path.stat().st_mode | 0o111)

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def _find_executable_under(root: Path) -> Path | None:
        suffix = ".exe" if sys.platform == "win32" else ""
        names = {f"llama-server{suffix}", f"llama{suffix}", "llama-server", "llama"}
        for item in root.rglob("*"):
            if item.is_file() and item.name in names:
                return item
        return None

    def _managed_binary_candidates(self) -> list[Path]:
        platform_key = self._platform_key()
        suffix = ".exe" if sys.platform == "win32" else ""
        names = [f"llama-server{suffix}", f"llama{suffix}"]
        return [root / name for root in self._managed_binary_roots() for name in names]

    def _managed_binary_roots(self) -> list[Path]:
        package_dir = Path(__file__).resolve().parent
        data_dir = Path(self.config.qwen_persona_data_dir)
        platform_key = self._platform_key()
        return [
            package_dir / "bin" / platform_key,
            package_dir / "bin",
            data_dir / "bin" / platform_key,
            data_dir / "bin",
        ]

    @staticmethod
    def _platform_key() -> str:
        system = platform.system().lower() or "unknown"
        machine = platform.machine().lower().replace("amd64", "x86_64")
        if machine in {"arm64", "aarch64"}:
            machine = "arm64"
        elif machine in {"x86_64", "x64"}:
            machine = "x64"
        return f"{system}-{machine}"

    def _build_llama_command(self, executable: str) -> list[str]:
        exe_name = Path(executable).name.lower()
        command = [executable]
        if exe_name in {"llama", "llama.exe"}:
            command.append("serve")

        model_file = self._model_file()
        if model_file:
            command.extend(["-m", str(model_file)])
        else:
            command.extend(["-hf", self._model_ref()])

        command.extend(
            [
                "--host",
                self._host(),
                "--port",
                str(self._port()),
                "-c",
                str(max(512, int(self.config.qwen_persona_llama_ctx_size))),
            ]
        )
        api_key = self.config.qwen_persona_api_key.strip()
        if api_key and api_key.upper() != "EMPTY":
            command.extend(["--api-key", api_key])
        command.extend(self._extra_args())
        return command

    def _model_file(self) -> Path | None:
        configured = self.config.qwen_persona_llama_model_file.strip()
        if not configured:
            model_value = self.config.qwen_persona_llama_model.strip()
            if model_value.lower().endswith(".gguf"):
                configured = model_value
        if not configured:
            return None
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return self.model_dir / path

    def _model_ref(self) -> str:
        return self.config.qwen_persona_llama_model.strip() or "Qwen/Qwen3-0.6B-GGUF:Q8_0"

    def _extra_args(self) -> list[str]:
        raw = self.config.qwen_persona_llama_extra_args
        if isinstance(raw, str):
            return shlex.split(raw)
        return [str(item) for item in raw]

    def _host(self) -> str:
        return self.config.qwen_persona_llama_host.strip() or "127.0.0.1"

    def _port(self) -> int:
        return int(self.config.qwen_persona_llama_port)

    async def _wait_for_server_health(self) -> None:
        timeout = 180
        deadline = asyncio.get_running_loop().time() + timeout
        next_log = 0.0
        self._append_log("system", f"waiting for llama.cpp health: {effective_api_base(self.config).rstrip('/')}/models")
        while asyncio.get_running_loop().time() < deadline:
            if await self._service_healthy():
                self._append_log("system", "llama.cpp health check passed")
                return
            if self._server_process and self._server_process.returncode is not None:
                self._append_log("system", f"llama.cpp process exited with {self._server_process.returncode}")
                raise RuntimeError(f"llama process exited with {self._server_process.returncode}")
            now = asyncio.get_running_loop().time()
            if now >= next_log:
                remaining = max(0, int(deadline - now))
                self._append_log("system", f"llama.cpp not ready yet; waiting up to {remaining}s")
                next_log = now + 10
            await asyncio.sleep(2)
        self._append_log("system", f"llama.cpp health check timed out: {effective_api_base(self.config)}/models")
        raise TimeoutError(f"model server did not become healthy: {effective_api_base(self.config)}/models")

    async def _service_healthy(self) -> bool:
        return bool((await self._service_models()).get("ok"))

    async def _service_models_summary(self) -> str:
        probe = await self._service_models()
        models = probe.get("models") or []
        if models:
            listed = ", ".join(str(item) for item in models[:8])
            suffix = "" if len(models) <= 8 else f", +{len(models) - 8} more"
            return f"models: {listed}{suffix}"
        status_code = probe.get("status_code", "-")
        body = str(probe.get("body", "") or "").strip()
        if body:
            return f"/models status={status_code}, body={body[:240]}"
        error = str(probe.get("error", "") or "").strip()
        if error:
            return f"/models unavailable: {error}"
        return f"/models status={status_code}, no model ids reported"

    async def _service_models(self) -> dict[str, Any]:
        headers: dict[str, str] = {}
        api_key = self.config.qwen_persona_api_key.strip()
        if api_key and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{effective_api_base(self.config).rstrip('/')}/models", headers=headers)
            models: list[str] = []
            body = response.text
            try:
                data = response.json()
                models = self._extract_model_ids(data)
            except Exception:
                body = response.text[:500]
            return {
                "ok": response.status_code < 500,
                "status_code": response.status_code,
                "models": models,
                "body": self._redact_text(body[:500]),
            }
        except Exception as exc:
            return {
                "ok": False,
                "status_code": 0,
                "models": [],
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _extract_model_ids(self, data: Any) -> list[str]:
        ids: list[str] = []
        if isinstance(data, dict):
            items = data.get("data")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        value = item.get("id") or item.get("model")
                        if value:
                            ids.append(str(value))
            for key in ("id", "model"):
                value = data.get(key)
                if isinstance(value, str):
                    ids.append(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    value = item.get("id") or item.get("model")
                    if value:
                        ids.append(str(value))
        return list(dict.fromkeys(ids))

    def _port_accepts_tcp(self) -> bool:
        try:
            with socket.create_connection((self._host(), self._port()), timeout=1.0):
                return True
        except OSError:
            return False

    def _detect_managed_orphan(self) -> ManagedLlamaProcess | None:
        if self.config.qwen_persona_runtime != "llama_cpp":
            return None
        process = self._server_process
        if process and process.returncode is None:
            return None
        for pid in self._listener_pids(self._port()):
            command = self._process_command(pid)
            if command and self._is_managed_llama_command(command):
                return ManagedLlamaProcess(pid=pid, command=command)
        return None

    def _listener_pids(self, port: int) -> list[int]:
        if sys.platform == "win32":
            return self._listener_pids_windows(port)
        if not shutil.which("lsof"):
            return []
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []
        pids: list[int] = []
        for line in result.stdout.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                pids.append(int(line[1:]))
        return list(dict.fromkeys(pids))

    def _listener_pids_windows(self, port: int) -> list[int]:
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return []
        pids: list[int] = []
        needle = f":{port}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper().startswith("TCP") and parts[3].upper() == "LISTENING":
                local = parts[1]
                if local.endswith(needle) and parts[-1].isdigit():
                    pids.append(int(parts[-1]))
        return list(dict.fromkeys(pids))

    def _process_command(self, pid: int) -> str:
        if sys.platform == "win32":
            return self._process_command_windows(pid)
        if not shutil.which("ps"):
            return ""
        try:
            result = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return ""
        return result.stdout.strip()

    def _process_command_windows(self, pid: int) -> str:
        try:
            result = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/value"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return ""
        for line in result.stdout.splitlines():
            if line.startswith("CommandLine="):
                return line.removeprefix("CommandLine=").strip()
        return ""

    def _is_managed_llama_command(self, command: str) -> bool:
        try:
            parts = shlex.split(command, posix=sys.platform != "win32")
        except ValueError:
            parts = command.split()
        if not parts:
            return False
        executable = parts[0]
        name = Path(executable).name.lower()
        if name not in {"llama-server", "llama-server.exe", "llama", "llama.exe"}:
            return False
        executable_path = Path(executable).expanduser()
        candidates: list[Path] = []
        if executable_path.is_absolute():
            candidates.append(executable_path)
        else:
            candidates.append((Path.cwd() / executable_path).resolve())
        text = command.replace("\\", "/")
        for root in self._managed_binary_roots():
            try:
                resolved_root = root.resolve()
            except Exception:
                resolved_root = root
            for candidate in candidates:
                try:
                    if self._is_relative_to(candidate.resolve(), resolved_root):
                        return True
                except Exception:
                    continue
            root_text = str(root).replace("\\", "/").rstrip("/")
            if root_text and root_text in text:
                return True
        data_bin = f"{str(Path(self.config.qwen_persona_data_dir)).replace(chr(92), '/')}/bin"
        return executable.replace("\\", "/").startswith(data_bin)

    def _terminate_pid(self, pid: int) -> None:
        sigterm = signal.SIGTERM
        try:
            os.kill(pid, sigterm)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                break
            time.sleep(0.2)
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        try:
            os.kill(pid, sigkill)
        except ProcessLookupError:
            return
