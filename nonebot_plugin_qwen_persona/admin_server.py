from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import MLX_MODEL_PRESETS, normalized_lora_modules
from .runtime_config import (
    admin_config_path,
    config_field_names,
    config_to_dict,
    config_to_python_dict,
)

logger = logging.getLogger(__name__)

RELOAD_SENSITIVE_FIELDS = {
    "qwen_persona_mlx_model",
    "qwen_persona_mlx_adapter_path",
    "qwen_persona_mlx_auto_load",
    "qwen_persona_mlx_trust_remote_code",
    "qwen_persona_model_dir",
    "qwen_persona_admin_enabled",
    "qwen_persona_admin_host",
    "qwen_persona_admin_port",
    "qwen_persona_data_dir",
    "qwen_persona_train_times",
    "qwen_persona_enable_training",
    "qwen_persona_learning_priority",
    "qwen_persona_reply_priority",
}

DANGER_FIELDS = {
    "qwen_persona_lora_train_command",
    "qwen_persona_after_train_command",
    "qwen_persona_lora_helper_api_key",
    "qwen_persona_admin_token",
}


def _json_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _static_root(static_dir: Path | None = None) -> Path:
    return static_dir or Path(__file__).resolve().parent / "webui_static"


def _safe_static_file(root: Path, request_path: str) -> Path | None:
    relative = request_path.lstrip("/") or "index.html"
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _authorized(config: Any, token: str | None, header_token: str | None) -> bool:
    expected = str(getattr(config, "qwen_persona_admin_token", "") or "")
    if not expected:
        return True
    return expected in {token or "", header_token or ""}


def _require_auth(config: Any, token: str | None, header_token: str | None) -> None:
    if not _authorized(config, token, header_token):
        raise HTTPException(status_code=401, detail="unauthorized")


def _runtime_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime": status.get("runtime", "mlx_vlm"),
        "state": status.get("state", "unknown"),
        "status": status.get("status", ""),
        "healthy": bool(status.get("healthy")),
        "model": status.get("model", ""),
        "loaded_model": status.get("loaded_model", ""),
        "adapter_path": status.get("adapter_path", ""),
        "loaded_adapter_path": status.get("loaded_adapter_path", ""),
        "last_error": status.get("last_error", ""),
        "operation": status.get("operation"),
        "control": status.get("control", {}),
        "metrics": status.get("metrics", {"last": None, "history": []}),
        "model_presets": MLX_MODEL_PRESETS,
    }


def _training_operation(trainer: Any | None) -> dict[str, Any]:
    progress = (
        trainer.progress_snapshot()
        if trainer is not None and hasattr(trainer, "progress_snapshot")
        else None
    )
    return {
        "running": bool(progress and progress.get("status") == "running"),
        "available": trainer is not None,
        "progress": progress,
    }


def _training_running(trainer: Any | None) -> bool:
    if trainer is None or not hasattr(trainer, "progress_snapshot"):
        return False
    progress = trainer.progress_snapshot()
    return bool(progress and progress.get("status") == "running")


def _training_config(config: Any) -> dict[str, Any]:
    return {
        "target_configured": True,
        "helper_configured": bool(str(getattr(config, "qwen_persona_lora_helper_api_url", "") or "").strip()),
        "min_train_samples": int(getattr(config, "qwen_persona_min_train_samples", 0) or 0),
        "dataset_limit": int(getattr(config, "qwen_persona_dataset_limit", 0) or 0),
        "external_corpus_enabled": bool(getattr(config, "qwen_persona_external_corpus_enabled", False)),
        "external_corpus_limit": int(getattr(config, "qwen_persona_external_corpus_limit", 0) or 0),
        "multimodal_training_enabled": bool(getattr(config, "qwen_persona_multimodal_training_enabled", False)),
        "mlx_lora_modules": normalized_lora_modules(config),
    }


def _log_tail_entries(log_tail: str, *, stream: str = "train") -> list[dict[str, str]]:
    lines = [line.strip() for line in str(log_tail or "").splitlines() if line.strip()]
    return [
        {"time": "", "stream": stream, "message": line}
        for line in lines[-300:]
    ]


def _plugin_log_path(config: Any) -> Path:
    data_dir = Path(str(getattr(config, "qwen_persona_data_dir", "data/qwen_persona") or "data/qwen_persona"))
    return data_dir / "bot_runtime.log"


def _file_log_entries(path: Path, *, tail: int, stream: str) -> list[dict[str, str]]:
    limit = max(0, min(int(tail), 1000))
    if limit <= 0 or not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return [{"time": "", "stream": stream, "message": line} for line in lines[-limit:]]


def _settings_payload(config: Any) -> dict[str, Any]:
    return {
        "values": config_to_dict(config),
        "override_file": str(admin_config_path(config)),
    }


def _validate_setting_value(config: Any, field_name: str, value: Any) -> Any:
    allowed = set(config_field_names(config))
    if field_name not in allowed:
        raise HTTPException(status_code=400, detail=f"unknown setting: {field_name}")
    if not hasattr(config, "__class__"):
        return value
    before = config_to_python_dict(config)
    try:
        validated = config.__class__(**{**before, field_name: value})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid setting value: {field_name}") from exc
    return config_to_dict(validated).get(field_name)


def create_admin_app(
    *,
    config: Any,
    save_callback: Callable[[dict[str, Any]], dict[str, Any]],
    mlx_runtime: Any | None = None,
    store: Any | None = None,
    trainer: Any | None = None,
    qwen_client: Any | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Qwen Persona Admin", docs_url=None, redoc_url=None)
    root = _static_root(static_dir)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = str(exc.detail or "error")
        return _json_error(detail, exc.status_code)

    def authorize(token: str | None, x_admin_token: str | None) -> None:
        _require_auth(config, token, x_admin_token)

    @app.get("/api/operations/overview")
    async def operations_overview(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        runtime_status = (
            await _maybe_await(mlx_runtime.status_snapshot())
            if mlx_runtime and hasattr(mlx_runtime, "status_snapshot")
            else {"state": "mlx_runtime_unavailable", "healthy": False}
        )
        dashboard = (
            await _maybe_await(store.dashboard_stats())
            if store and hasattr(store, "dashboard_stats")
            else {"totals": {}, "training": {}, "charts": {}}
        )
        events = (
            await _maybe_await(store.list_inference_events(limit=300, offset=0))
            if store and hasattr(store, "list_inference_events")
            else {"items": [], "total": 0, "limit": 300, "offset": 0}
        )
        client_metrics = (
            qwen_client.metrics_snapshot()
            if qwen_client and hasattr(qwen_client, "metrics_snapshot")
            else {"last": None, "history": []}
        )
        return {
            "runtime": _runtime_summary(dict(runtime_status)),
            "learning": dict(dashboard.get("totals") or {}),
            "training": {
                "summary": dashboard.get("training") or {},
                "operation": _training_operation(trainer),
                "config": _training_config(config),
            },
            "charts": dashboard.get("charts") or {},
            "inference": {"client": client_metrics},
            "inference_events": events,
            "errors": [
                runtime_status.get("last_error")
            ] if runtime_status.get("last_error") else [],
        }

    @app.get("/api/runtime/status")
    async def runtime_status(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        status = (
            await _maybe_await(mlx_runtime.status_snapshot())
            if mlx_runtime and hasattr(mlx_runtime, "status_snapshot")
            else {"state": "mlx_runtime_unavailable", "healthy": False}
        )
        return _runtime_summary(dict(status))

    @app.get("/api/runtime/logs")
    async def runtime_logs(
        tail: int = Query(default=300),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        logs = mlx_runtime.logs(tail=max(0, min(int(tail), 1000))) if mlx_runtime and hasattr(mlx_runtime, "logs") else []
        return {"logs": logs}

    async def run_runtime_action(action: str, confirmed: bool = False):
        if action in {"reload", "unload"} and not confirmed:
            return JSONResponse(
                {
                    "error": "confirmation required",
                    "confirmation_required": True,
                    "action": action,
                },
                status_code=409,
            )
        if not mlx_runtime or not hasattr(mlx_runtime, action):
            raise HTTPException(status_code=503, detail="mlx runtime unavailable")
        model = str(getattr(config, "qwen_persona_mlx_model", "") or "").strip() or "unknown"
        result = await _maybe_await(getattr(mlx_runtime, action)())
        runtime = _runtime_summary(dict(result or {}))
        return {
            "action": action,
            "message": f"{action} model '{model}'",
            "runtime": runtime,
        }

    @app.post("/api/runtime/load")
    async def runtime_load(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ):
        authorize(token, x_admin_token)
        return await run_runtime_action("load", confirmed=True)

    @app.post("/api/runtime/reload")
    async def runtime_reload(
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ):
        authorize(token, x_admin_token)
        payload = await request.json()
        return await run_runtime_action("reload", confirmed=bool(payload.get("confirmed")))

    @app.post("/api/runtime/unload")
    async def runtime_unload(
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ):
        authorize(token, x_admin_token)
        payload = await request.json()
        return await run_runtime_action("unload", confirmed=bool(payload.get("confirmed")))

    @app.get("/api/plugin/logs")
    async def plugin_logs(
        tail: int = Query(default=300),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        path = _plugin_log_path(config)
        return {
            "logs": _file_log_entries(path, tail=tail, stream="plugin"),
            "path": str(path),
            "exists": path.is_file(),
        }

    @app.get("/api/training/logs")
    async def training_logs(
        tail: int = Query(default=300),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        progress = (
            trainer.progress_snapshot()
            if trainer is not None and hasattr(trainer, "progress_snapshot")
            else None
        )
        logs = _log_tail_entries(
            str((progress or {}).get("log_tail", "")),
            stream="train",
        )
        limit = max(0, min(int(tail), 1000))
        return {"logs": logs[-limit:]}

    @app.post("/api/training/run")
    async def training_run(
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        if trainer is None or not hasattr(trainer, "run_training"):
            raise HTTPException(status_code=503, detail="trainer unavailable")
        if _training_running(trainer):
            raise HTTPException(status_code=409, detail="training already running")
        try:
            payload = await request.json()
        except Exception as exc:
            logger.warning("qwen persona admin training request body ignored: %r", exc)
            payload = {}
        if not isinstance(payload, dict):
            logger.warning("qwen persona admin training request body ignored: expected object")
            payload = {}
        include_external_corpus = payload.get("include_external_corpus")
        if include_external_corpus is not None:
            include_external_corpus = bool(include_external_corpus)
        full_external_corpus = bool(payload.get("full_external_corpus"))
        reason = "admin_full_external" if full_external_corpus else "admin"

        async def run_background() -> None:
            try:
                await trainer.run_training(
                    reason=reason,
                    include_external_corpus=include_external_corpus,
                    full_external_corpus=full_external_corpus,
                )
            except Exception:
                logger.exception("qwen persona admin training failed")

        asyncio.create_task(run_background())
        await asyncio.sleep(0)
        action = "run_full_external" if full_external_corpus else "run"
        return {
            "action": action,
            "message": (
                "start full external corpus training"
                if full_external_corpus
                else "start training"
            ),
            "training": _training_operation(trainer),
        }

    @app.get("/api/inference-events")
    async def inference_events(
        limit: int = Query(default=300),
        offset: int = Query(default=0),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        if not store or not hasattr(store, "list_inference_events"):
            return {"items": [], "total": 0, "limit": min(limit, 300), "offset": max(0, offset)}
        return await _maybe_await(store.list_inference_events(limit=limit, offset=offset))

    @app.get("/api/stickers")
    async def stickers(
        limit: int = Query(default=80),
        offset: int = Query(default=0),
        q: str = Query(default=""),
        type: str = Query(default=""),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        if not store or not hasattr(store, "list_stickers"):
            return {"items": [], "total": 0, "limit": limit, "offset": offset, "query": q, "type": type, "type_counts": {}}
        return await _maybe_await(store.list_stickers(limit=limit, offset=offset, query=q, segment_type=type))

    @app.get("/api/stickers/events")
    async def sticker_events(
        limit: int = Query(default=100),
        offset: int = Query(default=0),
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        if not store or not hasattr(store, "list_sticker_events"):
            return {"items": [], "total": 0, "limit": min(limit, 300), "offset": max(0, offset)}
        return await _maybe_await(store.list_sticker_events(limit=limit, offset=offset))

    @app.get("/api/stickers/{sticker_key:path}/preview")
    async def sticker_preview(
        sticker_key: str,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> Response:
        authorize(token, x_admin_token)
        if not store or not hasattr(store, "get_sticker_preview_path"):
            raise HTTPException(status_code=404, detail="sticker preview not found")
        repo_dir = getattr(config, "qwen_persona_sticker_repo_dir", "data/qwen_persona/sticker_repo")
        path = await _maybe_await(store.get_sticker_preview_path(sticker_key, repo_dir=repo_dir))
        if path is None or not Path(path).is_file():
            raise HTTPException(status_code=404, detail="sticker preview not found")
        return FileResponse(Path(path))

    @app.patch("/api/stickers/{sticker_key:path}")
    async def update_sticker(
        sticker_key: str,
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        payload = await request.json()
        if not isinstance(payload, dict) or "summary" not in payload:
            raise HTTPException(status_code=400, detail="summary is required")
        if not store or not hasattr(store, "update_sticker_summary"):
            raise HTTPException(status_code=503, detail="sticker store unavailable")
        return await _maybe_await(
            store.update_sticker_summary(sticker_key, str(payload["summary"]))
        )

    @app.delete("/api/stickers")
    async def delete_stickers(
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, int]:
        authorize(token, x_admin_token)
        payload = await request.json()
        raw_keys = payload.get("keys") if isinstance(payload, dict) else None
        keys = [raw_keys] if isinstance(raw_keys, str) else raw_keys
        if not isinstance(keys, list):
            raise HTTPException(status_code=400, detail="keys is required")
        if not store or not hasattr(store, "delete_stickers"):
            raise HTTPException(status_code=503, detail="sticker store unavailable")
        try:
            return await _maybe_await(store.delete_stickers([str(key) for key in keys]))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/settings")
    async def settings(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        return _settings_payload(config)

    @app.patch("/api/settings/fields/{field_name}")
    async def update_setting(
        field_name: str,
        request: Request,
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        payload = await request.json()
        if not isinstance(payload, dict) or "value" not in payload:
            raise HTTPException(status_code=400, detail="value is required")
        value = _validate_setting_value(config, field_name, payload["value"])
        requires_confirmation = field_name in RELOAD_SENSITIVE_FIELDS or field_name in DANGER_FIELDS
        if requires_confirmation and not bool(payload.get("confirmed")):
            return JSONResponse(
                {
                    "error": "confirmation required",
                    "confirmation_required": True,
                    "field": field_name,
                    "reload_required": field_name in RELOAD_SENSITIVE_FIELDS,
                    "danger": field_name in DANGER_FIELDS,
                },
                status_code=409,
            )
        result = save_callback({field_name: value})
        return {
            "field": field_name,
            "value": value,
            "confirmation_required": False,
            **result,
        }

    @app.get("/api/diagnostics")
    async def diagnostics(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        runtime_status = (
            await _maybe_await(mlx_runtime.status_snapshot())
            if mlx_runtime and hasattr(mlx_runtime, "status_snapshot")
            else {"state": "mlx_runtime_unavailable", "healthy": False}
        )
        runtime_logs = (
            mlx_runtime.logs(tail=300)
            if mlx_runtime and hasattr(mlx_runtime, "logs")
            else []
        )
        return {
            "runtime": {
                "status": _runtime_summary(dict(runtime_status)),
                "logs": runtime_logs,
            },
            "settings": {
                "values": config_to_dict(config),
                "override_file": str(admin_config_path(config)),
            },
            "static": {
                "root": str(root),
                "built": (root / "index.html").is_file(),
                "index": str(root / "index.html"),
            },
            "build": {
                "admin_server": "fastapi-vite",
            },
        }

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def static_or_fallback(path: str, request: Request) -> Response:
        if path.startswith("api/"):
            return _json_error("not found", 404)
        if not root.exists():
            return Response("webui static assets not built", status_code=404)
        file_path = _safe_static_file(root, path)
        if file_path:
            return FileResponse(file_path)
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        return Response("webui index not found", status_code=404)

    return app


class AdminServer:
    def __init__(
        self,
        *,
        config: Any,
        save_callback: Callable[[dict[str, Any]], dict[str, Any]],
        logger: Any,
        mlx_runtime: Any | None = None,
        store: Any | None = None,
        qwen_client: Any | None = None,
        trainer: Any | None = None,
        static_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.save_callback = save_callback
        self.logger = logger
        self.mlx_runtime = mlx_runtime
        self.store = store
        self.qwen_client = qwen_client
        self.trainer = trainer
        self.static_dir = static_dir
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not getattr(self.config, "qwen_persona_admin_enabled", True):
            return
        if self._thread and self._thread.is_alive():
            return
        app = create_admin_app(
            config=self.config,
            save_callback=self.save_callback,
            mlx_runtime=self.mlx_runtime,
            store=self.store,
            trainer=self.trainer,
            qwen_client=self.qwen_client,
            static_dir=self.static_dir,
        )
        host = str(getattr(self.config, "qwen_persona_admin_host", "127.0.0.1"))
        port = int(getattr(self.config, "qwen_persona_admin_port", 8082))
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self.logger.info(f"qwen persona admin server listening on http://{host}:{port}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None
