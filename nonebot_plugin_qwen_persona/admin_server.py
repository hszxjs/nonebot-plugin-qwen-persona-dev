from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import MLX_MODEL_PRESETS, normalized_lora_modules
from .runtime_config import admin_config_path, config_to_dict

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


def _field_payload(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int) and not isinstance(value, bool):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    elif isinstance(value, (list, tuple, set, dict)):
        kind = "json"
    else:
        kind = "str"
    return {
        "name": name,
        "kind": kind,
        "value": value,
        "reload_sensitive": name in RELOAD_SENSITIVE_FIELDS,
        "danger": name in DANGER_FIELDS,
    }


def _settings_sections(config: Any) -> list[dict[str, Any]]:
    values = config_to_dict(config)
    groups = [
        ("model", "模型与推理", [
            "qwen_persona_mlx_model",
            "qwen_persona_mlx_adapter_path",
            "qwen_persona_mlx_auto_load",
            "qwen_persona_mlx_trust_remote_code",
            "qwen_persona_model_dir",
            "qwen_persona_temperature",
            "qwen_persona_top_p",
            "qwen_persona_max_tokens",
            "qwen_persona_request_timeout",
        ]),
        ("learning", "学习与回复", [
            "qwen_persona_enabled_groups",
            "qwen_persona_enable_learning",
            "qwen_persona_enable_reply",
            "qwen_persona_reply_probability",
            "qwen_persona_follow_reply_probability",
            "qwen_persona_context_scope",
            "qwen_persona_context_messages",
            "qwen_persona_trigger_keywords",
            "qwen_persona_system_prompt",
        ]),
        ("training", "训练策略", [
            "qwen_persona_enable_training",
            "qwen_persona_train_times",
            "qwen_persona_dataset_limit",
            "qwen_persona_dataset_history_window",
            "qwen_persona_external_corpus_enabled",
            "qwen_persona_external_corpus_limit",
            "qwen_persona_min_train_samples",
            "qwen_persona_multimodal_training_enabled",
            "qwen_persona_mlx_lora_train_modules",
        ]),
        ("stickers", "表情行为", [
            "qwen_persona_enable_sending_stickers",
            "qwen_persona_sticker_segment_types",
            "qwen_persona_sticker_prompt_limit",
            "qwen_persona_max_reply_stickers",
            "qwen_persona_text_sticker_reply_probability",
            "qwen_persona_media_image_fallback_text",
            "qwen_persona_media_sticker_fallback_text",
        ]),
        ("rules", "规则过滤", [
            "qwen_persona_command_prefixes",
            "qwen_persona_command_regexes",
            "qwen_persona_auto_read_picmenu_config",
            "qwen_persona_blacklist_words",
            "qwen_persona_blacklist_regexes",
            "qwen_persona_blacklist_user_ids",
            "qwen_persona_blacklist_group_ids",
        ]),
        ("system", "系统后台", [
            "qwen_persona_admin_enabled",
            "qwen_persona_admin_host",
            "qwen_persona_admin_port",
            "qwen_persona_admin_token",
            "qwen_persona_admin_config_file",
            "qwen_persona_data_dir",
        ]),
    ]
    return [
        {
            "key": key,
            "title": title,
            "fields": [_field_payload(name, values[name]) for name in names if name in values],
        }
        for key, title, names in groups
    ]


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

    @app.get("/api/settings/sections")
    async def settings_sections(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        authorize(token, x_admin_token)
        return {
            "sections": _settings_sections(config),
            "override_file": str(admin_config_path(config)),
        }

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
        result = save_callback({field_name: payload["value"]})
        return {
            "field": field_name,
            "value": payload["value"],
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
                "sections": _settings_sections(config),
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


class AdminWebServer:
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
        self.logger.info(f"qwen persona admin web listening on http://{host}:{port}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None
