"""admin API 装配器：FastAPI 应用工厂 + uvicorn 守护线程服务器。

端点路径与载荷契约与拆分前完全一致（webui 契约见 docs/webui-data-contract.md）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .auth import make_auth_dependency
from .routes import backends, misc, operations, role_cards, runtime, settings, stickers, training
from .routes import model_manager as model_manager_routes


def _static_root(static_dir: Path | None = None) -> Path:
    return static_dir or Path(__file__).resolve().parents[1] / "webui_static"


def _safe_static_file(root: Path, request_path: str) -> Path | None:
    relative = request_path.lstrip("/") or "index.html"
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def create_admin_app(
    *,
    config: Any,
    save_callback: Callable[[dict[str, Any]], dict[str, Any]],
    mlx_runtime: Any | None = None,
    store: Any | None = None,
    trainer: Any | None = None,
    qwen_client: Any | None = None,
    role_card_service: Any | None = None,
    tool_registry: Any | None = None,
    provider_instances: Any | None = None,
    provider_bindings: Any | None = None,
    model_manager: Any | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Companion Admin", docs_url=None, redoc_url=None)
    root = _static_root(static_dir)
    auth = make_auth_dependency(config)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = str(exc.detail or "error")
        return JSONResponse({"error": detail}, status_code=exc.status_code)

    app.include_router(
        operations.build_router(
            config=config,
            auth=auth,
            store=store,
            trainer=trainer,
            mlx_runtime=mlx_runtime,
            qwen_client=qwen_client,
        )
    )
    app.include_router(runtime.build_router(config=config, auth=auth, mlx_runtime=mlx_runtime, qwen_client=qwen_client))
    app.include_router(training.build_router(config=config, auth=auth, trainer=trainer))
    app.include_router(stickers.build_router(config=config, auth=auth, store=store))
    app.include_router(settings.build_router(config=config, auth=auth, save_callback=save_callback))
    app.include_router(
        misc.build_router(config=config, auth=auth, store=store, mlx_runtime=mlx_runtime, static_root=root)
    )
    app.include_router(
        role_cards.build_router(auth=auth, role_card_service=role_card_service, tool_registry=tool_registry)
    )
    app.include_router(backends.build_router(auth=auth, instances=provider_instances, bindings=provider_bindings))
    app.include_router(model_manager_routes.build_router(auth=auth, model_manager=model_manager))

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def static_or_fallback(path: str, request: Request) -> Response:
        if path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
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
        role_card_service: Any | None = None,
        tool_registry: Any | None = None,
        provider_instances: Any | None = None,
        provider_bindings: Any | None = None,
        model_manager: Any | None = None,
        static_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.save_callback = save_callback
        self.logger = logger
        self.mlx_runtime = mlx_runtime
        self.store = store
        self.qwen_client = qwen_client
        self.trainer = trainer
        self.role_card_service = role_card_service
        self.tool_registry = tool_registry
        self.provider_instances = provider_instances
        self.provider_bindings = provider_bindings
        self.model_manager = model_manager
        self.static_dir = static_dir
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not getattr(self.config, "ai_companion_admin_enabled", True):
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
            role_card_service=self.role_card_service,
            tool_registry=self.tool_registry,
            provider_instances=self.provider_instances,
            provider_bindings=self.provider_bindings,
            model_manager=self.model_manager,
            static_dir=self.static_dir,
        )
        host = str(getattr(self.config, "ai_companion_admin_host", "127.0.0.1"))
        port = int(getattr(self.config, "ai_companion_admin_port", 8082))
        uvicorn_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self.logger.info(f"ai companion admin server listening on http://{host}:{port}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None
