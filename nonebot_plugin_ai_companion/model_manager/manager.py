"""模型文件管理器（文档十二.2）：后台下载 + 进度操作 + 磁盘占用 + 删除。

下载复用 HFDownloadManager（huggingface_hub 拉取到共享模型目录），
进度以 RuntimeOperation 暴露（percent/downloaded_bytes/total_bytes），
SSE 端点按秒推送快照，支持取消。
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import Any

from ..config import AiCompanionConfig
from ..contracts import OperationProgress, RuntimeOperation
from .download import HFDownloadManager, is_hf_repo_ref, local_repo_dir
from .presets import MLX_MODEL_PRESETS


class ModelOperationError(RuntimeError):
    """模型管理操作失败（重复下载/非法引用/删除失败等）。"""


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


class ModelManager:
    def __init__(self, config: AiCompanionConfig, download_manager: HFDownloadManager | None = None) -> None:
        self.config = config
        self.download_manager = download_manager or HFDownloadManager()
        self._operation: RuntimeOperation | None = None
        self._cancel_event = threading.Event()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # ---- 查询 ----
    def model_dir(self) -> Path:
        return Path(str(self.config.ai_companion_model_dir or "data/qwen_persona/models"))

    def list_local_models(self) -> list[dict[str, Any]]:
        """共享模型目录下的一级子目录：名称 + 磁盘占用 + 可读大小。"""
        root = self.model_dir()
        items: list[dict[str, Any]] = []
        if not root.is_dir():
            return items
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            size = _dir_size_bytes(child)
            items.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "size_bytes": size,
                    "size_display": _human_size(size),
                }
            )
        return items

    def presets(self) -> list[dict[str, str]]:
        return list(MLX_MODEL_PRESETS)

    def operation_snapshot(self) -> dict[str, Any] | None:
        return self._operation.to_dict() if self._operation is not None else None

    # ---- 下载 ----
    async def start_download(self, model_ref: str) -> dict[str, Any]:
        model_ref = str(model_ref or "").strip()
        if not model_ref:
            raise ModelOperationError("model ref is required")
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise ModelOperationError("another download is running")
            if not is_hf_repo_ref(model_ref):
                raise ModelOperationError(f"not a huggingface repo ref: {model_ref}")
            if local_repo_dir(model_ref, self.model_dir()).is_dir():
                raise ModelOperationError(f"model already exists locally: {model_ref}")

            self._cancel_event = threading.Event()
            self._operation = RuntimeOperation(
                type="model_download",
                phase="starting",
                message=f"downloading {model_ref}",
                target_model=model_ref,
            )
            self._task = asyncio.create_task(self._run_download(model_ref))
        return self.operation_snapshot() or {}

    async def _run_download(self, model_ref: str) -> None:
        operation = self._operation
        assert operation is not None

        def progress_update(**kwargs: Any) -> None:
            operation.update(**kwargs)

        progress = OperationProgress(progress_update)

        def should_cancel() -> bool:
            return self._cancel_event.is_set()

        try:
            prepared = await asyncio.to_thread(
                self.download_manager.prepare_model,
                model_ref,
                model_dir=self.model_dir(),
                progress=progress,
                cancel_check=should_cancel,
            )
        except Exception as exc:  # noqa: BLE001 — 失败记录到 operation 并结束
            operation.finish(
                state="failed" if not should_cancel() else "cancelled",
                phase="failed",
                message=f"download failed: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        operation.update(local_path=str(prepared.local_path))
        operation.finish(
            state="success" if not should_cancel() else "cancelled",
            phase="done",
            message=f"model ready at {prepared.local_path}",
        )

    def cancel_download(self) -> dict[str, Any]:
        if self._operation is None or (self._task is not None and self._task.done()):
            raise ModelOperationError("no running download")
        self._cancel_event.set()
        return {"cancelling": True}

    # ---- 删除 ----
    async def delete_model(self, model_ref: str) -> dict[str, Any]:
        target = local_repo_dir(str(model_ref or "").strip(), self.model_dir())
        if not target.is_dir():
            raise ModelOperationError(f"model not found locally: {model_ref}")
        if self._operation is not None and self._operation.state == "running":
            raise ModelOperationError("download in progress")

        def _delete() -> int:
            size = _dir_size_bytes(target)
            shutil.rmtree(target)
            return size

        try:
            freed = await asyncio.to_thread(_delete)
        except OSError as exc:
            raise ModelOperationError(f"delete failed: {exc}") from exc
        return {"deleted": model_ref, "freed_bytes": freed, "freed_display": _human_size(freed)}


def _human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"
