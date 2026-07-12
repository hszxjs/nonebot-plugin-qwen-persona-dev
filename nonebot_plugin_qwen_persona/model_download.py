from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MLX_VLM_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
]


class DownloadProgress(Protocol):
    def update(self, **kwargs: Any) -> None:
        ...


@dataclass(frozen=True)
class PreparedModel:
    local_path: Path
    source: str


def is_existing_local_model(model_ref: str) -> bool:
    return Path(str(model_ref or "")).expanduser().exists()


def is_hf_repo_ref(model_ref: str) -> bool:
    value = str(model_ref or "").strip()
    if not value or is_existing_local_model(value):
        return False
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith(("./", "../", "~")):
        return False
    return "/" in value and ":" not in value


def local_repo_dir(model_ref: str) -> str:
    return (
        str(model_ref or "")
        .strip()
        .replace("/", "--")
        .replace("\\", "--")
        .replace(":", "--")
    )


class HFDownloadManager:
    def __init__(
        self,
        *,
        snapshot_download: Any | None = None,
        allow_patterns: list[str] | None = None,
    ) -> None:
        self._snapshot_download = snapshot_download
        self._allow_patterns = allow_patterns or list(MLX_VLM_ALLOW_PATTERNS)

    def prepare_model(
        self,
        model_ref: str,
        *,
        model_dir: Path,
        progress: DownloadProgress,
    ) -> PreparedModel:
        if is_existing_local_model(model_ref):
            local_path = Path(model_ref).expanduser()
            progress.update(phase="preparing", message=f"using local model {local_path}", local_path=str(local_path))
            return PreparedModel(local_path=local_path, source="local")
        if not is_hf_repo_ref(model_ref):
            progress.update(phase="preparing", message=f"using direct model ref {model_ref}")
            return PreparedModel(local_path=Path(model_ref), source="direct")

        root = Path(model_dir).expanduser()
        target = root / local_repo_dir(model_ref)
        root.mkdir(parents=True, exist_ok=True)
        progress.update(phase="preparing", message=f"preparing {model_ref}", local_path=str(target))

        snapshot_download = self._download_function()
        total_bytes = self._dry_run_total_bytes(
            snapshot_download,
            model_ref=model_ref,
            target=target,
            progress=progress,
        )
        progress.update(
            phase="downloading",
            message=f"downloading {model_ref}",
            downloaded_bytes=0,
            total_bytes=total_bytes,
        )
        downloaded = {"bytes": 0}
        tqdm_class = self._progress_tqdm(progress, downloaded, total_bytes)

        kwargs: dict[str, Any] = {
            "repo_id": model_ref,
            "local_dir": str(target),
            "allow_patterns": self._allow_patterns,
        }
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            local_path = Path(snapshot_download(**kwargs))
        except TypeError as exc:
            logger.warning("HF snapshot_download tqdm_class unsupported for %s: %r", model_ref, exc)
            progress.update(
                phase="downloading",
                message=f"download progress callback unsupported: {type(exc).__name__}",
            )
            kwargs.pop("tqdm_class", None)
            local_path = Path(snapshot_download(**kwargs))
        progress.update(
            phase="downloaded",
            message=f"downloaded {model_ref}",
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
            percent=100.0 if total_bytes else None,
            local_path=str(local_path),
        )
        return PreparedModel(local_path=local_path, source="hf")

    def _download_function(self) -> Any:
        if self._snapshot_download is not None:
            return self._snapshot_download
        from huggingface_hub import snapshot_download

        return snapshot_download

    def _dry_run_total_bytes(
        self,
        snapshot_download: Any,
        *,
        model_ref: str,
        target: Path,
        progress: DownloadProgress,
    ) -> int | None:
        try:
            items = snapshot_download(
                repo_id=model_ref,
                local_dir=str(target),
                allow_patterns=self._allow_patterns,
                dry_run=True,
            )
        except TypeError as exc:
            logger.warning("HF snapshot_download dry-run unsupported for %s: %r", model_ref, exc)
            progress.update(phase="preparing", message=f"download dry-run unsupported: {type(exc).__name__}")
            return None
        except Exception as exc:
            logger.warning("HF download dry-run failed for %s: %s", model_ref, exc)
            progress.update(phase="preparing", message=f"download dry-run skipped: {type(exc).__name__}")
            return None
        total = 0
        for item in items or []:
            size = getattr(item, "file_size", None)
            if size is None:
                size = getattr(item, "size", None)
            if size:
                total += int(size)
        return total or None

    @staticmethod
    def _progress_tqdm(
        progress: DownloadProgress,
        downloaded: dict[str, int],
        total_bytes: int | None,
    ) -> Any:
        try:
            from tqdm.auto import tqdm
        except Exception as exc:
            logger.warning("tqdm unavailable for HF download progress: %r", exc)
            return None

        class DownloadTqdm(tqdm):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("disable", True)
                super().__init__(*args, **kwargs)

            def update(self, n: int | float = 1) -> Any:
                result = super().update(n)
                downloaded["bytes"] += int(n or 0)
                progress.update(
                    phase="downloading",
                    message=str(getattr(self, "desc", "") or "downloading"),
                    downloaded_bytes=downloaded["bytes"],
                    total_bytes=total_bytes,
                )
                return result

        return DownloadTqdm
