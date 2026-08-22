from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .model import is_transient_image_sticker_data, meaningful_protocol_summary
from .classifier import is_sendable_sticker_record


MAX_STICKER_MEDIA_BYTES = 16 * 1024 * 1024
CONTENT_TYPE_EXTENSIONS = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
logger = logging.getLogger(__name__)


def sticker_repo_manifest_path(repo_dir: str | Path) -> Path:
    return Path(repo_dir) / "manifest.jsonl"


async def cleanup_sticker_repo_manifest(repo_dir: str | Path) -> dict[str, int]:
    return await asyncio.to_thread(_cleanup_sticker_repo_manifest_sync, repo_dir)


def _cleanup_sticker_repo_manifest_sync(repo_dir: str | Path) -> dict[str, int]:
    path = sticker_repo_manifest_path(repo_dir)
    if not path.exists():
        return {
            "scanned": 0,
            "kept": 0,
            "removed": 0,
            "media_deleted": 0,
            "summary_updated": 0,
        }

    repo_path = Path(repo_dir).resolve()
    kept_lines: list[str] = []
    scanned = 0
    removed = 0
    media_deleted = 0
    summary_updated = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scanned += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        if _is_plain_image_manifest_record(record):
            removed += 1
            media_deleted += _delete_local_manifest_media(repo_path, record)
            continue
        protocol_summary = meaningful_protocol_summary(dict(record.get("data") or {}))
        if protocol_summary and str(record.get("summary") or "") != protocol_summary:
            record["summary"] = protocol_summary
            summary_updated += 1
        kept_lines.append(json.dumps(record, ensure_ascii=False, default=str))

    if removed or summary_updated:
        content = "\n".join(kept_lines)
        path.write_text((content + "\n") if content else "", encoding="utf-8", newline="\n")
    return {
        "scanned": scanned,
        "kept": len(kept_lines),
        "removed": removed,
        "media_deleted": media_deleted,
        "summary_updated": summary_updated,
    }


def _is_plain_image_manifest_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if str(record.get("type") or "") != "image":
        return False
    return not is_sendable_sticker_record(record)


def _delete_local_manifest_media(repo_dir: Path, record: dict[str, Any]) -> int:
    uri = str(record.get("uri") or "").strip()
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return 0
    media_dir = (repo_dir / "media").resolve()
    try:
        media_path = Path(parsed.path).resolve()
        if not media_path.is_relative_to(media_dir) or not media_path.is_file():
            return 0
        media_path.unlink()
    except OSError:
        return 0
    return 1


async def record_sticker_repo_manifest(
    *,
    repo_dir: str | Path,
    group_id: str | None,
    user_id: str | None,
    stickers: list[dict[str, Any]],
) -> None:
    if not stickers:
        return
    await asyncio.to_thread(_record_sticker_repo_manifest_sync, repo_dir, group_id, user_id, stickers)


def _record_sticker_repo_manifest_sync(
    repo_dir: str | Path,
    group_id: str | None,
    user_id: str | None,
    stickers: list[dict[str, Any]],
) -> None:
    path = sticker_repo_manifest_path(repo_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    records: list[dict[str, Any]] = []
    kept_stickers: list[dict[str, Any]] = []
    for sticker in stickers:
        _persist_sticker_media_sync(path.parent, sticker)
        if _is_unpersisted_transient_sticker(sticker):
            continue
        kept_stickers.append(sticker)
        records.append(
            {
                "seen_at": now,
                "group_id": group_id,
                "user_id": user_id,
                "key": sticker.get("key"),
                "type": sticker.get("type"),
                "summary": sticker.get("summary"),
                "uri": sticker.get("uri"),
                "data": sticker.get("data") or {},
            }
        )
    stickers[:] = kept_stickers
    if not records:
        return
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _persist_sticker_media_sync(repo_dir: Path, sticker: dict[str, Any]) -> None:
    if str(sticker.get("type") or "") != "image":
        return
    uri = str(sticker.get("uri") or "").strip()
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return

    media_dir = repo_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sticker.get("key") or "sticker"))
    existing = next(media_dir.glob(f"{safe_key}.*"), None)
    if existing is not None and existing.is_file():
        sticker["uri"] = existing.resolve().as_uri()
        return

    try:
        request = urllib.request.Request(
            uri,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            content = response.read(MAX_STICKER_MEDIA_BYTES + 1)
            content_type = _response_content_type(response)
    except Exception as exc:
        logger.warning("ai companion sticker media download failed: %s: %r", uri, exc)
        return
    if not content or len(content) > MAX_STICKER_MEDIA_BYTES:
        return

    extension = _media_extension(sticker, content_type)
    target = media_dir / f"{safe_key}{extension}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(target)
    sticker["uri"] = target.resolve().as_uri()


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", {})
    if hasattr(headers, "get_content_type"):
        return str(headers.get_content_type() or "").lower()
    value = headers.get("Content-Type", "") if hasattr(headers, "get") else ""
    return str(value or "").split(";", 1)[0].strip().lower()


def _media_extension(sticker: dict[str, Any], content_type: str) -> str:
    if content_type in CONTENT_TYPE_EXTENSIONS:
        return CONTENT_TYPE_EXTENSIONS[content_type]
    data = dict(sticker.get("data") or {})
    for value in (data.get("file"), sticker.get("uri"), data.get("url")):
        suffix = Path(urlparse(str(value or "")).path).suffix.lower()
        if suffix in {".gif", ".jpg", ".jpeg", ".png", ".webp"}:
            return ".jpg" if suffix == ".jpeg" else suffix
    return ".bin"


def _is_unpersisted_transient_sticker(sticker: dict[str, Any]) -> bool:
    if str(sticker.get("type") or "") != "image":
        return False
    if not is_transient_image_sticker_data(dict(sticker.get("data") or {})):
        return False
    return urlparse(str(sticker.get("uri") or "")).scheme in {"http", "https"}
