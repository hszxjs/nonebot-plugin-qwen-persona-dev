"""提醒卡片的 HTML 构建（纯函数）与渲染（htmlrender，延迟导入）。"""

from __future__ import annotations

import base64
import html
import logging
import time
from pathlib import Path
from typing import Any

from .model import PingCard

logger = logging.getLogger(__name__)

# 单张找人的文本上限与图片上限：卡片是提醒，不是聊天记录备份
MAX_TEXT_CHARS = 160
MAX_IMAGES = 2
MAX_IMAGE_DIM = 480
CARD_MAX_AGE_SECONDS = 86400.0

_CSS = """
body { margin:0; padding:0; background:transparent; font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif; }
.card { width:560px; background:#1a1a22; border-radius:18px; padding:20px 22px; color:#e8e8ef; }
.title { font-size:22px; font-weight:600; margin:2px 4px 16px; color:#f2f2f7; }
.finder { display:flex; gap:10px; margin-bottom:14px; align-items:flex-start; }
.avatar { width:38px; height:38px; border-radius:50%; background:#4d6bfe; color:#fff; font-size:18px;
          display:flex; align-items:center; justify-content:center; flex-shrink:0; font-weight:600; }
.right { min-width:0; flex:1; }
.fname { font-size:13px; color:#9a9aa8; margin:0 4px 5px; }
.bubble { background:#2b2b35; border-radius:4px 14px 14px 14px; padding:10px 14px; font-size:15px;
          line-height:1.55; word-break:break-word; color:#e8e8ef; }
.bubble img { max-width:320px; max-height:240px; border-radius:8px; display:block; margin-top:8px; }
.hint { font-size:12px; color:#6f6f7c; margin:6px 4px 0; text-align:right; }
"""


def _initial(name: str) -> str:
    clean = (name or "").strip()
    return clean[:1] if clean else "？"


def _downscale_to_data_url(image_bytes: bytes) -> str | None:
    """图片字节 → 缩放后的 data URL（进 HTML 用 base64，避免文件白名单问题）。"""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB") if image.mode not in {"RGB", "RGBA"} else image
            width, height = image.size
            scale = MAX_IMAGE_DIM / max(width, height, 1)
            if scale < 1.0:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001 — 单张图处理失败只少一张图，不废整卡
        logger.debug(f"ping card image embed failed: {exc!r}")
        return None


def build_card_html(card: PingCard, *, image_bytes_list: list[list[bytes]] | None = None) -> str:
    """PingCard → 自包含 HTML。image_bytes_list 与 card.finders 一一对应。"""
    finder_blocks: list[str] = []
    lists = image_bytes_list or [[] for _ in card.finders]
    for index, finder in enumerate(card.finders):
        name = html.escape(finder.name or "群友")
        text = html.escape((finder.text or "").strip()[:MAX_TEXT_CHARS])
        images = lists[index] if index < len(lists) else []
        img_tags = []
        for raw in images[:MAX_IMAGES]:
            data_url = _downscale_to_data_url(raw)
            if data_url:
                img_tags.append(f'<img src="{data_url}" alt="图片" />')
        body = text.replace("\n", "<br />")
        if img_tags:
            body = (body + "<br />" if body else "") + "".join(img_tags)
        body = body or "（图片消息）"
        reason_note = "引用了你" if finder.reason == "quote" else ""
        finder_blocks.append(
            f'<div class="finder">'
            f'<div class="avatar">{html.escape(_initial(finder.name))}</div>'
            f'<div class="right"><div class="fname">{name}{(" · " + reason_note) if reason_note else ""}</div>'
            f'<div class="bubble">{body}</div></div>'
            f"</div>"
        )
    inner = "".join(finder_blocks) or '<div class="bubble">（没有可展示的内容）</div>'
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head>"
        f"<body><div class='card'>"
        f"<div class='title'>{html.escape(card.title)}</div>"
        f"{inner}"
        f"<div class='hint'>{time.strftime('%H:%M')}</div>"
        f"</div></body></html>"
    )


def card_output_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "ping_cards"


def save_card_png(data_dir: str | Path, png_bytes: bytes, *, target_user_id: str) -> Path:
    """落盘并清理超过一天的旧卡片图。"""
    out_dir = card_output_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ping-{target_user_id}-{int(time.time() * 1000)}.png"
    path.write_bytes(png_bytes)
    now = time.time()
    for old in out_dir.glob("ping-*.png"):
        try:
            if now - old.stat().st_mtime > CARD_MAX_AGE_SECONDS:
                old.unlink(missing_ok=True)
        except OSError:
            continue
    return path


async def render_card_png(card: PingCard, *, image_bytes_list: list[list[bytes]] | None = None) -> bytes:
    """渲染成 PNG。htmlrender 延迟导入：没有 driver 的单测环境不会触发。"""
    from nonebot import require

    require("nonebot_plugin_htmlrender")
    from nonebot_plugin_htmlrender import html_to_pic

    page_html = build_card_html(card, image_bytes_list=image_bytes_list)
    return await html_to_pic(page_html, type="png", device_scale_factor=2)


def image_bytes_from_part(part: dict[str, Any]) -> bytes:
    """OneBot 图片段 → 原始字节（触发当下下载；本地 file:// 直接读）。失败返回空。"""
    import httpx

    data = part.get("data") if isinstance(part.get("data"), dict) else {}
    for field in ("url", "file", "file_id", "uri"):
        value = str((data or {}).get(field) or "").strip()
        if not value or value.startswith("base64://"):
            continue
        try:
            if value.startswith("file://"):
                from urllib.parse import unquote, urlparse

                return Path(unquote(urlparse(value).path)).read_bytes()
            if value.startswith("http://") or value.startswith("https://"):
                response = httpx.get(value, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                return response.content
            path = Path(value)
            if path.is_file():
                return path.read_bytes()
        except Exception as exc:  # noqa: BLE001 — 下载失败只少一张图
            logger.debug(f"ping card image download failed ({field}): {exc!r}")
            continue
    return b""


__all__ = [
    "build_card_html",
    "card_output_dir",
    "image_bytes_from_part",
    "render_card_png",
    "save_card_png",
]
