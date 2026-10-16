"""提醒卡片：HTML 构建（纯函数）与渲染（htmlrender，延迟导入）。

卡片是"离线摘要"：目标上线时一次性看到被找时的上下文（±N 条，相连区间合并），
触发行高亮标记，找的人发的图片以 data URL 内嵌（触发当下已下载，QQ URL 会过期）。
"""

from __future__ import annotations

import base64
import html
import io
import logging
import time
from pathlib import Path
from typing import Any

from ..conversation.window import render_turn_line, safe_speaker_name
from .digest import DigestBlock

logger = logging.getLogger(__name__)

# 单行文本与图片上限：卡片是提醒摘要，不是聊天记录备份
MAX_TEXT_CHARS = 200
MAX_IMAGES = 2
MAX_IMAGE_DIM = 480
CARD_MAX_AGE_SECONDS = 86400.0

_CSS = """
body { margin:0; padding:0; background:transparent; font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif; }
.card { width:580px; background:#1a1a22; border-radius:18px; padding:20px 22px; color:#e8e8ef; }
.title { font-size:22px; font-weight:600; margin:2px 4px 4px; color:#f2f2f7; }
.subtitle { font-size:13px; color:#8a8a98; margin:0 4px 14px; }
.block { margin-bottom:12px; }
.line { display:flex; gap:8px; margin-bottom:8px; align-items:flex-start; }
.speaker { color:#7fa2ff; font-size:13px; flex-shrink:0; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-top:2px; }
.content { min-width:0; flex:1; background:#2b2b35; border-radius:4px 12px 12px 12px; padding:8px 12px; font-size:14px; line-height:1.55; word-break:break-word; color:#e8e8ef; }
.content.trigger { background:#31304a; border:1px solid #4d6bfe66; }
.badge { display:inline-block; font-size:11px; color:#aab8ff; border:1px solid #4d6bfe88; border-radius:8px; padding:0 6px; margin-left:6px; vertical-align:1px; }
.content img { max-width:300px; max-height:220px; border-radius:8px; display:block; margin-top:6px; }
.divider { text-align:center; color:#55555f; font-size:12px; letter-spacing:6px; margin:10px 0; }
.truncated { font-size:12px; color:#6f6f7c; margin:0 4px 10px; }
.hint { font-size:12px; color:#6f6f7c; margin:6px 4px 0; text-align:right; }
"""


def _downscale_to_data_url(image_bytes: bytes) -> str | None:
    """图片字节 → 缩放后的 data URL（进 HTML 用 base64，避免文件白名单问题）。"""
    try:
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


def _files_to_data_urls(paths: list[str]) -> list[str]:
    urls: list[str] = []
    for path_text in paths[:MAX_IMAGES]:
        try:
            url = _downscale_to_data_url(Path(path_text).read_bytes())
            if url:
                urls.append(url)
        except OSError as exc:
            logger.debug(f"ping attachment read failed: {exc!r}")
    return urls


def _empty_turn_speaker(turn: Any, name_map: dict[str, str]) -> str:
    """空文本行（纯图片/表情消息）的说话人：拿不到行文本时的兜底渲染。"""
    if getattr(turn, "is_bot", False):
        return "你"
    name = getattr(turn, "speaker_name", "") or name_map.get(str(getattr(turn, "user_id", "") or ""), "")
    return safe_speaker_name(name, user_id=getattr(turn, "user_id", None))


def build_digest_html(
    *,
    title: str,
    subtitle: str,
    blocks: list[DigestBlock],
    names: dict[str, str] | None = None,
    attachments_by_message_id: dict[str, list[str]] | None = None,
    dropped_lines: int = 0,
) -> str:
    """投递卡：合并后的上下文块 + 触发行标记 + 附件图。纯字符串输出。"""
    name_map = names or {}
    attachment_map = attachments_by_message_id or {}
    block_htmls: list[str] = []
    for block in blocks:
        lines: list[str] = []
        for turn in block.turns:
            message_id = str(turn.get("source_message_id") or "")
            turn_obj = turn.get("turn")
            if turn_obj is None:
                continue
            line = render_turn_line(turn_obj, names=name_map)
            is_trigger = message_id in block.trigger_message_ids
            images = "".join(
                f'<img src="{url}" alt="图片" />' for url in _files_to_data_urls(attachment_map.get(message_id, []))
            )
            # 纯图片消息在时间线里是空文本行：触发行与带附件的行必须照常渲染，
            # 否则丢的恰好是被 @ 时发的那张图与「找你」标记本身。
            if not line and not images and not is_trigger:
                continue
            speaker, _, body = line.partition("：")
            if not line:
                speaker = _empty_turn_speaker(turn_obj, name_map)
            badge = '<span class="badge">找你</span>' if is_trigger else ""
            body_html = html.escape(body).replace("\n", "<br />") or "（图片消息）"
            lines.append(
                '<div class="line">'
                f'<span class="speaker">{html.escape(speaker)}</span>'
                f'<span class="content{" trigger" if is_trigger else ""}">{body_html}{badge}{images}</span>'
                "</div>"
            )
        if lines:
            block_htmls.append('<div class="block">' + "".join(lines) + "</div>")
    body = '<div class="divider">· · ·</div>'.join(block_htmls) or '<div class="bubble">（没有可展示的内容）</div>'
    truncation = f'<div class="truncated">（更早的 {dropped_lines} 条已省略）</div>' if dropped_lines > 0 else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='card'>"
        f"<div class='title'>{html.escape(title)}</div>"
        f"<div class='subtitle'>{html.escape(subtitle)}</div>"
        f"{truncation}{body}"
        f"<div class='hint'>{time.strftime('%H:%M')}</div>"
        "</div></body></html>"
    )


def card_output_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "ping_cards"


def local_file_uri(path: str | Path) -> str:
    """本地图片发送用的绝对 file:// URI。

    data_dir 是相对路径（data/qwen_persona），直接拼 `file://{path}` 时 NapCat
    会按自己的 CWD 解析相对路径、报 ENOENT——必须先 resolve 成绝对路径再发送。
    """
    return Path(path).resolve().as_uri()


def attachment_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "ping_attachments"


def save_card_png(data_dir: str | Path, png_bytes: bytes, *, target_user_id: str) -> Path:
    """落盘并清理超过一天的旧卡片图。"""
    out_dir = card_output_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ping-{target_user_id}-{int(time.time() * 1000)}.png"
    path.write_bytes(png_bytes)
    _cleanup_old(out_dir)
    return path


def save_attachment(data_dir: str | Path, image_bytes: bytes, *, message_id: str, index: int) -> Path | None:
    """触发时的图片落盘（QQ URL 会过期，必须当场存）。"""
    if not image_bytes:
        return None
    out_dir = attachment_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ping-{message_id or 'noid'}-{index}-{int(time.time() * 1000)}.jpg"
    path.write_bytes(image_bytes)
    _cleanup_old(out_dir)
    return path


def delete_attachments(paths: list[str]) -> int:
    removed = 0
    for path_text in paths:
        try:
            Path(path_text).unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def _cleanup_old(directory: Path) -> None:
    now = time.time()
    for old in directory.glob("*"):
        try:
            if now - old.stat().st_mtime > CARD_MAX_AGE_SECONDS:
                old.unlink(missing_ok=True)
        except OSError:
            continue


async def render_digest_png(**kwargs: Any) -> bytes:
    """渲染成 PNG。htmlrender 延迟导入：没有 driver 的单测环境不会触发。"""
    from nonebot import require

    require("nonebot_plugin_htmlrender")
    from nonebot_plugin_htmlrender import html_to_pic

    page_html = build_digest_html(**kwargs)
    return await html_to_pic(page_html, type="png", device_scale_factor=2)


def image_bytes_from_part(part: dict[str, Any]) -> bytes:
    """OneBot 图片段 → 原始字节（触发当下下载；本地 file:// 直接读）。失败返回空。"""
    from urllib.parse import unquote, urlparse

    import httpx

    data = part.get("data") if isinstance(part.get("data"), dict) else {}
    for field_name in ("url", "file", "file_id", "uri"):
        value = str((data or {}).get(field_name) or "").strip()
        if not value or value.startswith("base64://"):
            continue
        try:
            if value.startswith("file://"):
                return Path(unquote(urlparse(value).path)).read_bytes()
            if value.startswith(("http://", "https://")):
                response = httpx.get(value, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                return response.content
            path = Path(value)
            if path.is_file():
                return path.read_bytes()
        except Exception as exc:  # noqa: BLE001 — 下载失败只少一张图
            logger.debug(f"ping image download failed ({field_name}): {exc!r}")
            continue
    return b""


__all__ = [
    "build_digest_html",
    "card_output_dir",
    "delete_attachments",
    "image_bytes_from_part",
    "local_file_uri",
    "render_digest_png",
    "save_attachment",
    "save_card_png",
]
