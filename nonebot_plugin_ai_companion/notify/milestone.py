"""群聊成就（里程碑卡）：成员总发言突破阈值时发一张成就卡。

数据源是会话时间线（conversation_turns，成员消息，不含 bot 与命令噪声）：
- 总发言数 = 该成员在该群的时间线条数；
- 表情包数 = 其中 sticker_keys 非空的条数；
- 群内排名 = 按发言数在全群成员中的名次；击败百分比 = (总人数 - 排名) / 总人数。

触发：成员发言使总数恰好跨过某个阈值（100 / 1000 / 6666 / 10000 的整数倍…）。
计数是精确的：一条消息 +1，跨线那一刻只触发一次，无需"已播报"标记。
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认里程碑（趣味号 + 常规号）；>=10000 的整万数自动触发，无需配置
MILESTONE_DEFAULTS: tuple[int, ...] = (100, 500, 1000, 2000, 3000, 4000, 5000, 6666, 7777, 8888, 9999)

_CSS = """
body { margin:0; padding:0; background:transparent;
       font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif; }
.bg { width:640px; background:#7de3c4; padding:34px 38px 26px; }
.badge { display:inline-block; background:#ef3e5b; color:#fff; font-size:20px; font-weight:700;
         padding:7px 18px; border-radius:14px; letter-spacing:2px; }
.card { position:relative; background:#fff; border-radius:26px; margin-top:104px;
        padding:96px 44px 40px; color:#26262b; }
.ribbon { position:absolute; top:-124px; left:104px; width:38px; height:132px;
          background:#f9c9d4; transform:rotate(-24deg); z-index:0; }
.ribbon.right { left:206px; transform:rotate(24deg); }
.medal { position:absolute; top:-96px; left:56px; width:186px; height:186px; border-radius:50%;
         background:#f7c948; display:flex; align-items:center; justify-content:center;
         box-shadow:0 4px 10px rgba(0,0,0,0.08); z-index:1; }
.medal .inner { width:156px; height:156px; border-radius:50%; background:#fff; overflow:hidden;
                display:flex; align-items:center; justify-content:center; }
.medal img { width:100%; height:100%; object-fit:cover; }
.medal .initial { font-size:64px; font-weight:700; color:#f7c948; }
.name { font-size:34px; font-weight:700; margin:6px 0 26px; }
.label { color:#a9a9b2; font-size:17px; line-height:1.3; font-weight:600; }
.main { font-size:34px; font-weight:700; margin:6px 0 18px; }
.main .num { color:#ef3e5b; font-size:74px; font-weight:800; margin-left:10px; }
.row { font-size:23px; font-weight:600; margin:8px 0; }
.row .red { color:#ef3e5b; }
.footer { display:flex; justify-content:space-between; margin-top:18px;
          color:#1e5c4a; font-size:15px; font-weight:600; }
"""


def is_milestone(count: int, configured: list[int] | tuple[int, ...] = MILESTONE_DEFAULTS) -> bool:
    """是否跨过里程碑：命中配置号，或 >=10000 的整万数。"""
    value = int(count)
    if value <= 0:
        return False
    if value in {int(item) for item in configured}:
        return True
    return value >= 10000 and value % 10000 == 0


def _initial(name: str) -> str:
    clean = (name or "").strip()
    return clean[:1] if clean else "?"


async def fetch_avatar(qq: str) -> bytes | None:
    """QQ 头像（q1.qlogo.cn）；失败返回 None（卡片回落为首字金圈）。"""
    import httpx

    clean = str(qq or "").strip()
    if not clean:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            response = await client.get(f"https://q1.qlogo.cn/g?b=qq&nk={clean}&s=640")
            response.raise_for_status()
            return response.content if response.content else None
    except Exception as exc:  # noqa: BLE001 — 头像失败不废整卡
        logger.debug(f"milestone avatar download failed: {exc!r}")
        return None


def _avatar_tag(avatar: bytes | None, name: str) -> str:
    if avatar:
        import base64

        encoded = base64.b64encode(avatar).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" alt="头像" />'
    return f'<span class="initial">{html.escape(_initial(name))}</span>'


def build_milestone_html(
    *,
    badge: str,
    name: str,
    count: int,
    rank: int,
    total_members: int,
    sticker_count: int,
    avatar: bytes | None = None,
    footer_left: str = "",
    footer_right: str = "milestone card :)",
) -> str:
    """成就卡 HTML（纯函数；截图样式的还原）。"""
    beaten = max(0.0, (total_members - rank) / max(1, total_members) * 100)
    sticker_pct = sticker_count / max(1, count) * 100
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style></head><body><div class='bg'>"
        f"<div class='badge'>{html.escape(badge)}</div>"
        "<div class='card'>"
        "<div class='ribbon'></div><div class='ribbon right'></div>"
        f"<div class='medal'><div class='inner'>{_avatar_tag(avatar, name)}</div></div>"
        f"<div class='name'>{html.escape(name)}</div>"
        "<div class='label'>Total number<br />of messages</div>"
        f"<div class='main'>总发言破<span class='num'>{count}</span></div>"
        f"<div class='row'>群内排名 <span class='red'>#{rank}</span></div>"
        f"<div class='row'>{{ 击败了群内 <span class='red'>{beaten:.2f}%</span> 的活跃群友 }}</div>"
        f"<div class='row'>其中表情包 <span class='red'>{sticker_count}</span> 条，占比 <span class='red'>{sticker_pct:.1f}%</span></div>"
        "</div>"
        f"<div class='footer'><span>{html.escape(footer_left)}</span><span>{html.escape(footer_right)}</span></div>"
        "</div></body></html>"
    )


async def render_milestone_png(**kwargs: Any) -> bytes:
    """渲染成 PNG。htmlrender 延迟导入：没有 driver 的单测环境不会触发。"""
    from nonebot import require

    require("nonebot_plugin_htmlrender")
    from nonebot_plugin_htmlrender import html_to_pic

    page_html = build_milestone_html(**kwargs)
    return await html_to_pic(page_html, type="png", device_scale_factor=2)


def save_milestone_png(data_dir: str | Path, png_bytes: bytes, *, user_id: str) -> Path:
    out_dir = Path(data_dir) / "ping_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"milestone-{user_id}-{int(time.time() * 1000)}.png"
    path.write_bytes(png_bytes)
    now = time.time()
    for old in out_dir.glob("milestone-*.png"):
        try:
            if now - old.stat().st_mtime > 86400.0:
                old.unlink(missing_ok=True)
        except OSError:
            continue
    return path


async def fetch_member_stats(store: Any, *, group_id: str, user_id: str) -> dict[str, int] | None:
    """该成员在该群的（总发言，排名，活跃成员数，表情包数）。"""

    def _sync() -> dict[str, int] | None:

        with store.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT user_id, COUNT(*) AS c,
                       SUM(CASE WHEN sticker_keys != '[]' AND sticker_keys != '' THEN 1 ELSE 0 END) AS stickers
                FROM conversation_turns
                WHERE group_id = ? AND speaker_kind = 'member'
                  AND user_id IS NOT NULL AND user_id != ''
                GROUP BY user_id ORDER BY c DESC
                """,
                (group_id,),
            ).fetchall()
        total_members = len(rows)
        for position, row in enumerate(rows, start=1):
            if str(row["user_id"]) == str(user_id):
                stickers_raw = row["stickers"]
                sticker_count = int(stickers_raw) if stickers_raw is not None else 0
                # 兼容：sticker_keys 存的是 JSON 数组字符串
                return {
                    "total": int(row["c"]),
                    "rank": position,
                    "total_members": total_members,
                    "sticker_count": sticker_count,
                }
        return None

    return await asyncio.to_thread(_sync)


__all__ = [
    "MILESTONE_DEFAULTS",
    "build_milestone_html",
    "fetch_avatar",
    "fetch_member_stats",
    "is_milestone",
    "render_milestone_png",
    "save_milestone_png",
]
