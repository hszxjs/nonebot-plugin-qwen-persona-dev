"""群成员昵称解析（带 TTL 缓存）。

主语指代提取用：把 @ 的 qq 号、被引用消息的发送者解析成可读昵称，
让 bot 知道"谁在对谁说话"。避免每条消息都打 get_group_member_info API。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# 缓存 TTL（秒）。
_CACHE_TTL = 1800
# 单次查询超时（秒）。
_QUERY_TIMEOUT = 3.0


class NicknameResolver:
    """群成员昵称查询，带 in-memory TTL 缓存。线程/协程安全（GIL + dict 原子操作足够）。"""

    def __init__(self) -> None:
        # key: (group_id, user_id) -> (nickname, expire_ts)
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    async def resolve(self, bot: Any, group_id: str | None, user_id: str | None) -> str | None:
        """解析昵称。失败/超时返回 None（调用方降级为 qq 号）。"""
        if not group_id or not user_id:
            return None
        cache_key = (str(group_id), str(user_id))
        cached = self._cache.get(cache_key)
        if cached and cached[1] > time.time():
            return cached[0]

        nickname: str | None = None
        try:
            info = await asyncio.wait_for(
                bot.call_api(
                    "get_group_member_info",
                    group_id=int(group_id),
                    user_id=int(user_id),
                ),
                timeout=_QUERY_TIMEOUT,
            )
            # 群名片优先，其次昵称
            nickname = str(info.get("card") or info.get("nickname") or "").strip() or None
        except TimeoutError:
            logger.debug("nickname resolve timeout: group=%s user=%s", group_id, user_id)
        except (ValueError, TypeError) as exc:
            logger.debug("nickname resolve bad id: %r", exc)
        except Exception as exc:  # noqa: BLE001 — API 调用失败不阻塞主流程
            logger.debug("nickname resolve failed: group=%s user=%s err=%r", group_id, user_id, exc)

        # 即使是 None 也缓存（避免对查不到的 qq 反复打 API），但 TTL 短一些
        ttl = _CACHE_TTL if nickname else 300
        self._cache[cache_key] = (nickname or "", time.time() + ttl)
        # 控制缓存规模
        if len(self._cache) > 2000:
            now = time.time()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
        return nickname

    async def resolve_many(
        self,
        bot: Any,
        group_id: str | None,
        user_ids: list[str],
    ) -> dict[str, str | None]:
        """批量解析，返回 {user_id: nickname}。"""
        if not user_ids:
            return {}
        results = await asyncio.gather(
            *(self.resolve(bot, group_id, uid) for uid in user_ids),
            return_exceptions=True,
        )
        out: dict[str, str | None] = {}
        for uid, res in zip(user_ids, results, strict=False):
            out[uid] = res if isinstance(res, str) else None
        return out


# 模块级单例（插件内共享一个缓存池）。
nickname_resolver = NicknameResolver()
