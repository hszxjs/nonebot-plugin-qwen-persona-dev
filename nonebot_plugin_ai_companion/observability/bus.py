"""进程内事件总线：日志写入 → SSE 实时推送（文档十三节实时时间线）。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

_MAX_QUEUE = 200


class _Subscriber:
    def __init__(self, queue: asyncio.Queue[dict[str, Any]], loop: asyncio.AbstractEventLoop) -> None:
        self.queue = queue
        self.loop = loop


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[_Subscriber] = set()

    def publish(self, event: dict[str, Any]) -> None:
        """非阻塞发布；跨线程安全（非订阅 loop 线程经 call_soon_threadsafe 投递）。

        订阅队列满则丢弃该订阅的这条（SSE 客户端可重拉历史）。
        """
        for subscriber in list(self._subscribers):
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is subscriber.loop:
                self._offer(subscriber, event)
            else:
                subscriber.loop.call_soon_threadsafe(self._offer, subscriber, event)

    @staticmethod
    def _offer(subscriber: _Subscriber, event: dict[str, Any]) -> None:
        try:
            subscriber.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("event bus subscriber queue full, dropping event")

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
        subscriber = _Subscriber(queue, asyncio.get_running_loop())
        self._subscribers.add(subscriber)
        try:
            yield queue
        finally:
            self._subscribers.discard(subscriber)
