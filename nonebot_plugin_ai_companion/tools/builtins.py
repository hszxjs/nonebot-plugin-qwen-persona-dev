"""内置工具（文档四节收敛范围）：记忆查询、发送图片/表情、网页搜索。

不内置天气等示范性工具；全部经 ToolRegistry 注册，受角色卡 allowed_tools
白名单（权限矩阵）约束。
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import AiCompanionConfig
from ..store import PersonaStore
from .registry import ToolRegistry
from .search import SearchProviderError, build_search_provider

logger = logging.getLogger(__name__)

MEMORY_QUERY_TOOL = "memory_query"
STICKER_SEND_TOOL = "sticker_send"
WEB_SEARCH_TOOL = "web_search"


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    config: AiCompanionConfig,
    store: PersonaStore,
) -> None:
    """注册全部内置工具（幂等：重复注册同一 handler 不报错）。"""

    @registry.register(
        MEMORY_QUERY_TOOL,
        description="查询机器人对群聊历史的记忆：按关键词检索过往聊天记录片段",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"},
                "limit": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["query"],
        },
        permission_tags=("memory",),
    )
    async def memory_query(arguments: dict[str, Any]) -> list[dict[str, str]]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return []
        limit = arguments.get("limit")
        try:
            limit_value = max(1, min(int(limit), 10)) if limit is not None else 5
        except (TypeError, ValueError):
            limit_value = 5
        samples = await store.search_context_samples(group_id=None, query=query, limit=limit_value)
        return [
            {
                "user_id": str(item.get("user_id") or ""),
                "text": str(item.get("plain_text") or item.get("training_text") or "")[:200],
            }
            for item in samples
        ]

    @registry.register(
        STICKER_SEND_TOOL,
        description="按情绪标签挑选群聊表情包：返回可用表情的标识列表，供回复中引用",
        parameters={
            "type": "object",
            "properties": {
                "emotion": {"type": "string", "description": "情绪/场景标签，如：调侃、开心、无语"},
                "limit": {"type": "integer", "description": "候选数量，默认 3"},
            },
            "required": ["emotion"],
        },
        permission_tags=("sticker",),
    )
    async def sticker_send(arguments: dict[str, Any]) -> list[dict[str, Any]]:
        limit = arguments.get("limit")
        try:
            limit_value = max(1, min(int(limit), 8)) if limit is not None else 3
        except (TypeError, ValueError):
            limit_value = 3
        candidates = await store.top_stickers(group_id=None, limit=limit_value)
        return [
            {"key": str(item.get("key") or ""), "summary": str(item.get("summary") or "")[:80]}
            for item in candidates
            if item.get("key")
        ]

    search_provider = build_search_provider(config)
    if search_provider is not None:

        @registry.register(
            WEB_SEARCH_TOOL,
            description="网页搜索：获取互联网上的实时信息（新闻、资料、事实核查）",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
            permission_tags=("web",),
        )
        async def web_search(arguments: dict[str, Any]) -> list[dict[str, str]]:
            query = str(arguments.get("query") or "").strip()
            if not query:
                return []
            try:
                hits = await search_provider.search(query, limit=int(config.ai_companion_search_result_limit or 3))
            except SearchProviderError as exc:
                raise RuntimeError(str(exc)) from exc
            return [hit.to_dict() for hit in hits]

    else:
        logger.info("web_search tool not registered: ai_companion_search_provider not configured")
