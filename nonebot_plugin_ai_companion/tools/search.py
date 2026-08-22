"""网页搜索 Provider 抽象（文档四节）：Tavily / Serper / Bing / 自建 SearxNG 均可实现。

与推理 Provider 相同的注册表范式：新增搜索源只需实现 BaseSearchProvider
并在此注册，不改调用方。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..config import AiCompanionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class BaseSearchProvider(Protocol):
    async def search(self, query: str, *, limit: int = 3) -> list[SearchHit]: ...


class SearchProviderError(RuntimeError):
    """搜索源调用失败。"""


class SearxngSearchProvider:
    """自建 SearxNG（format=json 接口），无需 API key。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str, *, limit: int = 3) -> list[SearchHit]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "format": "json"},
                )
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"searxng request failed: {exc!r}") from exc
        if response.status_code != 200:
            raise SearchProviderError(f"searxng status {response.status_code}")
        try:
            results = response.json().get("results") or []
        except ValueError as exc:
            raise SearchProviderError(f"searxng invalid json: {exc!r}") from exc
        hits: list[SearchHit] = []
        for item in results[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or "")[:120],
                    url=str(item.get("url") or "")[:300],
                    snippet=str(item.get("content") or "")[:300],
                )
            )
        return hits


class TavilySearchProvider:
    """Tavily API（需要 key）。"""

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    async def search(self, query: str, *, limit: int = 3) -> list[SearchHit]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(
                    self.API_URL,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max(1, int(limit)),
                        "search_depth": "basic",
                    },
                )
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"tavily request failed: {exc!r}") from exc
        if response.status_code != 200:
            raise SearchProviderError(f"tavily status {response.status_code}: {response.text[:200]}")
        try:
            results = response.json().get("results") or []
        except ValueError as exc:
            raise SearchProviderError(f"tavily invalid json: {exc!r}") from exc
        hits: list[SearchHit] = []
        for item in results[: max(1, int(limit))]:
            if not isinstance(item, dict):
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or "")[:120],
                    url=str(item.get("url") or "")[:300],
                    snippet=str(item.get("content") or "")[:300],
                )
            )
        return hits


_SEARCH_PROVIDER_FACTORIES: dict[str, Any] = {}


def register_search_provider(name: str, factory: Any) -> None:
    existing = _SEARCH_PROVIDER_FACTORIES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"search provider {name!r} already registered")
    _SEARCH_PROVIDER_FACTORIES[name] = factory


def _build_searxng(config: AiCompanionConfig) -> BaseSearchProvider:
    return SearxngSearchProvider(config.ai_companion_searxng_url)


def _build_tavily(config: AiCompanionConfig) -> BaseSearchProvider:
    return TavilySearchProvider(config.ai_companion_tavily_api_key)


register_search_provider("searxng", _build_searxng)
register_search_provider("tavily", _build_tavily)


def build_search_provider(config: AiCompanionConfig) -> BaseSearchProvider | None:
    """按配置构建搜索源；未配置返回 None（web_search 工具不可用）。"""
    name = str(config.ai_companion_search_provider or "").strip().lower()
    if not name:
        return None
    factory = _SEARCH_PROVIDER_FACTORIES.get(name)
    if factory is None:
        logger.warning("unknown search provider %r (available: %s)", name, sorted(_SEARCH_PROVIDER_FACTORIES))
        return None
    try:
        return factory(config)
    except Exception as exc:  # noqa: BLE001 — 构建失败（缺 url/key 等）降级为不可用
        logger.warning("search provider %r build failed: %r", name, exc)
        return None
