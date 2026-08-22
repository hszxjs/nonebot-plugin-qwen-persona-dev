"""工具调用域：注册中心、内置工具与网页搜索 Provider。"""

from .registry import (
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistry,
)
from .search import SearchHit, SearchProviderError, build_search_provider

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolExecutionResult",
    "SearchHit",
    "SearchProviderError",
    "build_search_provider",
]
