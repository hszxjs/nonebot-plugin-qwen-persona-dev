"""工具调用域注册中心（文档四节）。

工具通过 ``@registry.register(name=..., parameters=..., permission_tags=[...])``
注册；角色卡级白名单（allowed_tools）决定每次会话哪些工具可用（权限矩阵），
注册中心与具体实现解耦，第三方插件可用同一接口接入。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema（type=object）
    permission_tags: tuple[str, ...] = ()
    handler: ToolHandler | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "permission_tags": list(self.permission_tags),
            "executable": self.handler is not None,
        }


class ToolExecutionError(RuntimeError):
    """工具执行失败（未注册 / 无 handler / handler 抛错）。"""


@dataclass
class ToolExecutionResult:
    name: str
    ok: bool
    payload: Any = None
    error: str = ""

    def render(self, *, max_chars: int = 1200) -> str:
        """把执行结果渲染为回填对话上下文的文本（超长截断）。"""
        if not self.ok:
            return f"工具 {self.name} 执行失败：{self.error}"
        try:
            text = (
                self.payload
                if isinstance(self.payload, str)
                else json.dumps(self.payload, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            text = str(self.payload)
        text = str(text or "").strip()
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 1)] + "…"
        return text or "（无结果）"


class ToolRegistry:
    """工具注册中心；旧版 ``tools`` 列表属性以只读视图兼容。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    @property
    def tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    def register(
        self,
        name: str,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        permission_tags: tuple[str, ...] | list[str] = (),
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register_tool(
                ToolDefinition(
                    name=name,
                    description=description or getattr(handler, "__doc__", "") or name,
                    parameters=dict(parameters or {"type": "object", "properties": {}}),
                    permission_tags=tuple(permission_tags),
                    handler=handler,
                )
            )
            return handler

        return decorator

    def register_tool(self, tool: ToolDefinition) -> None:
        existing = self._tools.get(tool.name)
        if existing is not None and existing is not tool:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self, *, allowed: list[str] | set[str] | None = None) -> list[str]:
        return [tool.name for tool in self.list_tools(allowed=allowed)]

    def list_tools(self, *, allowed: list[str] | set[str] | None = None) -> list[ToolDefinition]:
        if allowed is None:
            return list(self._tools.values())
        allowed_set = {str(item) for item in allowed}
        return [tool for name, tool in self._tools.items() if name in allowed_set]

    def describe_for_prompt(self, allowed: list[str] | set[str] | None = None) -> str:
        """渲染工具说明进系统提示词（仅白名单内的工具）。"""
        entries: list[str] = []
        for tool in self.list_tools(allowed=allowed):
            params = ", ".join(
                f"{key}({spec.get('type', 'any')})"
                for key, spec in (tool.parameters.get("properties") or {}).items()
                if isinstance(spec, dict)
            )
            param_text = f"（参数：{params}）" if params else ""
            entries.append(f"- {tool.name}：{tool.description}{param_text}")
        return "\n".join(entries)

    def openai_tools_payload(self, allowed: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
        """生成 OpenAI/GLM function calling 的 tools 载荷（仅白名单内）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.list_tools(allowed=allowed)
        ]

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolExecutionResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(name, ok=False, error="tool not registered")
        if tool.handler is None:
            return ToolExecutionResult(name, ok=False, error="tool has no handler")
        try:
            payload = await tool.handler(dict(arguments or {}))
        except Exception as exc:  # noqa: BLE001 — 工具异常回填为失败结果，不打断主流程
            logger.warning("tool %s execution failed: %r", name, exc)
            return ToolExecutionResult(name, ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolExecutionResult(name, ok=True, payload=payload)
