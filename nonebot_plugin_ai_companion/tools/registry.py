"""工具注册域：PromptTool 聚合与提示词渲染。"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import PromptTool


@dataclass
class ToolRegistry:
    tools: list[PromptTool] | None = None

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)

    def describe_for_prompt(self) -> str:
        if not self.tools:
            return ""
        descriptions = [tool.describe_for_prompt().strip() for tool in self.tools if tool.describe_for_prompt().strip()]
        return "\n".join(descriptions)
