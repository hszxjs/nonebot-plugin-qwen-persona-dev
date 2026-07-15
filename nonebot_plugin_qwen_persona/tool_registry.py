from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PromptTool(Protocol):
    name: str

    def describe_for_prompt(self) -> str:
        ...


@dataclass
class ToolRegistry:
    tools: list[PromptTool] | None = None

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)

    def describe_for_prompt(self) -> str:
        if not self.tools:
            return ""
        descriptions = [
            tool.describe_for_prompt().strip()
            for tool in self.tools
            if tool.describe_for_prompt().strip()
        ]
        return "\n".join(descriptions)
