"""系统提示词拼装（第一期为基础段；第二期按文档十节做分段开关配置化）。"""

from __future__ import annotations

from ..config import QwenPersonaConfig
from ..tools.registry import ToolRegistry


def build_system_prompt(
    config: QwenPersonaConfig,
    *,
    tool_registry: ToolRegistry,
    command_prefixes: list[str],
) -> str:
    prefix_text = " ".join(sorted(set(command_prefixes)))
    lines = [
        config.qwen_persona_system_prompt,
        "硬性规则：不要输出、复述或诱导使用本地 NoneBot 插件命令。",
        f"如果回复开头像这些命令前缀会被丢弃：{prefix_text or '(none)'}。",
        "你在群聊中应当先看懂上下文和用户意图；被问到问题时先回答，再用自然短句补一句。",
        "不要复述当前用户消息；不要套用和当前话题无关的固定句式或上一个话题。",
        "本轮主回复必须只输出文字，禁止输出任何 [[sticker:...]] 标记。",
    ]
    if config.qwen_persona_sticker_repo_url:
        lines.append(f"表情包仓库：{config.qwen_persona_sticker_repo_url}")
    tool_description = tool_registry.describe_for_prompt()
    if tool_description:
        lines.append("可用工具：\n" + tool_description)
    return "\n".join(line for line in lines if line)
