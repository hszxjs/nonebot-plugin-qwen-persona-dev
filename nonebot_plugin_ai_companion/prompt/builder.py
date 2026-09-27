"""系统提示词分段拼装（文档十节）。

段落注册表 + 可配置开关/排序：identity（身份角色卡段）、selfcheck（发送前自检段）、
naturalness（反拟人化/自然对话段）、sticker（表情决策段）、tools（工具段）为默认段，
顺序与一期输出逐字节兼容；memory（记忆注入段）、scene（场景感知段）、motivation
（接话动机段）为二期新增段，仅在提供内容/未被禁用时拼入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import AiCompanionConfig
from ..tools.registry import ToolRegistry

DEFAULT_SEGMENT_ORDER: tuple[str, ...] = (
    "identity",
    "mode",
    "selfcheck",
    "naturalness",
    "sticker",
    "tools",
    "memory",
    "scene",
)
KNOWN_SEGMENTS: tuple[str, ...] = DEFAULT_SEGMENT_ORDER + ("motivation",)

MOTIVATION_LINES: tuple[str, ...] = (
    "只有存在明确话头才主动展开新话题；没有天然话头时优先简短呼应或提出小问题，而非表演式发言。",
    "禁止用「说到这个」「突然想起」等生硬转折词硬接话题。",
    "把握不大时允许只简短回应甚至不接话。",
)


@dataclass
class PromptContext:
    """一次提示词拼装的输入（chat 服务填充，运营配置在 config 中）。"""

    config: AiCompanionConfig
    tool_registry: ToolRegistry
    command_prefixes: list[str] = field(default_factory=list)
    identity_text: str | None = None  # 角色卡渲染后的身份段；None 时回退 config 系统提示词
    allowed_tools: list[str] | None = None  # 角色卡工具白名单（None = 全部）
    memory_lines: list[str] = field(default_factory=list)  # 记忆注入段（长期记忆/画像摘要）
    scene_lines: list[str] = field(default_factory=list)  # 场景感知段（群/私聊、话题摘要等）
    mode: str = "member"  # member = 群里一起聊；assistant = 被点名/私聊，直给答案


def _identity_lines(ctx: PromptContext) -> list[str]:
    return [ctx.identity_text if ctx.identity_text is not None else ctx.config.ai_companion_system_prompt]


MEMBER_MODE_LINES: tuple[str, ...] = (
    "你现在在群里和群友一起聊天：可以接梗、互损、短句插话，不确定就少说。",
    "不要像客服或助手那样逐条回答，也不要每句都提自己是 AI；真要答问题就直说答案。",
)
ASSISTANT_MODE_LINES: tuple[str, ...] = (
    "现在有人点名或在私聊问你，按「助手」口气回复：先直接给答案，再补一句闲聊。",
    "不要用语气词绕开问题，不要用鱼话/梗糊弄事实；不确定就说不确定。",
)


def _mode_lines(ctx: PromptContext) -> list[str]:
    return list(ASSISTANT_MODE_LINES if str(ctx.mode or "member") == "assistant" else MEMBER_MODE_LINES)


def _selfcheck_lines(ctx: PromptContext) -> list[str]:
    prefix_text = " ".join(sorted(set(ctx.command_prefixes)))
    return [
        "硬性规则：不要输出、复述或诱导使用本地 NoneBot 插件命令。",
        f"如果回复开头像这些命令前缀会被丢弃：{prefix_text or '(none)'}。",
    ]


def _naturalness_lines(ctx: PromptContext) -> list[str]:
    return [
        "你在群聊中应当先看懂上下文和用户意图；被问到问题时先回答，再用自然短句补一句。",
        "不要复述当前用户消息；不要套用和当前话题无关的固定句式或上一个话题。",
    ]


def _sticker_lines(ctx: PromptContext) -> list[str]:
    lines = ["本轮主回复必须只输出文字，禁止输出任何 [[sticker:...]] 标记。"]
    if ctx.config.ai_companion_sticker_repo_url:
        lines.append(f"表情包仓库：{ctx.config.ai_companion_sticker_repo_url}")
    return lines


def _tools_lines(ctx: PromptContext) -> list[str]:
    description = ctx.tool_registry.describe_for_prompt(ctx.allowed_tools)
    if not description:
        return []
    return [
        "可用工具（由系统自动执行，你只需要正常说话）：\n" + description,
        "绝不要在回复文本中描述、假装或演示调用工具；不要输出「调用工具」「工具返回」「生成最终回复」等过程字样。",
    ]


def _memory_lines(ctx: PromptContext) -> list[str]:
    if not ctx.memory_lines:
        return []
    return ["以下是背景记忆信息，仅供参照，不要逐字复述：", *ctx.memory_lines]


def _scene_lines(ctx: PromptContext) -> list[str]:
    return list(ctx.scene_lines)


def _motivation_lines(ctx: PromptContext) -> list[str]:
    return list(MOTIVATION_LINES)


_SEGMENT_BUILDERS: dict[str, Callable[[PromptContext], list[str]]] = {
    "identity": _identity_lines,
    "selfcheck": _selfcheck_lines,
    "naturalness": _naturalness_lines,
    "sticker": _sticker_lines,
    "tools": _tools_lines,
    "memory": _memory_lines,
    "scene": _scene_lines,
    "motivation": _motivation_lines,
    "mode": _mode_lines,
}


def effective_segment_order(config: AiCompanionConfig) -> list[str]:
    """解析生效的段顺序：配置优先，未知段名忽略，缺失段按默认顺序补齐在末尾。"""
    configured = [str(item).strip() for item in config.ai_companion_prompt_segment_order if str(item).strip()]
    order = [name for name in configured if name in _SEGMENT_BUILDERS]
    for name in DEFAULT_SEGMENT_ORDER:
        if name not in order:
            order.append(name)
    return order


def build_system_prompt(ctx: PromptContext) -> str:
    disabled = {str(item).strip() for item in ctx.config.ai_companion_disabled_prompt_segments}
    lines: list[str] = []
    for segment in effective_segment_order(ctx.config):
        if segment in disabled:
            continue
        lines.extend(line for line in _SEGMENT_BUILDERS[segment](ctx) if line)
    return "\n".join(lines)


# ---- 兼容入口：一期签名（chat_service 已迁移；保留给外部调用方）----


def build_system_prompt_legacy(
    config: AiCompanionConfig,
    *,
    tool_registry: ToolRegistry,
    command_prefixes: list[str],
) -> str:
    return build_system_prompt(
        PromptContext(
            config=config,
            tool_registry=tool_registry,
            command_prefixes=command_prefixes,
        )
    )
