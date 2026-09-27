"""过滤"关于机器人自己的运行机制"的记忆与摘要。

抽取与摘要的提示词里都写了不要记这类东西，但模型仍会偶尔违反（实测出现过
「机器人曾因替换字符问题一度无法发言」「群bot 70% 概率触发回复」）。
这种记忆一旦注入，会让 bot 在群里背自己的参数，所以代码侧再兜一道。
注意：只挡"机制描述"，群友拿它开玩笑的称呼（如「鸡哥」）要留着。
"""

from __future__ import annotations

import re

# 群记忆 / 摘要：连"插件玩法"这类也算元信息（群层面该记的是"在玩什么"，不是命令用法）
_GROUP_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(机器人|bot|Bot|BOT|AI|ai).{0,14}(概率|触发|回复率|参数|规则|指令|插件|上下文|记忆|模型|版本|训练|复读|静默|发言)"
    ),
    re.compile(r"(回复|触发|搭话).{0,6}概率"),
    re.compile(r"插件.{0,8}(指令|命令|用法|参数|功能)"),
    re.compile(r"(提示词|系统提示|上下文窗口|温度参数|max_tokens|token 上限)"),
    re.compile(r"(模型|训练).{0,6}(替换字符|乱码|失联|报错|超时)"),
)

# 成员记忆：只挡"机器人自身运行机制"。
# 「他是插件的开发者」这类是关于这个人的事实，正是 bot 该记住的，不能误伤。
_MEMBER_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(机器人|bot|Bot|BOT|AI|ai).{0,14}(概率|触发|回复率|上下文窗口|提示词|max_tokens)"),
    re.compile(r"(回复|触发|搭话).{0,6}概率"),
    re.compile(r"(提示词|系统提示|上下文窗口|温度参数|max_tokens|token 上限)"),
)


def looks_like_bot_meta(text: str, *, scope: str = "group") -> bool:
    """scope=group 严格（群层面不记机制）；scope=member 宽松（保留关于这个人的事实）。"""
    clean = " ".join(str(text or "").split())
    if not clean:
        return False
    patterns = _GROUP_META_PATTERNS if str(scope) == "group" else _MEMBER_META_PATTERNS
    return any(pattern.search(clean) for pattern in patterns)


__all__ = ["looks_like_bot_meta"]
