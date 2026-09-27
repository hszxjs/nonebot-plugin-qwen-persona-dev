"""会话时间线域：群/私聊对话记录（谁在什么时候说了什么）+ 昵称快照。

对外入口：`ConversationRepo`（写入/窗口读取/回填）、`MemberNameRepo`（昵称表）、
`window` 里的渲染函数（时间线 → 提示词行）。
"""

from .model import SPEAKER_BOT, SPEAKER_MEMBER, ConversationTurn, session_key_for
from .repo import ConversationRepo, MemberNameRepo
from .window import render_history_lines, render_turn_line, safe_speaker_name

__all__ = [
    "SPEAKER_BOT",
    "SPEAKER_MEMBER",
    "ConversationRepo",
    "ConversationTurn",
    "MemberNameRepo",
    "render_history_lines",
    "render_turn_line",
    "safe_speaker_name",
    "session_key_for",
]
