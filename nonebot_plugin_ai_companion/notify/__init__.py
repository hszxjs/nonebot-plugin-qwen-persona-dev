"""「有人找你」提醒域：群里有人 @ / 引用了某群友，bot 替他攒一张提醒卡片。

职责边界：这里只负责"找到该提醒谁、把谁的话拼进卡片、画出来、发出去、留痕"。
不负责 bot 自己被 @ 的回复（那是 chat_service 的 force_reply 路径）。
"""

from .detector import detect_ping_candidates
from .model import FinderEntry, PingCandidate, PingDecision

__all__ = [
    "FinderEntry",
    "PingCandidate",
    "PingDecision",
    "detect_ping_candidates",
]
