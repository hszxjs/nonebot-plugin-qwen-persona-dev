"""会话上下文域：消息信封、合并转发展开、昵称解析、媒体部件解析。"""

from .envelope import MessageContextEnvelope, build_message_context
from .forward import count_forward_segments, resolve_forward
from .media_parts import message_to_parts

__all__ = [
    "MessageContextEnvelope",
    "build_message_context",
    "count_forward_segments",
    "resolve_forward",
    "message_to_parts",
]
