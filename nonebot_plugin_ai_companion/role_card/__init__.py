"""角色卡域：Bot 人格（bot_persona）与群友模仿（member_mimic）的统一抽象。"""

from .model import RoleCard, RoleCardCreate, RoleType, SessionRoleBinding, session_key_for
from .service import RoleCardService

__all__ = [
    "RoleCard",
    "RoleCardCreate",
    "RoleType",
    "SessionRoleBinding",
    "RoleCardService",
    "session_key_for",
]
