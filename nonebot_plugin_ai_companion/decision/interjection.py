"""主动搭话的加权与限流。

群聊里 bot 是"成员"，但 0.7 的固定概率会让它对几乎每句话都插嘴（历史数据显示
94.9% 的回复都是主动插话，且高度集中在 1-2 个人身上）。这里把概率拆成：
基础概率 × 热度/亲近度加权，再叠每群冷却与每小时上限，超限时只回被叫的。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# 加权系数（固定经验值，避免配置项爆炸）
MULTIPLIER_ADDRESSED_RECENT = 1.5  # 刚才有人在跟它说话
MULTIPLIER_HEAT = 1.5  # 群里话题正热
MULTIPLIER_SAME_SPEAKER = 0.5  # 刚回过同一个人，别追着同一人说


@dataclass(frozen=True)
class InterjectionInput:
    base_probability: float
    addressed_recently: bool = False
    heat_messages: int = 0
    heat_min_messages: int = 5
    same_speaker_as_last_reply: bool = False


@dataclass
class InterjectionResult:
    probability: float
    multipliers: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return ",".join(self.multipliers) if self.multipliers else "base"


def weighted_probability(data: InterjectionInput) -> InterjectionResult:
    """基础概率 × 各加权项，封顶 1.0。"""
    try:
        probability = max(0.0, min(1.0, float(data.base_probability)))
    except (TypeError, ValueError):
        probability = 0.0
    multipliers: list[str] = []
    if probability > 0:
        if data.addressed_recently:
            probability *= MULTIPLIER_ADDRESSED_RECENT
            multipliers.append(f"addressed×{MULTIPLIER_ADDRESSED_RECENT}")
        if int(data.heat_messages) >= max(1, int(data.heat_min_messages)):
            probability *= MULTIPLIER_HEAT
            multipliers.append(f"heat×{MULTIPLIER_HEAT}")
        if data.same_speaker_as_last_reply:
            probability *= MULTIPLIER_SAME_SPEAKER
            multipliers.append(f"same-speaker×{MULTIPLIER_SAME_SPEAKER}")
    return InterjectionResult(probability=min(1.0, probability), multipliers=multipliers)


@dataclass
class _GroupState:
    last_reply_at: float = 0.0
    last_addressed_at: float = 0.0
    last_reply_user_id: str = ""
    last_message_at: float = 0.0
    recent_message_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    reply_times: deque[float] = field(default_factory=lambda: deque(maxlen=500))


class InterjectionLimiter:
    """每群冷却 + 每小时上限（内存态，进程重启即归零，够用且零依赖）。

    阈值支持"热读"：传 config 时每次判定都取当前值，改配置立即生效（不必重启）。
    """

    def __init__(
        self,
        *,
        cooldown_seconds: float = 45.0,
        hourly_limit: int = 20,
        heat_window_seconds: float = 120.0,
        config: Any | None = None,
    ) -> None:
        self._config = config
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._hourly_limit = max(0, int(hourly_limit))
        self._heat_window_seconds = max(1.0, float(heat_window_seconds))
        self._groups: dict[str, _GroupState] = {}

    @property
    def cooldown_seconds(self) -> float:
        if self._config is not None:
            return max(0.0, float(getattr(self._config, "ai_companion_group_cooldown_seconds", 45) or 0))
        return self._cooldown_seconds

    @property
    def hourly_limit(self) -> int:
        if self._config is not None:
            return max(0, int(getattr(self._config, "ai_companion_group_hourly_limit", 20) or 0))
        return self._hourly_limit

    @property
    def heat_window_seconds(self) -> float:
        if self._config is not None:
            return max(1.0, float(getattr(self._config, "ai_companion_heat_window_seconds", 120) or 1))
        return self._heat_window_seconds

    def _state(self, group_id: str) -> _GroupState:
        return self._groups.setdefault(str(group_id), _GroupState())

    def observe_message(self, *, group_id: str, user_id: str = "", now: float | None = None) -> None:
        """记录一条群消息（用于热度统计）。"""
        if not group_id:
            return
        current = float(now if now is not None else time.time())
        state = self._state(group_id)
        state.last_message_at = current
        state.recent_message_times.append(current)

    def observe_reply(self, *, group_id: str, user_id: str = "", now: float | None = None) -> None:
        """记录一次 bot 回复（用于冷却与上限）。"""
        if not group_id:
            return
        current = float(now if now is not None else time.time())
        state = self._state(group_id)
        state.last_reply_at = current
        state.last_reply_user_id = str(user_id or "")
        state.reply_times.append(current)

    def heat_messages(self, *, group_id: str, now: float | None = None) -> int:
        if not group_id:
            return 0
        current = float(now if now is not None else time.time())
        state = self._state(group_id)
        cutoff = current - self.heat_window_seconds
        return sum(1 for stamp in state.recent_message_times if stamp >= cutoff)

    def mark_addressed(self, *, group_id: str, now: float | None = None) -> None:
        """记录"被叫过"（@ / 被回复）：短时间内主动搭话加权。"""
        if not group_id:
            return
        self._state(group_id).last_addressed_at = float(now if now is not None else time.time())

    def was_addressed_recently(self, *, group_id: str, within_seconds: float = 120.0, now: float | None = None) -> bool:
        if not group_id:
            return False
        current = float(now if now is not None else time.time())
        last = self._state(group_id).last_addressed_at
        return bool(last and current - last <= max(1.0, float(within_seconds)))

    def same_speaker_as_last_reply(self, *, group_id: str, user_id: str) -> bool:
        if not group_id or not user_id:
            return False
        return self._state(group_id).last_reply_user_id == str(user_id)

    def block_reason(self, *, group_id: str, now: float | None = None) -> str:
        """返回限流原因；空串表示放行。"""
        if not group_id:
            return ""
        current = float(now if now is not None else time.time())
        state = self._state(group_id)
        if self.cooldown_seconds > 0 and current - state.last_reply_at < self.cooldown_seconds:
            return "cooldown"
        if self.hourly_limit > 0:
            cutoff = current - 3600.0
            recent = [stamp for stamp in state.reply_times if stamp >= cutoff]
            if len(recent) >= self.hourly_limit:
                return "hourly_limit"
        return ""


__all__ = [
    "InterjectionInput",
    "InterjectionLimiter",
    "InterjectionResult",
    "MULTIPLIER_ADDRESSED_RECENT",
    "MULTIPLIER_HEAT",
    "MULTIPLIER_SAME_SPEAKER",
    "weighted_probability",
]
