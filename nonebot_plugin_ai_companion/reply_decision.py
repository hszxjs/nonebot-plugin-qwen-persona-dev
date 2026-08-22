from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyDecision:
    allowed: bool
    reason: str


def _clamp_probability(value: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def should_attempt_reply(
    *,
    enable_reply: bool,
    group_enabled: bool,
    self_event: bool,
    blacklisted: bool,
    command_like: bool,
    force_reply: bool,
    has_text: bool,
    reply_probability: float,
    follow_reply_probability: float,
    random_value: float,
) -> ReplyDecision:
    if not enable_reply:
        return ReplyDecision(False, "reply_disabled")
    if not group_enabled:
        return ReplyDecision(False, "group_disabled")
    if self_event:
        return ReplyDecision(False, "self_event")
    if blacklisted:
        return ReplyDecision(False, "blacklisted")
    if force_reply:
        return ReplyDecision(True, "force_reply")
    if command_like:
        return ReplyDecision(False, "command_like")
    if not has_text:
        return ReplyDecision(False, "empty")

    probability = _clamp_probability(reply_probability)
    reason = "random_reply"
    if probability <= 0:
        probability = _clamp_probability(follow_reply_probability)
        reason = "follow_reply"
    if probability > 0 and float(random_value) < probability:
        return ReplyDecision(True, reason)
    return ReplyDecision(False, "probability_miss")
