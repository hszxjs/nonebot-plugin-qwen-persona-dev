"""触发判定：一条消息里谁被"找"了。

纯函数、零 IO，全部判定依据来自已解析的 envelope 与 bot 身份。
规则（用户已确认）：
- @ 段必算找人；
- 引用消息也算找人（引用谁 = 找谁）；
- bot 自己被 @ 时不找人（那走正常回复路径，一次发言只触发一种行为）；
- 自己找自己、@全体、找 bot 都排除。
"""

from __future__ import annotations

from ..context.envelope import MessageContextEnvelope
from .model import PingCandidate, PingDecision


def detect_ping_candidates(
    *,
    envelope: MessageContextEnvelope,
    sender_user_id: str | None,
    bot_self_id: str | None,
    quote_enabled: bool = True,
) -> PingDecision:
    sender = str(sender_user_id or "").strip()
    self_id = str(bot_self_id or "").strip()

    targets: list[str] = []
    for item in envelope.at_targets:
        qq = str(item.get("qq") or "").strip()
        if qq and qq != "all":
            targets.append(qq)

    quote_target = ""
    if quote_enabled and envelope.reply_sender:
        quote_target = str(envelope.reply_sender.get("user_id") or "").strip()

    # bot 自己被 @（或被引用）：整条消息走正常回复路径，不产生提醒
    if self_id and (self_id in targets or (quote_target and quote_target == self_id)):
        return PingDecision(skip_reason="bot_targeted")

    candidates: list[PingCandidate] = []
    seen: set[str] = set()
    for qq in targets:
        if not qq or qq == sender or qq in seen:
            continue
        seen.add(qq)
        # @ 目标的显示名在收集层从 member_names / envelope 兜底，这里只定目标
        candidates.append(PingCandidate(target_user_id=qq, reason="at", finder_user_id=sender, finder_name=""))

    if quote_target and quote_target != sender and quote_target not in seen:
        seen.add(quote_target)
        quote_name = str((envelope.reply_sender or {}).get("nickname") or "")
        candidates.append(
            PingCandidate(target_user_id=quote_target, reason="quote", finder_user_id=sender, finder_name=quote_name)
        )

    return PingDecision(candidates=tuple(candidates))


__all__ = ["detect_ping_candidates"]
