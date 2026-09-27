"""批 3：搭话加权与限流、滚动摘要、助手/成员模式、在线失败降级。"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


# ---- 搭话加权 ----


def test_weighted_probability_applies_multipliers() -> None:
    from nonebot_plugin_ai_companion.decision.interjection import InterjectionInput, weighted_probability

    base = weighted_probability(InterjectionInput(base_probability=0.2))
    assert abs(base.probability - 0.2) < 1e-9 and base.describe() == "base"

    hot = weighted_probability(InterjectionInput(base_probability=0.2, heat_messages=6))
    assert abs(hot.probability - 0.3) < 1e-9 and "heat" in hot.describe()

    both = weighted_probability(
        InterjectionInput(base_probability=0.2, heat_messages=6, same_speaker_as_last_reply=True)
    )
    assert abs(both.probability - 0.15) < 1e-9  # 0.2 × 1.5 × 0.5

    clamped = weighted_probability(
        InterjectionInput(base_probability=0.9, addressed_recently=True, heat_messages=99)
    )
    assert clamped.probability == 1.0

    # 概率为 0 时不做加权（配置关掉主动搭话就别加）
    off = weighted_probability(InterjectionInput(base_probability=0.0, heat_messages=99, addressed_recently=True))
    assert off.probability == 0.0


def test_interjection_limiter_cooldown_and_hourly_cap() -> None:
    from nonebot_plugin_ai_companion.decision.interjection import InterjectionLimiter

    limiter = InterjectionLimiter(cooldown_seconds=45, hourly_limit=3, heat_window_seconds=120)
    assert limiter.block_reason(group_id="100", now=1000.0) == ""  # 没回过，放行

    limiter.observe_reply(group_id="100", user_id="42", now=1000.0)
    assert limiter.block_reason(group_id="100", now=1020.0) == "cooldown"
    assert limiter.block_reason(group_id="100", now=1050.0) == ""  # 冷却过了
    assert limiter.same_speaker_as_last_reply(group_id="100", user_id="42") is True
    assert limiter.same_speaker_as_last_reply(group_id="100", user_id="77") is False

    # 每小时上限：第 3 条后开始拦
    for stamp in (1100.0, 1200.0, 1300.0):
        limiter.observe_reply(group_id="100", user_id="42", now=stamp)
    assert limiter.block_reason(group_id="100", now=1400.0) == "hourly_limit"
    assert limiter.block_reason(group_id="100", now=1000.0 + 3800.0) == ""  # 早先的回复滑出一小时窗口


def test_interjection_limiter_heat_counts_recent_messages() -> None:
    from nonebot_plugin_ai_companion.decision.interjection import InterjectionLimiter

    limiter = InterjectionLimiter(heat_window_seconds=120)
    for index in range(6):
        limiter.observe_message(group_id="100", user_id=str(index), now=1000.0 + index)
    assert limiter.heat_messages(group_id="100", now=1010.0) == 6
    assert limiter.heat_messages(group_id="100", now=1200.0) == 0  # 窗口外
    assert limiter.heat_messages(group_id="", now=1010.0) == 0


# ---- 滚动摘要 ----


def test_clean_summary_text_strips_wrappers() -> None:
    from nonebot_plugin_ai_companion.conversation.summary import clean_summary_text

    assert clean_summary_text("摘要：群里在打战争雷霆，约了今晚开黑") == "群里在打战争雷霆，约了今晚开黑"
    assert clean_summary_text('“  两行\n文字  ”') == "两行 文字"
    assert clean_summary_text("") == ""


async def test_group_summary_updates_incrementally(persona_store) -> None:
    from nonebot_plugin_ai_companion.conversation.model import SPEAKER_MEMBER
    from nonebot_plugin_ai_companion.conversation.summary import run_summary_round

    for index, text in enumerate(["雷雷新版本出了", "今晚八点开黑吗", "行，我先吃饭"]):
        await persona_store.conversation.record_turn(
            session_key="group:100",
            speaker_kind=SPEAKER_MEMBER,
            text=text,
            group_id="100",
            user_id="42",
            speaker_name="阿强",
            created_at=100.0 + index,
        )

    calls: list[str] = []

    async def fake_chat(prompt: str) -> str:
        calls.append(prompt)
        return "群里在聊战争雷霆新版本，约了今晚八点开黑"

    results = await run_summary_round(
        conversation=persona_store.conversation,
        names_repo=persona_store.member_names,
        repo=persona_store.group_summaries,
        chat=fake_chat,
        min_new_turns=2,
        interval_seconds=0.0,
        now=1000.0,
    )
    assert [item["group_id"] for item in results] == ["100"]
    stored = await persona_store.group_summaries.get(group_id="100")
    assert stored is not None and stored.summary.startswith("群里在聊战争雷霆")
    assert stored.turns_covered == 3
    assert "今晚八点开黑吗" in calls[0]

    # 没有新对话 → 不再调用模型
    again = await run_summary_round(
        conversation=persona_store.conversation,
        names_repo=persona_store.member_names,
        repo=persona_store.group_summaries,
        chat=fake_chat,
        min_new_turns=2,
        interval_seconds=0.0,
        now=2000.0,
    )
    assert again == [] and len(calls) == 1
    # 旧摘要会作为增量上下文喂回去
    await persona_store.conversation.record_turn(
        session_key="group:100",
        speaker_kind=SPEAKER_MEMBER,
        text="打赢了",
        group_id="100",
        user_id="42",
        created_at=2001.0,
    )
    await run_summary_round(
        conversation=persona_store.conversation,
        names_repo=persona_store.member_names,
        repo=persona_store.group_summaries,
        chat=fake_chat,
        min_new_turns=1,
        interval_seconds=0.0,
        now=3000.0,
    )
    assert "群里在聊战争雷霆" in calls[-1]


# ---- 助手 / 成员模式 ----


def _fake_chat_service(*, self_id: str = "999"):
    from nonebot_plugin_ai_companion.chat_service import ChatService

    return SimpleNamespace(env=SimpleNamespace(cached_self_id=lambda: self_id)), ChatService


def test_reply_mode_switches_on_private_addressed_and_reply() -> None:
    from nonebot_plugin_ai_companion.context.envelope import MessageContextEnvelope

    fake_self, ChatService = _fake_chat_service()

    def envelope(*, forced: bool = False, reply_user: str = "") -> MessageContextEnvelope:
        return MessageContextEnvelope(
            current_text="在吗",
            quoted_texts=[],
            forwarded_texts=[],
            media_parts=[],
            sticker_keys=[],
            forced_by_at=forced,
            context_sources=[],
            reply_sender={"user_id": reply_user, "nickname": ""} if reply_user else None,
        )

    assert ChatService._reply_mode(fake_self, group_id="100", envelope=envelope()) == "member"
    assert ChatService._reply_mode(fake_self, group_id="100", envelope=envelope(forced=True)) == "assistant"
    assert ChatService._reply_mode(fake_self, group_id=None, envelope=envelope()) == "assistant"
    # 有人在回复 bot 自己 → 助手口气
    assert ChatService._reply_mode(fake_self, group_id="100", envelope=envelope(reply_user="999")) == "assistant"
    # 回复的是别人 → 还是成员口气
    assert ChatService._reply_mode(fake_self, group_id="100", envelope=envelope(reply_user="42")) == "member"


# ---- 在线失败降级 ----


async def test_fallback_to_local_skips_unloaded_runtime() -> None:
    from nonebot_plugin_ai_companion.chat_service import ChatService
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.prompt.planner import ReplyIntent, ReplyPlan

    plan = ReplyPlan(intent=ReplyIntent.PERSONA_TEXT, messages=[], media_parts=[], allow_sticker=False)

    class Runtime:
        def __init__(self, state: str) -> None:
            self.state = state
            self.calls = 0

        def status_snapshot(self) -> dict[str, str]:
            return {"state": self.state}

        async def chat(self, messages, *, media_parts=None) -> str:
            self.calls += 1
            return "本地兜底回复"

    not_loaded = Runtime("not_loaded")
    fake_self = SimpleNamespace(
        config=AiCompanionConfig(ai_companion_fallback_to_local=True),
        mlx_runtime=not_loaded,
        _online_backend_active=lambda: True,
    )
    assert await ChatService._fallback_to_local(fake_self, plan, error=RuntimeError("429")) is None
    assert not_loaded.calls == 0  # 不主动拉起未加载的模型

    loaded = Runtime("loaded")
    fake_self.mlx_runtime = loaded
    assert await ChatService._fallback_to_local(fake_self, plan, error=RuntimeError("429")) == "本地兜底回复"
    assert loaded.calls == 1

    # 配置关闭降级 → 不调用本地
    fake_self.config = AiCompanionConfig(ai_companion_fallback_to_local=False)
    assert await ChatService._fallback_to_local(fake_self, plan, error=RuntimeError("429")) is None
    assert loaded.calls == 1
