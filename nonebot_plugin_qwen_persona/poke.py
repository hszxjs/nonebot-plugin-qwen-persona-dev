from __future__ import annotations

from typing import Any


def _str_id(value: Any) -> str:
    return str(value or "").strip()


def _int_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_poke_at_self(event: Any, self_id: str | int | None) -> bool:
    if _str_id(getattr(event, "notice_type", "")) != "notify":
        return False
    if _str_id(getattr(event, "sub_type", "")) != "poke":
        return False
    if not _str_id(getattr(event, "user_id", "")):
        return False
    target_id = _str_id(getattr(event, "target_id", ""))
    wanted = _str_id(self_id or getattr(event, "self_id", ""))
    return bool(target_id and wanted and target_id == wanted)


async def send_counter_poke(bot: Any, event: Any) -> bool:
    user_id = _int_id(getattr(event, "user_id", None))
    if user_id is None:
        return False

    group_id = _int_id(getattr(event, "group_id", None))
    if group_id is not None:
        api_candidates = (
            ("group_poke", {"group_id": group_id, "user_id": user_id}),
            ("send_group_poke", {"group_id": group_id, "user_id": user_id}),
            ("send_poke", {"group_id": group_id, "user_id": user_id}),
        )
    else:
        api_candidates = (
            ("friend_poke", {"user_id": user_id}),
            ("send_friend_poke", {"user_id": user_id}),
            ("send_poke", {"user_id": user_id}),
        )

    for api_name, params in api_candidates:
        try:
            await bot.call_api(api_name, **params)
            return True
        except Exception:
            continue
    return False
