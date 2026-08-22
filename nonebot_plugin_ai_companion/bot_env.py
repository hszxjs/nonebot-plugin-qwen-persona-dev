"""共享环境助手：事件 ID 提取、黑白名单、命令识别、@ 判定等。

chat 服务与 learning 服务共同使用的"环境策略"集中在此，
依赖 driver.config（NoneBot 全局配置）与插件配置。
"""

from __future__ import annotations

import re
from typing import Any

from nonebot.adapters import Bot, Event

from .command_discovery import discover_picmenu_command_regexes
from .config import AiCompanionConfig
from .prompt.text import is_command_like


class BotEnvironment:
    def __init__(self, driver_config: Any, config: AiCompanionConfig) -> None:
        self.driver_config = driver_config
        self.config = config

    # ---- 事件标识 ----
    def dump_event_ids(self, event: Event) -> dict[str, str | None]:
        def as_str(value: Any) -> str | None:
            return None if value is None else str(value)

        try:
            user_id = event.get_user_id()
        except Exception:
            # 无 user_id 的事件（如 notice）回退到属性读取，读取失败属预期路径。
            user_id = getattr(event, "user_id", None)

        return {
            "platform": getattr(event, "platform", None) or event.__class__.__module__,
            "user_id": as_str(user_id),
            "group_id": as_str(
                getattr(event, "group_id", None)
                or getattr(event, "guild_id", None)
                or getattr(event, "channel_id", None)
            ),
            "message_id": as_str(getattr(event, "message_id", None)),
        }

    @staticmethod
    def bot_self_id(bot: Bot) -> str | None:
        value = str(getattr(bot, "self_id", "") or "")
        return value or None

    def is_self_event(self, bot: Bot, ids: dict[str, str | None]) -> bool:
        self_id = self.bot_self_id(bot)
        return bool(self_id and ids.get("user_id") == self_id)

    # ---- 群与命令 ----
    def group_enabled(self, group_id: str | None) -> bool:
        enabled = {str(item) for item in self.config.ai_companion_enabled_groups}
        return not enabled or (group_id is not None and group_id in enabled)

    def command_start_values(self) -> list[str]:
        values = list(self.config.ai_companion_command_prefixes)
        values.extend(
            str(item) for item in getattr(self.driver_config, "command_start", set()) or set()
        )
        return values

    def command_regex_values(self) -> list[str]:
        regexes = list(self.config.ai_companion_command_regexes)
        if self.config.ai_companion_auto_read_picmenu_config:
            regexes.extend(
                discover_picmenu_command_regexes(
                    self.driver_config,
                    config_keys=self.config.ai_companion_picmenu_config_keys,
                    command_keys=self.config.ai_companion_picmenu_command_keys,
                    prefixes=self.command_start_values(),
                )
            )
        return list(dict.fromkeys(regexes))

    def is_command_message(self, text: str) -> bool:
        return is_command_like(
            text,
            prefixes=self.command_start_values(),
            regexes=self.command_regex_values(),
        )

    # ---- 黑名单 ----
    def is_blacklisted(self, user_id: str | None, group_id: str | None, text: str) -> bool:
        if user_id and user_id in {
            str(item) for item in self.config.ai_companion_blacklist_user_ids
        }:
            return True
        if group_id and group_id in {
            str(item) for item in self.config.ai_companion_blacklist_group_ids
        }:
            return True
        for word in self.config.ai_companion_blacklist_words:
            word = str(word).strip()
            if word and word in text:
                return True
        for pattern in self.config.ai_companion_blacklist_regexes:
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                continue
        return False

    def remove_blacklisted_reply_lines(self, text: str) -> str:
        kept: list[str] = []
        for line in (text or "").splitlines():
            if not self.is_blacklisted(None, None, line):
                kept.append(line)
        return "\n".join(kept).strip()

    # ---- @ 命中 ----
    def is_to_me(self, event: Event, plain_text: str) -> bool:
        value = getattr(event, "to_me", False)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = False
        if bool(value):
            return True

        nicknames = [
            str(item) for item in getattr(self.driver_config, "nickname", set()) or set()
        ]
        nicknames.extend(self.config.ai_companion_trigger_keywords)
        stripped = plain_text.lstrip()
        return bool(stripped and any(stripped.startswith(name) for name in nicknames if name))
