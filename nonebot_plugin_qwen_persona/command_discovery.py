from __future__ import annotations

import re
from typing import Any, Iterable

PICMENU_NAME_HINTS = ("picmenu", "pic_menu", "pic-menu", "nonebot_plugin_picmenu")
TOKEN_SPLIT_RE = re.compile(r"[\s,，;；|、]+")
PREFIX_TOKEN_RE = re.compile(r"(?<!\S)([/!！#＃.。][^\s,，;；|、]+)")


def discover_picmenu_command_regexes(
    driver_config: Any,
    *,
    config_keys: Iterable[str],
    command_keys: Iterable[str],
    prefixes: Iterable[str],
) -> list[str]:
    commands = discover_picmenu_commands(
        driver_config,
        config_keys=config_keys,
        command_keys=command_keys,
        prefixes=prefixes,
    )
    return [command_to_regex(command) for command in commands]


def discover_picmenu_commands(
    driver_config: Any,
    *,
    config_keys: Iterable[str],
    command_keys: Iterable[str],
    prefixes: Iterable[str],
) -> list[str]:
    explicit_config_keys = {normalize_key(item) for item in config_keys if str(item).strip()}
    command_key_hints = {normalize_key(item) for item in command_keys if str(item).strip()}
    prefix_values = tuple(str(item) for item in prefixes if str(item))
    found: set[str] = set()

    for name, value in iter_config_items(driver_config):
        normalized = normalize_key(name)
        if not is_picmenu_config_name(normalized, explicit_config_keys):
            continue
        collect_commands(
            value,
            found=found,
            command_key_hints=command_key_hints,
            prefixes=prefix_values,
            in_command_context=False,
        )

    return sorted(found, key=lambda item: (len(item), item))


def iter_config_items(config: Any) -> list[tuple[str, Any]]:
    if hasattr(config, "model_dump"):
        data = config.model_dump()
        if isinstance(data, dict):
            return list(data.items())
    if hasattr(config, "dict"):
        data = config.dict()
        if isinstance(data, dict):
            return list(data.items())
    if isinstance(config, dict):
        return list(config.items())

    items: list[tuple[str, Any]] = []
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        items.append((name, value))
    return items


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).lower())


def is_picmenu_config_name(normalized: str, explicit_config_keys: set[str]) -> bool:
    if normalized in explicit_config_keys:
        return True
    return any(normalize_key(hint) in normalized for hint in PICMENU_NAME_HINTS)


def is_command_key(key: Any, command_key_hints: set[str]) -> bool:
    normalized = normalize_key(key)
    return any(hint and hint in normalized for hint in command_key_hints)


def collect_commands(
    value: Any,
    *,
    found: set[str],
    command_key_hints: set[str],
    prefixes: tuple[str, ...],
    in_command_context: bool,
) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            collect_commands(
                item,
                found=found,
                command_key_hints=command_key_hints,
                prefixes=prefixes,
                in_command_context=in_command_context or is_command_key(key, command_key_hints),
            )
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            collect_commands(
                item,
                found=found,
                command_key_hints=command_key_hints,
                prefixes=prefixes,
                in_command_context=in_command_context,
            )
        return
    if isinstance(value, str):
        for command in extract_commands_from_string(value, prefixes=prefixes, trusted=in_command_context):
            found.add(command)


def extract_commands_from_string(text: str, *, prefixes: tuple[str, ...], trusted: bool) -> set[str]:
    stripped = text.strip()
    if not stripped:
        return set()

    found = {token.strip() for token in PREFIX_TOKEN_RE.findall(stripped) if is_reasonable_command(token)}
    if not trusted:
        return found

    parts = TOKEN_SPLIT_RE.split(stripped)
    if len(parts) == 1 and is_reasonable_command(stripped, allow_no_prefix=True):
        found.add(stripped)
    else:
        for part in parts:
            part = part.strip()
            if is_reasonable_command(part, allow_no_prefix=True):
                found.add(part)

    for prefix in prefixes:
        if prefix and stripped.startswith(prefix) and is_reasonable_command(stripped):
            found.add(stripped)
    return found


def is_reasonable_command(text: str, *, allow_no_prefix: bool = False) -> bool:
    command = text.strip()
    if not command or len(command) > 40:
        return False
    if "://" in command or "{" in command or "}" in command:
        return False
    if any(ch in command for ch in "\r\n\t"):
        return False
    if command[0] in "/!！#＃.。":
        return len(command) > 1
    if not allow_no_prefix:
        return False
    return bool(re.fullmatch(r"[\w\-\u4e00-\u9fff]{1,24}", command))


def command_to_regex(command: str) -> str:
    escaped = re.escape(command.strip())
    return rf"^\s*{escaped}(\s|$)"
