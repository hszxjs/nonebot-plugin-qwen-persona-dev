from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def config_field_names(config: Any) -> list[str]:
    model_fields = getattr(type(config), "model_fields", None)
    if model_fields is not None:
        return list(model_fields)
    legacy_fields = getattr(type(config), "__fields__", None)
    if legacy_fields is not None:
        return list(legacy_fields)
    return [name for name in vars(config) if name.startswith("qwen_persona_")]


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if hasattr(config, "dict"):
        return config.dict()
    return {name: getattr(config, name) for name in config_field_names(config)}


def config_to_python_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    return {name: getattr(config, name) for name in config_field_names(config)}


def admin_config_path(config: Any) -> Path:
    configured = str(getattr(config, "qwen_persona_admin_config_file", "") or "").strip()
    if configured:
        return Path(configured)
    data_dir = Path(str(getattr(config, "qwen_persona_data_dir", "data/qwen_persona")))
    return data_dir / "admin_config.json"


def load_config_overrides(config: Any) -> dict[str, Any]:
    path = admin_config_path(config)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in set(config_field_names(config))}


def save_config_overrides(config: Any, values: dict[str, Any]) -> Path:
    path = admin_config_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed = set(config_field_names(config))
    clean_values = {key: value for key, value in values.items() if key in allowed}
    path.write_text(
        json.dumps(clean_values, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def apply_config_values(config: Any, values: dict[str, Any]) -> list[str]:
    allowed = set(config_field_names(config))
    incoming = {key: value for key, value in values.items() if key in allowed}
    if not incoming:
        return []

    before = config_to_python_dict(config)
    merged = {**before, **incoming}
    validated = config.__class__(**merged)
    validated_data = config_to_python_dict(validated)

    changed: list[str] = []
    for key in config_field_names(config):
        old = before.get(key)
        new = validated_data.get(key)
        if old != new:
            changed.append(key)
        setattr(config, key, new)
    return changed
