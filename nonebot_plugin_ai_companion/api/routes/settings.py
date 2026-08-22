"""配置路由：读取 / 按字段更新（含敏感字段二次确认）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ...runtime_config import (
    admin_config_path,
    config_field_names,
    config_to_dict,
    config_to_python_dict,
)

RELOAD_SENSITIVE_FIELDS = {
    "ai_companion_mlx_model",
    "ai_companion_mlx_adapter_path",
    "ai_companion_mlx_auto_load",
    "ai_companion_mlx_trust_remote_code",
    "ai_companion_model_dir",
    "ai_companion_admin_enabled",
    "ai_companion_admin_host",
    "ai_companion_admin_port",
    "ai_companion_data_dir",
    "ai_companion_train_times",
    "ai_companion_enable_training",
    "ai_companion_learning_priority",
    "ai_companion_reply_priority",
}

DANGER_FIELDS = {
    "ai_companion_lora_train_command",
    "ai_companion_after_train_command",
    "ai_companion_lora_helper_api_key",
    "ai_companion_admin_token",
}


def _validate_setting_value(config: Any, field_name: str, value: Any) -> Any:
    allowed = set(config_field_names(config))
    if field_name not in allowed:
        raise HTTPException(status_code=400, detail=f"unknown setting: {field_name}")
    if not hasattr(config, "__class__"):
        return value
    before = config_to_python_dict(config)
    try:
        validated = config.__class__(**{**before, field_name: value})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid setting value: {field_name}") from exc
    return config_to_dict(validated).get(field_name)


def build_router(
    *,
    config: Any,
    auth,
    save_callback: Callable[[dict[str, Any]], dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/api/settings", dependencies=[Depends(auth)])

    @router.get("")
    async def settings() -> dict[str, Any]:
        return {
            "values": config_to_dict(config),
            "override_file": str(admin_config_path(config)),
        }

    @router.patch("/fields/{field_name}", response_model=None)
    async def update_setting(field_name: str, request: Request) -> dict[str, Any] | JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict) or "value" not in payload:
            raise HTTPException(status_code=400, detail="value is required")
        value = _validate_setting_value(config, field_name, payload["value"])
        requires_confirmation = field_name in RELOAD_SENSITIVE_FIELDS or field_name in DANGER_FIELDS
        if requires_confirmation and not bool(payload.get("confirmed")):
            return JSONResponse(
                {
                    "error": "confirmation required",
                    "confirmation_required": True,
                    "field": field_name,
                    "reload_required": field_name in RELOAD_SENSITIVE_FIELDS,
                    "danger": field_name in DANGER_FIELDS,
                },
                status_code=409,
            )
        result = save_callback({field_name: value})
        return {
            "field": field_name,
            "value": value,
            "confirmation_required": False,
            **result,
        }

    return router
