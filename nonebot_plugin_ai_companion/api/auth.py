"""admin API 鉴权：token 查询参数或 X-Admin-Token 头（空 token 配置 = 开放）。"""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Query


def make_auth_dependency(config: Any):
    def authorize(
        token: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> None:
        expected = str(getattr(config, "ai_companion_admin_token", "") or "")
        if not expected:
            return
        if expected not in {token or "", x_admin_token or ""}:
            raise HTTPException(status_code=401, detail="unauthorized")

    return authorize
