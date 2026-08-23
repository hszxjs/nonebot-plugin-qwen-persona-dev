"""Baseline：既有 schema 由运行时 ensure_schema（IF NOT EXISTS）创建。

对已有数据库执行 `alembic stamp head` 对齐版本；全新库无需任何迁移
（启动时 ensure_schema 自动建表）。此后新增表/列的变更应新建 revision。

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-23
"""

from __future__ import annotations

from typing import Sequence, Union

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 空基线：所有现存表（chat_samples…message_logs 等 14 张）由
    # nonebot_plugin_ai_companion/memory/db.py 的幂等 DDL 负责。
    pass


def downgrade() -> None:
    pass
