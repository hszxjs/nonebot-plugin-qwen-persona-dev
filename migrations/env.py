"""Alembic 环境：SQLite（persona.sqlite3），路径可经 AI_COMPANION_DATA_DIR 覆盖。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if not config.get_main_option("sqlalchemy.url"):
    data_dir = os.environ.get("AI_COMPANION_DATA_DIR", "data/qwen_persona")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{data_dir}/persona.sqlite3")

# 便于离线生成 SQL 时的导入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

target_metadata = None  # 插件使用原始 SQL DDL（memory/db.py），无 ORM metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
