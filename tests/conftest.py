"""Shared pytest fixtures and the package-stub bootstrap for the test suite.

The real ``nonebot_plugin_ai_companion`` package ``__init__`` requires a nonebot
runtime, so tests stub the package in ``sys.modules`` and import submodules
directly. This bootstrap runs once when pytest loads ``conftest.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_ai_companion"

package = types.ModuleType("nonebot_plugin_ai_companion")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_ai_companion", package)


@pytest.fixture
async def persona_store(tmp_path: Path):
    """An initialized PersonaStore rooted in a per-test temporary directory."""
    from nonebot_plugin_ai_companion.store import PersonaStore

    store = PersonaStore(tmp_path)
    await store.init()
    return store
