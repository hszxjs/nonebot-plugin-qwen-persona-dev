"""可观测性域：caption 调试日志（data/qwen_persona/caption_debug.log）。"""

from __future__ import annotations

import time
from pathlib import Path


def append_caption_debug_log(message: str, *, data_dir: str | Path = "data/qwen_persona") -> None:
    """追加一行 caption 调试日志；写失败静默（诊断用途，不值得打断主流程）。"""
    try:
        path = Path(data_dir) / "caption_debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")
    except OSError:
        pass
