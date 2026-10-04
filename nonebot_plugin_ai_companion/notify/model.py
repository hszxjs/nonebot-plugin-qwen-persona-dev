"""找人提醒的数据模型：候选、判定、卡片。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PingCandidate:
    """一次"找某人"的候选：谁被找、怎么找的。"""

    target_user_id: str
    reason: str  # "at" | "quote"
    finder_user_id: str = ""
    finder_name: str = ""  # 解析失败的昵称由收集层用 safe_speaker_name 兜底


@dataclass(frozen=True)
class PingDecision:
    """检测结果：一条消息最多产生一批候选（可空）。"""

    candidates: tuple[PingCandidate, ...] = ()
    skip_reason: str = ""  # 非空表示整条消息不产生提醒（如 bot 自己被 @）


@dataclass
class FinderEntry:
    """卡片里的一条"找你的人"。"""

    user_id: str = ""
    name: str = ""
    text: str = ""
    image_data_urls: list[str] = field(default_factory=list)
    reason: str = "at"


@dataclass
class PingCard:
    """一张待渲染的提醒卡。"""

    group_id: str
    target_user_id: str
    target_name: str
    finders: list[FinderEntry] = field(default_factory=list)

    @property
    def finder_count(self) -> int:
        return len(self.finders)

    @property
    def title(self) -> str:
        name = self.target_name or "群友"
        suffix = "们" if self.finder_count > 1 else ""
        return f"{name}同学{suffix}，有 {self.finder_count} 人找你哦"
