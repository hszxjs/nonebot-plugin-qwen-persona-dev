"""PersonaStore：领域仓库的聚合容器（依赖注入根）。

对外保持旧 storage.py 的完整公共 API（委托各仓库），调用方可逐步迁移到
具名仓库属性（store.samples / store.stickers / store.events ...）。
数据库 schema 与 persona.sqlite3 文件位置不变。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .finetune.dataset import SftDatasetExporter
from .finetune.runs import TrainingRunRepo
from .memory.corpus import ExternalCorpusRepo
from .memory.db import PersonaDatabase
from .memory.patterns import ConversationPatternRepo
from .memory.samples import ChatSampleRepo
from .memory.stats import StatsRepo
from .observability.events import InferenceEventRepo, StickerEventRepo
from .sticker.assets import StickerAssetRepo


class PersonaStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db = PersonaDatabase(data_dir)
        # 具名仓库：新代码请优先使用这些属性
        self.samples = ChatSampleRepo(self.db)
        self.patterns = ConversationPatternRepo(self.db)
        self.corpus = ExternalCorpusRepo(self.db)
        self.stickers = StickerAssetRepo(self.db)
        self.events = InferenceEventRepo(self.db)
        self.sticker_events = StickerEventRepo(self.db)
        self.training_runs = TrainingRunRepo(self.db)
        self.dataset = SftDatasetExporter(self.db)
        self.stats_repo = StatsRepo(self.db)

    @property
    def db_path(self) -> Path:
        return self.db.db_path

    async def init(self) -> None:
        await self.db.init()

    # ---- 可观测性：推理事件 ----
    async def record_inference_event(self, **kwargs: Any) -> dict[str, Any]:
        return await self.events.record_inference_event(**kwargs)

    async def list_inference_events(self, *, limit: int = 300, offset: int = 0) -> dict[str, Any]:
        return await self.events.list_inference_events(limit=limit, offset=offset)

    # ---- 可观测性：表情事件 ----
    async def record_sticker_event(self, **kwargs: Any) -> dict[str, Any]:
        return await self.sticker_events.record_sticker_event(**kwargs)

    async def list_sticker_events(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        return await self.sticker_events.list_sticker_events(limit=limit, offset=offset)

    # ---- 维护：清理 ----
    async def cleanup_empty_chat_artifacts(self) -> dict[str, int]:
        samples_deleted = await self.samples.cleanup_empty_samples()
        patterns_deleted = await self.patterns.cleanup_empty_patterns()
        return {"samples_deleted": samples_deleted, "patterns_deleted": patterns_deleted}

    async def cleanup_training_noise(
        self,
        *,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        samples_deleted = await self.samples.cleanup_noise_samples(
            command_prefixes=command_prefixes,
            command_regexes=command_regexes,
        )
        patterns_deleted = await self.patterns.cleanup_noise_patterns(
            command_prefixes=command_prefixes,
            command_regexes=command_regexes,
        )
        return {"samples_deleted": samples_deleted, "patterns_deleted": patterns_deleted}

    # ---- 记忆：样本 / 语料 / 模式 ----
    async def record_sample(
        self,
        *,
        platform: str | None,
        user_id: str | None,
        group_id: str | None,
        message_id: str | None,
        plain_text: str,
        training_text: str,
        raw_segments: list[dict[str, Any]],
        stickers: list[dict[str, Any]],
        learn_reason: str,
    ) -> None:
        await self.samples.record_sample(
            platform=platform,
            user_id=user_id,
            group_id=group_id,
            message_id=message_id,
            plain_text=plain_text,
            training_text=training_text,
            raw_segments=raw_segments,
            stickers=stickers,
            learn_reason=learn_reason,
        )

    async def import_external_corpus_samples(
        self,
        *,
        source: str,
        records: Iterable[dict[str, Any]],
        min_chars: int,
        max_chars: int,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
    ) -> dict[str, int]:
        return await self.corpus.import_external_corpus_samples(
            source=source,
            records=records,
            min_chars=min_chars,
            max_chars=max_chars,
            command_prefixes=command_prefixes,
            command_regexes=command_regexes,
        )

    async def record_conversation_pattern(self, **kwargs: Any) -> None:
        await self.patterns.record_conversation_pattern(**kwargs)

    async def get_conversation_pattern(self, **kwargs: Any) -> dict[str, Any] | None:
        return await self.patterns.get_conversation_pattern(**kwargs)

    async def recent_samples(self, *, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await self.samples.recent_samples(group_id=group_id, limit=limit)

    async def search_context_samples(self, *, group_id: str | None, query: str, limit: int) -> list[dict[str, Any]]:
        return await self.samples.search_context_samples(group_id=group_id, query=query, limit=limit)

    # ---- 表情域：资产与表情 ----
    async def record_media_asset(self, **kwargs: Any) -> dict[str, int]:
        return await self.stickers.record_media_asset(**kwargs)

    async def record_sticker_asset(self, **kwargs: Any) -> dict[str, int]:
        return await self.stickers.record_sticker_asset(**kwargs)

    async def search_media_assets(self, *, group_id: str | None, query: str, limit: int) -> list[dict[str, Any]]:
        return await self.stickers.search_media_assets(group_id=group_id, query=query, limit=limit)

    async def top_stickers(self, *, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await self.stickers.top_stickers(group_id=group_id, limit=limit)

    async def get_sticker(self, key: str) -> dict[str, Any] | None:
        return await self.stickers.get_sticker(key)

    async def list_stickers(self, **kwargs: Any) -> dict[str, Any]:
        return await self.stickers.list_stickers(**kwargs)

    async def get_sticker_preview_path(self, key: str, *, repo_dir: str | Path) -> Path | None:
        return await self.stickers.get_sticker_preview_path(key, repo_dir=repo_dir)

    async def delete_stickers(self, keys: list[str]) -> dict[str, int]:
        return await self.stickers.delete_stickers(keys)

    async def update_sticker_summary(self, key: str, summary: str) -> dict[str, Any]:
        return await self.stickers.update_sticker_summary(key, summary)

    async def normalize_sticker_summaries_from_protocol(self) -> dict[str, int]:
        return await self.stickers.normalize_sticker_summaries_from_protocol()

    async def deduplicate_stickers_by_canonical_key(self, sticker_segment_types: Iterable[str]) -> dict[str, int]:
        return await self.stickers.deduplicate_stickers_by_canonical_key(sticker_segment_types)

    async def cleanup_legacy_media_artifacts(self) -> dict[str, int]:
        return await self.stickers.cleanup_legacy_media_artifacts()

    async def backfill_stickers_from_raw_segments(self, sticker_segment_types: Iterable[str]) -> dict[str, int]:
        return await self.stickers.backfill_stickers_from_raw_segments(sticker_segment_types)

    # ---- 微调域：数据集与运行 ----
    async def export_sft_dataset(self, **kwargs: Any) -> int:
        return await self.dataset.export_sft_dataset(**kwargs)

    async def begin_training_run(self, **kwargs: Any) -> None:
        await self.training_runs.begin_training_run(**kwargs)

    async def finish_training_run(self, *, run_id: str, status: str, log: str) -> None:
        await self.training_runs.finish_training_run(run_id=run_id, status=status, log=log)

    # ---- 统计 ----
    async def stats(self) -> dict[str, int]:
        return await self.stats_repo.stats()

    async def dashboard_stats(self, *, days: int = 14, group_limit: int = 8) -> dict[str, Any]:
        return await self.stats_repo.dashboard_stats(days=days, group_limit=group_limit)
