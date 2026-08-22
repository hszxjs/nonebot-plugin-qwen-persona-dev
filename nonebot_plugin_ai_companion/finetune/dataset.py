"""微调域：SFT 数据集导出（聊天样本 + 外部语料 + 多模态资产 → JSONL）。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..memory.corpus import external_corpus_rows
from ..memory.db import PersonaDatabase
from ..prompt.media_text import context_text_from_sample
from ..prompt.text import normalize_text
from ..providers.generation_safety import is_low_quality_generated_text
from ..sticker.assets import media_asset_rows

CHAT_LOG_PREFIX_RE = re.compile(r"^\s*\d{5,12}\s*[:：]\s*")


def _is_excluded_dataset_row(
    row: dict[str, Any],
    *,
    excluded_user_ids: set[str],
    excluded_message_ids: set[str],
    assistant_text: str | None = None,
) -> bool:
    user_id = str(row.get("user_id") or "")
    message_id = str(row.get("message_id") or "")
    if user_id and user_id in excluded_user_ids:
        return True
    if message_id and message_id in excluded_message_ids:
        return True
    if assistant_text is not None and CHAT_LOG_PREFIX_RE.match(str(assistant_text or "")):
        return True
    return False


def _strip_chat_log_prefix(text: str) -> str:
    return CHAT_LOG_PREFIX_RE.sub("", str(text or "")).strip()


def _has_chat_log_prefix(text: str) -> bool:
    return bool(CHAT_LOG_PREFIX_RE.match(str(text or "")))


def _sanitize_metadata_chat_log_prefixes(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_metadata_chat_log_prefixes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata_chat_log_prefixes(item) for item in value]
    if isinstance(value, str):
        return _strip_chat_log_prefix(value)
    return value


class SftDatasetExporter:
    def __init__(self, db: PersonaDatabase) -> None:
        self.db = db

    async def export_sft_dataset(
        self,
        *,
        output_path: Path,
        limit: int,
        history_window: int,
        system_prompt: str,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
        external_limit: int = 0,
        include_multimodal_assets: bool = False,
        excluded_user_ids: Iterable[str] = (),
        excluded_message_ids: Iterable[str] = (),
    ) -> int:
        return await asyncio.to_thread(
            self._export_sft_dataset_sync,
            output_path,
            limit,
            history_window,
            system_prompt,
            command_prefixes,
            command_regexes,
            external_limit,
            include_multimodal_assets,
            excluded_user_ids,
            excluded_message_ids,
        )

    def _export_sft_dataset_sync(
        self,
        output_path: Path,
        limit: int,
        history_window: int,
        system_prompt: str,
        command_prefixes: Iterable[str] = (),
        command_regexes: Iterable[str] = (),
        external_limit: int = 0,
        include_multimodal_assets: bool = False,
        excluded_user_ids: Iterable[str] = (),
        excluded_message_ids: Iterable[str] = (),
    ) -> int:
        self.db.ensure_schema()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_limit = max(int(limit or 0), 0)
        unlimited_dataset = dataset_limit <= 0
        excluded_users = {str(item) for item in excluded_user_ids if str(item).strip()}
        excluded_messages = {str(item) for item in excluded_message_ids if str(item).strip()}
        with self.db.connection() as conn:
            if unlimited_dataset:
                rows = conn.execute(
                    """
                    SELECT id, user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                    FROM chat_samples
                    WHERE training_text != ''
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            else:
                fetch_limit = max(dataset_limit + history_window + 32, dataset_limit)
                rows = conn.execute(
                    """
                    SELECT id, user_id, group_id, message_id, plain_text, training_text, sticker_keys, created_at
                    FROM chat_samples
                    WHERE training_text != ''
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (fetch_limit,),
                ).fetchall()

        ordered = [dict(row) for row in reversed(rows)]
        exported = 0
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for idx, row in enumerate(ordered):
                if not unlimited_dataset and exported >= dataset_limit:
                    break
                history = ordered[max(0, idx - history_window) : idx]
                history_lines = []
                for item in history:
                    if _is_excluded_dataset_row(
                        item,
                        excluded_user_ids=excluded_users,
                        excluded_message_ids=excluded_messages,
                    ):
                        continue
                    context_text = context_text_from_sample(item)
                    if _is_excluded_dataset_row(
                        item,
                        excluded_user_ids=excluded_users,
                        excluded_message_ids=excluded_messages,
                        assistant_text=context_text,
                    ):
                        continue
                    context_text = _strip_chat_log_prefix(context_text)
                    if context_text and not self.db.is_noise_text(
                        context_text,
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    ):
                        history_lines.append(f"上一句：{context_text}")
                prompt = "\n".join(history_lines) or "群聊刚开始，请直接接住当前话题。"
                assistant_text = context_text_from_sample(row)
                if _is_excluded_dataset_row(
                    row,
                    excluded_user_ids=excluded_users,
                    excluded_message_ids=excluded_messages,
                    assistant_text=assistant_text,
                ):
                    continue
                assistant_text = _strip_chat_log_prefix(assistant_text)
                if (
                    not assistant_text
                    or self.db.is_noise_text(
                        assistant_text,
                        command_prefixes=command_prefixes,
                        command_regexes=command_regexes,
                    )
                    or is_low_quality_generated_text(assistant_text)
                ):
                    continue
                record = {
                    "images": [],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "下面是群聊上下文。请优先理解当前语境，按自然群聊方式回复：\n" + prompt,
                        },
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "metadata": {
                        "source": "chat_samples",
                        "sample_id": row["id"],
                        "group_id": row.get("group_id"),
                        "user_id": row.get("user_id"),
                        "created_at": row.get("created_at"),
                    },
                }
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                exported += 1
            external_limit = int(external_limit or 0)
            external_enabled = external_limit >= 0
            has_dataset_room = unlimited_dataset or exported < dataset_limit
            if external_enabled and has_dataset_room:
                if external_limit > 0:
                    row_limit = external_limit if unlimited_dataset else min(external_limit, dataset_limit - exported)
                else:
                    row_limit = 0 if unlimited_dataset else max(dataset_limit - exported, 0)
                for row in external_corpus_rows(self.db, limit=row_limit):
                    raw_text = normalize_text(str(row.get("text") or ""))
                    if (
                        not raw_text
                        or _has_chat_log_prefix(raw_text)
                        or self.db.is_noise_text(
                            raw_text,
                            command_prefixes=command_prefixes,
                            command_regexes=command_regexes,
                        )
                        or is_low_quality_generated_text(raw_text)
                    ):
                        continue
                    text = _strip_chat_log_prefix(raw_text)
                    if not text:
                        continue
                    metadata = _sanitize_metadata_chat_log_prefixes(self.db.json_object(row.get("metadata_json")))
                    user_prompt = normalize_text(str(metadata.get("user_prompt") or ""))
                    user_content = (
                        "下面是外部公开社区语境，请按这种口吻自然接话：\n" + user_prompt
                        if user_prompt
                        else "下面是一条公开社区语料，请学习其自然口吻，不要输出来源信息："
                    )
                    record = {
                        "images": [],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": text},
                        ],
                        "metadata": {
                            "source": row.get("source"),
                            "source_id": row.get("source_id"),
                            "external_sample_id": row.get("id"),
                            "created_at": row.get("created_at"),
                            "external": True,
                            "extra": metadata,
                        },
                    }
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    exported += 1
                    if not unlimited_dataset and exported >= dataset_limit:
                        break
            has_dataset_room = unlimited_dataset or exported < dataset_limit
            if include_multimodal_assets and has_dataset_room:
                if unlimited_dataset:
                    media_limit = 0
                else:
                    media_limit = max(dataset_limit - exported, 0)
                for row in media_asset_rows(self.db, limit=media_limit):
                    uri = str(row.get("uri") or "").strip()
                    raw_answer = normalize_text(str(row.get("answer") or ""))
                    if (
                        not uri
                        or not raw_answer
                        or _has_chat_log_prefix(raw_answer)
                        or is_low_quality_generated_text(raw_answer)
                    ):
                        continue
                    answer = _strip_chat_log_prefix(raw_answer)
                    prompt = _strip_chat_log_prefix(normalize_text(str(row.get("prompt") or "")))
                    metadata = _sanitize_metadata_chat_log_prefixes(self.db.json_object(row.get("metadata_json")))
                    record = {
                        "images": [uri],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": uri},
                                    {
                                        "type": "text",
                                        "text": prompt or "请按群聊语境理解这张图片。",
                                    },
                                ],
                            },
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": {
                            "source": row.get("source"),
                            "source_key": row.get("source_key"),
                            "media_asset_id": row.get("id"),
                            "created_at": row.get("created_at"),
                            "multimodal": True,
                            "extra": metadata,
                        },
                    }
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    exported += 1
                    if not unlimited_dataset and exported >= dataset_limit:
                        break
        return exported
