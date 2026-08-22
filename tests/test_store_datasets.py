from __future__ import annotations

import json
from pathlib import Path


async def test_export_sft_dataset_filters_self_and_generated_chat_log_outputs(tmp_path: Path, persona_store) -> None:
    await persona_store.record_sample(
        platform="qq",
        user_id="10001",
        group_id="g1",
        message_id="m1",
        plain_text="前面聊拉面",
        training_text="前面聊拉面",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    await persona_store.record_sample(
        platform="qq",
        user_id="bot",
        group_id="g1",
        message_id="m2",
        plain_text="我是 bot 输出",
        training_text="我是 bot 输出",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    await persona_store.record_sample(
        platform="qq",
        user_id="10002",
        group_id="g1",
        message_id="m3",
        plain_text="3305167706: 先煮面再调汤吧",
        training_text="3305167706: 先煮面再调汤吧",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    await persona_store.record_sample(
        platform="qq",
        user_id="10003",
        group_id="g1",
        message_id="m4",
        plain_text="兰州拉面一般先熬汤再煮面",
        training_text="兰州拉面一般先熬汤再煮面",
        raw_segments=[],
        stickers=[],
        learn_reason="accepted",
    )
    output_path = tmp_path / "dataset.jsonl"

    count = await persona_store.export_sft_dataset(
        output_path=output_path,
        limit=0,
        history_window=4,
        system_prompt="system",
        excluded_user_ids={"bot"},
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assistant_texts = [row["messages"][-1]["content"] for row in rows]
    user_prompts = [row["messages"][1]["content"] for row in rows]
    assert count == 2
    assert assistant_texts == ["前面聊拉面", "兰州拉面一般先熬汤再煮面"]
    assert all("3305167706:" not in prompt for prompt in user_prompts)
    assert all("先煮面再调汤" not in prompt for prompt in user_prompts)
    assert all("10001:" not in prompt for prompt in user_prompts)


async def test_export_sft_dataset_skips_low_quality_media_asset_answers(tmp_path: Path, persona_store) -> None:
    await persona_store.record_media_asset(
        source="sticker",
        source_key="bad",
        uri="file:///tmp/bad.webp",
        prompt="请描述这个表情。",
        answer="mp窗 Turing - pleas nodeType <think> Gam Gam Gam",
    )
    await persona_store.record_media_asset(
        source="sticker",
        source_key="good",
        uri="file:///tmp/good.webp",
        prompt="请描述这个表情。",
        answer="一个动漫角色惊讶地张大嘴巴。",
    )
    output_path = tmp_path / "dataset.jsonl"

    count = await persona_store.export_sft_dataset(
        output_path=output_path,
        limit=0,
        history_window=0,
        system_prompt="system",
        include_multimodal_assets=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert count == 1
    assert len(rows) == 1
    assert rows[0]["metadata"]["source_key"] == "good"
    assert rows[0]["messages"][-1]["content"] == "一个动漫角色惊讶地张大嘴巴。"


async def test_export_sft_dataset_filters_external_and_media_chat_log_prefixes(tmp_path: Path, persona_store) -> None:
    await persona_store.import_external_corpus_samples(
        source="external",
        records=[
            {
                "source_id": "bad-external",
                "text": "3305167706: 这是模型生成污染",
            },
            {
                "source_id": "good-external",
                "text": "自然外部语料",
                "metadata": {"user_prompt": "3305167706: 外部上下文"},
            },
        ],
        min_chars=1,
        max_chars=200,
    )
    await persona_store.record_media_asset(
        source="sticker",
        source_key="bad-media",
        uri="file:///tmp/bad.webp",
        prompt="请描述图片",
        answer="3305167706: 这是污染图片描述",
    )
    await persona_store.record_media_asset(
        source="sticker",
        source_key="good-media",
        uri="file:///tmp/good.webp",
        prompt="3305167706: 请描述图片",
        answer="一个动漫角色惊讶地张大嘴巴。",
    )
    output_path = tmp_path / "dataset.jsonl"

    count = await persona_store.export_sft_dataset(
        output_path=output_path,
        limit=0,
        history_window=0,
        system_prompt="system",
        external_limit=0,
        include_multimodal_assets=True,
    )

    content = output_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]

    assert count == 2
    assert {row["metadata"]["source"] for row in rows} == {"external", "sticker"}
    assert "3305167706:" not in content
    assert "模型生成污染" not in content
    assert "污染图片描述" not in content
