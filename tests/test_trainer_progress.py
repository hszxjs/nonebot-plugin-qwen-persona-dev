from __future__ import annotations

import json
from pathlib import Path


class FailingHelperTrainerMixin:
    helper_attempts: int

    async def _assist_lora_record(self, *, api_url: str, record: dict[str, object]) -> dict[str, object]:
        self.helper_attempts += 1
        raise RuntimeError("helper unavailable")


def test_trainer_passes_dataset_excluded_user_ids_to_export(tmp_path: Path) -> None:
    import asyncio

    from nonebot_plugin_ai_companion.config import QwenPersonaConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class Store:
        def __init__(self) -> None:
            self.export_kwargs: dict[str, object] = {}

        async def export_sft_dataset(self, **kwargs: object) -> int:
            self.export_kwargs = kwargs
            return 0

        async def begin_training_run(self, **_kwargs: object) -> None:
            return None

        async def finish_training_run(self, **_kwargs: object) -> None:
            return None

    store = Store()
    trainer = PersonaTrainer(
        QwenPersonaConfig(
            qwen_persona_data_dir=str(tmp_path),
            qwen_persona_dataset_excluded_user_ids={"bot-self"},
        ),
        store=store,  # type: ignore[arg-type]
    )

    asyncio.run(trainer.run_training(reason="unit", include_external_corpus=False))

    assert store.export_kwargs["excluded_user_ids"] == {"bot-self"}


def test_progress_log_records_stage_changes_without_waiting_for_final_log() -> None:
    from nonebot_plugin_ai_companion.config import QwenPersonaConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    trainer = PersonaTrainer(QwenPersonaConfig(), store=object())  # type: ignore[arg-type]

    trainer._start_progress(reason="admin_full_external")
    trainer._set_progress(
        stage="dataset_exported",
        sample_count=80650,
        dataset_path="data/qwen_persona/datasets/run.jsonl",
        output_dir="data/qwen_persona/lora_runs/run",
    )
    trainer._set_progress(stage="assisting_dataset", sample_count=80650)

    log_tail = trainer.progress_snapshot()["log_tail"]

    assert "training started: reason=admin_full_external" in log_tail
    assert "stage=dataset_exported sample_count=80650" in log_tail
    assert "stage=assisting_dataset sample_count=80650" in log_tail


def test_lora_helper_limit_bounds_attempts_not_successes(tmp_path: Path) -> None:
    import asyncio

    from nonebot_plugin_ai_companion.config import QwenPersonaConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class FailingTrainer(FailingHelperTrainerMixin, PersonaTrainer):
        pass

    raw_dataset_path = tmp_path / "raw.jsonl"
    records = [
        {"messages": [{"role": "user", "content": f"u-{index}"}]}
        for index in range(5)
    ]
    raw_dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    trainer = FailingTrainer(
        QwenPersonaConfig(
            qwen_persona_lora_helper_api_url="http://helper.invalid/v1",
            qwen_persona_lora_helper_sample_limit=2,
        ),
        store=object(),  # type: ignore[arg-type]
    )
    trainer.helper_attempts = 0

    assisted_path, log = asyncio.run(
        trainer._maybe_build_assisted_dataset(
            raw_dataset_path=raw_dataset_path,
            run_id="limit-test",
        )
    )

    assert trainer.helper_attempts == 2
    assert len(assisted_path.read_text(encoding="utf-8").splitlines()) == 5
    assert "attempted=2" in log
    assert "processed=0" in log
    assert "fallback=2" in log


def test_lora_helper_base_v1_url_posts_to_chat_completions_endpoint(tmp_path: Path) -> None:
    import asyncio

    from nonebot_plugin_ai_companion.config import QwenPersonaConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class CapturingTrainer(PersonaTrainer):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.helper_urls: list[str] = []

        async def _assist_lora_record(self, *, api_url: str, record: dict[str, object]) -> dict[str, object]:
            self.helper_urls.append(api_url)
            return record

    raw_dataset_path = tmp_path / "raw.jsonl"
    record = {"messages": [{"role": "user", "content": "u"}]}
    raw_dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    trainer = CapturingTrainer(
        QwenPersonaConfig(
            qwen_persona_lora_helper_api_url="https://dm-fox.rjj.cc/codex/v1",
            qwen_persona_lora_helper_sample_limit=1,
        ),
        store=object(),  # type: ignore[arg-type]
    )

    asyncio.run(
        trainer._maybe_build_assisted_dataset(
            raw_dataset_path=raw_dataset_path,
            run_id="url-test",
        )
    )

    assert (
        trainer.helper_urls
        == [
            "https://dm-fox.rjj.cc/codex/v1/chat/completions",
        ]
    )


def test_lora_helper_fallback_log_includes_http_error_body(tmp_path: Path) -> None:
    import asyncio

    import httpx

    from nonebot_plugin_ai_companion.config import QwenPersonaConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class HttpFailingTrainer(PersonaTrainer):
        async def _assist_lora_record(self, *, api_url: str, record: dict[str, object]) -> dict[str, object]:
            request = httpx.Request("POST", api_url)
            response = httpx.Response(
                404,
                request=request,
                json={"error": {"message": "model test is not available"}},
            )
            raise httpx.HTTPStatusError("not found", request=request, response=response)

    raw_dataset_path = tmp_path / "raw.jsonl"
    record = {"messages": [{"role": "user", "content": "u"}]}
    raw_dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    trainer = HttpFailingTrainer(
        QwenPersonaConfig(
            qwen_persona_lora_helper_api_url="https://helper.test/v1",
            qwen_persona_lora_helper_sample_limit=1,
        ),
        store=object(),  # type: ignore[arg-type]
    )

    asyncio.run(
        trainer._maybe_build_assisted_dataset(
            raw_dataset_path=raw_dataset_path,
            run_id="http-error-test",
        )
    )

    assert "model test is not available" in trainer.progress_snapshot()["log_tail"]
