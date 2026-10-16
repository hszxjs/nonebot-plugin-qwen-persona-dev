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

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
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
        AiCompanionConfig(
            ai_companion_data_dir=str(tmp_path),
            ai_companion_dataset_excluded_user_ids={"bot-self"},
        ),
        store=store,  # type: ignore[arg-type]
    )

    asyncio.run(trainer.run_training(reason="unit", include_external_corpus=False))

    assert store.export_kwargs["excluded_user_ids"] == {"bot-self"}


def test_progress_log_records_stage_changes_without_waiting_for_final_log() -> None:
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    trainer = PersonaTrainer(AiCompanionConfig(), store=object())  # type: ignore[arg-type]

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

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class FailingTrainer(FailingHelperTrainerMixin, PersonaTrainer):
        pass

    raw_dataset_path = tmp_path / "raw.jsonl"
    records = [{"messages": [{"role": "user", "content": f"u-{index}"}]} for index in range(5)]
    raw_dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    trainer = FailingTrainer(
        AiCompanionConfig(
            ai_companion_lora_helper_api_url="http://helper.invalid/v1",
            ai_companion_lora_helper_sample_limit=2,
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

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
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
        AiCompanionConfig(
            ai_companion_lora_helper_api_url="https://dm-fox.rjj.cc/codex/v1",
            ai_companion_lora_helper_sample_limit=1,
        ),
        store=object(),  # type: ignore[arg-type]
    )

    asyncio.run(
        trainer._maybe_build_assisted_dataset(
            raw_dataset_path=raw_dataset_path,
            run_id="url-test",
        )
    )

    assert trainer.helper_urls == [
        "https://dm-fox.rjj.cc/codex/v1/chat/completions",
    ]


def test_lora_helper_fallback_log_includes_http_error_body(tmp_path: Path) -> None:
    import asyncio

    import httpx

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
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
        AiCompanionConfig(
            ai_companion_lora_helper_api_url="https://helper.test/v1",
            ai_companion_lora_helper_sample_limit=1,
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


def test_lora_helper_payload_disables_thinking_and_caps_tokens(tmp_path: Path) -> None:
    """清洗请求默认禁 thinking + 按样本长度设输出上限；extra_body 可覆盖；400 去参后不再携带。"""
    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class Store:
        pass

    record = {"messages": [{"role": "assistant", "content": "你好" * 200}]}
    trainer = PersonaTrainer(
        AiCompanionConfig(
            ai_companion_data_dir=str(tmp_path),
            ai_companion_lora_helper_system_prompt="sp",
        ),
        Store(),
    )
    payload = trainer._build_lora_helper_payload(record)
    assert payload["thinking"] == {"type": "disabled"}
    assert int(payload["max_tokens"]) >= 1024
    assert payload["response_format"] == {"type": "json_object"}

    # extra_body 是逃生门：可覆盖任意默认字段
    override = PersonaTrainer(
        AiCompanionConfig(
            ai_companion_data_dir=str(tmp_path),
            ai_companion_lora_helper_extra_body={"max_tokens": 77, "thinking": {"type": "enabled"}},
        ),
        Store(),
    )
    overridden = override._build_lora_helper_payload(record)
    assert overridden["max_tokens"] == 77
    assert overridden["thinking"] == {"type": "enabled"}

    # 与参数无关的 400 不去参（绝不能把禁思考悄悄删掉，否则后续样本退回烧钱模式）
    assert trainer._drop_lora_helper_unsupported(payload, body_text="context length exceeded") == []
    assert "thinking" in payload and "max_tokens" in payload

    # 端点点名某参数才去参，并记住（后续样本不再携带；没被点名的保留）
    named = '{"error":{"message":"Unrecognized request argument supplied: thinking"}}'
    assert trainer._drop_lora_helper_unsupported(payload, body_text=named) == ["thinking"]
    assert "thinking" not in payload and "max_tokens" in payload
    rebuilt = trainer._build_lora_helper_payload(record)
    assert "thinking" not in rebuilt and "max_tokens" in rebuilt

    # 去参记忆覆盖 extra_body 提供的同名字段（否则端点拒绝时每条样本都要 400 白试一次）
    override._lora_helper_dropped_params.add("thinking")
    filtered = override._build_lora_helper_payload(record)
    assert "thinking" not in filtered and filtered["max_tokens"] == 77


async def test_assist_lora_record_drops_named_param_only(tmp_path: Path, monkeypatch) -> None:
    """假 transport：400 点名 thinking → 去参重试成功；后续样本不再带 thinking、max_tokens 保留。"""
    import httpx

    from nonebot_plugin_ai_companion.config import AiCompanionConfig
    from nonebot_plugin_ai_companion.finetune import trainer as trainer_module
    from nonebot_plugin_ai_companion.finetune.trainer import PersonaTrainer

    class Store:
        pass

    calls: list[dict] = []
    scripted: list[tuple[str, str]] = [
        ("400", '{"error":{"message":"Unrecognized request argument supplied: thinking"}}'),
        ("200", ""),
    ]

    class FakeResponse:
        def __init__(self, status_code: int, text: str = "", payload: dict | None = None) -> None:
            self.status_code = status_code
            self.text = text
            self._payload = payload or {}

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "boom",
                    request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(self.status_code),
                )

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> bool:
            return False

        async def post(self, url, headers=None, json=None):
            calls.append(dict(json or {}))
            kind, text = scripted.pop(0)
            if kind == "400":
                return FakeResponse(400, text=text)
            return FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"messages":[{"role":"assistant","content":"ok"}]}'}}]},
            )

    monkeypatch.setattr(trainer_module.httpx, "AsyncClient", FakeClient)
    trainer = PersonaTrainer(AiCompanionConfig(ai_companion_data_dir=str(tmp_path)), Store())
    record = {"messages": [{"role": "user", "content": "hi"}]}

    out = await trainer._assist_lora_record(api_url="http://x/v1/chat/completions", record=record)
    assert out["messages"][0]["content"] == "ok"
    assert len(calls) == 2  # 400 → 去参 → 重发
    assert calls[0]["thinking"] == {"type": "disabled"} and "thinking" not in calls[1]
    assert trainer._lora_helper_dropped_params == {"thinking"}

    # 下一个样本：不再携带 thinking（只发 1 次），max_tokens 仍在
    calls.clear()
    scripted.append(("200", ""))
    await trainer._assist_lora_record(api_url="http://x/v1/chat/completions", record=record)
    assert len(calls) == 1
    assert "thinking" not in calls[0] and "max_tokens" in calls[0]
