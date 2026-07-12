from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
import tempfile

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "nonebot_plugin_qwen_persona"
package = types.ModuleType("nonebot_plugin_qwen_persona")
package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules.setdefault("nonebot_plugin_qwen_persona", package)


class FailingHelperTrainerMixin:
    helper_attempts: int

    async def _assist_lora_record(self, *, api_url: str, record: dict[str, object]) -> dict[str, object]:
        self.helper_attempts += 1
        raise RuntimeError("helper unavailable")


class TrainerProgressTests(unittest.TestCase):
    def test_progress_log_records_stage_changes_without_waiting_for_final_log(self) -> None:
        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.trainer import PersonaTrainer

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

        self.assertIn("training started: reason=admin_full_external", log_tail)
        self.assertIn("stage=dataset_exported sample_count=80650", log_tail)
        self.assertIn("stage=assisting_dataset sample_count=80650", log_tail)

    def test_lora_helper_limit_bounds_attempts_not_successes(self) -> None:
        import asyncio
        import json

        from nonebot_plugin_qwen_persona.config import QwenPersonaConfig
        from nonebot_plugin_qwen_persona.trainer import PersonaTrainer

        class FailingTrainer(FailingHelperTrainerMixin, PersonaTrainer):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            raw_dataset_path = Path(tmp) / "raw.jsonl"
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

            self.assertEqual(trainer.helper_attempts, 2)
            self.assertEqual(len(assisted_path.read_text(encoding="utf-8").splitlines()), 5)
            self.assertIn("attempted=2", log)
            self.assertIn("processed=0", log)
            self.assertIn("fallback=2", log)


if __name__ == "__main__":
    unittest.main()
