from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import QwenPersonaConfig, effective_chat_model, normalized_lora_modules
from ..storage import PersonaStore

logger = logging.getLogger(__name__)

# 匹配 GLM 等模型常用 ```json ... ``` 代码块围栏，用于容错解析辅助清洗结果。
_JSON_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```",
    re.DOTALL,
)


@dataclass(frozen=True)
class TrainingResult:
    status: str
    sample_count: int
    dataset_path: str
    output_dir: str
    log: str = ""


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def parse_time_slot(slot: str) -> tuple[int, int]:
    hour_text, minute_text = slot.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("time must be HH:MM")
    return hour, minute


class PersonaTrainer:
    def __init__(self, config: QwenPersonaConfig, store: PersonaStore) -> None:
        self.config = config
        self.store = store
        self._lock = asyncio.Lock()
        self._progress: dict[str, Any] = {
            "stage": "idle",
            "status": "",
            "reason": "",
            "sample_count": 0,
            "dataset_path": "",
            "output_dir": "",
            "log_tail": "",
            "started_at": None,
            "finished_at": None,
        }

    async def run_training(
        self,
        *,
        reason: str,
        include_external_corpus: bool | None = None,
        full_external_corpus: bool = False,
    ) -> TrainingResult:
        async with self._lock:
            self._start_progress(reason=reason)
            try:
                return await self._run_training_locked(
                    reason=reason,
                    include_external_corpus=include_external_corpus,
                    full_external_corpus=full_external_corpus,
                )
            except Exception as exc:
                self._finish_progress(
                    status="failed",
                    log=f"{type(exc).__name__}: {exc}",
                )
                raise

    def progress_snapshot(self) -> dict[str, Any]:
        return dict(self._progress)

    def _start_progress(self, *, reason: str) -> None:
        now = time.time()
        self._progress = {
            "stage": "exporting_dataset",
            "status": "running",
            "reason": reason,
            "sample_count": 0,
            "dataset_path": "",
            "output_dir": "",
            "log_tail": f"training started: reason={reason}",
            "started_at": now,
            "finished_at": None,
        }

    def _set_progress(
        self,
        *,
        stage: str,
        status: str = "running",
        sample_count: int | None = None,
        dataset_path: str | None = None,
        output_dir: str | None = None,
        log: str = "",
    ) -> None:
        self._progress["stage"] = stage
        self._progress["status"] = status
        if sample_count is not None:
            self._progress["sample_count"] = sample_count
        if dataset_path is not None:
            self._progress["dataset_path"] = dataset_path
        if output_dir is not None:
            self._progress["output_dir"] = output_dir
        self._append_progress_log(
            self._progress_stage_log(stage=stage, status=status)
        )
        if log:
            self._append_progress_log(log)

    def _finish_progress(
        self,
        *,
        status: str,
        sample_count: int | None = None,
        dataset_path: str | None = None,
        output_dir: str | None = None,
        log: str = "",
    ) -> None:
        self._set_progress(
            stage="finished" if status != "failed" else "failed",
            status=status,
            sample_count=sample_count,
            dataset_path=dataset_path,
            output_dir=output_dir,
            log=log,
        )
        self._progress["finished_at"] = time.time()

    def _append_progress_log(self, log: str) -> None:
        current = str(self._progress.get("log_tail") or "")
        joined = (current + "\n" + log).strip() if current else log.strip()
        self._progress["log_tail"] = joined[-4000:]

    def _progress_stage_log(self, *, stage: str, status: str) -> str:
        parts = [f"stage={stage}"]
        sample_count = self._progress.get("sample_count")
        if sample_count:
            parts.append(f"sample_count={sample_count}")
        parts.append(f"status={status}")
        dataset_path = str(self._progress.get("dataset_path") or "")
        if dataset_path:
            parts.append(f"dataset={dataset_path}")
        output_dir = str(self._progress.get("output_dir") or "")
        if output_dir:
            parts.append(f"output={output_dir}")
        return " ".join(parts)

    async def _run_training_locked(
        self,
        *,
        reason: str,
        include_external_corpus: bool | None = None,
        full_external_corpus: bool = False,
    ) -> TrainingResult:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        data_dir = Path(self.config.qwen_persona_data_dir)
        raw_dataset_path = data_dir / "datasets" / f"qwen_persona_{run_id}.jsonl"
        output_dir = data_dir / "lora_runs" / run_id
        self._set_progress(
            stage="exporting_dataset",
            dataset_path=str(raw_dataset_path),
            output_dir=str(output_dir),
        )

        external_enabled = (
            bool(self.config.qwen_persona_external_corpus_enabled)
            if include_external_corpus is None
            else bool(include_external_corpus)
        )
        if not external_enabled:
            external_limit = -1
        elif full_external_corpus:
            external_limit = 0
        else:
            external_limit = self.config.qwen_persona_external_corpus_limit
        sample_count = await self.store.export_sft_dataset(
            output_path=raw_dataset_path,
            limit=0 if full_external_corpus else self.config.qwen_persona_dataset_limit,
            history_window=self.config.qwen_persona_dataset_history_window,
            system_prompt=self.config.qwen_persona_system_prompt,
            command_prefixes=self.config.qwen_persona_command_prefixes,
            command_regexes=self.config.qwen_persona_command_regexes,
            external_limit=external_limit,
            include_multimodal_assets=bool(
                self.config.qwen_persona_multimodal_training_enabled
            ),
            excluded_user_ids=self.config.qwen_persona_dataset_excluded_user_ids,
        )
        self._set_progress(stage="dataset_exported", sample_count=sample_count)

        if sample_count < self.config.qwen_persona_min_train_samples:
            log = (
                f"sample_count={sample_count}, "
                f"min_train_samples={self.config.qwen_persona_min_train_samples}"
            )
            await self.store.begin_training_run(
                run_id=run_id,
                reason=reason,
                dataset_path=str(raw_dataset_path),
                output_dir=str(output_dir),
                sample_count=sample_count,
            )
            await self.store.finish_training_run(
                run_id=run_id,
                status="skipped_not_enough_samples",
                log=log,
            )
            self._finish_progress(
                status="skipped_not_enough_samples",
                sample_count=sample_count,
                dataset_path=str(raw_dataset_path),
                output_dir=str(output_dir),
                log=log,
            )
            return TrainingResult(
                status="skipped_not_enough_samples",
                sample_count=sample_count,
                dataset_path=str(raw_dataset_path),
                output_dir=str(output_dir),
            )

        command_template = self.config.qwen_persona_lora_train_command.strip()

        if self.config.qwen_persona_lora_helper_api_url.strip():
            self._set_progress(stage="assisting_dataset", sample_count=sample_count)
        dataset_path, helper_log = await self._maybe_build_assisted_dataset(
            raw_dataset_path=raw_dataset_path,
            run_id=run_id,
        )
        if helper_log:
            self._set_progress(
                stage="dataset_ready",
                dataset_path=str(dataset_path),
                log=helper_log,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        await self.store.begin_training_run(
            run_id=run_id,
            reason=reason,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            sample_count=sample_count,
        )

        mapping = SafeFormatDict(
            run_id=run_id,
            dataset=str(dataset_path),
            raw_dataset=str(raw_dataset_path),
            output_dir=str(output_dir),
            model=effective_chat_model(self.config),
            data_dir=str(data_dir),
        )
        self._set_progress(stage="running_train", sample_count=sample_count)
        if command_template:
            train_command = command_template.format_map(mapping)
            train_status, train_log = await self._run_shell(train_command, mapping)
        else:
            train_status, train_log = await self._run_mlx_lora(mapping, sample_count)

        final_status = train_status
        final_log = (helper_log + "\n\n" if helper_log else "") + train_log
        if train_status == "success" and self.config.qwen_persona_after_train_command.strip():
            self._set_progress(stage="running_after_train", log=train_log)
            after_command = self.config.qwen_persona_after_train_command.format_map(mapping)
            after_status, after_log = await self._run_shell(after_command, mapping)
            final_status = "success" if after_status == "success" else f"after_train_{after_status}"
            final_log = final_log + "\n\n[after_train]\n" + after_log

        await self.store.finish_training_run(run_id=run_id, status=final_status, log=final_log)
        self._finish_progress(
            status=final_status,
            sample_count=sample_count,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            log=final_log,
        )
        return TrainingResult(
            status=final_status,
            sample_count=sample_count,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            log=final_log,
        )

    async def _maybe_build_assisted_dataset(
        self,
        *,
        raw_dataset_path: Path,
        run_id: str,
    ) -> tuple[Path, str]:
        api_url = self._normalize_lora_helper_api_url(
            self.config.qwen_persona_lora_helper_api_url
        )
        if not api_url:
            return raw_dataset_path, ""

        assisted_path = raw_dataset_path.with_name(f"qwen_persona_{run_id}.assisted.jsonl")
        limit = self.config.qwen_persona_lora_helper_sample_limit
        attempted = 0
        processed = 0
        fallback = 0
        total = 0
        skipped = 0
        last_logged_attempt = 0
        assisted_path.parent.mkdir(parents=True, exist_ok=True)

        with raw_dataset_path.open("r", encoding="utf-8") as source, assisted_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    # 外部语料可能含非 UTF-8 字节（如贴吧 GBK 中文）导致单行损坏。
                    # 跳过坏行并记日志，不让整个训练崩溃。
                    skipped += 1
                    if skipped <= 3:
                        logger.warning(
                            "lora_helper: skip malformed dataset line %d: %r",
                            total, str(exc),
                        )
                    continue
                if limit > 0 and attempted >= limit:
                    target.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    if total % 10000 == 0:
                        self._append_progress_log(
                            self._lora_helper_progress_log(
                                total=total,
                                attempted=attempted,
                                processed=processed,
                                fallback=fallback,
                            )
                        )
                    continue
                attempted += 1
                try:
                    assisted = await self._assist_lora_record(api_url=api_url, record=record)
                    target.write(json.dumps(assisted, ensure_ascii=False, default=str) + "\n")
                    processed += 1
                except Exception as exc:
                    fallback += 1
                    target.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    if fallback <= 3:
                        message = (
                            "qwen persona lora helper fallback: "
                            f"{self._format_lora_helper_error(exc)}"
                        )
                        logger.warning(message)
                        self._append_progress_log(message)
                if attempted - last_logged_attempt >= 10:
                    last_logged_attempt = attempted
                    self._append_progress_log(
                        self._lora_helper_progress_log(
                            total=total,
                            attempted=attempted,
                            processed=processed,
                            fallback=fallback,
                        )
                    )

        log = (
            "[lora_helper]\n"
            f"model={self.config.qwen_persona_lora_helper_model}\n"
            f"raw_dataset={raw_dataset_path}\n"
            f"assisted_dataset={assisted_path}\n"
            f"total={total} attempted={attempted} processed={processed} fallback={fallback} skipped={skipped}"
        )
        return assisted_path, log

    @staticmethod
    def _normalize_lora_helper_api_url(api_url: str) -> str:
        value = str(api_url or "").strip()
        if not value:
            return ""
        parts = urlsplit(value)
        path = parts.path.rstrip("/")
        if path.endswith("/chat/completions"):
            normalized_path = path
        elif path.endswith("/v1"):
            normalized_path = f"{path}/chat/completions"
        else:
            return value
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                normalized_path,
                parts.query,
                parts.fragment,
            )
        )

    @staticmethod
    def _format_lora_helper_error(exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text.replace("\n", " ").strip()
            if len(body) > 500:
                body = body[:497] + "..."
            return (
                f"HTTPStatusError(status_code={exc.response.status_code}, "
                f"url={exc.request.url}, body={body!r})"
            )
        return repr(exc)

    @staticmethod
    def _lora_helper_progress_log(
        *,
        total: int,
        attempted: int,
        processed: int,
        fallback: int,
    ) -> str:
        return (
            "[lora_helper_progress] "
            f"read={total} attempted={attempted} processed={processed} fallback={fallback}"
        )

    async def _assist_lora_record(self, *, api_url: str, record: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.qwen_persona_lora_helper_api_key.strip():
            headers["Authorization"] = (
                f"Bearer {self.config.qwen_persona_lora_helper_api_key.strip()}"
            )

        extra_body = dict(self.config.qwen_persona_lora_helper_extra_body)
        # response_format 对 GLM 系列（含 glm-5.2）的 OpenAI 兼容端点是可选的，
        # 部分中转/官方网关对 json_object 支持不稳定。默认开启，用户可在
        # extra_body 里用 _disable_response_format=true 关闭，或直接覆盖 response_format。
        payload: dict[str, Any] = {
            "model": self.config.qwen_persona_lora_helper_model,
            "messages": [
                {
                    "role": "system",
                    "content": self.config.qwen_persona_lora_helper_system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(record, ensure_ascii=False, default=str),
                },
            ],
            "temperature": 0.2,
        }
        disable_rf = bool(extra_body.pop("_disable_response_format", False))
        if not disable_rf and "response_format" not in extra_body:
            payload["response_format"] = {"type": "json_object"}
        # 用户自定义字段最后合并，可覆盖任何上述默认值。
        payload.update(extra_body)

        timeout = float(self.config.qwen_persona_lora_helper_timeout)
        # 仅对限流和网关错误退避重试；4xx（鉴权/参数错误等）直接抛出走 fallback。
        retryable_status = {429, 500, 502, 503, 504}
        max_retries = 3
        async with httpx.AsyncClient(timeout=timeout) as client:
            data: dict[str, Any] | None = None
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(api_url, headers=headers, json=payload)
                    if response.status_code in retryable_status:
                        raise httpx.HTTPStatusError(
                            f"helper status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPStatusError as exc:
                    # 4xx 等不可重试错误：立刻向上抛，交由 _maybe_build_assisted_dataset 走 fallback。
                    if exc.response is None or exc.response.status_code not in retryable_status:
                        raise
                    last_exc = exc
                    if attempt >= max_retries:
                        raise
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                except httpx.TransportError as exc:
                    # 网络层错误（连接/超时/读中断）通常可重试。
                    last_exc = exc
                    if attempt >= max_retries:
                        raise
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            else:
                # 理论上不会到达，保险起见。
                raise last_exc  # type: ignore[misc]
            if data is None:  # type: ignore[unreachable]
                raise last_exc  # type: ignore[misc]

        content = self._extract_helper_content(data)
        assisted = self._parse_helper_json(content)
        if not isinstance(assisted, dict) or not isinstance(assisted.get("messages"), list):
            raise ValueError("assistant response must be a JSON object with messages")
        if "metadata" not in assisted and "metadata" in record:
            assisted["metadata"] = record["metadata"]
        return assisted

    @staticmethod
    def _extract_helper_content(data: dict[str, Any]) -> str:
        """兼容 OpenAI 标准与部分 GLM 中转的字段差异，提取回复文本。"""
        try:
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("empty choices in helper response")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # 部分接口把 content 拆成多段 dict（{"type":"text","text":...}）。
                parts: list[str] = []
                for seg in content:
                    if isinstance(seg, dict):
                        parts.append(str(seg.get("text") or seg.get("content") or ""))
                    else:
                        parts.append(str(seg))
                content = "".join(parts)
            if not content:
                raise ValueError("empty content in helper response")
            return str(content)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unrecognized helper response shape: {exc!r}") from exc

    @staticmethod
    def _parse_helper_json(content: str) -> dict[str, Any]:
        """从 GLM 回复中解析 JSON，容忍前后说明文字与 ```json 代码块。"""
        text = content.strip()
        # 1) 直接解析（最理想：纯 JSON）。
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2) 去掉 ```json ... ``` / ``` ... ``` 代码块围栏。
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            inner = fence_match.group("body").strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass
        # 3) 抓取第一个 { 到最后一个 } 之间的子串。
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"failed to parse JSON from helper response: {text[:200]!r}")

    async def _run_mlx_lora(
        self,
        mapping: SafeFormatDict,
        sample_count: int,
    ) -> tuple[str, str]:
        if importlib.util.find_spec("mlx_vlm") is None:
            log = (
                "mlx-vlm is not installed; install project dependencies before MLX LoRA training."
            )
            return "skipped_missing_mlx_vlm", log

        output_dir = Path(str(mapping["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "adapters.safetensors"
        command = [
            sys.executable,
            "-m",
            "mlx_vlm.lora",
            "--model-path",
            str(mapping["model"]),
            "--dataset",
            str(mapping["dataset"]),
            "--split",
            "train",
            "--output-path",
            str(output_path),
            "--learning-rate",
            str(float(self.config.qwen_persona_mlx_lora_learning_rate)),
            "--batch-size",
            str(max(1, int(self.config.qwen_persona_mlx_lora_batch_size))),
            "--steps-per-report",
            str(max(1, int(self.config.qwen_persona_mlx_lora_steps_per_report))),
            "--steps-per-save",
            str(max(1, int(self.config.qwen_persona_mlx_lora_save_every))),
            "--max-seq-length",
            str(max(128, int(self.config.qwen_persona_mlx_lora_max_seq_length))),
            "--gradient-accumulation-steps",
            str(max(1, int(self.config.qwen_persona_mlx_lora_grad_accum_steps))),
            "--lora-alpha",
            str(float(self.config.qwen_persona_mlx_lora_scale)),
            "--lora-rank",
            str(max(1, int(self.config.qwen_persona_mlx_lora_rank))),
            "--lora-dropout",
            str(float(self.config.qwen_persona_mlx_lora_dropout)),
        ]
        if int(self.config.qwen_persona_mlx_lora_epochs) > 0:
            command.extend(["--epochs", str(int(self.config.qwen_persona_mlx_lora_epochs))])
        else:
            command.extend(["--iters", str(max(1, int(self.config.qwen_persona_mlx_lora_iters or 1)))])
        if self.config.qwen_persona_mlx_lora_grad_checkpoint:
            command.append("--grad-checkpoint")
        if self.config.qwen_persona_mlx_lora_mask_prompt:
            command.append("--train-on-completions")
        adapter_path = self.config.qwen_persona_mlx_adapter_path.strip()
        if adapter_path:
            command.extend(["--adapter-path", adapter_path])

        modules = normalized_lora_modules(self.config)
        warnings: list[str] = []
        if "vision_tower" in modules:
            command.append("--train-vision")
            warnings.append(
                "vision_tower training is enabled; this can exceed 16GB unified memory."
            )
        if "projection" in modules:
            warnings.append(
                "projection module selection is recorded for review; mlx-vlm CLI trains language LoRA and optionally unfreezes the vision stack."
            )

        env = os.environ.copy()
        env.update(
            {
                "QWEN_PERSONA_RUN_ID": str(mapping["run_id"]),
                "QWEN_PERSONA_DATASET": str(mapping["dataset"]),
                "QWEN_PERSONA_RAW_DATASET": str(mapping["raw_dataset"]),
                "QWEN_PERSONA_OUTPUT_DIR": str(mapping["output_dir"]),
                "QWEN_PERSONA_MODEL": str(mapping["model"]),
                "QWEN_PERSONA_SAMPLE_COUNT": str(sample_count),
            }
        )
        cwd = self.config.qwen_persona_train_cwd.strip() or None
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            timeout = self.config.qwen_persona_train_timeout_seconds or None
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            stdout, _ = await process.communicate()
            log = stdout.decode(errors="replace")
            return "timeout", self._mlx_train_log(command, warnings, log)
        log = stdout.decode(errors="replace")
        return (
            "success" if process.returncode == 0 else f"failed_{process.returncode}",
            self._mlx_train_log(command, warnings, log),
        )

    @staticmethod
    def _mlx_train_log(command: list[str], warnings: list[str], output: str) -> str:
        prefix = "[mlx_vlm_lora]\ncommand=" + json.dumps(command, ensure_ascii=False)
        if warnings:
            prefix += "\nwarnings=" + json.dumps(warnings, ensure_ascii=False)
        return prefix + "\n\n" + output

    async def _run_shell(
        self,
        command: str,
        mapping: SafeFormatDict,
    ) -> tuple[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "QWEN_PERSONA_RUN_ID": str(mapping["run_id"]),
                "QWEN_PERSONA_DATASET": str(mapping["dataset"]),
                "QWEN_PERSONA_RAW_DATASET": str(mapping["raw_dataset"]),
                "QWEN_PERSONA_OUTPUT_DIR": str(mapping["output_dir"]),
                "QWEN_PERSONA_MODEL": str(mapping["model"]),
            }
        )
        cwd = self.config.qwen_persona_train_cwd.strip() or None
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            timeout = self.config.qwen_persona_train_timeout_seconds or None
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            stdout, _ = await process.communicate()
            return "timeout", stdout.decode(errors="replace")

        log = stdout.decode(errors="replace")
        return ("success" if process.returncode == 0 else f"failed_{process.returncode}", log)
