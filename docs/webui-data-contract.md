# WebUI Data Contract

更新时间：2026-07-12

本文只记录当前后台 WebUI 依赖的数据结构、字段与类型，作为后续前后端拆分时的数据契约基线。

## 通用对象

### ApiError

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `error` | `string` | 后端返回的错误文本。 |

### NoticeState

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `"info" \| "success" \| "error" \| string` | 前端提示状态。 |
| `text` | `string` | 前端提示内容。 |

## 配置数据

### ConfigPayload

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `values` | `Record<string, ConfigValue>` | 配置字段当前值。 |
| `override_file` | `string` | 当前覆盖配置文件路径。 |
| `changed` | `string[]` | 保存后变更的字段名；仅保存响应中存在。 |
| `restart_required` | `string[]` | 保存后需要重启才完全生效的字段名；仅保存响应中存在。 |

后端配置接口不传页面分区、label、description、widget 或选项。配置页和规则页的呈现语义由前端 `settings-schema.ts` 按字段名维护。

### ConfigValue

当前配置值会被序列化为 JSON 兼容类型：

- `boolean`
- `number`
- `string`
- `string[]`
- `Record<string, unknown>`

`set[str]` 字段经 JSON 序列化后表现为 `string[]`。

### 配置字段清单

#### 后台网页

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_admin_enabled` | `bool` |
| `qwen_persona_admin_host` | `str` |
| `qwen_persona_admin_port` | `int` |
| `qwen_persona_admin_token` | `str` |
| `qwen_persona_admin_config_file` | `str` |

#### 模型部署

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_mlx_model` | `str` |
| `qwen_persona_mlx_adapter_path` | `str` |
| `qwen_persona_mlx_auto_load` | `bool` |
| `qwen_persona_mlx_trust_remote_code` | `bool` |
| `qwen_persona_model_dir` | `str` |

#### 模型调用

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_temperature` | `float` |
| `qwen_persona_top_p` | `float` |
| `qwen_persona_max_tokens` | `int` |

#### 学习与回复

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_enabled_groups` | `set[str]` |
| `qwen_persona_enable_learning` | `bool` |
| `qwen_persona_enable_reply` | `bool` |
| `qwen_persona_learning_priority` | `int` |
| `qwen_persona_reply_priority` | `int` |
| `qwen_persona_reply_probability` | `float` |
| `qwen_persona_follow_reply_probability` | `float` |
| `qwen_persona_pattern_reply_threshold` | `int` |
| `qwen_persona_context_scope` | `"group" \| "global" \| "mixed"` |
| `qwen_persona_context_messages` | `int` |
| `qwen_persona_min_sample_chars` | `int` |
| `qwen_persona_max_sample_chars` | `int` |
| `qwen_persona_system_prompt` | `str` |

#### 表情包

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_text_sticker_reply_probability` | `float` |
| `qwen_persona_sticker_segment_types` | `set[str]` |
| `qwen_persona_sticker_prompt_limit` | `int` |
| `qwen_persona_sticker_repo_dir` | `str` |
| `qwen_persona_sticker_repo_url` | `str` |
| `qwen_persona_enable_sending_stickers` | `bool` |
| `qwen_persona_max_reply_stickers` | `int` |
| `qwen_persona_media_sticker_fallback_text` | `str` |

#### LoRA 训练

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_enable_training` | `bool` |
| `qwen_persona_train_times` | `list[str]` |
| `qwen_persona_lora_train_command` | `str` |
| `qwen_persona_after_train_command` | `str` |
| `qwen_persona_train_cwd` | `str` |
| `qwen_persona_train_timeout_seconds` | `int` |
| `qwen_persona_min_train_samples` | `int` |
| `qwen_persona_multimodal_training_enabled` | `bool` |
| `qwen_persona_mlx_lora_train_modules` | `list[str]` |
| `qwen_persona_mlx_lora_rank` | `int` |
| `qwen_persona_mlx_lora_scale` | `float` |
| `qwen_persona_mlx_lora_dropout` | `float` |
| `qwen_persona_mlx_lora_learning_rate` | `float` |
| `qwen_persona_mlx_lora_batch_size` | `int` |
| `qwen_persona_mlx_lora_grad_accum_steps` | `int` |
| `qwen_persona_mlx_lora_epochs` | `int` |
| `qwen_persona_mlx_lora_iters` | `int` |
| `qwen_persona_mlx_lora_max_seq_length` | `int` |
| `qwen_persona_mlx_lora_grad_checkpoint` | `bool` |
| `qwen_persona_mlx_lora_mask_prompt` | `bool` |
| `qwen_persona_mlx_lora_save_every` | `int` |
| `qwen_persona_mlx_lora_steps_per_report` | `int` |
| `qwen_persona_lora_helper_api_url` | `str` |
| `qwen_persona_lora_helper_api_key` | `str` |
| `qwen_persona_lora_helper_model` | `str` |
| `qwen_persona_lora_helper_timeout` | `float` |
| `qwen_persona_lora_helper_sample_limit` | `int` |
| `qwen_persona_lora_helper_extra_body` | `dict[str, Any]` |
| `qwen_persona_lora_helper_system_prompt` | `str` |

#### 黑名单

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_blacklist_words` | `list[str]` |
| `qwen_persona_blacklist_regexes` | `list[str]` |
| `qwen_persona_blacklist_user_ids` | `set[str]` |
| `qwen_persona_blacklist_group_ids` | `set[str]` |

#### 其他

| 字段 | 类型 |
| --- | --- |
| `qwen_persona_request_timeout` | `float` |
| `qwen_persona_data_dir` | `str` |
| `qwen_persona_trigger_keywords` | `list[str]` |
| `qwen_persona_enable_repeat` | `bool` |
| `qwen_persona_repeat_min_count` | `int` |
| `qwen_persona_repeat_probability` | `float` |
| `qwen_persona_repeat_cooldown_seconds` | `int` |
| `qwen_persona_command_prefixes` | `list[str]` |
| `qwen_persona_command_regexes` | `list[str]` |
| `qwen_persona_auto_read_picmenu_config` | `bool` |
| `qwen_persona_picmenu_config_keys` | `list[str]` |
| `qwen_persona_picmenu_command_keys` | `list[str]` |
| `qwen_persona_media_image_fallback_text` | `str` |
| `qwen_persona_dataset_limit` | `int` |
| `qwen_persona_dataset_history_window` | `int` |
| `qwen_persona_external_corpus_enabled` | `bool` |
| `qwen_persona_external_corpus_limit` | `int` |

### 配置实现备注

- 后端只负责配置值读取、保存、字段白名单校验、类型校验和危险操作确认。
- 前端设置页不显示规则字段；规则字段只在独立规则过滤页面呈现。
- 前端字段没有说明时不显示说明行，不使用通用 fallback 文案。

## 运行统计数据

### RuntimeStats

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `totals` | `RuntimeTotals` | 累计统计。 |
| `training` | `TrainingSummary` | 最近训练摘要。 |
| `charts` | `RuntimeCharts` | 趋势和分布数据。 |
| `inference` | `InferenceSummary` | 推理性能统计。 |
| `training_operation` | `TrainingOperation` | 当前训练任务状态。 |
| `training_config` | `TrainingConfigSummary` | 训练相关配置摘要。 |

### RuntimeTotals

| 字段 | 类型 |
| --- | --- |
| `learned_messages` | `number` |
| `trainable_samples` | `number` |
| `pending_training_samples` | `number` |
| `stickers` | `number` |
| `external_corpus_samples` | `number` |
| `media_assets` | `number` |
| `training_runs` | `number` |
| `groups` | `number` |
| `users` | `number` |
| `samples_with_stickers` | `number` |
| `today_samples` | `number` |
| `last_24h_samples` | `number` |
| `first_sample_at` | `number \| null` |
| `last_sample_at` | `number \| null` |
| `avg_training_chars` | `number \| null` |

### TrainingSummary

| 字段 | 类型 |
| --- | --- |
| `last_success` | `TrainingRun \| null` |
| `last_run` | `TrainingRun \| null` |
| `pending_since` | `number \| null` |

### TrainingRun

| 字段 | 类型 |
| --- | --- |
| `id` | `string` |
| `status` | `string` |
| `reason` | `string` |
| `started_at` | `number` |
| `finished_at` | `number \| null` |
| `sample_count` | `number` |
| `dataset_path` | `string` |
| `output_dir` | `string` |

### RuntimeCharts

| 字段 | 类型 |
| --- | --- |
| `learning_trend` | `CountPoint[]` |
| `group_contribution` | `CountPoint[]` |
| `performance_trend` | `InferenceMetric[]` |

### CountPoint

| 字段 | 类型 |
| --- | --- |
| `label` | `string` |
| `count` | `number` |

### InferenceSummary

| 字段 | 类型 |
| --- | --- |
| `last_tps` | `InferenceMetric \| null` |
| `client` | `InferenceMetricSnapshot` |
| `mlx_runtime` | `InferenceMetricSnapshot` |

### InferenceMetricSnapshot

| 字段 | 类型 |
| --- | --- |
| `last` | `InferenceMetric \| null` |
| `history` | `InferenceMetric[]` |

### InferenceMetric

| 字段 | 类型 |
| --- | --- |
| `time` | `string` |
| `model` | `string` |
| `latency_seconds` | `number` |
| `completion_tokens` | `number \| null` |
| `response_chars` | `number` |
| `tps` | `number` |
| `source` | `string` |

### TrainingOperation

| 字段 | 类型 |
| --- | --- |
| `running` | `boolean` |
| `action` | `string` |
| `reason` | `string` |
| `last_result` | `TrainingResult \| null` |
| `available` | `boolean` |
| `progress` | `TrainingProgress \| null` |

### TrainingProgress

| 字段 | 类型 |
| --- | --- |
| `stage` | `string` |
| `status` | `string` |
| `reason` | `string` |
| `sample_count` | `number` |
| `dataset_path` | `string` |
| `output_dir` | `string` |
| `log_tail` | `string` |
| `started_at` | `number \| null` |
| `finished_at` | `number \| null` |

### TrainingResult

| 字段 | 类型 |
| --- | --- |
| `status` | `string` |
| `sample_count` | `number` |
| `dataset_path` | `string` |
| `output_dir` | `string` |
| `log` | `string` |
| `action` | `string?` |
| `reason` | `string?` |

### TrainingConfigSummary

| 字段 | 类型 |
| --- | --- |
| `target_configured` | `boolean` |
| `helper_configured` | `boolean` |
| `min_train_samples` | `number` |
| `dataset_limit` | `number` |
| `external_corpus_enabled` | `boolean` |
| `external_corpus_limit` | `number` |
| `multimodal_training_enabled` | `boolean` |
| `mlx_lora_modules` | `string[]` |

## MLX 运行时数据

### MlxStatus

| 字段 | 类型 |
| --- | --- |
| `runtime` | `"mlx_vlm"` |
| `status` | `string` |
| `state` | `"loaded" \| "loading" \| "not_loaded" \| "missing_dependency" \| "failed" \| "mlx_runtime_unavailable" \| string` |
| `healthy` | `boolean` |
| `model` | `string` |
| `loaded_model` | `string` |
| `adapter_path` | `string` |
| `loaded_adapter_path` | `string` |
| `last_error` | `string` |
| `control` | `MlxControl` |
| `operation` | `string?` |
| `metrics` | `InferenceMetricSnapshot` |
| `model_presets` | `ModelPreset[]` |

### MlxControl

| 字段 | 类型 |
| --- | --- |
| `can_load` | `boolean` |
| `can_unload` | `boolean` |
| `can_reload` | `boolean` |
| `reason` | `string?` |

### ModelPreset

| 字段 | 类型 |
| --- | --- |
| `value` | `string` |
| `label` | `string` |
| `description` | `string` |

### MlxLogEntry

| 字段 | 类型 |
| --- | --- |
| `time` | `string` |
| `stream` | `string` |
| `message` | `string` |

### LogsPayload

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `logs` | `MlxLogEntry[]` | 日志行。 |
| `path` | `string?` | 文件日志来源路径；仅插件运行日志返回。 |
| `exists` | `boolean?` | 文件日志来源是否存在；仅插件运行日志返回。 |

### MlxActionResult

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `action` | `"load" \| "reload" \| "unload"` | 动作名。 |
| `message` | `string` | 可直接展示的操作文案，例如 `load model 'xxx'`。 |
| `runtime` | `MlxStatus` | 动作后的运行时状态摘要。 |
| `confirmation_required` | `boolean?` | reload/unload 未确认时返回。 |
| `error` | `string?` | 错误文本。 |

## 训练触发数据

### TrainingActionRequest

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `reason` | `string?` | 触发原因。 |
| `include_external_corpus` | `boolean?` | 是否合并外部语料。 |
| `full_external_corpus` | `boolean?` | 是否忽略语料上限并使用全量外部语料。 |

### TrainingActionResult

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | `string` | 任务提交或训练结果状态。 |
| `action` | `string?` | 通常为 `training`。 |
| `reason` | `string?` | 触发原因。 |
| `sample_count` | `number?` | 同步结果中的样本数。 |
| `dataset_path` | `string?` | 同步结果中的数据集路径。 |
| `output_dir` | `string?` | 同步结果中的输出目录。 |
| `log` | `string?` | 同步结果中的日志尾部。 |

## 表情仓库数据

### StickerGalleryPayload

| 字段 | 类型 |
| --- | --- |
| `items` | `StickerItem[]` |
| `total` | `number` |
| `limit` | `number` |
| `offset` | `number` |
| `query` | `string` |
| `type` | `string` |
| `type_counts` | `Record<string, number>` |

### StickerItem

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `key` | `string` | 表情唯一键。 |
| `type` | `string` | 消息段类型。 |
| `summary` | `string` | 表情提示词，可编辑。 |
| `uri` | `string` | 表情资源 URI。 |
| `data` | `Record<string, unknown>` | 协议原始数据或扩展数据。 |
| `group_id` | `string \| null` | 来源群。 |
| `user_id` | `string \| null` | 来源用户。 |
| `first_seen` | `number \| null` | 首次记录时间戳。 |
| `last_seen` | `number \| null` | 最近记录时间戳。 |
| `count` | `number` | 出现次数。 |

### StickerItem.data 约定字段

`data` 是开放对象。当前展示逻辑会读取以下约定字段：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `summary` | `string?` | 标题和提示词回退。 |
| `name` | `string?` | 标题回退。 |
| `text` | `string?` | 标题回退。 |
| `id` | `string?` | 标题和无图缩略回退。 |
| `emoji_id` | `string?` | 无图缩略回退。 |
| `file` | `string?` | 无图缩略回退。 |
| `url` | `string?` | 缩略图地址。 |
| `preview` | `string?` | 缩略图地址回退。 |
| `thumb` | `string?` | 缩略图地址回退。 |

### StickerUpdateRequest

| 字段 | 类型 |
| --- | --- |
| `key` | `string` |
| `summary` | `string` |

### StickerUpdateResult

| 字段 | 类型 |
| --- | --- |
| `updated` | `number` |
| `item` | `StickerItem \| null` |

### StickerDeleteRequest

| 字段 | 类型 |
| --- | --- |
| `keys` | `string[] \| string` |

### StickerDeleteResult

| 字段 | 类型 |
| --- | --- |
| `requested` | `number` |
| `deleted` | `number` |

## 前端状态数据

这些对象是当前前端为了交互维护的本地状态。重构时可以保留语义，但不需要继承旧变量名。

### ShellState

| 字段 | 类型 |
| --- | --- |
| `values` | `Record<string, ConfigValue>` |
| `fieldSchema` | frontend internal |
| `logsPaused` | `boolean` |
| `modelFieldsDirty` | `boolean` |
| `runtimeStats` | `RuntimeStats \| null` |
| `mlxStatus` | `MlxStatus \| null` |
| `chartMode` | `"learning" \| "performance" \| "groups"` |

### StickerState

| 字段 | 类型 |
| --- | --- |
| `items` | `StickerItem[]` |
| `total` | `number` |
| `limit` | `number` |
| `offset` | `number` |
| `query` | `string` |
| `type` | `string` |
| `typeCounts` | `Record<string, number>` |
| `selectedKeys` | `string[]` |
| `loading` | `boolean` |
| `notice` | `NoticeState` |
| `preview` | `StickerItem \| null` |
| `summaryText` | `string` |
| `saving` | `boolean` |

## 数据使用备注

- 表情仓库中只有 `summary` 是当前 WebUI 明确编辑的业务字段。
- `RuntimeTotals.external_corpus_samples`、`RuntimeTotals.media_assets`、`RuntimeTotals.first_sample_at`、`RuntimeTotals.avg_training_chars` 当前属于已返回但未重点展示的数据。
- `StickerItem.data` 必须作为开放结构保留，不能在前端强行收窄为固定协议字段。
- `InferenceMetric` 同时来自客户端包装层和 MLX 运行时，聚合时需要保留 `source`。
- 配置字段 UI schema 是前端内部契约；后端配置 API 保持 k-v 数据边界。
