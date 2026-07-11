# WebUI Component Design Contract

更新时间：2026-07-12

状态：临时设计契约。本文基于 `webui-data-contract.md` 和 shadcn 官方 registry 查询结果整理，用于后续 FastAPI + Vite WebUI 重构的信息呈现和组件选择。

组件查询方式：`pnpx shadcn@latest list @shadcn`。

已确认可用的关键 shadcn 组件包括：

- 结构：`Sidebar`, `Card`, `Item`, `Separator`, `Tabs`, `Accordion`, `Collapsible`, `Resizable`, `ScrollArea`
- 表单：`Field`, `Form`, `Input`, `InputGroup`, `Textarea`, `Switch`, `Slider`, `Checkbox`, `ToggleGroup`, `Select`, `NativeSelect`, `Combobox`, `Command`
- 展示：`Badge`, `Chart`, `Progress`, `Avatar`, `AspectRatio`, `Pagination`, `Empty`, `Skeleton`, `Spinner`, `Kbd`
- 反馈与浮层：`Alert`, `AlertDialog`, `Dialog`, `Drawer`, `Sheet`, `Popover`, `HoverCard`, `Tooltip`, `Sonner`
- 菜单：`DropdownMenu`, `ContextMenu`, `NavigationMenu`, `Button`, `ButtonGroup`

## 选择原则

- 布尔值使用 `Switch`，不使用 `RadioGroup`。
- 2 到 5 个短标签选项使用 `ToggleGroup`。
- 长标签、动态选项、可搜索选项使用 `Combobox`，必要时由 `Command + Popover` 组合实现。
- 小范围数值使用 `Slider + InputGroup`。
- 精确技术数值使用 `Field + InputGroup`，并标注单位和特殊值语义。
- 动态列表使用 `ItemGroup` 风格的 `Item` 列表、badge token、或 `Command` 多选，不直接塞进 textarea。
- 长文本、命令、prompt、JSON 使用 `Textarea`，但默认放在 `Sheet`、`Dialog`、`Accordion` 或 `Collapsible` 中。
- 表情 entry 使用 `Card + Item + AspectRatio`，不以普通 table 作为主呈现。
- 破坏性操作使用 `AlertDialog`。
- 非重点数据放进 `HoverCard`、`Tooltip` 或详情 layer。
- 调试型原始数据可以使用 `ScrollArea` 和只读 `Textarea`，不要影响主流程。

## 全局结构层

| Intent | Candidate | Decision |
| --- | --- | --- |
| 后台主导航，承载运行、模型、训练、学习、表情、规则、系统 | `Sidebar`, `NavigationMenu`, `Tabs` | 使用 `Sidebar`。后台信息密度高，主导航应常驻；二级内容内部再用 `Tabs`。 |
| 首屏运行摘要 | `Card`, `Item`, `Badge`, `HoverCard` | 使用 `Card` 分区，内部用 `Item` 表达状态行；次要字段藏入 `HoverCard`。 |
| 主要操作，如加载模型、卸载、训练、保存 | `Button`, `ButtonGroup`, `AlertDialog`, `Spinner` | 使用 `ButtonGroup` 聚合相关动作；危险动作进 `AlertDialog`；异步时 `Spinner + disabled`。 |
| 分隔信息层级 | `Separator`, `Card`, `Accordion`, `Collapsible` | 大区域用 `Card`，区域内组间用 `Separator`，高级内容用 `Accordion/Collapsible`。 |
| 异常和成功反馈 | `Alert`, `Sonner`, `Badge` | 阻塞错误用 `Alert`，保存/训练提交用 `Sonner`，状态常驻用 `Badge`。 |

## 运行状态字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `MlxStatus.state` | 传达当前运行状态 | `Badge`, `Alert`, `Item` | `Badge` 作为主状态；异常时升级为 `Alert`。 |
| `MlxStatus.status` | 动作结果或内部状态 | `Badge`, `HoverCard` | 放入状态 `HoverCard`，避免和 `state` 混淆。 |
| `MlxStatus.healthy` | 判断是否可用 | `Badge`, `Switch` | 只读 `Badge`，不要用 `Switch`，因为它不是用户控制项。 |
| `MlxStatus.runtime` | 技术运行时 | `Badge`, `HoverCard` | 在 `HoverCard` 中展示，不占主层级。 |
| `MlxStatus.last_error` | 错误定位 | `Alert`, `HoverCard`, `ScrollArea` | 有错误时显示 `Alert` 摘要，长错误进 `HoverCard/ScrollArea`。 |
| `model` | 当前目标模型 | `Combobox`, `Item`, `Badge` | 展示态用 `Item`，编辑态用 `Combobox`。 |
| `loaded_model` | 实际已加载模型 | `Item`, `Tooltip` | 和目标模型并列成对比 `Item`；不一致时加 `Badge`。 |
| `adapter_path` | 目标 adapter | `InputGroup`, `Tooltip`, `HoverCard` | 展示态截断 + `Tooltip`，编辑态 `InputGroup`。 |
| `loaded_adapter_path` | 实际已加载 adapter | `Item`, `Tooltip` | 和 `adapter_path` 并列显示，不放输入框。 |
| `control.can_load` | 动作可用性 | `Button disabled`, `Tooltip` | 控制加载按钮 disabled，原因放 `Tooltip`。 |
| `control.can_unload` | 动作可用性 | `Button disabled`, `AlertDialog` | 卸载是破坏性操作，用 danger button + `AlertDialog`。 |
| `control.can_reload` | 动作可用性 | `Button disabled`, `Tooltip` | 重载按钮 disabled，原因放 `Tooltip`。 |
| `control.reason` | 不可操作原因 | `Tooltip`, `HoverCard` | 放在 disabled 按钮 `Tooltip`。 |
| `operation` | 后台操作中状态 | `Badge`, `Progress`, `Spinner` | `Badge + Spinner`，长任务才显示 `Progress`。 |
| `model_presets` | 模型候选 | `Combobox`, `Command`, `Select` | `Combobox`，底层用 `Command + Popover`；支持搜索和自定义值。 |

## 推理与日志字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `InferenceMetric.tps` | 核心性能指标 | `Card`, `Badge`, `Chart` | 主层 KPI `Card`，趋势用 `Chart` line。 |
| `latency_seconds` | 性能诊断 | `HoverCard`, `Chart` | 主层藏到 `HoverCard`，趋势可叠加在性能详情。 |
| `completion_tokens` | 技术细节 | `HoverCard` | 不做主指标，仅在性能详情中展示。 |
| `response_chars` | 输出量判断 | `HoverCard` | 和 TPS 同卡片的详情。 |
| `source` | 指标来源 | `Badge`, `Tooltip` | 小 `Badge`。 |
| `metrics.last` | 最近一次推理 | `Card`, `Item` | 性能卡片主内容。 |
| `metrics.history` | 推理历史 | `Chart` | `Chart` line，配 tooltip。 |
| `MlxLogEntry.time` | 日志时间 | `Item`, `ScrollArea` | 日志层用 `Item` 行。 |
| `MlxLogEntry.stream` | 日志来源 | `Badge` | `Badge`，如 system/stderr。 |
| `MlxLogEntry.message` | 日志内容 | `ScrollArea`, `HoverCard` | 专用日志 layer 用 `ScrollArea`，不用 table。 |

## 运行统计字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `learned_messages` | 总学习量 | `Card` | 主层 KPI。 |
| `trainable_samples` | 可训练样本量 | `Card`, `HoverCard` | 和已入库消息同卡，作为副指标。 |
| `pending_training_samples` | 是否需要训练 | `Card`, `Badge`, `Progress` | 主层重点 KPI；达到阈值时高亮 `Badge`。 |
| `stickers` | 表情库规模 | `Card` | 主层 KPI。 |
| `external_corpus_samples` | 外部语料量 | `HoverCard`, `Card` | 非重点，放训练卡 HoverCard。 |
| `media_assets` | 多模态资产量 | `HoverCard` | 非重点，放表情/训练详情。 |
| `training_runs` | 训练历史数量 | `Card`, `HoverCard` | 训练卡副指标。 |
| `groups` | 覆盖群数 | `Card` | 覆盖范围卡主字段。 |
| `users` | 覆盖用户数 | `Card` | 覆盖范围卡副字段。 |
| `samples_with_stickers` | 表情参与学习情况 | `Card`, `HoverCard` | 表情卡副指标。 |
| `today_samples` | 今日学习量 | `Card` | 活跃度卡主字段。 |
| `last_24h_samples` | 近 24h 学习量 | `Card` | 活跃度卡副字段。 |
| `first_sample_at` | 样本起始时间 | `HoverCard` | 不占主层级。 |
| `last_sample_at` | 最近样本时间 | `Card`, `Tooltip` | 活跃度卡展示相对时间，完整时间进 `Tooltip`。 |
| `avg_training_chars` | 平均训练文本长度 | `HoverCard` | 数据质量详情，不进主层级。 |

## 图表字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `learning_trend` | 学习趋势 | `Chart` line/bar | 样本日趋势用 `Chart` bar；移动端可横向 bar。 |
| `group_contribution` | 群贡献分布 | `Chart` bar, `Item` list | 用 horizontal bar；旁边可给 Top groups `Item` 列表。 |
| `performance_trend` | TPS/性能趋势 | `Chart` line | 用 line chart，tooltip 展示 `tps/latency/source`。 |

## 训练状态字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `training.running` / `training_operation.running` | 是否训练中 | `Badge`, `Progress`, `Spinner` | `Badge + Progress`，训练按钮显示 `Spinner`。 |
| `training.last_success` | 最近成功训练 | `Item`, `HoverCard` | 训练摘要卡中的 `Item`。 |
| `training.last_run` | 最近一次训练 | `Item`, `Badge` | 展示 status + 时间；失败时 `Alert`。 |
| `training.pending_since` | 待训练起点 | `HoverCard` | 放在待训练样本卡详情。 |
| `progress.stage` | 当前阶段 | `Progress`, `Item` | 独立训练 layer 的阶段主标签。 |
| `progress.status` | 阶段状态 | `Badge` | 与 stage 并列。 |
| `progress.reason` | 触发原因 | `HoverCard`, `Item` | 训练详情展示，不放主层级。 |
| `progress.sample_count` | 当前训练样本数 | `Card`, `Progress` | 训练 layer 主指标。 |
| `progress.dataset_path` | 数据集路径 | `InputGroup` readonly, `Tooltip` | 详情区只读 `InputGroup`，支持复制按钮。 |
| `progress.output_dir` | 输出路径 | `InputGroup` readonly, `Tooltip` | 同上。 |
| `progress.log_tail` | 训练日志 | `ScrollArea`, `Collapsible` | 独立日志层，不用 table。 |
| `TrainingResult.status` | 训练结果 | `Badge`, `Alert` | 成功 badge，失败 alert。 |
| `TrainingResult.log` | 结果日志 | `ScrollArea`, `Drawer` | 放训练详情 `Drawer`。 |

## 训练配置摘要字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `target_configured` | 是否能训练 | `Badge`, `Alert` | false 时训练区顶部 `Alert`。 |
| `helper_configured` | 是否启用辅助清洗 | `Badge`, `HoverCard` | 训练策略卡副状态。 |
| `min_train_samples` | 训练门槛 | `Slider`, `InputGroup` | 小范围数值，`Slider + InputGroup`。 |
| `dataset_limit` | 数据集上限 | `Slider`, `InputGroup` | `Slider + InputGroup`，0 表示不限需有说明。 |
| `external_corpus_enabled` | 外部语料是否参与 | `Switch` | 布尔配置用 `Switch`。 |
| `external_corpus_limit` | 外部语料上限 | `Slider`, `InputGroup` | `Slider + InputGroup`，0 表示不限。 |
| `multimodal_training_enabled` | 多模态训练 | `Switch`, `Alert` | `Switch`，开启时展示风险/说明 `Alert`。 |
| `mlx_lora_modules` | LoRA 模块 | `ToggleGroup`, `Checkbox`, `Field choice card` | 用 `Field choice card + Checkbox`，因为每项需要风险描述，不适合短 Toggle。 |

## 表情仓库字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `StickerGalleryPayload.items` | 表情集合 | `Item`, `Card`, `ScrollArea`, `Pagination` | 卡片网格，每个 entry 用 `Card + Item`，不要直接 table。 |
| `total` | 总数 | `Badge`, `Card` | 仓库标题旁 `Badge`。 |
| `limit` | 每页数量 | `Select`, `Pagination` | `Select`，选项 48/80/120。 |
| `offset` | 分页位置 | `Pagination` | 只由 `Pagination` 控制，不直接展示。 |
| `query` | 搜索 | `InputGroup`, `Command` | 普通搜索用 `InputGroup`；复杂筛选可升级 `Command`。 |
| `type` | 类型筛选 | `ToggleGroup`, `Combobox` | 类型少时 `ToggleGroup`，类型多时 `Combobox`。 |
| `type_counts` | 类型计数 | `Badge`, `ToggleGroup` | 类型按钮内显示 count badge。 |
| `StickerItem.key` | 唯一键 | `Tooltip`, `HoverCard`, `Kbd` | 次要字段，放详情和 tooltip，不做主标题。 |
| `StickerItem.type` | 表情类型 | `Badge` | 卡片角标。 |
| `StickerItem.summary` | 提示词 | `ItemTitle`, `Textarea` | 卡片主标题；编辑时 `Textarea`。 |
| `StickerItem.uri` | 资源地址 | `Tooltip`, `InputGroup readonly` | 详情中只读 + 复制。 |
| `StickerItem.data` | 原始协议数据 | `Accordion`, `Textarea readonly`, `ScrollArea` | 详情底部折叠原始 JSON。 |
| `group_id` | 来源群 | `HoverCard`, `Badge` | 详情中展示，列表不主显。 |
| `user_id` | 来源用户 | `HoverCard`, `Badge` | 同上。 |
| `first_seen` | 首次出现 | `ItemDescription`, `Tooltip` | 详情中展示完整时间。 |
| `last_seen` | 最近出现 | `ItemDescription`, `Tooltip` | 卡片副信息展示相对时间。 |
| `count` | 使用次数 | `Badge` | 卡片上突出显示。 |
| `data.url/preview/thumb` | 图片源 | `AspectRatio`, `Card` | 卡片顶部固定比例预览。 |
| `data.id/emoji_id/file` | 缩略回退 | `Avatar`, `Badge` | 无图时用 `AvatarFallback` 或 fallback block。 |

## 表情操作字段

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `StickerUpdateRequest.key` | 指定编辑对象 | hidden state, `Sheet` context | 不展示为输入，由打开的详情上下文提供。 |
| `StickerUpdateRequest.summary` | 编辑提示词 | `Textarea`, `Field` | `Field + Textarea`，带字数限制和保存按钮。 |
| `StickerUpdateResult.updated` | 保存结果 | `Sonner`, `Badge` | 成功/失败用 toast。 |
| `StickerUpdateResult.item` | 更新后对象 | state refresh | 不直接展示结构，更新卡片和详情。 |
| `StickerDeleteRequest.keys` | 批量删除对象 | `Checkbox`, `AlertDialog` | 每卡 `Checkbox`，删除前 `AlertDialog`。 |
| `StickerDeleteResult.requested/deleted` | 删除结果 | `Sonner` | toast 显示删除数量。 |

## 配置字段：后台与系统

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_admin_enabled` | 后台开关 | `Switch` | `Field + Switch`。 |
| `qwen_persona_admin_host` | 监听地址 | `InputGroup`, `Combobox` | `InputGroup`，可给 localhost/0.0.0.0 作为 suggestions。 |
| `qwen_persona_admin_port` | 端口 | `InputGroup` | 数字 `InputGroup`，带端口范围校验。 |
| `qwen_persona_admin_token` | 访问令牌 | `InputGroup`, `Button` | password `InputGroup` + 显示/复制/生成按钮。 |
| `qwen_persona_admin_config_file` | 覆盖配置路径 | `InputGroup`, `Tooltip` | path `InputGroup` + 复制按钮。 |
| `qwen_persona_data_dir` | 数据目录 | `InputGroup` | path `InputGroup`，标记重启提示。 |
| `override_file` | 当前覆盖文件 | `HoverCard`, `InputGroup readonly` | 设置底部只读，不进主表单。 |
| `changed` | 保存变更 | `Sonner`, `Alert` | 保存后 toast；多项变更用 `Alert`。 |
| `restart_required` | 需重启项 | `Alert`, `Badge` | 保存后警告 alert，字段旁 `Badge`。 |

## 配置字段：模型部署与推理

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_mlx_model` | 选择模型 | `Combobox`, `Command`, `Select` | `Combobox`，动态/长 label/可自定义。 |
| `qwen_persona_mlx_adapter_path` | Adapter 路径 | `InputGroup`, `Combobox` | `InputGroup` + 最近路径 `Popover`。 |
| `qwen_persona_mlx_auto_load` | 启动预加载 | `Switch` | `Switch`。 |
| `qwen_persona_mlx_trust_remote_code` | 信任远程代码 | `Switch`, `Alert` | `Switch` + 开启时 `Alert` 风险提示。 |
| `qwen_persona_model_dir` | 模型目录 | `InputGroup` | path `InputGroup`。 |
| `qwen_persona_temperature` | 回复随机度 | `Slider`, `InputGroup` | `Slider + 数字 InputGroup`，范围 0-2。 |
| `qwen_persona_top_p` | 采样范围 | `Slider`, `InputGroup` | `Slider + 数字 InputGroup`，范围 0-1。 |
| `qwen_persona_max_tokens` | 最大回复长度 | `Slider`, `InputGroup` | 常用范围用 `Slider`，允许手填。 |
| `qwen_persona_request_timeout` | 请求超时 | `InputGroup`, `Slider` | `InputGroup` 秒单位；不建议 slider，因为跨度可能大。 |

## 配置字段：学习与回复

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_enabled_groups` | 限定群列表 | `Combobox`, `Command`, `Badge` | 多选 `Combobox/Command` + selected badges。 |
| `qwen_persona_enable_learning` | 学习开关 | `Switch` | `Switch`。 |
| `qwen_persona_enable_reply` | 回复开关 | `Switch` | `Switch`。 |
| `qwen_persona_learning_priority` | matcher 优先级 | `InputGroup` | 技术数字，放高级区数字输入。 |
| `qwen_persona_reply_priority` | matcher 优先级 | `InputGroup` | 同上。 |
| `qwen_persona_reply_probability` | 随机回复概率 | `Slider`, `InputGroup` | 概率用 `Slider + % InputGroup`。 |
| `qwen_persona_follow_reply_probability` | 跟风概率 | `Slider`, `InputGroup` | `Slider + % InputGroup`。 |
| `qwen_persona_trigger_keywords` | 触发关键词 | `ItemGroup`, `InputGroup`, `Badge` | token/chip 列表，新增用 `InputGroup`。 |
| `qwen_persona_pattern_reply_threshold` | 组合触发阈值 | `Slider`, `InputGroup` | 小整数，用 `Slider + InputGroup`。 |
| `qwen_persona_context_scope` | 上下文范围 | `ToggleGroup` | 3 个短选项，用 single `ToggleGroup`。 |
| `qwen_persona_context_messages` | 上下文消息数 | `Slider`, `InputGroup` | 小范围整数，用 `Slider`。 |
| `qwen_persona_min_sample_chars` | 最短样本 | `Slider`, `InputGroup` | 小范围数字，用 `Slider + InputGroup`。 |
| `qwen_persona_max_sample_chars` | 最长样本 | `Slider`, `InputGroup` | 和最短样本组成 range 设置。 |
| `qwen_persona_system_prompt` | 人格提示词 | `Textarea`, `Sheet` | 主区摘要，编辑进 `Sheet + Textarea`。 |

## 配置字段：复读

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_enable_repeat` | 是否参与复读 | `Switch` | `Switch`。 |
| `qwen_persona_repeat_min_count` | 复读触发次数 | `Slider`, `InputGroup` | 小整数，用 `Slider`。 |
| `qwen_persona_repeat_probability` | 复读概率 | `Slider`, `InputGroup` | 概率 slider。 |
| `qwen_persona_repeat_cooldown_seconds` | 复读冷却 | `InputGroup`, `Slider` | `InputGroup` 秒单位；可带快捷按钮 30/60/90。 |

## 配置字段：表情行为

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_text_sticker_reply_probability` | 文本触发表情概率 | `Slider`, `InputGroup` | 概率 slider。 |
| `qwen_persona_sticker_segment_types` | 表情段类型 | `Checkbox`, `FieldSet`, `ToggleGroup` | 用 `FieldSet + Checkbox`，因为是多选集合。 |
| `qwen_persona_sticker_prompt_limit` | 提示表情数量 | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_sticker_repo_dir` | 仓库目录 | `InputGroup` | path 输入 + 复制按钮。 |
| `qwen_persona_sticker_repo_url` | 仓库地址 | `InputGroup` | URL 输入。 |
| `qwen_persona_enable_sending_stickers` | 是否允许发图 | `Switch` | `Switch`。 |
| `qwen_persona_max_reply_stickers` | 单次最多表情 | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_media_image_fallback_text` | 图片降级文本 | `InputGroup` | 短文本，用 `InputGroup`。 |
| `qwen_persona_media_sticker_fallback_text` | 表情降级文本 | `InputGroup` | 短文本，用 `InputGroup`。 |

## 配置字段：训练策略

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_enable_training` | 定时训练开关 | `Switch` | `Switch`。 |
| `qwen_persona_train_times` | 训练时间点 | `ItemGroup`, `InputGroup`, `Button` | 时间 chip 列表，新增用 time `InputGroup`。 |
| `qwen_persona_lora_train_command` | 自定义训练命令 | `Textarea`, `Alert`, `Collapsible` | 高级区 `Textarea`，顶部 `Alert` 风险提示。 |
| `qwen_persona_after_train_command` | 训练后命令 | `Textarea`, `Alert`, `Collapsible` | 同上。 |
| `qwen_persona_train_cwd` | 训练工作目录 | `InputGroup` | path `InputGroup`。 |
| `qwen_persona_train_timeout_seconds` | 训练超时 | `InputGroup` | 秒单位数字输入。 |
| `qwen_persona_dataset_limit` | 样本上限 | `Slider`, `InputGroup` | `Slider + InputGroup`，0 表示不限。 |
| `qwen_persona_dataset_history_window` | 上下文窗口 | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_external_corpus_enabled` | 合并外部语料 | `Switch` | `Switch`。 |
| `qwen_persona_external_corpus_limit` | 外部语料上限 | `Slider`, `InputGroup` | `Slider + InputGroup`，0 表示不限。 |
| `qwen_persona_min_train_samples` | 最少训练样本 | `Slider`, `InputGroup` | `Slider + InputGroup`。 |
| `qwen_persona_multimodal_training_enabled` | 多模态资产训练 | `Switch`, `Alert` | `Switch` + 说明 `Alert`。 |
| `qwen_persona_mlx_lora_train_modules` | LoRA 模块 | `Field choice card`, `Checkbox` | 三个 choice card，每项带风险描述。 |
| `qwen_persona_mlx_lora_rank` | LoRA rank | `Slider`, `InputGroup` | 常用离散值可 slider。 |
| `qwen_persona_mlx_lora_scale` | LoRA alpha | `Slider`, `InputGroup` | slider + 数字。 |
| `qwen_persona_mlx_lora_dropout` | dropout | `Slider`, `InputGroup` | 0-1 slider。 |
| `qwen_persona_mlx_lora_learning_rate` | 学习率 | `InputGroup`, `Combobox` | 科学计数法不适合 slider，用输入 + preset 下拉。 |
| `qwen_persona_mlx_lora_batch_size` | batch size | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_mlx_lora_grad_accum_steps` | 梯度累积 | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_mlx_lora_epochs` | epochs | `Slider`, `InputGroup` | 小整数 slider。 |
| `qwen_persona_mlx_lora_iters` | iters | `InputGroup` | 0 语义特殊，用数字输入 + helper text。 |
| `qwen_persona_mlx_lora_max_seq_length` | 最大序列长度 | `Select`, `InputGroup` | 常见值 1024/2048/4096 用 `Select`，允许高级手填。 |
| `qwen_persona_mlx_lora_grad_checkpoint` | 梯度检查点 | `Switch` | `Switch`。 |
| `qwen_persona_mlx_lora_mask_prompt` | 只训练回复 | `Switch` | `Switch`。 |
| `qwen_persona_mlx_lora_save_every` | 保存间隔 | `InputGroup` | 精确数字输入。 |
| `qwen_persona_mlx_lora_steps_per_report` | 日志间隔 | `InputGroup` | 精确数字输入。 |

## 配置字段：辅助清洗

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_lora_helper_api_url` | 辅助接口地址 | `InputGroup` | URL 输入。 |
| `qwen_persona_lora_helper_api_key` | API key | `InputGroup`, `Button` | password `InputGroup` + 显示/清空。 |
| `qwen_persona_lora_helper_model` | 辅助模型 | `Combobox` | 可能动态/自定义，用 `Combobox`。 |
| `qwen_persona_lora_helper_timeout` | 请求超时 | `InputGroup` | 秒单位。 |
| `qwen_persona_lora_helper_sample_limit` | 样本上限 | `InputGroup`, `Slider` | 0 语义特殊，用数字输入。 |
| `qwen_persona_lora_helper_extra_body` | 附加 JSON | `Textarea`, `Accordion` | JSON editor textarea，默认折叠。 |
| `qwen_persona_lora_helper_system_prompt` | 清洗提示词 | `Textarea`, `Sheet` | 长文本，放高级 `Sheet`。 |

## 配置字段：过滤与安全

| 字段 | Intent | Candidate | Decision |
| --- | --- | --- | --- |
| `qwen_persona_command_prefixes` | 命令前缀列表 | `ItemGroup`, `Badge`, `InputGroup` | chip/item 列表。 |
| `qwen_persona_command_regexes` | 命令过滤正则 | `ItemGroup`, `Dialog`, `Textarea` | 规则列表，每条用 `Item`，编辑进 `Dialog`。 |
| `qwen_persona_auto_read_picmenu_config` | 自动读取 picmenu | `Switch` | `Switch`。 |
| `qwen_persona_picmenu_config_keys` | picmenu 配置键 | `ItemGroup`, `InputGroup` | token list。 |
| `qwen_persona_picmenu_command_keys` | picmenu 命令键 | `ItemGroup`, `InputGroup` | token list。 |
| `qwen_persona_blacklist_words` | 黑名单词 | `ItemGroup`, `InputGroup` | token list，支持批量粘贴。 |
| `qwen_persona_blacklist_regexes` | 黑名单正则 | `ItemGroup`, `Dialog`, `Textarea` | 规则列表 + 测试弹窗。 |
| `qwen_persona_blacklist_user_ids` | 黑名单用户 | `Combobox`, `ItemGroup`, `Badge` | 多选/手填 ID，展示成 badges。 |
| `qwen_persona_blacklist_group_ids` | 黑名单群 | `Combobox`, `ItemGroup`, `Badge` | 同上。 |

## Layer 拆分

| Layer | shadcn 组件组合 | 承载字段 |
| --- | --- | --- |
| Shell | `Sidebar + SidebarGroup + SidebarMenu + Badge` | 主模块导航、异常 badge、训练中 badge。 |
| Overview | `Card + Item + Badge + HoverCard + Chart` | 运行、学习、训练、表情、趋势摘要。 |
| Runtime | `Card + ButtonGroup + Combobox + InputGroup + Alert` | MLX 状态、模型、adapter、控制动作。 |
| Training | `Card + Progress + Item + ScrollArea + Accordion` | 训练进度、样本、日志、训练配置。 |
| Learning | `FieldGroup + Field + Switch + Slider + ToggleGroup` | 学习、回复、上下文、概率、复读。 |
| Stickers | `Card grid + Item + AspectRatio + Badge + Checkbox + Sheet + Pagination` | 表情 entry、筛选、编辑、删除、详情。 |
| Rules | `ItemGroup + Dialog + Textarea + Alert + Separator` | 正则、关键词、黑名单、picmenu。 |
| Advanced | `Accordion + Collapsible + FieldGroup + InputGroup + Textarea` | 路径、命令、helper、系统设置。 |
| Diagnostics | `Drawer + ScrollArea + Badge + Button` | 日志、原始 JSON、路径、复制。 |

## 关键取舍

- `Table` 只用于调试型密集数据或 raw data，不作为表情库主呈现。
- `RadioGroup` 基本不用；布尔值统一 `Switch`，短枚举优先 `ToggleGroup`。
- 小范围数值优先 `Slider + InputGroup`，精确技术数值才用数字输入。
- 动态列表和长 label 选项用 `Combobox/Command`，短固定选项才用 `Select/ToggleGroup`。
- 表情 entry 用 `Card + Item + AspectRatio`，详情用 `Sheet`，删除用 `AlertDialog`。
- 非重点运行概览字段放 `HoverCard`，不要把主层级做成字段倾倒。
