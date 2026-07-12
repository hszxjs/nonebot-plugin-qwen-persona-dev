# WebUI Rebuild Spec

更新时间：2026-07-12

本文定义 Qwen Persona 后台 WebUI 的彻底重构方案。新 WebUI 采用 FastAPI 后端 + Vite SSG 前端，前端技术栈为 React + TypeScript，HTTP 请求使用 `ky`，数据缓存与刷新使用 `swr`，组件库使用 shadcn/ui，图标库使用 `@tabler/icons-react`。

本文只参考数据契约与组件选择契约，不参考旧嵌入式后台实现。

## 目标

- 将后台从嵌入式 HTML/JS 重构为高内聚、低耦合的前后端分层架构。
- 以运维控制台为主体验，配置管理独立成页面。
- 首屏优先回答：bot 是否正常运行、推理是否正常、训练是否需要处理、最近是否有错误。
- 表情库、配置编辑、规则管理、诊断数据均从首屏下沉到独立工作区或详情 layer。
- 明确 UI 组件边界，禁止用 Tailwind className 手写视觉组件。

## 非目标

- 不迁移旧后台 HTML、CSS、JS 结构。
- 不兼容旧 WebUI payload。
- 不在首屏呈现表情库。
- 不把配置页做成后端字段的平铺表单。
- 不用普通 table 作为表情仓库主呈现。

## 技术架构

### Backend

后端采用插件内独立 FastAPI Admin Server。

- 同源提供领域 API 与 Vite SSG 静态产物。
- 独立于 NoneBot 主 driver，降低生命周期耦合。
- 负责鉴权、领域 schema 聚合、任务触发、状态快照、日志查询、静态文件托管。
- 不拼接 HTML，不返回前端组件状态。

### Static Host

FastAPI 负责托管前端静态产物。

- 静态资源命中文件时直接返回文件。
- 非 API、非静态资源命中的路径返回 `index.html`，支持 Browser Router 伪静态。
- 静态资源应支持合理缓存；`index.html` 应避免强缓存，保证发布后能拿到最新入口。

### Frontend

前端采用 Vite SSG + React + TypeScript。

- 使用 Browser Router。
- 使用 `ky` 封装 HTTP client，包括 token、错误解析、超时、JSON 类型处理。
- 使用 `swr` 管理请求缓存、分层轮询、乐观更新和错误重试。
- 使用 shadcn/ui 组件表达视觉与交互。
- 使用 `@tabler/icons-react` 作为唯一结构性图标来源。

## 强制 UI 规则

### Tailwind className 边界

Tailwind `className` 只能处理 layout composition：

- grid
- flex
- gap
- 响应式列
- 页面区域定位
- 跨组件间距
- 宽高约束
- 滚动区域尺寸

Tailwind `className` 不得用于构建任何视觉组件，包括但不限于：

- 表面
- 边框
- 圆角
- 阴影
- 色彩
- 按钮
- 卡片
- 输入框
- 弹窗
- 标签
- 状态徽标
- alert/callout
- hover/active/disabled 视觉状态

视觉表达必须由 shadcn 组件、组件 variant、语义 token 和组件组合承担。任何例外必须在 spec 或代码注释中明确说明原因。

### shadcn 组件纪律

- 表单使用 `Field` / `FieldGroup` / `FieldSet`。
- 布尔值使用 `Switch`，不使用 `RadioGroup`。
- 2 到 5 个短选项使用 `ToggleGroup`。
- 动态、长标签、可搜索选项使用 `Combobox`，必要时组合 `Command + Popover`。
- 小范围数值使用 `Slider + InputGroup`。
- 精确技术数值使用 `Field + InputGroup`，标明单位和特殊值语义。
- 长文本、命令、prompt、JSON 使用 `Textarea`，并默认放在 `Sheet`、`Dialog`、`Accordion` 或 `Collapsible` 中。
- 破坏性确认使用 modal 形态，优先 `AlertDialog`，不使用简单 `alert()`。
- 操作反馈使用 `Sonner`。
- 空状态使用 `Empty`。
- 加载状态使用 `Skeleton` 或 `Spinner`。
- 分隔使用 `Separator`，不使用手写 border。
- 表情 entry 使用 `Card + Item + AspectRatio`，不使用普通 table 作为主呈现。

### Icon 规则

- 统一使用 `@tabler/icons-react`。
- 图标只作为语义辅助，不用 emoji 作为结构图标。
- Button 内图标遵守 shadcn 图标规则，使用 `data-icon` 标记图标位置。
- 不在图标上手写尺寸 class；遵守组件内部图标尺寸体系。

## 信息架构

### 主结构

- Shell：`Sidebar`
- 首屏：运维控制台
- 独立工作区：表情仓库、配置管理、规则过滤、诊断
- 详情层：`Sheet` / `Drawer` / `Dialog`
- 危险确认：`AlertDialog`

### 首屏运维控制台

首屏采用舒适低密度设计，避免字段倾倒。

首屏展示：

- 模型健康状态
- 主推理事件日志
- 训练状态
- 学习与样本摘要
- 最近错误
- 必要操作按钮

首屏不展示：

- 表情库
- 配置编辑
- 原始 JSON
- regex 列表
- 长路径
- 长日志
- 完整聊天上下文

### 配置管理

配置管理放在独立页面，采用语义分区 + 高级折叠。

建议分区：

- 模型与推理
- 学习与回复
- 训练策略
- 表情行为
- 系统后台
- 高级/危险设置

规则过滤是独立工作区，不进入配置页 tabs。

配置页不应平铺后端字段。字段呈现必须按用户意图和操作风险组织。

## 数据刷新策略

使用 SWR 分层轮询。

| 数据域 | 刷新策略 |
| --- | --- |
| runtime 状态 | 3 到 5 秒 |
| inference event log | 2 到 5 秒 |
| training progress | 2 到 3 秒 |
| overview stats | 10 到 15 秒 |
| stickers | 用户操作或筛选变化时刷新 |
| settings | 首次进入和保存后刷新 |
| folded runtime/training logs | 展开时刷新，可手动刷新 |

日志刷新必须支持暂停。

## 领域 API

新 API 采用领域化 schema，不兼容旧 WebUI payload。

### operations overview

服务首屏，聚合运行、训练、学习、最近错误和关键计数，避免首屏发起过多请求。

应包含：

- runtime 摘要
- training 摘要
- learning/sample 摘要
- 最近 inference event log
- 最近错误摘要
- 关键操作可用性

### runtime

负责 MLX 状态和模型控制。

应包含：

- runtime state
- healthy
- target model
- loaded model
- target adapter
- loaded adapter
- last error
- control availability
- runtime metrics
- folded MLX runtime log

### inference-events

负责主推理事件日志。

这是首屏主日志，不复用 MLX runtime log。

每条 event 建议包含：

- event id
- time
- group/user 摘要
- trigger reason
- model
- adapter 摘要
- latency
- TPS
- response chars
- media count
- status
- error summary
- truncated input
- truncated output

默认保留和展示最近 300 条。

首屏只展示元信息和截断摘要；点击后进入详情抽屉查看截断输入/输出、媒体数量、错误和相关模型信息。

### training

负责训练状态、训练触发、训练配置摘要和训练日志。

应包含：

- current operation
- progress
- last run
- last success
- pending samples
- dataset/output path 摘要
- folded training log
- training action availability

### stickers

负责表情仓库。

应包含：

- sticker cards
- type counts
- search
- type filter
- pagination
- summary edit
- batch delete
- sticker detail
- raw protocol data in folded detail

### settings

负责语义化配置读取与保存。

配置 API 应按前端语义分区输出，不直接暴露后端原始字段列表作为 UI 结构。字段仍需保留稳定 key，便于保存和变更追踪。

### diagnostics

负责低频调试数据。

应包含：

- raw JSON
- full paths
- internal logs
- version/build info
- static asset/build metadata

诊断数据不进入首屏。

## 自动保存与确认

配置页采用自动保存。

### 自动保存规则

- 普通字段变更后 debounce 自动保存。
- 保存成功使用 `Sonner` 提示。
- 保存失败在字段附近显示错误，并使用 `Alert` 呈现阻塞错误。
- 保存后如果字段需要重启或 reload，必须明确提示。

### Modal 确认规则

危险操作和会触发完整 reload 的字段不能静默自动保存，必须通过 modal 二次确认。不要使用简单 alert。

需要 modal 确认的操作：

- 卸载模型
- 重载模型
- 全量外部语料训练
- 删除表情
- 批量删除表情
- 修改自定义训练命令
- 修改训练后命令
- 修改 helper API key
- 修改后台 token
- 修改可能导致模型完整 reload 的字段

可能导致完整 reload 或服务重启的字段：

- `qwen_persona_mlx_model`
- `qwen_persona_mlx_adapter_path`
- `qwen_persona_mlx_auto_load`
- `qwen_persona_mlx_trust_remote_code`
- `qwen_persona_model_dir`
- `qwen_persona_admin_enabled`
- `qwen_persona_admin_host`
- `qwen_persona_admin_port`
- `qwen_persona_data_dir`
- `qwen_persona_train_times`
- `qwen_persona_enable_training`
- `qwen_persona_learning_priority`
- `qwen_persona_reply_priority`

Modal 文案必须说明：

- 会改变什么
- 是否会触发 reload 或重启
- 是否可能中断当前任务
- 用户确认后才提交保存

## 首屏设计

### Runtime Status Card

组件：

- `Card`
- `Item`
- `Badge`
- `ButtonGroup`
- `Tooltip`
- `HoverCard`
- `Alert`
- `Collapsible`

展示：

- state
- healthy
- target model
- loaded model
- adapter summary
- last error
- load/reload/unload actions

折叠内容：

- MLX runtime log
- full model/adapter path
- control reason

### Inference Event Log Card

组件：

- `Card`
- `Item`
- `Badge`
- `ScrollArea`
- `HoverCard`
- `Sheet`
- `Skeleton`
- `Empty`

展示：

- 最近 300 条推理事件
- time
- status
- trigger reason
- latency
- TPS
- media count
- truncated input/output summary

详情：

- truncated input
- truncated output
- error summary
- group/user 摘要
- model/adapter
- timing metrics

### Training Card

组件：

- `Card`
- `Progress`
- `Item`
- `Badge`
- `ButtonGroup`
- `Collapsible`
- `ScrollArea`
- `Alert`

展示：

- running/status/stage
- pending samples
- sample count
- last run
- last success
- normal training action
- full external corpus action

折叠内容：

- training log
- dataset path
- output dir
- reason

### Learning Summary Card

组件：

- `Card`
- `Item`
- `Badge`
- `HoverCard`
- `Chart`

展示：

- learned messages
- trainable samples
- pending training samples
- today samples
- last 24h samples
- groups
- users

HoverCard：

- first sample time
- last sample time
- avg training chars
- external corpus samples
- media assets

## 表情仓库设计

表情仓库独立于首屏。

组件：

- `Card`
- `Item`
- `AspectRatio`
- `Badge`
- `Checkbox`
- `InputGroup`
- `Combobox`
- `ToggleGroup`
- `Pagination`
- `Sheet`
- `AlertDialog`
- `Sonner`
- `Empty`
- `Skeleton`

Entry 呈现：

- 图片预览
- summary
- type badge
- count badge
- last seen
- checkbox selection
- quick edit action

详情 Sheet：

- summary editor
- uri
- group/user
- first/last seen
- folded raw data JSON

## 配置页设计

配置页使用 `FieldGroup` 和语义分区。

### 模型与推理

- model：`Combobox`
- adapter path：`InputGroup`
- auto load：`Switch`
- trust remote code：`Switch + Alert`
- model dir：`InputGroup`
- temperature：`Slider + InputGroup`
- top_p：`Slider + InputGroup`
- max tokens：`Slider + InputGroup`
- request timeout：`InputGroup`

### 学习与回复

- enable learning：`Switch`
- enable reply：`Switch`
- enabled groups：`Combobox + Badge`
- context scope：`ToggleGroup`
- context messages：`Slider + InputGroup`
- reply probabilities：`Slider + InputGroup`
- trigger keywords：`Item` token list
- system prompt：`Sheet + Textarea`

### 训练策略

- enable training：`Switch`
- train times：time token list
- dataset limit：`Slider + InputGroup`
- history window：`Slider + InputGroup`
- external corpus enabled：`Switch`
- external corpus limit：`Slider + InputGroup`
- multimodal training：`Switch + Alert`
- LoRA modules：choice cards with `Checkbox`
- LoRA hyperparams：advanced accordion
- custom commands：danger accordion with modal confirmation

### 表情行为

- sending stickers：`Switch`
- segment types：`FieldSet + Checkbox`
- prompt limit：`Slider + InputGroup`
- max reply stickers：`Slider + InputGroup`
- fallback texts：`InputGroup`
- repo dir/url：advanced `InputGroup`

## 规则过滤页面

- prefixes：token list
- regexes：`Item` rule list + editor dialog
- picmenu keys：token list
- blacklist words：token list
- blacklist regexes：rule list + test dialog
- blacklist users/groups：`Combobox + Badge`

### 系统后台

- admin enabled：`Switch`
- host：`InputGroup`
- port：`InputGroup`
- token：password `InputGroup`
- config file：readonly/path `InputGroup`
- data dir：path `InputGroup`

## 错误、空状态与加载

### 空状态

- 无推理事件：`Empty`，提示等待 bot 触发或检查回复开关。
- 无训练样本：`Empty`，展示当前阈值和样本数。
- 无表情：`Empty`，说明表情学习边界。
- 无日志：`Empty`，不显示空白区域。

### 错误状态

- API 失败：`Alert` + 重试按钮。
- token 失败：进入 auth 状态，提示重新输入 token。
- MLX 未加载：展示加载动作和模型摘要。
- 训练失败：训练 Card 内 `Alert`，详情保留日志。

### 加载状态

- 首屏 Card 用 `Skeleton`。
- 操作按钮内用 `Spinner`。
- 表情卡片加载用 card skeleton。

## 响应式

- 桌面：`Sidebar` 常驻。
- 窄屏：`Sidebar` 折叠为 sheet/drawer 形态。
- 首屏 Card 从多列降为单列。
- 表情仓库卡片网格降为单列。
- 日志使用固定高度 `ScrollArea`，避免撑爆页面。

## 计划阶段

### Phase 1：骨架与边界

- 建立 FastAPI Admin Server 边界。
- 建立 Vite SSG React TS 前端骨架。
- 建立静态托管和 Browser Router fallback。
- 建立 ky client 和 SWR provider。
- 建立 shadcn 组件与 Tabler icon 约束。

### Phase 2：领域 API

- 定义 Pydantic schema。
- 实现 operations overview。
- 实现 runtime、training、inference-events、stickers、settings、diagnostics API。
- 加入 token 鉴权和统一错误格式。

### Phase 3：运维控制台

- 实现首屏。
- 接入 SWR 分层轮询。
- 实现 Inference Event Log。
- 实现 runtime/training folded logs。

### Phase 4：表情仓库

- 卡片式表情列表。
- 搜索、筛选、分页。
- summary 编辑。
- 批量选择和删除确认。
- 详情 Sheet。

### Phase 5：配置管理

- 语义分区配置页。
- 自动保存。
- reload/danger 字段 modal 确认。
- 字段级错误和保存反馈。

### Phase 6：诊断与收尾

- diagnostics layer。
- empty/error/loading 状态补齐。
- 响应式检查。
- 可访问性检查。
- 移除旧嵌入式后台入口。

## 验收标准

- 首屏不包含表情库和配置表单。
- 首屏展示 Inference Event Log，默认最近 300 条。
- MLX runtime log 折叠在 runtime Card 下。
- training log 折叠在 training Card 下。
- 配置页自动保存。
- 危险操作和 reload 字段必须 modal 二次确认。
- 表情 entry 以 card 形式呈现。
- shadcn 组件承担视觉表达。
- Tailwind className 未被用于手写视觉组件。
- 图标统一来自 `@tabler/icons-react`。
- FastAPI 能正确托管 SSG 静态产物并支持伪静态。
- 新 WebUI 不依赖旧嵌入式后台实现。
