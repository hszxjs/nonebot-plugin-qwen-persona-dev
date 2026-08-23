# 架构说明与分期路线（第一期重构完成稿）

本插件按"高内聚低耦合"的领域划分组织：**按领域而非技术层切分模块**，域之间只依赖
`contracts.py` 中声明的接口与 DTO，禁止跨域 import 具体实现类；NoneBot matcher 层只做
路由 + 依赖装配，业务编排下沉到 `chat_service` / `learning_service`，领域逻辑在各领域子包。

## 领域依赖图（箭头 = 允许的依赖方向）

```
                     ┌────────────── nonebot matcher 层（__init__.py）──────────────┐
                     │  matcher 路由 + DI 装配 + 生命周期钩子                        │
                     └───┬───────────────┬────────────────┬────────────────┬────────┘
                         ▼               ▼                ▼                ▼
                  chat_service    learning_service     api/server      finetune/schedule
                         └───────────────┴────────┬───────┴────────────────┘
                                                   ▼
        ┌──────────────────────────── contracts.py（跨域 Protocol / DTO）────────────────────────────┐
        │  BaseProvider·ProviderCapabilities·MediaCaptioner·StickerSelectionRuntime·PromptTool·        │
        │  DownloadProgress·RuntimeOperation·InferenceMetric 等                                        │
        └───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┘
                ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼
            providers  model_manager memory    context    sticker    decision    prompt    tools
           (推理来源)   (模型文件)  (样本记忆) (会话上下文) (表情包)   (接话决策)  (提示词)   (工具)
                │          │          │          │          │          │          │          │
                └──────────┴─────┬────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                                 ▼
                    finetune（微调：数据集/训练/运行记录）
                    observability（事件与日志）· delivery（投递 + OneBot v11）

依赖规则：
- 任意域 → contracts.py：允许（且只允许接口/DTO）
- providers ↔ prompt：generation_profiles 依赖 prompt/budget（采样档位需要 token 预算）
- memory → sticker（写入表情/媒体资产）、sticker → memory/db（共享 PersonaDatabase 连接层）
- delivery/onebot_v11 是唯一的 OneBot v11 平台代码收口（poke/转发/表情消息构造）
- store.py = PersonaStore 聚合容器（DI 根），组合各领域仓库并保留旧公共 API 以便渐进迁移
```

## 与重构前（nonebot-plugin-qwen-persona 0.1.x）的对应关系

| 旧模块（平铺） | 新位置 |
|---|---|
| `storage.py`（2627 行） | `memory/{db,samples,patterns,corpus,stats}.py`、`sticker/assets.py`、`observability/events.py`、`finetune/{runs,dataset}.py`、`store.py`（聚合） |
| `__init__.py`（1611 行） | `__init__.py`（368 行）+ `chat_service.py` + `learning_service.py` + `bot_env.py` + `prompt/builder.py` + `context/relevance.py` + `delivery/onebot_v11.py` + `finetune/schedule.py` + `observability/debug_log.py` |
| `admin_server.py`（658 行） | `api/{auth,helpers,server}.py` + `api/routes/{operations,runtime,training,stickers,settings,misc}.py` |
| `qwen_client.py` | `providers/client.py`（经注册表构建） |
| `mlx_backend.py` / `mlx_runtime.py` / `glm_backend.py` | `providers/` 同名模块（已注册 local/remote 类型） |
| `filters.py` | `prompt/text.py`（文本工具）+ `decision/gates.py`（门控） |
| `media.py` | `context/media_parts.py` + `memory/ingest.py` + `prompt/media_text.py` |
| `message_context.py` / `forward_resolver.py` / `nickname_resolver.py` | `context/envelope.py` / `context/forward.py` / `context/nicknames.py` |
| `sticker*.py` / `media_analysis.py` | `sticker/`（model/classifier/repo_files/selector/analysis/assets） |
| `reply_decision.py` / `repeat.py` | `decision/gates.py` / `decision/repeat.py` |
| `reply_planner.py` / `prompt_budget.py` / `meme_gate.py` | `prompt/planner.py` / `prompt/budget.py` / `prompt/meme_gate.py` |
| `tool_registry.py` | `tools/registry.py`（PromptTool 移入 contracts） |
| `trainer.py` | `finetune/trainer.py`（+runs/dataset/schedule） |
| `model_download.py` + presets | `model_manager/download.py` + `model_manager/presets.py` |
| `inference_events.py` / `runtime_state.py` | `observability/inference_events.py` / `contracts.py` |
| `reply_delivery.py` / `poke.py` | `delivery/plan.py` / `delivery/onebot_v11.py` |

## 第一期不变量（已验证）

1. **数据零迁移**：persona.sqlite3 schema 字节级不变（shasum 校验）；models/datasets/lora_runs/sticker_repo 路径不变；数据目录仍为 `data/qwen_persona`
2. **环境变量兼容**：旧 `QWEN_PERSONA_*` 键经 `AiCompanionConfig` 的 before-validator 自动改写为 `ai_companion_*`
3. **Admin API 契约不变**：路径/载荷/token 语义与拆分前一致（webui 未改一行业务代码即通过契约测试）
4. **行为不变**：matcher 优先级、回复概率、命令名、表情选择固定走本地 MLX 等现状全部保留（`chat_service._select_reply_sticker_key` 有注释标注）

## 工具链（CI 与本地强制项）

```bash
python -m pytest tests -q                       # 86 用例（含 provider registry 9 例）
ruff check nonebot_plugin_ai_companion tests    # lint
ruff format --check nonebot_plugin_ai_companion tests
pyrefly check nonebot_plugin_ai_companion       # 类型检查（0 error；mlx.core 原生扩展行内忽略）
pnpm --dir webui test                           # 前端 11 用例
```

## 第二期交付（已完成）

- **role_card 域**：`RoleCard` 统一抽象（`bot_persona` / `member_mimic` 共享字段），
  增量表 `role_cards` / `session_role_bindings`（按惯例先备份 persona.sqlite3）；
  会话绑定切换后短期上下文从 `bound_at` 重新计（长期检索不受影响）
- **member_mimic**：`style_analyzer` 用辅助 LLM（providers/aux_llm.py，glm-4.5-flash）
  从该群友近期样本增量提炼口头禅/语气/话题偏好/情绪基线，支持一键"刷新语言风格"
- **提示词分段**（文档十节）：identity/selfcheck/naturalness/sticker/tools 默认段
  与一期输出逐字节兼容（有测试锁定）；memory/scene/motivation 新段按
  `ai_companion_prompt_segment_order` / `ai_companion_disabled_prompt_segments` 配置拼入
- **工具域 v2**（文档四节）：`@register(name, parameters, permission_tags)` 注册表、
  角色卡 `allowed_tools` 白名单（权限矩阵，webui 勾选编辑）、内置
  memory_query / sticker_send / web_search；`BaseSearchProvider` 抽象 + SearxNG + Tavily
- **GLM 工具调用回路**：`chat_with_tools` → 执行 → 结果回填 → 二次推理，
  `ai_companion_enable_tool_calls` 默认关闭，轮次/超时/结果长度有上限防死循环
- **API/webui**：`/api/role-cards`（CRUD/克隆/绑定/生成/刷新风格）+ `/api/tools`；
  webui 新增"角色卡"页（第 6 个工作区）

## 第三期交付（已完成）

- **Ollama / llama.cpp 适配器**：`providers/openai_compat.py` 共用基类（chat /
  chat_with_tools / 重试 / 健康检查 / 多模态 base64），`@register` 注册为 local 类型，
  base_url 自动补 /v1，内置按文档十二.3 梯度整理的默认模型列表
- **InferenceBackendConfig 持久化**（文档三.3）：`inference_backend_configs` 增量表，
  同一后端类型可建多实例；`ProviderInstanceManager` 缓存实例（updated_at 失效）
- **RoleProviderBinding 三角色绑定**（文档三.4）：`role_provider_bindings` 表，
  chat / vision / data_cleaning 各自绑定实例 + 模型覆盖 + 降级实例；
  `ProviderResolver` 解析：绑定实例（chat 含降级链与模型覆盖）→ 无绑定回退旧路径
  （QwenClient / MLX 运行时 / 辅助 LLM），未配置时行为与二期完全一致
- **一期债务清偿**：表情选择器与后台识图改走 vision 角色解析（默认仍为本地 MLX，
  绑定实例后可切 Ollama 视觉模型等）；member_mimic 风格分析改走 data_cleaning 角色
- **模型管理器**（文档十二.2）：后台 HF 下载（tqdm 钩子取消）、RuntimeOperation 进度、
  磁盘占用统计、删除清理；`/api/models/manage/download/events` SSE 流式推送
- **webui**：新增「推理后端」页（实例 CRUD/健康徽章/角色绑定/优先级）与
  「模型管理」页（EventSource 下载进度）

## 第四期交付（已完成）

- **消息溯源**（文档十三节）：`message_logs` 增量表记录 trigger/reply 平台消息 ID，
  并关联当时生效的角色卡、决策原因与明细、Provider/模型、耗时、贴纸与输入输出预览；
  环形淘汰条数由 `ai_companion_log_max_records` 配置。发送层改造为
  `{ok, reply_message_id}` 返回（从 OneBot send 结果提取消息 ID），
  chat/repeat/pattern 三条回复路径均记录
- **引用命令**：`/trace`（引用 Bot 回复 → 按 reply_message_id 反查溯源卡片）；
  `/添加表情 [描述]`（引用图片消息 → 从历史样本或引用本体收录进表情资产）
- **统一日志时间线**：`GET /api/logs` 跨 message/inference/sticker 三源合并分页查询；
  `GET /api/logs/stream` SSE 实时增量（心跳保活、可选 max_seconds 有限流、线程安全
  EventBus）；webui「日志时间线」页 = HTTP 历史 + EventSource 实时前插合并（去重、
  暂停/重连、来源筛选、明细展开）
- **黑名单词库页**：词汇/正则增删（经 settings PATCH 持久化）+ 样本文本命中测试
  端点（`POST /api/blacklist/test` 返回命中规则明细）

## 第五期交付（已完成，全量收官）

- **LoraAdapter 版本管理**（文档十一节）：`lora_adapters` 表以「角色卡 + 源语言」为
  维度维护版本（zh / en / zh-mixed-en 检测见 `detect_language`）；
  `trainer.run_training(role_card_id=, source_language=)` 成功后自动登记新版本；
  激活 = 同维度互斥；回滚 = 激活早于当前活跃版本的最新历史版本；webui 运维页
  提供版本分组、一键激活/回滚/登记
- **长期结构化记忆**（文档六节）：`long_term_memories` 表（称呼/偏好/重要事件/事实，
  重要度 1–5）；`extract_memories` 经 data_cleaning 角色产出结构化 JSON（回退辅助
  LLM）；入库自动去重（字符 2-gram Jaccard ≥ 0.6 合并，取更高重要度并刷新时效）；
  `decay` 按半衰期衰减重要度、低于阈值标记遗忘；**prompt 记忆注入段的数据源**
  （memory 段启用时取发送者未衰减 top 记忆）；webui 会话监控页含记忆管理 tab
- **ToolCallLog + KPI**（文档十三/十四节）：`tool_call_logs` 记录工具名/成败/耗时
  （环形淘汰）；`GET /api/dashboard/kpi` 聚合消息量、按触发原因拆分的接话统计、
  按模型拆分的延迟/TPS、后端健康徽章、工具成功率、活跃会话数；运维首页新增
  KPI 卡片区
- **会话监控页**（文档十四节）：按会话（群/私聊）聚合的最近回复记录 + 下钻时间线、
  手动触发接话（`POST /api/sessions/poke`，经在线 bot 生成并发送主动消息）、
  记忆提取/衰减操作
- **Alembic 迁移体系**：alembic.ini + migrations/env.py（数据目录可经
  `AI_COMPANION_DATA_DIR` 覆盖）+ 0001 空基线（存量库 `alembic stamp head` 对齐；
  运行时 ensure_schema 幂等 DDL 仍作兜底；后续 schema 变更换新 revision）

## 状态

五期路线全部完成。本文档的"后续分期路线"归零；新的增强需求建议从
`docs/architecture.md` 的领域依赖图出发评估落点。

## 已知的现状债务（刻意保留，分期解决）

- 表情选择与后台识图固定直连 `mlx_runtime`（GLM 模式下本地视觉仍被使用）——第三期随
  RoleProviderBinding 的 vision 角色绑定一起解决
- 工具调用回路在 GLM 与 OpenAI 兼容实例（Ollama/llama.cpp）上可用；本地 MLX 后端
  无 function calling，工具仅作为提示词描述存在
- `sticker/assets.py` 的去重/回填/清理维护例程会跨表重写 chat_samples（同事务、表情域语义，
  已在模块 docstring 标注）；第五期 Alembic 化时收敛
- prompt 各段（身份/记忆/场景/接话动机/自检）目前是单一 builder 函数，第二期拆分段落注册表
