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

## 后续分期路线（源自总体架构文档，未实施）

- **第二期：role_card 域**。`RoleCard` 统一抽象（`bot_persona` / `member_mimic` 共享字段结构）、
  角色卡 × 工具权限矩阵、`BaseSearchProvider` 搜索工具域、系统提示词分段开关配置化
  （prompt/builder.py 已预留为分段拼装的落点）
- **第三期：Provider 域扩展**。Ollama / llama.cpp 适配器（`@register` 装饰器即插即用）、
  `InferenceBackendConfig` 数据库持久化与故障降级链、`RoleProviderBinding`
  （chat/vision/data_cleaning 三角色绑定）、模型管理器页面（SSE 流式下载进度）。
  `providers/base.py` 的 capabilities()/default_models() 已为前端展示预留
- **第四期：observability 增强**。`MessageLog` 记录 trigger/reply 平台消息 ID、
  `/trace` 引用溯源命令、历史（HTTP）+ 实时（SSE）合并时间线、反拟人化黑名单词库页
- **第五期：finetune 增强 + 长期记忆**。`LoraAdapter` 角色卡+语种绑定与版本回滚、
  长期结构化记忆（重要度/时效性/衰减）、Alembic 迁移体系、KPI Dashboard 首页

## 已知的现状债务（刻意保留，分期解决）

- 表情选择与后台识图固定直连 `mlx_runtime`（GLM 模式下本地视觉仍被使用）——第三期随
  RoleProviderBinding 的 vision 角色绑定一起解决
- `sticker/assets.py` 的去重/回填/清理维护例程会跨表重写 chat_samples（同事务、表情域语义，
  已在模块 docstring 标注）；第五期 Alembic 化时收敛
- prompt 各段（身份/记忆/场景/接话动机/自检）目前是单一 builder 函数，第二期拆分段落注册表
