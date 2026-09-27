# nonebot-plugin-ai-companion

AI 对话伙伴 NoneBot2 插件：本地 MLX / 在线 LLM 双推理后端（Provider 注册表抽象），被动学习群友语气与表情包，支持 LoRA 定时训练。前身为 `nonebot-plugin-qwen-persona`，0.2.0 起按领域分包重构（重构说明见 `docs/architecture.md`）。

## 安装

```bash
# 在宿主 bot 项目（uv workspace 成员）
uv sync

# 单独安装
uv pip install -e .                # 本体
uv pip install -e ".[onebot]"      # + OneBot v11 适配器
uv pip install -e ".[scheduler]"   # + 定时训练（nonebot-plugin-apscheduler）
uv pip install -e ".[dev]"         # + pytest/ruff/pyrefly
```

宿主 `pyproject.toml` 注册：

```toml
[tool.nonebot.plugins]
nonebot-plugin-ai-companion = ["nonebot_plugin_ai_companion"]
```

## 配置

所有配置键前缀为 `ai_companion_*`（如 `AI_COMPANION_MLX_MODEL`）。旧前缀 `QWEN_PERSONA_*` 的环境变量与 `admin_config.json` 键**继续兼容**（自动改写为新前缀，admin 配置在下次保存时完成迁移）。

关键配置（完整列表见 webui 设置页或 `config.py`）：

| 键 | 默认 | 说明 |
|---|---|---|
| `ai_companion_inference_backend` | `mlx` | `mlx`（本地）/ `llm`（在线 OpenAI 兼容 API；旧值 `glm` 仍兼容） |
| `ai_companion_mlx_model` | `mlx-community/Qwen3.5-9B-MLX-4bit` | 本地 MLX-VLM 模型 |
| `ai_companion_online_api_url` / `_api_key` / `_model` | - | 在线 LLM 后端（DeepSeek / GLM / Qwen 等） |
| `ai_companion_data_dir` | `data/qwen_persona` | 运行数据目录（历史路径保持不变） |
| `ai_companion_admin_host` / `_port` | `127.0.0.1` / `8082` | 管理 WebUI |
| `ai_companion_train_times` | `["06:10", "23:50"]` | LoRA 定时训练时刻 |

两个推理后端配置示例：

```env
# 本地 MLX（Apple Silicon）
AI_COMPANION_INFERENCE_BACKEND=mlx
AI_COMPANION_MLX_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit
AI_COMPANION_MLX_AUTO_LOAD=true

# 在线 LLM
AI_COMPANION_INFERENCE_BACKEND=llm
AI_COMPANION_ONLINE_API_URL=https://api.z.ai/api/coding/paas/v4/chat/completions
AI_COMPANION_ONLINE_API_KEY=sk-xxx
AI_COMPANION_ONLINE_MODEL=deepseek-flash
```

## 使用

- @机器人 或私聊即可对话；命中称呼关键词、复读、对话模式也会触发
- 命令：`人格状态`、`人格训练`（SUPERUSER）、`人格模型状态`、`人格部署模型`（SUPERUSER）
- 溯源命令（第四期）：引用机器人的一条回复发送 `/trace`，可查看该回复使用的角色卡、
  触发原因、模型与耗时；引用一条图片/表情消息发送 `/添加表情 [描述]`，可将其收录进表情仓库
- 管理 WebUI：`http://127.0.0.1:8082`（运维控制台（含 KPI 首屏与 LoRA adapter 版本
  管理）/ 表情仓库 / 角色卡 / 推理后端 / 模型管理 / 日志时间线 / 黑名单词库 /
  会话监控（含长期记忆管理）/ 配置 / 规则 / 诊断）；日志时间线页为 HTTP 历史 +
  SSE 实时合并视图

### 长期记忆与微调（第五期）

- **长期记忆**：会话监控页可按 QQ 号提取结构化记忆（称呼/偏好/重要事件/事实，
  经 data_cleaning 角色分析），入库自动去重、按半衰期衰减遗忘；启用提示词
  memory 段后自动注入 bot 回复上下文
- **LoRA adapter 版本管理**：训练可关联「角色卡 + 语种」，成功后自动登记新版本；
  运维页支持按版本一键激活/回滚
- **数据库迁移**：schema 变更走 Alembic（存量库执行 `alembic stamp head` 对齐），
  运行时幂等 DDL 兜底

### 推理后端与角色绑定（第三期）

- **多后端实例**：管理页可为 Ollama / llama.cpp（本地）与在线 LLM API 创建任意多个实例
  （自定义 base_url / 模型 / API key / 优先级），一键健康检查；新后端类型只需在
  `providers/` 加一个 `@register` 适配器文件
- **三角色绑定**：`chat`（主对话）/ `vision`（识图与表情选择）/ `data_cleaning`（风格
  分析等后台任务）各自可绑定实例 + 模型覆盖 + 降级实例；chat 支持故障降级链
  （主实例失败自动换备，全挂回退默认配置）。未绑定时行为与旧版完全一致
- **模型管理**：HF 模型下载（SSE 实时进度、可取消）、磁盘占用统计、一键删除；
  快速候选含 Qwen3 / Gemma 3 / Qwen2.5-VL / LLaVA 等梯度列表
- 默认端点可用环境变量覆盖：`AI_COMPANION_OLLAMA_API_URL`、
  `AI_COMPANION_LLAMA_CPP_API_URL`

### 角色卡与工具（第二期）

- **角色卡**：管理页可创建 `bot_persona` 人格（系统提示词支持 `{persona_name}` / `{user_nickname}`
  / `{time}` / `{memory_summary}` 变量），或按 QQ 号一键生成 `member_mimic` 群友模仿卡
  （自动分析口头禅/语气/话题偏好，可随时"刷新语言风格"）。绑定到群后生效，
  切换绑定会重置该会话短期上下文
- **工具权限矩阵**：每张角色卡可勾选允许的工具（memory_query / sticker_send / web_search）；
  `AI_COMPANION_ENABLE_TOOL_CALLS=true` 时在线 LLM 后端启用 function calling
  回路（执行工具 → 回填 → 二次推理，轮次/超时受限）
- **网页搜索**：`AI_COMPANION_SEARCH_PROVIDER=searxng|tavily` 并配置
  `AI_COMPANION_SEARXNG_URL` 或 `AI_COMPANION_TAVILY_API_KEY` 后，web_search 工具可用
- **提示词分段**：`AI_COMPANION_PROMPT_SEGMENT_ORDER` / `AI_COMPANION_DISABLED_PROMPT_SEGMENTS`
  可调整/关闭系统提示词各段（identity/selfcheck/naturalness/sticker/tools/memory/scene/motivation）

## 开发

```bash
uv sync --extra dev          # 或在宿主项目 dev 组中安装

# 测试（86 个用例）
python -m pytest tests -q

# 静态检查（CI 与本地强制项）
ruff check nonebot_plugin_ai_companion tests
ruff format --check nonebot_plugin_ai_companion tests
pyrefly check nonebot_plugin_ai_companion

# WebUI 开发与重建（产物写入 nonebot_plugin_ai_companion/webui_static/）
cd webui && pnpm install && pnpm dev    # 开发
cd webui && pnpm build                  # 构建
pnpm --dir webui test                   # 前端 vitest
```

### 目录结构

```
nonebot_plugin_ai_companion/
├── __init__.py          # matcher 路由 + DI 装配 + 生命周期（~370 行）
├── chat_service.py      # 回复主管线编排
├── learning_service.py  # 被动学习管线编排
├── bot_env.py           # 事件 ID / 黑名单 / 命令识别等环境策略
├── contracts.py         # 跨域 Protocol 与 DTO（唯一允许的跨域依赖）
├── store.py             # PersonaStore：领域仓库聚合容器
├── config.py            # ai_companion_* 配置（含旧前缀兼容）
├── runtime_config.py    # admin_config.json 覆盖持久化
├── providers/           # 推理来源域：base/registry/client + mlx/llm 后端
├── model_manager/       # 模型下载与默认模型列表
├── memory/              # 样本/模式/语料仓库 + 共享 DB 层 + 摄入管线
├── context/             # 消息信封/转发展开/昵称/媒体解析/图文相关性
├── sticker/             # 表情建模/分类/仓库/选择/后台分析/资产
├── decision/            # 接话门控 + 复读与模式跟踪
├── prompt/              # 文本工具/预算/意图规划/梗选择/媒体文本/系统提示词
├── tools/               # 工具注册（PromptTool）
├── finetune/            # LoRA 训练/数据集导出/训练记录/定时任务
├── observability/       # 推理与表情事件、调试日志
├── delivery/            # 投递规划 + OneBot v11 平台发送
└── api/                 # admin FastAPI（auth/helpers/routes/server）
```

领域划分与依赖方向、后续分期路线见 [docs/architecture.md](docs/architecture.md)。
