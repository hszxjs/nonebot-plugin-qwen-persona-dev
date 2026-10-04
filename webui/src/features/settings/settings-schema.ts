import type { ConfigValue } from "@/lib/types"

export type FieldControl =
  | "switch"
  | "text"
  | "password"
  | "number"
  | "textarea"
  | "toggle"
  | "checkboxes"
  | "tokens"
  | "select"

export type SettingsSectionKey =
  | "model"
  | "learning"
  | "training"
  | "stickers"
  | "system"
  | "online"
  | "anime"

export type FieldOption = {
  value: string
  label: string
  description?: string
}

export type FieldSchema = {
  name: string
  label: string
  description?: string
  section?: SettingsSectionKey
  control: FieldControl
  min?: number
  max?: number
  step?: number
  options?: FieldOption[]
  reloadSensitive?: boolean
  danger?: boolean
}

export type BoundField = FieldSchema & {
  value: ConfigValue
}

export const SETTINGS_SECTIONS: { key: SettingsSectionKey; title: string }[] = [
  { key: "model", title: "模型与推理" },
  { key: "online", title: "在线 LLM 推理" },
  { key: "anime", title: "动漫图片识别" },
  { key: "learning", title: "学习与回复" },
  { key: "training", title: "训练策略" },
  { key: "stickers", title: "表情行为" },
  { key: "system", title: "系统后台" },
]

export const RELOAD_SENSITIVE_FIELDS = new Set([
  "ai_companion_mlx_model",
  "ai_companion_mlx_adapter_path",
  "ai_companion_mlx_auto_load",
  "ai_companion_mlx_trust_remote_code",
  "ai_companion_model_dir",
  "ai_companion_admin_enabled",
  "ai_companion_admin_host",
  "ai_companion_admin_port",
  "ai_companion_data_dir",
  "ai_companion_train_times",
  "ai_companion_enable_training",
  "ai_companion_learning_priority",
  "ai_companion_reply_priority",
])

export const DANGER_FIELDS = new Set([
  "ai_companion_lora_train_command",
  "ai_companion_after_train_command",
  "ai_companion_lora_helper_api_key",
  "ai_companion_admin_token",
  "ai_companion_tavily_api_key",
  "ai_companion_exa_api_key",
])

const contextScopeOptions = [
  { value: "group", label: "group" },
  { value: "global", label: "global" },
  { value: "mixed", label: "mixed" },
]

const inferenceBackendOptions = [
  { value: "mlx", label: "本地 MLX", description: "使用本地加载的 MLX 模型推理。" },
  { value: "llm", label: "在线 LLM", description: "使用在线 LLM API 推理（OpenAI 兼容，如 DeepSeek / GLM）。" },
]

const loraModuleOptions = [
  { value: "language", label: "language", description: "语言层 LoRA" },
  { value: "projection", label: "projection", description: "多模态投影层" },
  { value: "vision_tower", label: "vision tower", description: "视觉塔" },
]

const stickerTypeOptions = [
  { value: "face", label: "face" },
  { value: "mface", label: "mface" },
  { value: "emoji", label: "emoji" },
  { value: "sticker", label: "sticker" },
  { value: "image", label: "image" },
]

export const SETTINGS_FIELDS: FieldSchema[] = [
  { name: "ai_companion_mlx_model", label: "目标模型", description: "MLX-VLM 推理模型。", section: "model", control: "text" },
  { name: "ai_companion_mlx_adapter_path", label: "Adapter 路径", description: "当前加载的 LoRA adapter。", section: "model", control: "text" },
  { name: "ai_companion_mlx_auto_load", label: "启动预加载", section: "model", control: "switch" },
  { name: "ai_companion_mlx_trust_remote_code", label: "信任远程代码", description: "仅对需要自定义模型代码的仓库开启。", section: "model", control: "switch" },
  { name: "ai_companion_model_dir", label: "模型目录", section: "model", control: "text" },
  { name: "ai_companion_temperature", label: "随机度", section: "model", control: "number", min: 0, max: 2, step: 0.05 },
  { name: "ai_companion_top_p", label: "Top P", section: "model", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "ai_companion_max_tokens", label: "最大回复 tokens", section: "model", control: "number", min: 1, max: 4096, step: 1 },
  { name: "ai_companion_request_timeout", label: "请求超时", section: "model", control: "number", min: 1, max: 600, step: 1 },

  { name: "ai_companion_inference_backend", label: "推理后端", description: "选择对话推理走本地 MLX 还是在线 LLM API。切换即时生效，无需重启。", section: "online", control: "toggle", options: inferenceBackendOptions },
  { name: "ai_companion_online_api_url", label: "在线 API URL", description: "OpenAI 兼容 chat/completions 地址。", section: "online", control: "text" },
  { name: "ai_companion_online_api_key", label: "在线 API Key", section: "online", control: "password" },
  { name: "ai_companion_online_model", label: "在线模型", description: "从 model list 拉取，失败时显示兜底列表。", section: "online", control: "select" },
  { name: "ai_companion_online_timeout", label: "在线超时(秒)", section: "online", control: "number", min: 5, max: 600, step: 5 },
  { name: "ai_companion_online_multimodal", label: "多模态", description: "开启后图片转 base64 发送。Coding 订阅端点不支持，默认关闭（图片丢弃仅发文字）。", section: "online", control: "switch" },

  { name: "ai_companion_search_provider", label: "网页搜索源", description: "启用 web_search 工具的搜索来源：exa / tavily / searxng。留空表示不启用。", section: "online", control: "select", options: [
    { value: "", label: "不启用" },
    { value: "exa", label: "Exa.ai（推荐，免费 1000 次/月）" },
    { value: "tavily", label: "Tavily" },
    { value: "searxng", label: "SearxNG（自建）" },
  ] },
  { name: "ai_companion_exa_api_key", label: "Exa API Key", description: "exa.ai 控制台获取；搜索源选 exa 时必填。", section: "online", control: "password" },
  { name: "ai_companion_tavily_api_key", label: "Tavily API Key", description: "搜索源选 tavily 时必填。", section: "online", control: "password" },
  { name: "ai_companion_searxng_url", label: "SearxNG 地址", description: "如 http://127.0.0.1:8888；搜索源选 searxng 时必填。", section: "online", control: "text" },
  { name: "ai_companion_search_result_limit", label: "搜索结果条数", section: "online", control: "number", min: 1, max: 10, step: 1 },
  { name: "ai_companion_enable_tool_calls", label: "工具调用（function calling）", description: "开启后在线 LLM 后端可真实调用已注册工具（搜索/记忆查询/表情）。改后需重启以注册搜索工具。", section: "online", control: "switch" },
  { name: "ai_companion_tool_max_rounds", label: "工具调用轮次上限", section: "online", control: "number", min: 1, max: 5, step: 1 },
  { name: "ai_companion_tool_call_timeout_seconds", label: "单次工具超时(秒)", section: "online", control: "number", min: 3, max: 60, step: 1 },
  { name: "ai_companion_tool_result_max_chars", label: "工具结果回填上限(字符)", section: "online", control: "number", min: 200, max: 4000, step: 100 },

  { name: "ai_companion_anime_caption_enabled", label: "动漫图片识别", description: "开启后图片会走多路识别（AI视觉 + SauceNAO + iqdb），结果进 prompt。", section: "anime", control: "switch" },
  { name: "ai_companion_anime_vision_enabled", label: "AI 视觉识图", description: "用多模态视觉模型理解任意图：角色/场景/动作。弥补以图搜源对冷门图命中率低。", section: "anime", control: "switch" },
  { name: "ai_companion_anime_vision_model", label: "视觉模型", description: "多模态在线模型，如 deepseek-flash。", section: "anime", control: "text" },
  { name: "ai_companion_anime_vision_api_url", label: "视觉 API URL", section: "anime", control: "text" },
  { name: "ai_companion_anime_vision_api_key", label: "视觉 API Key", description: "空则复用在线 LLM 推理 key。", section: "anime", control: "password" },
  { name: "ai_companion_saucenao_api_key", label: "SauceNAO API Key", description: "saucenao.com 免费注册获取。留空则跳过 SauceNAO。", section: "anime", control: "password" },
  { name: "ai_companion_saucenao_dbmask", label: "SauceNAO 库掩码", description: "999=动漫库默认全开。", section: "anime", control: "number", min: 1, max: 999999, step: 1 },
  { name: "ai_companion_anime_min_similarity", label: "SauceNAO 命中阈值", description: "低于此值视为未识别。视觉识图不受此限制。", section: "anime", control: "number", min: 0, max: 100, step: 1 },
  { name: "ai_companion_forward_max_messages", label: "转发最多展开条数", description: "合并转发消息展开的上限，防止爆 token。", section: "anime", control: "number", min: 1, max: 100, step: 1 },
  { name: "ai_companion_forward_max_chars", label: "转发文字最大字符", section: "anime", control: "number", min: 100, max: 8000, step: 100 },
  { name: "ai_companion_media_image_limit", label: "图片识别上限", description: "单条消息图片超过此数则跳过识别，只处理文字。", section: "anime", control: "number", min: 1, max: 30, step: 1 },
  { name: "ai_companion_media_relevance_check", label: "图文相关性判断", description: "识别图片后判断是否与文字相关，无关则丢弃图片描述。多一次轻量 API 调用。", section: "anime", control: "switch" },

  { name: "ai_companion_enabled_groups", label: "启用群组", description: "留空表示不限制群组。", section: "learning", control: "tokens" },
  { name: "ai_companion_relevant_context_max_age_hours", label: "相关性检索时效(小时)", description: "「相关历史/历史图片理解」只命中该窗口内的内容，防止陈年图片描述串场。0 表示不限。", section: "learning", control: "number", min: 0, max: 720, step: 6 },
  { name: "ai_companion_enable_learning", label: "学习开关", section: "learning", control: "switch" },
  { name: "ai_companion_enable_reply", label: "回复开关", section: "learning", control: "switch" },
  { name: "ai_companion_learning_priority", label: "学习优先级", section: "learning", control: "number", min: 1, max: 20000, step: 1 },
  { name: "ai_companion_reply_priority", label: "回复优先级", section: "learning", control: "number", min: 1, max: 20000, step: 1 },
  { name: "ai_companion_reply_probability", label: "随机回复概率", section: "learning", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "ai_companion_follow_reply_probability", label: "跟随回复概率", section: "learning", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "ai_companion_pattern_reply_threshold", label: "模式触发阈值", section: "learning", control: "number", min: 1, max: 20, step: 1 },
  {
    name: "ai_companion_context_scope",
    label: "上下文范围（已废弃）",
    description: "会话时间线始终按会话（群 / 私聊）作用域，不再跨群混合；该开关仅为兼容旧配置保留。",
    section: "learning",
    control: "toggle",
    options: contextScopeOptions,
  },
  {
    name: "ai_companion_context_messages",
    label: "上下文条数",
    description: "送进提示词的最近对话条数（带说话人）。在线 LLM 用满该值，本地 MLX 按序列长度自动压小。",
    section: "learning",
    control: "number",
    min: 1,
    max: 80,
    step: 1,
  },
  {
    name: "ai_companion_memory_enabled",
    label: "长期记忆",
    description: "群记忆 + 成员记忆：批量增量抽取，回复时按敏感度分级注入。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_memory_extract_min_turns",
    label: "记忆抽取批量",
    description: "累计多少条新对话触发一次抽取（配合间隔一起判断）。",
    section: "learning",
    control: "number",
    min: 5,
    max: 200,
    step: 5,
  },
  {
    name: "ai_companion_memory_extract_interval_seconds",
    label: "记忆抽取间隔（秒）",
    description: "距上次抽取超过该间隔且有待抽取内容时也会跑一次。",
    section: "learning",
    control: "number",
    min: 300,
    max: 21600,
    step: 300,
  },
  {
    name: "ai_companion_memory_member_limit",
    label: "每人记忆上限",
    description: "超出后按重要度与最近提起时间软淘汰（标过期，可在记忆页恢复）。",
    section: "learning",
    control: "number",
    min: 3,
    max: 100,
    step: 1,
  },
  {
    name: "ai_companion_memory_group_limit",
    label: "每群记忆上限",
    description: "同上，针对群层面记忆。",
    section: "learning",
    control: "number",
    min: 5,
    max: 200,
    step: 5,
  },
  {
    name: "ai_companion_memory_half_life_days",
    label: "记忆半衰期（天）",
    description: "重要度按此半衰期衰减，低于阈值标记为过期不再注入。",
    section: "learning",
    control: "number",
    min: 1,
    max: 720,
    step: 1,
  },
  {
    name: "ai_companion_memory_disabled_groups",
    label: "不记记忆的群",
    description: "这些群既不抽取记忆也不注入（每行/空格分隔群号）。",
    section: "learning",
    control: "tokens",
  },
  {
    name: "ai_companion_summary_enabled",
    label: "话题摘要",
    description: "每群维护一份「最近在聊什么」，跨小时接得上话题（注入提示词）。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_summary_min_turns",
    label: "摘要批量",
    description: "累计多少条新对话触发一次摘要更新。",
    section: "learning",
    control: "number",
    min: 10,
    max: 300,
    step: 10,
  },
  {
    name: "ai_companion_group_cooldown_seconds",
    label: "搭话冷却（秒）",
    description: "主动插话的每群冷却；被 @ 或被回复不受限制。",
    section: "learning",
    control: "number",
    min: 0,
    max: 600,
    step: 5,
  },
  {
    name: "ai_companion_group_hourly_limit",
    label: "搭话每小时上限",
    description: "每群每小时主动插话条数上限，超过后只回被叫的。",
    section: "learning",
    control: "number",
    min: 0,
    max: 200,
    step: 1,
  },
  {
    name: "ai_companion_heat_min_messages",
    label: "话题热度阈值",
    description: "窗口内消息数达到该值时，主动搭话概率加权 1.5 倍。",
    section: "learning",
    control: "number",
    min: 2,
    max: 50,
    step: 1,
  },
  {
    name: "ai_companion_fallback_to_local",
    label: "失败降级到本地 MLX",
    description: "在线端点失败时用已加载的本地模型兜底；未加载则跳过（不主动拉起）。",
    section: "online",
    control: "switch",
  },
  {
    name: "ai_companion_ping_enabled",
    label: "有人找你提醒",
    description: "群里有人 @ 或引用某群友（10 分钟没说话）时静默记下原话与上下 ±3 条；等他发言（=上线）时 bot @ 他，把错过的内容画成卡片一次性发给他。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_ping_quote_enabled",
    label: "引用也算找人",
    description: "关闭后只有 @ 才触发提醒。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_ping_idle_seconds",
    label: "在场判定窗口（秒）",
    description: "目标在该窗口内说过话就不提醒（人还在场）。",
    section: "learning",
    control: "number",
    min: 0,
    max: 3600,
    step: 60,
  },
  {
    name: "ai_companion_ping_context_lines",
    label: "上下文条数",
    description: "每条 @/引用记录其前后各多少条消息（相连的自动合并成连续段）。",
    section: "learning",
    control: "number",
    min: 0,
    max: 10,
    step: 1,
  },
  {
    name: "ai_companion_ping_max_lines",
    label: "卡片行数上限",
    description: "投递卡片最多展示多少行，超出的从最老的开始省略。",
    section: "learning",
    control: "number",
    min: 10,
    max: 100,
    step: 5,
  },
  {
    name: "ai_companion_ping_max_pending",
    label: "信箱容量",
    description: "每个目标最多积压多少条找人记录，超了丢最老的。",
    section: "learning",
    control: "number",
    min: 5,
    max: 100,
    step: 5,
  },
  {
    name: "ai_companion_ping_ttl_hours",
    label: "信箱过期（小时）",
    description: "目标一直不上线的话，找人记录最多保留多久。",
    section: "learning",
    control: "number",
    min: 1,
    max: 168,
    step: 1,
  },
  {
    name: "ai_companion_ping_at_target",
    label: "提醒时 @ 目标",
    description: "卡片消息带 @，确保目标收到群通知；关闭则只发图。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_thread_enabled",
    label: "话题线程",
    description: "把消息串成「谁在接谁的话」，历史里渲染成「阿强 → 小红：…」，回复时聚焦一条线。",
    section: "learning",
    control: "switch",
  },
  {
    name: "ai_companion_thread_window_seconds",
    label: "线程推断窗口（秒）",
    description: "没有引用/@ 时，多久内的上一条消息算作「在接这句话」。",
    section: "learning",
    control: "number",
    min: 30,
    max: 3600,
    step: 30,
  },
  {
    name: "ai_companion_reply_quote",
    label: "回复引用",
    description: "回复时引用触发它的那条消息，群里看得出在接谁的话；长文合并转发无法引用。",
    section: "learning",
    control: "switch",
  },
  { name: "ai_companion_min_sample_chars", label: "最小样本长度", section: "learning", control: "number", min: 1, max: 200, step: 1 },
  { name: "ai_companion_max_sample_chars", label: "最大样本长度", section: "learning", control: "number", min: 20, max: 4000, step: 10 },
  { name: "ai_companion_trigger_keywords", label: "触发关键词", section: "learning", control: "tokens" },
  { name: "ai_companion_system_prompt", label: "系统提示词", section: "learning", control: "textarea" },

  { name: "ai_companion_enable_training", label: "定时训练", section: "training", control: "switch" },
  { name: "ai_companion_lora_auto_deploy", label: "训练后自动部署 adapter", description: "训练成功即激活新版本、持久化 adapter 路径，并在 MLX 后端模式下热重载。", section: "training", control: "switch" },
  { name: "ai_companion_member_mimic_model", label: "风格分析模型", description: "生成/刷新群友模仿卡用的模型；留空默认在线模型。", section: "training", control: "text" },
  { name: "ai_companion_member_mimic_sample_limit", label: "风格分析取样条数", section: "training", control: "number", min: 20, max: 500, step: 10 },
  { name: "ai_companion_train_times", label: "训练时间", description: "每行一个 HH:MM。", section: "training", control: "tokens" },
  { name: "ai_companion_dataset_limit", label: "数据集上限", section: "training", control: "number", min: 0, max: 20000, step: 100 },
  { name: "ai_companion_dataset_history_window", label: "历史窗口", section: "training", control: "number", min: 0, max: 30, step: 1 },
  { name: "ai_companion_external_corpus_enabled", label: "外部语料", section: "training", control: "switch" },
  { name: "ai_companion_external_corpus_limit", label: "外部语料上限", description: "0 表示不限制。", section: "training", control: "number", min: 0, max: 1000000, step: 1000 },
  { name: "ai_companion_min_train_samples", label: "最小训练样本", section: "training", control: "number", min: 1, max: 1000, step: 1 },
  { name: "ai_companion_multimodal_training_enabled", label: "多模态训练", section: "training", control: "switch" },
  { name: "ai_companion_mlx_lora_train_modules", label: "LoRA 模块", section: "training", control: "checkboxes", options: loraModuleOptions },
  { name: "ai_companion_mlx_lora_rank", label: "LoRA rank", section: "training", control: "number", min: 1, max: 128, step: 1 },
  { name: "ai_companion_mlx_lora_scale", label: "LoRA scale", section: "training", control: "number", min: 1, max: 128, step: 1 },
  { name: "ai_companion_mlx_lora_dropout", label: "LoRA dropout", section: "training", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "ai_companion_mlx_lora_learning_rate", label: "学习率", section: "training", control: "number", min: 0, max: 0.001, step: 0.000001 },
  { name: "ai_companion_mlx_lora_batch_size", label: "Batch size", section: "training", control: "number", min: 1, max: 16, step: 1 },
  { name: "ai_companion_mlx_lora_grad_accum_steps", label: "梯度累积", section: "training", control: "number", min: 1, max: 64, step: 1 },
  { name: "ai_companion_mlx_lora_epochs", label: "Epochs", section: "training", control: "number", min: 1, max: 20, step: 1 },
  { name: "ai_companion_mlx_lora_iters", label: "Iters", description: "0 表示按 epochs 训练。", section: "training", control: "number", min: 0, max: 100000, step: 100 },
  { name: "ai_companion_mlx_lora_max_seq_length", label: "最大序列长度", section: "training", control: "number", min: 256, max: 8192, step: 128 },
  { name: "ai_companion_mlx_lora_grad_checkpoint", label: "梯度检查点", section: "training", control: "switch" },
  { name: "ai_companion_mlx_lora_mask_prompt", label: "Mask prompt", section: "training", control: "switch" },
  { name: "ai_companion_mlx_lora_save_every", label: "保存间隔", section: "training", control: "number", min: 1, max: 10000, step: 10 },
  { name: "ai_companion_mlx_lora_steps_per_report", label: "报告间隔", section: "training", control: "number", min: 1, max: 1000, step: 1 },
  { name: "ai_companion_lora_train_command", label: "自定义训练命令", section: "training", control: "textarea" },
  { name: "ai_companion_after_train_command", label: "训练后命令", section: "training", control: "textarea" },
  { name: "ai_companion_train_cwd", label: "训练工作目录", section: "training", control: "text" },
  { name: "ai_companion_train_timeout_seconds", label: "训练超时", section: "training", control: "number", min: 0, max: 86400, step: 60 },
  { name: "ai_companion_lora_helper_api_url", label: "Helper API URL", section: "training", control: "text" },
  { name: "ai_companion_lora_helper_api_key", label: "Helper API Key", section: "training", control: "password" },
  { name: "ai_companion_lora_helper_model", label: "Helper 模型", section: "training", control: "text" },
  { name: "ai_companion_lora_helper_timeout", label: "Helper 超时", section: "training", control: "number", min: 1, max: 600, step: 1 },
  { name: "ai_companion_lora_helper_sample_limit", label: "Helper 样本上限", section: "training", control: "number", min: 0, max: 20000, step: 100 },
  { name: "ai_companion_lora_helper_extra_body", label: "Helper 额外参数", section: "training", control: "textarea" },
  { name: "ai_companion_lora_helper_system_prompt", label: "Helper 系统提示词", section: "training", control: "textarea" },

  { name: "ai_companion_enable_sending_stickers", label: "发送表情", section: "stickers", control: "switch" },
  { name: "ai_companion_sticker_segment_types", label: "表情段类型", section: "stickers", control: "checkboxes", options: stickerTypeOptions },
  { name: "ai_companion_sticker_prompt_limit", label: "提示词数量", section: "stickers", control: "number", min: 0, max: 100, step: 1 },
  { name: "ai_companion_sticker_repo_dir", label: "表情仓库目录", section: "stickers", control: "text" },
  { name: "ai_companion_sticker_repo_url", label: "表情仓库 URL", section: "stickers", control: "text" },
  { name: "ai_companion_max_reply_stickers", label: "单次表情上限", section: "stickers", control: "number", min: 0, max: 10, step: 1 },
  { name: "ai_companion_text_sticker_reply_probability", label: "文本带表情概率", section: "stickers", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "ai_companion_media_image_fallback_text", label: "图片占位文本", section: "stickers", control: "text" },
  { name: "ai_companion_media_sticker_fallback_text", label: "表情占位文本", section: "stickers", control: "text" },

  { name: "ai_companion_admin_enabled", label: "后台开关", section: "system", control: "switch" },
  { name: "ai_companion_log_max_records", label: "日志保留条数", description: "推理/表情/溯源日志各自的滚动淘汰上限。", section: "system", control: "number", min: 100, max: 10000, step: 100 },
  { name: "ai_companion_prompt_segment_order", label: "提示词分段顺序", description: "逗号/换行分隔。默认 identity,selfcheck,naturalness,sticker,tools；加入 memory/scene/motivation 启用对应段（memory 段注入长期记忆）。", section: "system", control: "tokens" },
  { name: "ai_companion_disabled_prompt_segments", label: "禁用的提示词段", description: "留空即可；填段落名以关闭对应段。", section: "system", control: "tokens" },
  { name: "ai_companion_admin_host", label: "监听地址", section: "system", control: "text" },
  { name: "ai_companion_admin_port", label: "监听端口", section: "system", control: "number", min: 1, max: 65535, step: 1 },
  { name: "ai_companion_admin_token", label: "访问令牌", section: "system", control: "password" },
  { name: "ai_companion_admin_config_file", label: "覆盖配置文件", section: "system", control: "text" },
  { name: "ai_companion_data_dir", label: "数据目录", section: "system", control: "text" },
]

export const RULE_FIELDS: FieldSchema[] = [
  { name: "ai_companion_command_prefixes", label: "命令前缀", description: "每行一个前缀。", control: "tokens" },
  { name: "ai_companion_command_regexes", label: "命令正则", description: "每行一条正则。", control: "tokens" },
  { name: "ai_companion_auto_read_picmenu_config", label: "读取菜单配置", control: "switch" },
  { name: "ai_companion_picmenu_config_keys", label: "菜单配置键", control: "tokens" },
  { name: "ai_companion_picmenu_command_keys", label: "菜单命令键", control: "tokens" },
  { name: "ai_companion_blacklist_words", label: "黑名单词", control: "tokens" },
  { name: "ai_companion_blacklist_regexes", label: "黑名单正则", control: "tokens" },
  { name: "ai_companion_blacklist_user_ids", label: "黑名单用户", control: "tokens" },
  { name: "ai_companion_blacklist_group_ids", label: "黑名单群组", control: "tokens" },
]

export function bindSettingsSections(values: Record<string, ConfigValue>) {
  return SETTINGS_SECTIONS.map((section) => ({
    ...section,
    fields: SETTINGS_FIELDS.filter((field) => field.section === section.key && field.name in values)
      .map((field) => bindField(field, values)),
  })).filter((section) => section.fields.length > 0)
}

export function bindRuleFields(values: Record<string, ConfigValue>) {
  return RULE_FIELDS.filter((field) => field.name in values).map((field) => bindField(field, values))
}

export function bindField(field: FieldSchema, values: Record<string, ConfigValue>): BoundField {
  return {
    ...field,
    reloadSensitive: field.reloadSensitive ?? RELOAD_SENSITIVE_FIELDS.has(field.name),
    danger: field.danger ?? DANGER_FIELDS.has(field.name),
    value: values[field.name] ?? null,
  }
}

export function rawNameBadge(name: string) {
  const parts = name.split("_").filter(Boolean)
  if (parts.length <= 3) return name
  return `${parts[0]}...${parts.at(-2)}...${parts.at(-1)}`
}
