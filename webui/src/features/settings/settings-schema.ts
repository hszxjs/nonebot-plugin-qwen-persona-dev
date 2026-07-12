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

export type SettingsSectionKey =
  | "model"
  | "learning"
  | "training"
  | "stickers"
  | "system"

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
  { key: "learning", title: "学习与回复" },
  { key: "training", title: "训练策略" },
  { key: "stickers", title: "表情行为" },
  { key: "system", title: "系统后台" },
]

export const RELOAD_SENSITIVE_FIELDS = new Set([
  "qwen_persona_mlx_model",
  "qwen_persona_mlx_adapter_path",
  "qwen_persona_mlx_auto_load",
  "qwen_persona_mlx_trust_remote_code",
  "qwen_persona_model_dir",
  "qwen_persona_admin_enabled",
  "qwen_persona_admin_host",
  "qwen_persona_admin_port",
  "qwen_persona_data_dir",
  "qwen_persona_train_times",
  "qwen_persona_enable_training",
  "qwen_persona_learning_priority",
  "qwen_persona_reply_priority",
])

export const DANGER_FIELDS = new Set([
  "qwen_persona_lora_train_command",
  "qwen_persona_after_train_command",
  "qwen_persona_lora_helper_api_key",
  "qwen_persona_admin_token",
])

const contextScopeOptions = [
  { value: "group", label: "group" },
  { value: "global", label: "global" },
  { value: "mixed", label: "mixed" },
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
  { name: "qwen_persona_mlx_model", label: "目标模型", description: "MLX-VLM 推理模型。", section: "model", control: "text" },
  { name: "qwen_persona_mlx_adapter_path", label: "Adapter 路径", description: "当前加载的 LoRA adapter。", section: "model", control: "text" },
  { name: "qwen_persona_mlx_auto_load", label: "启动预加载", section: "model", control: "switch" },
  { name: "qwen_persona_mlx_trust_remote_code", label: "信任远程代码", description: "仅对需要自定义模型代码的仓库开启。", section: "model", control: "switch" },
  { name: "qwen_persona_model_dir", label: "模型目录", section: "model", control: "text" },
  { name: "qwen_persona_temperature", label: "随机度", section: "model", control: "number", min: 0, max: 2, step: 0.05 },
  { name: "qwen_persona_top_p", label: "Top P", section: "model", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "qwen_persona_max_tokens", label: "最大回复 tokens", section: "model", control: "number", min: 1, max: 4096, step: 1 },
  { name: "qwen_persona_request_timeout", label: "请求超时", section: "model", control: "number", min: 1, max: 600, step: 1 },

  { name: "qwen_persona_enabled_groups", label: "启用群组", description: "留空表示不限制群组。", section: "learning", control: "tokens" },
  { name: "qwen_persona_enable_learning", label: "学习开关", section: "learning", control: "switch" },
  { name: "qwen_persona_enable_reply", label: "回复开关", section: "learning", control: "switch" },
  { name: "qwen_persona_learning_priority", label: "学习优先级", section: "learning", control: "number", min: 1, max: 20000, step: 1 },
  { name: "qwen_persona_reply_priority", label: "回复优先级", section: "learning", control: "number", min: 1, max: 20000, step: 1 },
  { name: "qwen_persona_reply_probability", label: "随机回复概率", section: "learning", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "qwen_persona_follow_reply_probability", label: "跟随回复概率", section: "learning", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "qwen_persona_pattern_reply_threshold", label: "模式触发阈值", section: "learning", control: "number", min: 1, max: 20, step: 1 },
  { name: "qwen_persona_context_scope", label: "上下文范围", section: "learning", control: "toggle", options: contextScopeOptions },
  { name: "qwen_persona_context_messages", label: "上下文条数", section: "learning", control: "number", min: 1, max: 80, step: 1 },
  { name: "qwen_persona_min_sample_chars", label: "最小样本长度", section: "learning", control: "number", min: 1, max: 200, step: 1 },
  { name: "qwen_persona_max_sample_chars", label: "最大样本长度", section: "learning", control: "number", min: 20, max: 4000, step: 10 },
  { name: "qwen_persona_trigger_keywords", label: "触发关键词", section: "learning", control: "tokens" },
  { name: "qwen_persona_system_prompt", label: "系统提示词", section: "learning", control: "textarea" },

  { name: "qwen_persona_enable_training", label: "定时训练", section: "training", control: "switch" },
  { name: "qwen_persona_train_times", label: "训练时间", description: "每行一个 HH:MM。", section: "training", control: "tokens" },
  { name: "qwen_persona_dataset_limit", label: "数据集上限", section: "training", control: "number", min: 0, max: 20000, step: 100 },
  { name: "qwen_persona_dataset_history_window", label: "历史窗口", section: "training", control: "number", min: 0, max: 30, step: 1 },
  { name: "qwen_persona_external_corpus_enabled", label: "外部语料", section: "training", control: "switch" },
  { name: "qwen_persona_external_corpus_limit", label: "外部语料上限", description: "0 表示不限制。", section: "training", control: "number", min: 0, max: 1000000, step: 1000 },
  { name: "qwen_persona_min_train_samples", label: "最小训练样本", section: "training", control: "number", min: 1, max: 1000, step: 1 },
  { name: "qwen_persona_multimodal_training_enabled", label: "多模态训练", section: "training", control: "switch" },
  { name: "qwen_persona_mlx_lora_train_modules", label: "LoRA 模块", section: "training", control: "checkboxes", options: loraModuleOptions },
  { name: "qwen_persona_mlx_lora_rank", label: "LoRA rank", section: "training", control: "number", min: 1, max: 128, step: 1 },
  { name: "qwen_persona_mlx_lora_scale", label: "LoRA scale", section: "training", control: "number", min: 1, max: 128, step: 1 },
  { name: "qwen_persona_mlx_lora_dropout", label: "LoRA dropout", section: "training", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "qwen_persona_mlx_lora_learning_rate", label: "学习率", section: "training", control: "number", min: 0, max: 0.001, step: 0.000001 },
  { name: "qwen_persona_mlx_lora_batch_size", label: "Batch size", section: "training", control: "number", min: 1, max: 16, step: 1 },
  { name: "qwen_persona_mlx_lora_grad_accum_steps", label: "梯度累积", section: "training", control: "number", min: 1, max: 64, step: 1 },
  { name: "qwen_persona_mlx_lora_epochs", label: "Epochs", section: "training", control: "number", min: 1, max: 20, step: 1 },
  { name: "qwen_persona_mlx_lora_iters", label: "Iters", description: "0 表示按 epochs 训练。", section: "training", control: "number", min: 0, max: 100000, step: 100 },
  { name: "qwen_persona_mlx_lora_max_seq_length", label: "最大序列长度", section: "training", control: "number", min: 256, max: 8192, step: 128 },
  { name: "qwen_persona_mlx_lora_grad_checkpoint", label: "梯度检查点", section: "training", control: "switch" },
  { name: "qwen_persona_mlx_lora_mask_prompt", label: "Mask prompt", section: "training", control: "switch" },
  { name: "qwen_persona_mlx_lora_save_every", label: "保存间隔", section: "training", control: "number", min: 1, max: 10000, step: 10 },
  { name: "qwen_persona_mlx_lora_steps_per_report", label: "报告间隔", section: "training", control: "number", min: 1, max: 1000, step: 1 },
  { name: "qwen_persona_lora_train_command", label: "自定义训练命令", section: "training", control: "textarea" },
  { name: "qwen_persona_after_train_command", label: "训练后命令", section: "training", control: "textarea" },
  { name: "qwen_persona_train_cwd", label: "训练工作目录", section: "training", control: "text" },
  { name: "qwen_persona_train_timeout_seconds", label: "训练超时", section: "training", control: "number", min: 0, max: 86400, step: 60 },
  { name: "qwen_persona_lora_helper_api_url", label: "Helper API URL", section: "training", control: "text" },
  { name: "qwen_persona_lora_helper_api_key", label: "Helper API Key", section: "training", control: "password" },
  { name: "qwen_persona_lora_helper_model", label: "Helper 模型", section: "training", control: "text" },
  { name: "qwen_persona_lora_helper_timeout", label: "Helper 超时", section: "training", control: "number", min: 1, max: 600, step: 1 },
  { name: "qwen_persona_lora_helper_sample_limit", label: "Helper 样本上限", section: "training", control: "number", min: 0, max: 20000, step: 100 },
  { name: "qwen_persona_lora_helper_extra_body", label: "Helper 额外参数", section: "training", control: "textarea" },
  { name: "qwen_persona_lora_helper_system_prompt", label: "Helper 系统提示词", section: "training", control: "textarea" },

  { name: "qwen_persona_enable_sending_stickers", label: "发送表情", section: "stickers", control: "switch" },
  { name: "qwen_persona_sticker_segment_types", label: "表情段类型", section: "stickers", control: "checkboxes", options: stickerTypeOptions },
  { name: "qwen_persona_sticker_prompt_limit", label: "提示词数量", section: "stickers", control: "number", min: 0, max: 100, step: 1 },
  { name: "qwen_persona_sticker_repo_dir", label: "表情仓库目录", section: "stickers", control: "text" },
  { name: "qwen_persona_sticker_repo_url", label: "表情仓库 URL", section: "stickers", control: "text" },
  { name: "qwen_persona_max_reply_stickers", label: "单次表情上限", section: "stickers", control: "number", min: 0, max: 10, step: 1 },
  { name: "qwen_persona_text_sticker_reply_probability", label: "文本带表情概率", section: "stickers", control: "number", min: 0, max: 1, step: 0.01 },
  { name: "qwen_persona_media_image_fallback_text", label: "图片占位文本", section: "stickers", control: "text" },
  { name: "qwen_persona_media_sticker_fallback_text", label: "表情占位文本", section: "stickers", control: "text" },

  { name: "qwen_persona_admin_enabled", label: "后台开关", section: "system", control: "switch" },
  { name: "qwen_persona_admin_host", label: "监听地址", section: "system", control: "text" },
  { name: "qwen_persona_admin_port", label: "监听端口", section: "system", control: "number", min: 1, max: 65535, step: 1 },
  { name: "qwen_persona_admin_token", label: "访问令牌", section: "system", control: "password" },
  { name: "qwen_persona_admin_config_file", label: "覆盖配置文件", section: "system", control: "text" },
  { name: "qwen_persona_data_dir", label: "数据目录", section: "system", control: "text" },
]

export const RULE_FIELDS: FieldSchema[] = [
  { name: "qwen_persona_command_prefixes", label: "命令前缀", description: "每行一个前缀。", control: "tokens" },
  { name: "qwen_persona_command_regexes", label: "命令正则", description: "每行一条正则。", control: "tokens" },
  { name: "qwen_persona_auto_read_picmenu_config", label: "读取菜单配置", control: "switch" },
  { name: "qwen_persona_picmenu_config_keys", label: "菜单配置键", control: "tokens" },
  { name: "qwen_persona_picmenu_command_keys", label: "菜单命令键", control: "tokens" },
  { name: "qwen_persona_blacklist_words", label: "黑名单词", control: "tokens" },
  { name: "qwen_persona_blacklist_regexes", label: "黑名单正则", control: "tokens" },
  { name: "qwen_persona_blacklist_user_ids", label: "黑名单用户", control: "tokens" },
  { name: "qwen_persona_blacklist_group_ids", label: "黑名单群组", control: "tokens" },
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
