export type MetricSnapshot = {
  last: Record<string, unknown> | null
  history: Record<string, unknown>[]
}

export type RuntimeOperation = {
  id: string
  type: string
  state: string
  phase: string
  message: string
  percent: number | null
  downloaded_bytes: number | null
  total_bytes: number | null
  started_at: string
  finished_at: string | null
  error: string
  target_model?: string
  local_path?: string
  previous_model?: string
}

export type RuntimeSummary = {
  runtime: string
  state: string
  status: string
  healthy: boolean
  model: string
  loaded_model: string
  adapter_path: string
  loaded_adapter_path: string
  last_error: string
  operation: RuntimeOperation | null
  control: {
    can_load?: boolean
    can_unload?: boolean
    can_reload?: boolean
    reason?: string
  }
  metrics: MetricSnapshot
  model_presets: { value: string; label: string; description?: string }[]
}

export type LearningSummary = {
  learned_messages?: number
  trainable_samples?: number
  pending_training_samples?: number
  stickers?: number
  today_samples?: number
  last_24h_samples?: number
  groups?: number
  users?: number
  external_corpus_samples?: number
  media_assets?: number
  first_sample_at?: number | null
  last_sample_at?: number | null
  avg_training_chars?: number | null
}

export type TrainingProgress = {
  stage: string
  status: string
  reason: string
  sample_count: number
  dataset_path: string
  output_dir: string
  log_tail: string
  started_at: number | null
  finished_at: number | null
}

export type TrainingSummary = {
  summary: Record<string, unknown>
  operation: {
    running: boolean
    available: boolean
    progress: TrainingProgress | null
  }
  config: Record<string, unknown>
}

export type InferenceEvent = {
  event_id: string
  time: string
  status: string
  trigger_reason: string
  model: string
  adapter: string
  latency_seconds: number | null
  tps: number | null
  media_count: number
  response_chars: number
  error_summary: string
  input_preview: string
  output_preview: string
  group_id: string | null
  user_id: string | null
}

export type InferenceEventsPayload = {
  items: InferenceEvent[]
  total: number
  limit: number
  offset: number
}

export type StickerClassification = {
  label: string
  reason: string
  sendable: boolean
  confidence?: number
}

export type OperationsOverview = {
  runtime: RuntimeSummary
  learning: LearningSummary
  training: TrainingSummary
  charts: Record<string, unknown>
  inference: Record<string, unknown>
  inference_events: InferenceEventsPayload
  errors: string[]
}

export type StickerItem = {
  key: string
  type: string
  summary: string
  uri: string
  data: Record<string, unknown>
  preview_url: string | null
  sendable: boolean
  classification: StickerClassification | null
  group_id: string | null
  user_id: string | null
  first_seen: number | null
  last_seen: number | null
  count: number
}

export type StickerGalleryPayload = {
  items: StickerItem[]
  total: number
  limit: number
  offset: number
  query: string
  type: string
  type_counts: Record<string, number>
}

export type StickerEvent = {
  event_id: string
  time: string
  created_at: number
  event_type: string
  status: string
  key: string | null
  group_id: string | null
  user_id: string | null
  message: string
  details: Record<string, unknown>
}

export type StickerEventsPayload = {
  items: StickerEvent[]
  total: number
  limit: number
  offset: number
}

export type ConfigValue =
  | boolean
  | number
  | string
  | string[]
  | Record<string, unknown>
  | null

export type SettingsField = {
  name: string
  kind: "bool" | "int" | "float" | "json" | "str" | string
  value: ConfigValue
  reload_sensitive: boolean
  danger: boolean
}

export type SettingsPayload = {
  values: Record<string, ConfigValue>
  override_file: string
}

export type DiagnosticsPayload = {
  runtime: Record<string, unknown>
  settings: SettingsPayload
  static: {
    root: string
    built: boolean
    index: string
  }
  build: Record<string, unknown>
}

export type LogEntry = {
  time: string
  stream: string
  message: string
}

export type TimelineLogSource = "message" | "inference" | "sticker"

export type MessageLogEntry = {
  source: "message"
  created_at: number
  time: string
  group_id: string | null
  user_id: string | null
  status?: string
  trigger_message_id?: number | string | null
  reply_message_id?: number | string | null
  reply_kind?: string
  trigger_reason?: string
  decision_detail?: Record<string, unknown> | null
  role_card_id?: string | null
  role_card_name?: string | null
  provider?: string | null
  model?: string | null
  latency_seconds?: number | null
  sticker_keys?: string[]
  input_preview?: string
  output_preview?: string
}

export type InferenceLogEntry = {
  source: "inference"
  created_at: number
  time: string
  group_id: string | null
  user_id: string | null
  status?: string
  event_id: string
  trigger_reason: string
  model: string
  adapter: string
  latency_seconds: number | null
  tps: number | null
  media_count: number
  response_chars: number
  error_summary: string
  input_preview: string
  output_preview: string
}

export type StickerLogEntry = {
  source: "sticker"
  created_at: number
  time: string
  group_id: string | null
  user_id: string | null
  status?: string
  event_id: string
  event_type: string
  key: string | null
  message: string
}

export type TimelineLogEntry = MessageLogEntry | InferenceLogEntry | StickerLogEntry

export type TimelineLogsPayload = {
  items: TimelineLogEntry[]
  total: number
}

export type TimelineLogFrame = {
  type: TimelineLogSource
  item: TimelineLogEntry
}

export type BlacklistMatch = {
  kind: "word" | "regex"
  value: string
}

export type BlacklistTestResult = {
  text: string
  matched: boolean
  matches: BlacklistMatch[]
  words: string[]
  regexes: string[]
}

export type LogsPayload = {
  logs: LogEntry[]
  path?: string
  exists?: boolean
}

export type RuntimeAction = "load" | "reload" | "unload"

export type RuntimeActionResult = {
  action: RuntimeAction
  message: string
  runtime: RuntimeSummary
  confirmation_required?: boolean
  error?: string
}

export type TrainingActionResult = {
  action: "run" | "run_full_external"
  message: string
  training: TrainingSummary["operation"]
  error?: string
}

export type ModelsPayload = {
  models: string[]
  source: "api" | "fallback"
}

export type RoleCardType = "bot_persona" | "member_mimic"

export type RoleCard = {
  id: string
  role_type: RoleCardType
  name: string
  system_prompt: string
  style_tags: string[]
  tone_examples: string[]
  catchphrases: string[]
  topic_preferences: string[]
  emotion_baseline: string
  relation_profile: string
  allowed_tools: string[]
  avatar: string
  enabled: boolean
  target_user_id: string | null
  created_at: number
  updated_at: number
  bindings: string[]
}

export type RoleCardPayload = {
  name: string
  role_type?: RoleCardType
  system_prompt?: string
  style_tags?: string[]
  tone_examples?: string[]
  allowed_tools?: string[]
  avatar?: string
  enabled?: boolean
  target_user_id?: string | null
}

export type RoleCardsPayload = {
  items: RoleCard[]
  total: number
  role_types: string[]
}

export type RoleCardBinding = {
  session_key: string
  role_card_id: string
  bound_at: number
}

export type RoleCardBindResult = {
  binding: RoleCardBinding
}

export type RoleCardUnbindResult = {
  unbound: 0 | 1
}

export type ToolInfo = {
  name: string
  description: string
  parameters: Record<string, unknown>
  permission_tags: string[]
  executable: boolean
}

export type ToolsPayload = {
  items: ToolInfo[]
  total: number
}

export type ProviderSpec = {
  name: string
  provider_type: "local" | "remote" | "unknown" | string
  display_name?: string
  docs_url?: string
}

export type BackendConfig = {
  id: string
  backend_type: string
  display_name: string
  settings: Record<string, unknown>
  enabled: boolean
  priority: number
  created_at: number
  updated_at: number
  spec: ProviderSpec
}

export type BackendsPayload = {
  items: BackendConfig[]
  total: number
  available_types: ProviderSpec[]
}

export type BackendPayload = {
  backend_type: string
  display_name: string
  settings: Record<string, unknown>
  enabled?: boolean
  priority?: number
}

export type BackendHealthCheckResult = {
  id: string
  healthy: boolean
  status: {
    healthy?: boolean
    model?: string
    last_error?: string | null
  } & Record<string, unknown>
}

export type ProviderRole = "chat" | "vision" | "data_cleaning"

export type RoleBinding = {
  role: ProviderRole | string
  provider_id: string
  model: string
  fallback_provider_id: string | null
  updated_at: number
}

export type ProviderBindingsPayload = {
  items: RoleBinding[]
  roles: ProviderRole[]
}

export type ProviderBindingPayload = {
  provider_id: string
  model?: string
  fallback_provider_id?: string | null
}

export type LocalModel = {
  name: string
  path: string
  size_bytes: number
  size_display: string
}

export type ModelPreset = {
  value: string
  label: string
  description?: string
}

export type ModelManageOperationState =
  | "running"
  | "success"
  | "failed"
  | "cancelled"
  | "idle"
  | string

export type ModelManageOperation = {
  type: string
  phase: string
  state: ModelManageOperationState
  message: string
  percent: number | null
  downloaded_bytes: number | null
  total_bytes: number | null
  error: string
  target_model?: string
  local_path?: string
  started_at: string
  finished_at: string | null
}

export type ModelsManagePayload = {
  local: LocalModel[]
  presets: ModelPreset[]
  total_size_bytes: number
  operation: ModelManageOperation | null
}

export type ModelDownloadResult = {
  operation: ModelManageOperation
}

export type ModelCancelResult = {
  cancelling: boolean
}

export type ModelDeleteResult = {
  deleted: string
  freed_bytes: number
  freed_display: string
}

export type KpiReplyReasonStat = {
  reason: string
  count: number
}

export type KpiProviderStat = {
  model: string
  total: number
  avg_latency_seconds: number
  avg_tps: number
}

export type KpiToolItem = {
  name: string
  total: number
  ok: number
  success_rate: number
  avg_latency_seconds: number
}

export type KpiToolStats = {
  total_calls: number
  overall_success_rate: number | null
  items: KpiToolItem[]
}

export type KpiBackendHealth = {
  healthy: boolean | null
  model: string
  backend: string
}

export type KpiPayload = {
  window_days: number
  total_replies: number
  reply_reasons: KpiReplyReasonStat[]
  providers: KpiProviderStat[]
  active_sessions: number
  learned_messages: number
  tools: KpiToolStats
  backend: KpiBackendHealth
}

export type SessionSummary = {
  session_key: string
  count: number
  latest: MessageLogEntry
}

export type SessionsPayload = {
  items: SessionSummary[]
  total: number
}

export type SessionPokeResult = {
  group_id: string
  sent: boolean
  reply_message_id: number | string
}

export type LongTermMemory = {
  id: number
  user_id: string
  kind: string
  content: string
  importance: number
  created_at: number
  last_seen_at: number
  decayed: boolean | number
}

export type MemoriesPayload = {
  items: LongTermMemory[]
  total: number
}

export type MemoryDeleteResult = {
  deleted: number
}

export type MemoryExtractResult = {
  written: string[]
  count: number
}

export type MemoryDecayResult = {
  decayed: number
  total: number
}

export type LoraAdapter = {
  id: string
  role_card_id: string
  source_language: string
  version: number
  path: string
  note: string
  active: boolean
  created_at: number
}

export type LoraAdaptersPayload = {
  items: LoraAdapter[]
  total: number
}

export type LoraAdapterPayload = {
  role_card_id: string
  source_language: string
  path: string
  note?: string
  activate?: boolean
}

export type LoraAdapterDeleteResult = {
  deleted: number
}
