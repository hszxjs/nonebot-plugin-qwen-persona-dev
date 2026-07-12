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
