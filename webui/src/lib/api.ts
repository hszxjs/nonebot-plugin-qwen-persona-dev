import ky, { HTTPError } from "ky"
import useSWR from "swr"

import type {
  BackendHealthCheckResult,
  BackendPayload,
  BackendsPayload,
  BlacklistTestResult,
  ConfigValue,
  DiagnosticsPayload,
  KpiPayload,
  LogsPayload,
  LoraAdapter,
  LoraAdapterDeleteResult,
  LoraAdapterPayload,
  LoraAdaptersPayload,
  MemoryDecayResult,
  MemoryDeleteResult,
  MemoryExtractResult,
  MemoriesPayload,
  ModelCancelResult,
  ModelDeleteResult,
  ModelDownloadResult,
  ModelsManagePayload,
  ModelsPayload,
  OperationsOverview,
  ProviderBindingPayload,
  ProviderBindingsPayload,
  RoleCard,
  RoleCardBindResult,
  RoleCardPayload,
  RoleCardUnbindResult,
  RoleCardsPayload,
  RuntimeAction,
  RuntimeActionResult,
  SessionPokeResult,
  SessionsPayload,
  SettingsPayload,
  StickerEventsPayload,
  StickerGalleryPayload,
  TimelineLogsPayload,
  ToolsPayload,
  TrainingActionResult,
} from "@/lib/types"

const tokenStorageKey = "ai-companion-admin-token"
const legacyTokenStorageKey = "qwen-persona-admin-token"

export function getAdminToken() {
  const token = window.localStorage.getItem(tokenStorageKey)
  if (token) return token
  // 旧版本 localStorage key 迁移：读到旧值则搬移到新 key
  const legacy = window.localStorage.getItem(legacyTokenStorageKey)
  if (legacy) {
    window.localStorage.setItem(tokenStorageKey, legacy)
    window.localStorage.removeItem(legacyTokenStorageKey)
    return legacy
  }
  return ""
}

export function setAdminToken(token: string) {
  window.localStorage.setItem(tokenStorageKey, token)
}

export const api = ky.create({
  prefix: "/api",
  timeout: 30_000,
  hooks: {
    beforeRequest: [
      ({ request }) => {
        const token = getAdminToken()
        if (token) {
          request.headers.set("X-Admin-Token", token)
        }
      },
    ],
  },
})

const fetchJson = <T>(path: string) => api.get(path).json<T>()

export function useOperationsOverview() {
  return useSWR<OperationsOverview>(
    "operations/overview",
    fetchJson,
    { refreshInterval: 5_000 }
  )
}

export function useStickers() {
  return useSWR<StickerGalleryPayload>("stickers", fetchJson)
}

export function useStickerEvents(open: boolean) {
  return useSWR<StickerEventsPayload>(
    open ? "stickers/events?limit=100" : null,
    fetchJson,
    { refreshInterval: open ? 5_000 : 0 }
  )
}

export function updateStickerSummary(key: string, summary: string) {
  return api.patch(`stickers/${encodeURIComponent(key)}`, { json: { summary } }).json()
}

export function deleteStickers(keys: string[]) {
  return api.delete("stickers", { json: { keys } }).json()
}

export function useSettings() {
  return useSWR<SettingsPayload>("settings", fetchJson)
}

export function useModels() {
  return useSWR<ModelsPayload>("models", fetchJson, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  })
}

export function saveSettingField(
  fieldName: string,
  value: ConfigValue,
  confirmed = false
) {
  return api
    .patch(`settings/fields/${encodeURIComponent(fieldName)}`, {
      json: { value, confirmed },
    })
    .json()
}

export function useDiagnostics() {
  return useSWR<DiagnosticsPayload>("diagnostics", fetchJson)
}

export function useRuntimeLogs(paused: boolean) {
  return useSWR<LogsPayload>(
    "runtime/logs?tail=300",
    fetchJson,
    { refreshInterval: paused ? 0 : 3_000 }
  )
}

export function usePluginLogs(paused: boolean) {
  return useSWR<LogsPayload>(
    "plugin/logs?tail=300",
    fetchJson,
    { refreshInterval: paused ? 0 : 3_000 }
  )
}

export function useTrainingLogs(paused: boolean) {
  return useSWR<LogsPayload>(
    "training/logs?tail=300",
    fetchJson,
    { refreshInterval: paused ? 0 : 3_000 }
  )
}

export function runRuntimeAction(action: RuntimeAction, confirmed = false) {
  return api
    .post(`runtime/${action}`, { json: { confirmed } })
    .json<RuntimeActionResult>()
}

export function runTrainingAction(fullExternalCorpus = false) {
  return api
    .post("training/run", {
      json: fullExternalCorpus
        ? { include_external_corpus: true, full_external_corpus: true }
        : {},
    })
    .json<TrainingActionResult>()
}

export async function apiErrorMessage(error: unknown): Promise<string> {
  if (error instanceof HTTPError) {
    try {
      const body = (await error.response.clone().json()) as {
        error?: string
        detail?: string
      }
      if (body?.error) return body.error
      if (body?.detail) return body.detail
    } catch {
      // 响应体不是 JSON 时退回默认 message
    }
  }
  return error instanceof Error ? error.message : String(error)
}

export function useRoleCards() {
  return useSWR<RoleCardsPayload>("role-cards", fetchJson)
}

export function useTools() {
  return useSWR<ToolsPayload>("tools", fetchJson, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  })
}

export function createRoleCard(payload: RoleCardPayload) {
  return api.post("role-cards", { json: payload }).json<{ item: RoleCard }>()
}

export function updateRoleCard(id: string, payload: Partial<RoleCardPayload>) {
  return api
    .patch(`role-cards/${encodeURIComponent(id)}`, { json: payload })
    .json<{ item: RoleCard }>()
}

export function deleteRoleCard(id: string) {
  return api.delete(`role-cards/${encodeURIComponent(id)}`).json<{ deleted: number }>()
}

export function cloneRoleCard(id: string, name: string) {
  return api
    .post(`role-cards/${encodeURIComponent(id)}/clone`, { json: { name } })
    .json<{ item: RoleCard }>()
}

export function bindRoleCard(id: string, body: { group_id?: string | null; user_id?: string | null }) {
  return api
    .post(`role-cards/${encodeURIComponent(id)}/bind`, { json: body })
    .json<RoleCardBindResult>()
}

export function unbindRoleCard(body: { group_id?: string; user_id?: string }) {
  return api.delete("role-cards/bindings", { json: body }).json<RoleCardUnbindResult>()
}

export function generateMemberMimic(userId: string, groupId?: string) {
  return api
    .post("role-cards/member-mimic", {
      json: groupId ? { user_id: userId, group_id: groupId } : { user_id: userId },
    })
    .json<{ item: RoleCard }>()
}

export function refreshMemberMimicStyle(id: string) {
  return api
    .post(`role-cards/${encodeURIComponent(id)}/refresh-style`)
    .json<{ item: RoleCard }>()
}

export function useBackends() {
  return useSWR<BackendsPayload>("backends", fetchJson)
}

export function createBackend(payload: BackendPayload) {
  return api.post("backends", { json: payload }).json<{ item: BackendsPayload["items"][number] }>()
}

export function updateBackend(id: string, payload: BackendPayload) {
  return api
    .patch(`backends/${encodeURIComponent(id)}`, { json: payload })
    .json<{ item: BackendsPayload["items"][number] }>()
}

export function deleteBackend(id: string) {
  return api.delete(`backends/${encodeURIComponent(id)}`).json<{ deleted: number }>()
}

export function healthCheckBackend(id: string) {
  return api
    .post(`backends/${encodeURIComponent(id)}/health-check`)
    .json<BackendHealthCheckResult>()
}

export function useProviderBindings() {
  return useSWR<ProviderBindingsPayload>("provider-bindings", fetchJson)
}

export function setProviderBinding(role: string, payload: ProviderBindingPayload) {
  return api
    .put(`provider-bindings/${encodeURIComponent(role)}`, { json: payload })
    .json<{ binding: ProviderBindingsPayload["items"][number] }>()
}

export function useModelsManage() {
  return useSWR<ModelsManagePayload>("models/manage", fetchJson)
}

export function startModelDownload(modelRef: string) {
  return api
    .post("models/manage/download", { json: { model_ref: modelRef } })
    .json<ModelDownloadResult>()
}

export function cancelModelDownload() {
  return api.post("models/manage/download/cancel").json<ModelCancelResult>()
}

export function deleteLocalModel(modelRef: string) {
  return api
    .delete("models/manage", { searchParams: { model_ref: modelRef } })
    .json<ModelDeleteResult>()
}

export function modelDownloadEventsUrl() {
  const token = getAdminToken()
  return `/api/models/manage/download/events${token ? `?token=${encodeURIComponent(token)}` : ""}`
}

export function useLogs(kinds: string, groupId: string, limit = 100) {
  const params = new URLSearchParams({
    kinds: kinds || "all",
    limit: String(limit),
    offset: "0",
  })
  const trimmedGroup = groupId.trim()
  if (trimmedGroup) params.set("group_id", trimmedGroup)
  return useSWR<TimelineLogsPayload>(`logs?${params.toString()}`, fetchJson, {
    revalidateOnFocus: false,
    revalidateIfStale: false,
  })
}

export function logsStreamUrl(maxSeconds = 0) {
  const token = getAdminToken()
  const search = new URLSearchParams({ max_seconds: String(maxSeconds) })
  if (token) search.set("token", token)
  return `/api/logs/stream?${search.toString()}`
}

export function testBlacklist(text: string) {
  return api.post("blacklist/test", { json: { text } }).json<BlacklistTestResult>()
}

export function useKpi(days = 7) {
  return useSWR<KpiPayload>(`dashboard/kpi?days=${days}`, fetchJson, {
    revalidateOnFocus: false,
    dedupingInterval: 30_000,
  })
}

export function useSessions(limit = 50) {
  return useSWR<SessionsPayload>(`sessions?limit=${limit}`, fetchJson, {
    revalidateOnFocus: false,
  })
}

export function pokeSession(groupId: string) {
  return api
    .post("sessions/poke", { json: { group_id: groupId } })
    .json<SessionPokeResult>()
}

// 会话下钻：群会话直接走 group_id 过滤；私聊会话（private:xxx）没有 group_id，
// 拉取最近的 message 日志后由组件按 user_id 过滤。
export function useSessionLogs(sessionKey: string, enabled: boolean) {
  const isPrivate = sessionKey.startsWith("private:")
  const params = new URLSearchParams({ kinds: "message", limit: "100", offset: "0" })
  if (!isPrivate) params.set("group_id", sessionKey)
  return useSWR<TimelineLogsPayload>(
    enabled ? `logs?${params.toString()}` : null,
    fetchJson,
    { revalidateOnFocus: false, revalidateIfStale: false }
  )
}

export function useMemories(userId: string, includeDecayed: boolean, limit = 200) {
  const params = new URLSearchParams({
    include_decayed: String(includeDecayed),
    limit: String(limit),
  })
  const trimmed = userId.trim()
  if (trimmed) params.set("user_id", trimmed)
  return useSWR<MemoriesPayload>(`memories?${params.toString()}`, fetchJson, {
    revalidateOnFocus: false,
  })
}

export function deleteMemory(id: number) {
  return api.delete(`memories/${id}`).json<MemoryDeleteResult>()
}

export function extractMemories(userId: string) {
  return api
    .post("memories/extract", { json: { user_id: userId } })
    .json<MemoryExtractResult>()
}

export function decayMemories(halfLifeDays = 30) {
  return api
    .post("memories/decay", { searchParams: { half_life_days: String(halfLifeDays) } })
    .json<MemoryDecayResult>()
}

export function useLoraAdapters(roleCardId = "") {
  const key = roleCardId
    ? `lora-adapters?role_card_id=${encodeURIComponent(roleCardId)}`
    : "lora-adapters"
  return useSWR<LoraAdaptersPayload>(key, fetchJson, {
    revalidateOnFocus: false,
  })
}

export function registerLoraAdapter(payload: LoraAdapterPayload) {
  return api.post("lora-adapters", { json: payload }).json<{ item: LoraAdapter }>()
}

export function activateLoraAdapter(id: string) {
  return api
    .post(`lora-adapters/${encodeURIComponent(id)}/activate`)
    .json<{ item: LoraAdapter }>()
}

export function rollbackLoraAdapter(roleCardId: string, sourceLanguage: string) {
  return api
    .post("lora-adapters/rollback", {
      json: { role_card_id: roleCardId, source_language: sourceLanguage },
    })
    .json<{ item: LoraAdapter }>()
}

export function deleteLoraAdapter(id: string) {
  return api
    .delete(`lora-adapters/${encodeURIComponent(id)}`)
    .json<LoraAdapterDeleteResult>()
}
