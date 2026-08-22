import ky from "ky"
import useSWR from "swr"

import type {
  ConfigValue,
  DiagnosticsPayload,
  LogsPayload,
  ModelsPayload,
  OperationsOverview,
  RuntimeAction,
  RuntimeActionResult,
  SettingsPayload,
  StickerEventsPayload,
  StickerGalleryPayload,
  TrainingActionResult,
} from "@/lib/types"

const tokenStorageKey = "qwen-persona-admin-token"

export function getAdminToken() {
  return window.localStorage.getItem(tokenStorageKey) || ""
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
