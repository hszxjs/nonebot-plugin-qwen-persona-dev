import ky from "ky"
import useSWR from "swr"

import type {
  ConfigValue,
  DiagnosticsPayload,
  OperationsOverview,
  SettingsSectionsPayload,
  StickerGalleryPayload,
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

export function updateStickerSummary(key: string, summary: string) {
  return api.patch(`stickers/${encodeURIComponent(key)}`, { json: { summary } }).json()
}

export function deleteStickers(keys: string[]) {
  return api.delete("stickers", { json: { keys } }).json()
}

export function useSettingsSections() {
  return useSWR<SettingsSectionsPayload>("settings/sections", fetchJson)
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
