import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { DiagnosticsWorkspaceView } from "@/features/diagnostics/diagnostics-workspace"
import { SettingsWorkspaceView } from "@/features/settings/settings-workspace"
import { StickersWorkspaceView } from "@/features/stickers/stickers-workspace"
import type {
  DiagnosticsPayload,
  SettingsSectionsPayload,
  StickerGalleryPayload,
} from "@/lib/types"

const stickers: StickerGalleryPayload = {
  items: [
    {
      key: "sticker-1",
      type: "mface",
      summary: "[肿么这样]",
      uri: "https://example.test/sticker.png",
      data: { url: "https://example.test/sticker.png" },
      group_id: "10",
      user_id: "20",
      first_seen: 100,
      last_seen: 200,
      count: 3,
    },
  ],
  total: 1,
  limit: 80,
  offset: 0,
  query: "",
  type: "",
  type_counts: { mface: 1 },
}

const settings: SettingsSectionsPayload = {
  override_file: "/tmp/qwen-persona.toml",
  sections: [
    {
      key: "system",
      title: "系统后台",
      fields: [
        {
          name: "qwen_persona_admin_enabled",
          kind: "bool",
          value: true,
          reload_sensitive: true,
          danger: false,
        },
      ],
    },
  ],
}

const diagnostics: DiagnosticsPayload = {
  runtime: { status: { state: "loaded" }, logs: [] },
  settings,
  static: { root: "/pkg/webui_static", built: true, index: "/pkg/webui_static/index.html" },
  build: { admin_server: "fastapi-vite" },
}

describe("admin workspaces", () => {
  afterEach(() => cleanup())

  it("renders stickers as card entries instead of a table", () => {
    render(
      <StickersWorkspaceView
        payload={stickers}
        selectedKeys={[]}
        onToggleSelected={vi.fn()}
        onEditSummary={vi.fn()}
        onDeleteSelected={vi.fn()}
      />
    )

    expect(screen.getByRole("heading", { name: "表情仓库" })).toBeInTheDocument()
    expect(screen.getByText("[肿么这样]")).toBeInTheDocument()
    expect(screen.getAllByText("mface").length).toBeGreaterThan(0)
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
  })

  it("renders settings by semantic section with reload confirmation cues", () => {
    render(<SettingsWorkspaceView payload={settings} onSaveField={vi.fn()} />)

    expect(screen.getByRole("heading", { name: "配置管理" })).toBeInTheDocument()
    expect(screen.getByText("系统后台")).toBeInTheDocument()
    expect(screen.getByRole("switch", { name: "qwen_persona_admin_enabled" })).toBeInTheDocument()
    expect(screen.getByText("需要确认")).toBeInTheDocument()
  })

  it("renders diagnostics payload without entering the first-screen dashboard", () => {
    render(<DiagnosticsWorkspaceView payload={diagnostics} />)

    expect(screen.getByRole("heading", { name: "诊断" })).toBeInTheDocument()
    expect(screen.getAllByText("fastapi-vite").length).toBeGreaterThan(0)
    expect(screen.getByText("/pkg/webui_static")).toBeInTheDocument()
  })
})
