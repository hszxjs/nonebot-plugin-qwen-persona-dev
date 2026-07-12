import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { DiagnosticsWorkspaceView } from "@/features/diagnostics/diagnostics-workspace"
import { OperationsDashboard } from "@/features/operations/operations-dashboard"
import { RulesWorkspaceView } from "@/features/rules/rules-workspace"
import { SettingsWorkspaceView } from "@/features/settings/settings-workspace"
import { StickersWorkspaceView } from "@/features/stickers/stickers-workspace"
import {
  RuntimeLogPanelView,
  TrainingLogPanelView,
} from "@/features/operations/operations-dashboard"
import type {
  DiagnosticsPayload,
  OperationsOverview,
  SettingsPayload,
  StickerGalleryPayload,
} from "@/lib/types"

const mocks = vi.hoisted(() => ({
  runRuntimeAction: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    runRuntimeAction: mocks.runRuntimeAction,
    useRuntimeLogs: () => ({ data: { logs: [] } }),
    useTrainingLogs: () => ({ data: { logs: [] } }),
  }
})

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

const settings: SettingsPayload = {
  override_file: "/tmp/qwen-persona.toml",
  values: {
    qwen_persona_admin_enabled: true,
    qwen_persona_command_prefixes: ["/", "!"],
    qwen_persona_command_regexes: ["^/help"],
    qwen_persona_blacklist_words: ["spam"],
  },
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
    expect(screen.getByRole("tablist")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "系统后台" })).toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: "规则过滤" })).not.toBeInTheDocument()
    expect(screen.getByText("后台开关")).toBeInTheDocument()
    expect(screen.getByTitle("qwen_persona_admin_enabled")).toBeInTheDocument()
    expect(screen.getByRole("switch", { name: "qwen_persona_admin_enabled" })).toBeInTheDocument()
    expect(screen.getByText("需要确认")).toBeInTheDocument()
    expect(screen.queryByText(/保存后可能触发 reload/)).not.toBeInTheDocument()
    expect(screen.queryByText(/变更后自动保存/)).not.toBeInTheDocument()
    expect(screen.queryByText(/高级配置项/)).not.toBeInTheDocument()
  })

  it("renders rules from frontend schema instead of backend sections", () => {
    render(
      <RulesWorkspaceView
        payload={settings}
        onSaveField={vi.fn()}
      />
    )

    expect(screen.getByRole("heading", { name: "规则过滤" })).toBeInTheDocument()
    expect(screen.getByText("命令前缀")).toBeInTheDocument()
    expect(screen.getAllByRole("textbox")[0].textContent).toBe("/\n!")
  })

  it("renders historical log panels with realtime controls", () => {
    render(
      <>
        <RuntimeLogPanelView
          paused={false}
          logs={[{ time: "now", stream: "system", message: "runtime ready" }]}
          onTogglePaused={vi.fn()}
        />
        <TrainingLogPanelView
          paused={false}
          logs={[{ time: "now", stream: "train", message: "training idle" }]}
          onTogglePaused={vi.fn()}
        />
      </>
    )

    expect(screen.getAllByText("历史日志").length).toBeGreaterThan(0)
    expect(screen.getAllByText("自动刷新").length).toBeGreaterThan(0)
    expect(screen.getByText("runtime ready")).toBeInTheDocument()
    expect(screen.getByText("training idle")).toBeInTheDocument()
  })

  it("renders diagnostics payload without entering the first-screen dashboard", () => {
    render(<DiagnosticsWorkspaceView payload={diagnostics} />)

    expect(screen.getByRole("heading", { name: "诊断" })).toBeInTheDocument()
    expect(screen.getAllByText("fastapi-vite").length).toBeGreaterThan(0)
    expect(screen.getByText("/pkg/webui_static")).toBeInTheDocument()
  })

  it("runs runtime load action and exposes the operation message", async () => {
    const overview: OperationsOverview = {
      runtime: {
        runtime: "mlx_vlm",
        state: "not_loaded",
        status: "not_loaded",
        healthy: false,
        model: "target-model",
        loaded_model: "",
        adapter_path: "",
        loaded_adapter_path: "",
        last_error: "",
        operation: null,
        control: { can_load: true, can_unload: false, can_reload: true },
        metrics: { last: null, history: [] },
        model_presets: [],
      },
      learning: {},
      training: {
        summary: {},
        operation: { running: false, available: true, progress: null },
        config: {},
      },
      charts: {},
      inference: {},
      inference_events: { items: [], total: 0, limit: 300, offset: 0 },
      errors: [],
    }
    mocks.runRuntimeAction.mockResolvedValueOnce({
      action: "load",
      message: "load model 'target-model'",
      runtime: { status: "loaded" },
    })

    render(<OperationsDashboard overview={overview} />)
    fireEvent.click(screen.getByRole("button", { name: /加载/ }))

    await waitFor(() => expect(mocks.runRuntimeAction).toHaveBeenCalledWith("load"))
    expect(await screen.findByText("load model 'target-model'")).toBeInTheDocument()
  })
})
