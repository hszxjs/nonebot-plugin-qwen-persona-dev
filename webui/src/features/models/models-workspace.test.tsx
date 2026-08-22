import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ModelsWorkspace } from "@/features/models/models-workspace"
import type { ModelManageOperation, ModelsManagePayload } from "@/lib/types"

const mocks = vi.hoisted(() => ({
  useModelsManage: vi.fn(),
  startModelDownload: vi.fn(),
  cancelModelDownload: vi.fn(),
  deleteLocalModel: vi.fn(),
  modelDownloadEventsUrl: vi.fn(),
  mutateModels: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    useModelsManage: mocks.useModelsManage,
    startModelDownload: mocks.startModelDownload,
    cancelModelDownload: mocks.cancelModelDownload,
    deleteLocalModel: mocks.deleteLocalModel,
    modelDownloadEventsUrl: mocks.modelDownloadEventsUrl,
  }
})

const eventsUrl = "/api/models/manage/download/events?token=test-token"

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  closed = false
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  close() {
    this.closed = true
  }
}

const runningOperation: ModelManageOperation = {
  type: "model_download",
  phase: "downloading",
  state: "running",
  message: "downloading Qwen/Qwen3-4B",
  percent: 42,
  downloaded_bytes: 42,
  total_bytes: 100,
  error: "",
  target_model: "Qwen/Qwen3-4B",
  local_path: "",
  started_at: "2026-08-23T00:00:00Z",
  finished_at: null,
}

const overview: ModelsManagePayload = {
  local: [
    {
      name: "Qwen/Qwen3-4B",
      path: "/models/Qwen/Qwen3-4B",
      size_bytes: 8 * 1024 * 1024 * 1024,
      size_display: "8.0 GiB",
    },
  ],
  presets: [
    {
      value: "Qwen/Qwen3-8B",
      label: "Qwen3 8B",
      description: "本地对话基座",
    },
  ],
  total_size_bytes: 8 * 1024 * 1024 * 1024,
  operation: runningOperation,
}

function renderWorkspace(payload: ModelsManagePayload = overview) {
  mocks.useModelsManage.mockReturnValue({
    data: payload,
    error: null,
    isLoading: false,
    mutate: mocks.mutateModels,
  })
  mocks.modelDownloadEventsUrl.mockReturnValue(eventsUrl)
  return render(<ModelsWorkspace />)
}

function emitFrame(source: MockEventSource, frame: unknown) {
  act(() => {
    source.onmessage?.(new MessageEvent("message", { data: JSON.stringify(frame) }))
  })
}

describe("models workspace", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    MockEventSource.instances.length = 0
    vi.unstubAllGlobals()
  })

  it("renders disk overview table and attaches SSE for a running download", () => {
    vi.stubGlobal("EventSource", MockEventSource)
    renderWorkspace()

    expect(screen.getByRole("heading", { name: "模型管理" })).toBeInTheDocument()
    expect(screen.getByRole("table")).toBeInTheDocument()
    expect(screen.getByText("Qwen/Qwen3-4B")).toBeInTheDocument()
    expect(screen.getAllByText("8.0 GiB").length).toBeGreaterThan(0)
    expect(screen.getByText("/models/Qwen/Qwen3-4B")).toBeInTheDocument()
    expect(screen.getByText("42%")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "取消下载" })).toBeInTheDocument()

    expect(MockEventSource.instances).toHaveLength(1)
    expect(MockEventSource.instances[0].url).toBe(eventsUrl)
  })

  it("updates progress from SSE frames and closes the stream on terminal state", async () => {
    vi.stubGlobal("EventSource", MockEventSource)
    renderWorkspace()
    const source = MockEventSource.instances[0]

    emitFrame(source, { ...runningOperation, percent: 77, downloaded_bytes: 77 })
    expect(await screen.findByText("77%")).toBeInTheDocument()

    emitFrame(source, {
      ...runningOperation,
      state: "success",
      phase: "done",
      message: "download complete",
      percent: 100,
      finished_at: "2026-08-23T00:01:00Z",
    })

    await waitFor(() => expect(source.closed).toBe(true))
    await waitFor(() => expect(mocks.mutateModels).toHaveBeenCalled())
    expect(await screen.findByText("成功")).toBeInTheDocument()
    expect(screen.getByText("download complete")).toBeInTheDocument()
  })

  it("starts a download with the entered model ref and attaches the stream", async () => {
    vi.stubGlobal("EventSource", MockEventSource)
    renderWorkspace({ ...overview, operation: null })
    expect(MockEventSource.instances).toHaveLength(0)

    mocks.startModelDownload.mockResolvedValue({ operation: runningOperation })
    fireEvent.change(screen.getByLabelText("模型引用"), {
      target: { value: "Qwen/Qwen3-4B" },
    })
    fireEvent.click(screen.getByRole("button", { name: /开始下载/ }))

    await waitFor(() =>
      expect(mocks.startModelDownload).toHaveBeenCalledWith("Qwen/Qwen3-4B")
    )
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    expect(await screen.findByText("42%")).toBeInTheDocument()
  })

  it("keeps the stream closed when the download request is rejected", async () => {
    vi.stubGlobal("EventSource", MockEventSource)
    renderWorkspace({ ...overview, operation: null })

    mocks.startModelDownload.mockRejectedValue(new Error("another download is running"))
    fireEvent.change(screen.getByLabelText("模型引用"), {
      target: { value: "Qwen/Qwen3-4B" },
    })
    fireEvent.click(screen.getByRole("button", { name: /开始下载/ }))

    await waitFor(() =>
      expect(mocks.startModelDownload).toHaveBeenCalledWith("Qwen/Qwen3-4B")
    )
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it("deletes a local model after confirmation", async () => {
    renderWorkspace({ ...overview, operation: null })

    fireEvent.click(screen.getByRole("button", { name: "删除" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/删除模型「Qwen\/Qwen3-4B」？/)
    ).toBeInTheDocument()

    mocks.deleteLocalModel.mockResolvedValue({
      deleted: "Qwen/Qwen3-4B",
      freed_bytes: 8 * 1024 * 1024 * 1024,
      freed_display: "8.0 GiB",
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "删除" }))

    await waitFor(() =>
      expect(mocks.deleteLocalModel).toHaveBeenCalledWith("Qwen/Qwen3-4B")
    )
    await waitFor(() => expect(mocks.mutateModels).toHaveBeenCalled())
  })
})
