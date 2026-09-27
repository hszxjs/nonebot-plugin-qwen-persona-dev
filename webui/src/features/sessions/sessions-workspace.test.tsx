import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OperationsDashboard } from "@/features/operations/operations-dashboard"
import { SessionsWorkspace } from "@/features/sessions/sessions-workspace"
import type {
  KpiPayload,
  LongTermMemory,
  LoraAdapter,
  MemoriesPayload,
  MessageLogEntry,
  OperationsOverview,
  RoleCard,
  RoleCardsPayload,
  SessionsPayload,
  TimelineLogsPayload,
} from "@/lib/types"

const mocks = vi.hoisted(() => ({
  useKpi: vi.fn(),
  useSessions: vi.fn(),
  useSessionLogs: vi.fn(),
  useMemories: vi.fn(),
  useLoraAdapters: vi.fn(),
  useRoleCards: vi.fn(),
  pokeSession: vi.fn(),
  extractMemories: vi.fn(),
  decayMemories: vi.fn(),
  deleteMemory: vi.fn(),
  registerLoraAdapter: vi.fn(),
  activateLoraAdapter: vi.fn(),
  rollbackLoraAdapter: vi.fn(),
  deleteLoraAdapter: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    useKpi: mocks.useKpi,
    useSessions: mocks.useSessions,
    useSessionLogs: mocks.useSessionLogs,
    useMemories: mocks.useMemories,
    useLoraAdapters: mocks.useLoraAdapters,
    useRoleCards: mocks.useRoleCards,
    pokeSession: mocks.pokeSession,
    extractMemories: mocks.extractMemories,
    decayMemories: mocks.decayMemories,
    deleteMemory: mocks.deleteMemory,
    registerLoraAdapter: mocks.registerLoraAdapter,
    activateLoraAdapter: mocks.activateLoraAdapter,
    rollbackLoraAdapter: mocks.rollbackLoraAdapter,
    deleteLoraAdapter: mocks.deleteLoraAdapter,
  }
})

const kpi: KpiPayload = {
  window_days: 7,
  total_replies: 123,
  reply_reasons: [
    { reason: "mention", count: 80 },
    { reason: "random", count: 43 },
  ],
  providers: [{ model: "deepseek-flash", total: 100, avg_latency_seconds: 1.234, avg_tps: 30.5 }],
  active_sessions: 5,
  learned_messages: 456,
  tools: {
    total_calls: 50,
    overall_success_rate: 0.87,
    items: [{ name: "search_web", total: 50, ok: 44, success_rate: 0.88, avg_latency_seconds: 0.9 }],
  },
  backend: { healthy: true, model: "deepseek-flash", backend: "llm_online" },
}

const overview: OperationsOverview = {
  runtime: {
    runtime: "llm_online",
    state: "loaded",
    status: "loaded",
    healthy: true,
    model: "target-model",
    loaded_model: "target-model",
    adapter_path: "",
    loaded_adapter_path: "",
    last_error: "",
    operation: null,
    control: { can_load: false, can_unload: false, can_reload: false },
    metrics: { last: null, history: [] },
    model_presets: [],
  },
  learning: { learned_messages: 10, trainable_samples: 8, pending_training_samples: 3 },
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

function buildRoleCard(): RoleCard {
  return {
    id: "card-1",
    role_type: "bot_persona",
    name: "毒舌管家",
    system_prompt: "",
    style_tags: [],
    tone_examples: [],
    catchphrases: [],
    topic_preferences: [],
    emotion_baseline: "",
    relation_profile: "",
    allowed_tools: [],
    avatar: "",
    enabled: true,
    target_user_id: null,
    created_at: 1783795200,
    updated_at: 1783795300,
    bindings: [],
  }
}

const roleCards: RoleCardsPayload = { items: [buildRoleCard()], total: 1, role_types: [] }

function buildAdapter(overrides: Partial<LoraAdapter> = {}): LoraAdapter {
  return {
    id: "a1",
    role_card_id: "card-1",
    source_language: "zh",
    version: 1,
    path: "/adapters/v1",
    note: "",
    active: false,
    created_at: 1783790000,
    ...overrides,
  }
}

const adapters = {
  items: [
    buildAdapter({ id: "a2", version: 2, path: "/adapters/v2", active: true, created_at: 1783795200 }),
    buildAdapter({ id: "a1", version: 1, path: "/adapters/v1", note: "旧版" }),
    buildAdapter({ id: "a3", source_language: "en", version: 1, path: "/adapters/en-v1" }),
  ],
  total: 3,
}

function buildMessage(overrides: Partial<MessageLogEntry> = {}): MessageLogEntry {
  return {
    source: "message",
    created_at: 1783795200,
    time: "2026-07-12T08:00:00Z",
    group_id: "g1",
    user_id: "u1",
    status: "success",
    trigger_reason: "mention",
    role_card_name: "毒舌管家",
    provider: "llm",
    model: "deepseek-flash",
    latency_seconds: 1.2,
    input_preview: "你好",
    output_preview: "你好呀",
    ...overrides,
  }
}

const sessions: SessionsPayload = {
  items: [
    {
      session_key: "g1",
      count: 3,
      latest: buildMessage(),
    },
    {
      session_key: "private:u1",
      count: 1,
      latest: buildMessage({
        group_id: null,
        user_id: "u1",
        trigger_reason: "private",
        role_card_name: null,
        latency_seconds: 0.8,
        time: "2026-07-12T07:00:00Z",
        created_at: 1783791600,
      }),
    },
  ],
  total: 2,
}

const sessionLogs: TimelineLogsPayload = {
  items: [
    buildMessage(),
    buildMessage({
      trigger_reason: "random",
      time: "2026-07-12T07:30:00Z",
      created_at: 1783793400,
      latency_seconds: 2.5,
    }),
  ],
  total: 2,
}

function buildMemory(overrides: Partial<LongTermMemory> = {}): LongTermMemory {
  return {
    id: 1,
    user_id: "100",
    kind: "profile",
    content: "喜欢猫",
    importance: 0.9,
    created_at: 1783795200,
    last_seen_at: 1783795300,
    decayed: 0,
    ...overrides,
  }
}

const memories: MemoriesPayload = {
  items: [
    buildMemory(),
    buildMemory({ id: 2, kind: "preference", content: "讨厌早八", importance: 0.3, decayed: 1 }),
  ],
  total: 2,
}

function renderOperationsDashboard() {
  mocks.useKpi.mockReturnValue({ data: kpi, error: null, isLoading: false })
  mocks.useLoraAdapters.mockReturnValue({ data: adapters, error: null, isLoading: false })
  mocks.useRoleCards.mockReturnValue({ data: roleCards, error: null, isLoading: false })
  return render(<OperationsDashboard overview={overview} />)
}

function renderSessionsWorkspace() {
  mocks.useSessions.mockReturnValue({ data: sessions, error: null, isLoading: false })
  mocks.useSessionLogs.mockReturnValue({ data: sessionLogs, error: null, isLoading: false })
  mocks.useMemories.mockReturnValue({ data: memories, error: null, isLoading: false })
  return render(<SessionsWorkspace />)
}

describe("phase5 kpi dashboard", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders kpi stat cards, reason distribution and provider latency table", () => {
    renderOperationsDashboard()

    const section = screen.getByTestId("kpi-section")
    expect(within(section).getByText("近 7 天")).toBeInTheDocument()
    const stats = within(section).getAllByTestId("kpi-stat")
    expect(stats).toHaveLength(5)
    expect(within(section).getByText("总回复数")).toBeInTheDocument()
    expect(within(section).getByText("123")).toBeInTheDocument()
    expect(within(section).getByText("活跃会话")).toBeInTheDocument()
    expect(within(section).getByText("5")).toBeInTheDocument()
    expect(within(section).getByText("学习消息量")).toBeInTheDocument()
    expect(within(section).getByText("456")).toBeInTheDocument()
    expect(within(section).getByText("工具成功率")).toBeInTheDocument()
    expect(within(section).getByText("87%")).toBeInTheDocument()
    expect(within(section).getByText("后端健康")).toBeInTheDocument()
    expect(within(section).getByText("健康")).toBeInTheDocument()
    expect(within(section).getAllByText("deepseek-flash").length).toBeGreaterThan(0)
    expect(within(section).getByText("llm_online")).toBeInTheDocument()

    const reasons = within(section).getByTestId("kpi-reasons")
    expect(within(reasons).getByText("mention")).toBeInTheDocument()
    expect(within(reasons).getByText("80")).toBeInTheDocument()
    expect(within(reasons).getByText("random")).toBeInTheDocument()

    const providers = within(section).getByTestId("kpi-providers")
    expect(within(providers).getByText("deepseek-flash")).toBeInTheDocument()
    expect(within(providers).getByText("1.23s")).toBeInTheDocument()
    expect(within(providers).getByText("30.50 tok/s")).toBeInTheDocument()
  })

  it("groups adapters by role card and language with active/activate actions", async () => {
    renderOperationsDashboard()

    const card = screen.getByTestId("adapters-card")
    const groups = within(card).getAllByTestId("adapter-group")
    expect(groups).toHaveLength(2)
    expect(groups[0]).toHaveAttribute("data-group", "card-1::zh")
    expect(groups[1]).toHaveAttribute("data-group", "card-1::en")

    expect(within(groups[0]).getByText("毒舌管家")).toBeInTheDocument()
    expect(within(groups[0]).getByText("zh")).toBeInTheDocument()
    expect(within(groups[1]).getByText("en")).toBeInTheDocument()

    const rows = within(card).getAllByTestId("adapter-version-row")
    expect(rows).toHaveLength(3)
    expect(within(rows[0]).getByText("v2")).toBeInTheDocument()
    expect(within(rows[0]).getByText("使用中")).toBeInTheDocument()
    expect(within(rows[0]).getByRole("button", { name: "激活" })).toBeDisabled()

    mocks.activateLoraAdapter.mockResolvedValue({ item: adapters.items[1] })
    fireEvent.click(within(rows[1]).getByRole("button", { name: "激活" }))
    await waitFor(() => expect(mocks.activateLoraAdapter).toHaveBeenCalledWith("a1"))
  })

  it("rolls back a group after confirmation and blocks single-version groups", async () => {
    renderOperationsDashboard()

    const groups = screen.getAllByTestId("adapter-group")
    expect(within(groups[0]).getByRole("button", { name: "回滚" })).toBeEnabled()
    expect(within(groups[1]).getByRole("button", { name: "回滚" })).toBeDisabled()

    mocks.rollbackLoraAdapter.mockResolvedValue({ item: adapters.items[1] })
    fireEvent.click(within(groups[0]).getByRole("button", { name: "回滚" }))
    const dialog = await screen.findByRole("alertdialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "确认回滚" }))

    await waitFor(() =>
      expect(mocks.rollbackLoraAdapter).toHaveBeenCalledWith("card-1", "zh")
    )
  })

  it("renders the sessions list with drill-down logs and group-only poke action", async () => {
    renderSessionsWorkspace()

    expect(screen.getByRole("heading", { name: "会话监控" })).toBeInTheDocument()
    const cards = screen.getAllByTestId("session-card")
    expect(cards).toHaveLength(2)

    const groupCard = within(cards[0])
    expect(groupCard.getByText("g1")).toBeInTheDocument()
    expect(groupCard.getByText("群聊")).toBeInTheDocument()
    expect(groupCard.getByText("3 条")).toBeInTheDocument()
    expect(groupCard.getByText("mention · 毒舌管家")).toBeInTheDocument()
    expect(groupCard.getByText("耗时 1.20s")).toBeInTheDocument()
    expect(within(cards[1]).getByText("private:u1")).toBeInTheDocument()
    expect(within(cards[1]).getByText("私聊")).toBeInTheDocument()

    // 只有群聊会话提供手动接话入口
    const pokeButtons = screen.getAllByRole("button", { name: /手动触发接话/ })
    expect(pokeButtons).toHaveLength(1)

    fireEvent.click(screen.getByRole("button", { name: "展开会话 g1 日志" }))
    const logRows = await screen.findAllByTestId("session-log-row")
    expect(logRows).toHaveLength(2)
    expect(mocks.useSessionLogs).toHaveBeenCalledWith("g1", true)

    mocks.pokeSession.mockResolvedValue({ group_id: "g1", sent: true, reply_message_id: 555 })
    fireEvent.click(pokeButtons[0])
    const dialog = await screen.findByRole("alertdialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "确认触发" }))
    await waitFor(() => expect(mocks.pokeSession).toHaveBeenCalledWith("g1"))
  })

})
