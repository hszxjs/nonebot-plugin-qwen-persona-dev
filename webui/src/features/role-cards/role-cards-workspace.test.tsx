import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  RoleCardsWorkspace,
  RoleCardsWorkspaceView,
} from "@/features/role-cards/role-cards-workspace"
import type { RoleCard, RoleCardsPayload, ToolsPayload } from "@/lib/types"

const mocks = vi.hoisted(() => ({
  useRoleCards: vi.fn(),
  useTools: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    useRoleCards: mocks.useRoleCards,
    useTools: mocks.useTools,
  }
})

function buildCard(overrides: Partial<RoleCard> = {}): RoleCard {
  return {
    id: "card-1",
    role_type: "bot_persona",
    name: "毒舌管家",
    system_prompt: "你是{persona_name}，用毒舌短句回复{user_nickname}。",
    style_tags: ["毒舌", "短句"],
    tone_examples: ["这么简单还要问？", "笑死，看看你。"],
    catchphrases: [],
    topic_preferences: [],
    emotion_baseline: "",
    relation_profile: "",
    allowed_tools: ["search_web"],
    avatar: "",
    enabled: true,
    target_user_id: null,
    created_at: 1783795200,
    updated_at: 1783795300,
    bindings: ["group:888"],
    ...overrides,
  }
}

const personaCard = buildCard()

const mimicCard = buildCard({
  id: "card-2",
  role_type: "member_mimic",
  name: "群友小 A",
  system_prompt: "模仿群友小 A 的语气：爱用省略号和颜文字…",
  style_tags: ["省略号"],
  tone_examples: ["今天好累…", "笑死(", ""],
  allowed_tools: [],
  target_user_id: "10086",
  bindings: [],
  enabled: false,
})

const payload: RoleCardsPayload = {
  items: [personaCard, mimicCard],
  total: 2,
  role_types: ["bot_persona", "member_mimic"],
}

const tools: ToolsPayload = {
  items: [
    {
      name: "search_web",
      description: "联网搜索网页信息",
      parameters: {},
      permission_tags: [],
      executable: true,
    },
    {
      name: "anime_search",
      description: "以图搜源动漫图片",
      parameters: {},
      permission_tags: [],
      executable: true,
    },
  ],
  total: 2,
}

function renderWorkspace() {
  mocks.useRoleCards.mockReturnValue({ data: payload, error: null, isLoading: false })
  mocks.useTools.mockReturnValue({ data: tools, error: null, isLoading: false })
  return render(<RoleCardsWorkspace />)
}

function noopPromise() {
  return Promise.resolve()
}

describe("role cards workspace", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders the card list with type badges, bindings and mimic actions", () => {
    renderWorkspace()

    expect(screen.getByRole("heading", { name: "角色卡" })).toBeInTheDocument()
    expect(screen.getByText("毒舌管家")).toBeInTheDocument()
    expect(screen.getByText("群友小 A")).toBeInTheDocument()
    expect(screen.getByText("人格")).toBeInTheDocument()
    expect(screen.getByText("模仿")).toBeInTheDocument()
    expect(screen.getByText("QQ 10086")).toBeInTheDocument()
    expect(screen.getByText("group:888")).toBeInTheDocument()
    expect(screen.getByText("总计 2")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "新建人格" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "生成群友模仿卡" })).toBeInTheDocument()
    // 仅 member_mimic 卡片提供语言风格刷新入口
    expect(screen.getAllByRole("button", { name: "刷新语言风格" })).toHaveLength(1)
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
  })

  it("shows the tool permission matrix inside the edit sheet", async () => {
    renderWorkspace()

    fireEvent.click(screen.getAllByRole("button", { name: "编辑" })[0])

    const sheet = await screen.findByRole("dialog", { name: "编辑角色卡" })
    expect(
      within(sheet).getByText("可用模板变量：{persona_name} {user_nickname} {time} {memory_summary}")
    ).toBeInTheDocument()
    const searchWeb = within(sheet).getByRole("checkbox", { name: "search_web" })
    expect(searchWeb).toBeChecked()
    const animeSearch = within(sheet).getByRole("checkbox", { name: "anime_search" })
    expect(animeSearch).not.toBeChecked()

    fireEvent.click(animeSearch)
    expect(animeSearch).toBeChecked()
  })

  it("keeps member mimic generated fields read-only in the edit sheet", async () => {
    renderWorkspace()

    fireEvent.click(screen.getAllByRole("button", { name: "编辑" })[1])

    const sheet = await screen.findByRole("dialog", { name: "编辑角色卡" })
    const systemPrompt = within(sheet).getByRole("textbox", {
      name: /系统提示词（分析生成，只读）/,
    })
    expect(systemPrompt).toHaveAttribute("readonly")
    expect(systemPrompt).toHaveValue("模仿群友小 A 的语气：爱用省略号和颜文字…")
    expect(within(sheet).getByText(/刷新语言风格/)).toBeInTheDocument()
    expect(within(sheet).getByRole("checkbox", { name: "search_web" })).toBeInTheDocument()
  })

  it("surfaces member mimic analysis failure inside the generation dialog", async () => {
    render(
      <RoleCardsWorkspaceView
        payload={payload}
        tools={tools.items}
        onCreate={noopPromise}
        onUpdate={noopPromise}
        onDelete={noopPromise}
        onClone={noopPromise}
        onBind={noopPromise}
        onUnbind={noopPromise}
        onGenerateMimic={vi.fn().mockRejectedValue(new Error("样本不足：该成员样本少于 20 条"))}
        onRefreshStyle={noopPromise}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "生成群友模仿卡" }))
    const dialog = await screen.findByRole("dialog", { name: "生成群友模仿卡" })
    fireEvent.change(within(dialog).getByLabelText("QQ 号"), {
      target: { value: "10086" },
    })
    fireEvent.click(within(dialog).getByRole("button", { name: /开始分析/ }))

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("样本不足：该成员样本少于 20 条")
    )
  })
})
