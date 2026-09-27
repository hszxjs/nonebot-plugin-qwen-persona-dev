import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { MemoriesWorkspace } from "@/features/memories/memories-workspace"
import type { PersonaMemoriesPayload, PersonaMemoryGroupsPayload } from "@/lib/types"

const mocks = vi.hoisted(() => ({
  usePersonaMemories: vi.fn(),
  usePersonaMemoryGroups: vi.fn(),
  mutateMemories: vi.fn(),
  extractPersonaMemories: vi.fn(),
  decayPersonaMemories: vi.fn(),
  deletePersonaMemory: vi.fn(),
  purgeMemberMemories: vi.fn(),
  updatePersonaMemory: vi.fn(),
  refreshConversationNames: vi.fn(),
  apiErrorMessage: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    usePersonaMemories: mocks.usePersonaMemories,
    usePersonaMemoryGroups: mocks.usePersonaMemoryGroups,
    extractPersonaMemories: mocks.extractPersonaMemories,
    decayPersonaMemories: mocks.decayPersonaMemories,
    deletePersonaMemory: mocks.deletePersonaMemory,
    purgeMemberMemories: mocks.purgeMemberMemories,
    updatePersonaMemory: mocks.updatePersonaMemory,
    refreshConversationNames: mocks.refreshConversationNames,
    apiErrorMessage: mocks.apiErrorMessage,
  }
})

const groups: PersonaMemoryGroupsPayload = {
  groups: ["1019859184", "994844677"],
  disabled: ["142904565"],
  sensitivities: ["neutral", "relation", "emotion", "private"],
}

const groupMemories: PersonaMemoriesPayload = {
  scope: "group",
  items: [
    {
      id: 1,
      scope: "group",
      group_id: "1019859184",
      user_id: "",
      kind: "梗",
      content: "打本指一起刷副本",
      importance: 4,
      sensitivity: "neutral",
      created_at: 1783795200,
      last_seen_at: 1783795200,
      decayed: false,
    },
    {
      id: 2,
      scope: "group",
      group_id: "1019859184",
      user_id: "",
      kind: "事件",
      content: "群主上周失恋了",
      importance: 5,
      sensitivity: "emotion",
      created_at: 1783795200,
      last_seen_at: 1783795200,
      decayed: false,
    },
  ],
}

const memberMemories: PersonaMemoriesPayload = {
  scope: "member",
  items: [
    {
      id: 7,
      scope: "member",
      group_id: "1019859184",
      user_id: "3305167706",
      kind: "偏好",
      content: "喜欢周末开黑",
      importance: 3,
      sensitivity: "relation",
      created_at: 1783795200,
      last_seen_at: 1783795200,
      decayed: false,
    },
  ],
}

function renderWorkspace(payload: PersonaMemoriesPayload = groupMemories) {
  mocks.usePersonaMemories.mockReturnValue({
    data: payload,
    error: null,
    isLoading: false,
    mutate: mocks.mutateMemories,
  })
  mocks.usePersonaMemoryGroups.mockReturnValue({ data: groups, error: null, isLoading: false })
  return render(<MemoriesWorkspace />)
}

describe("memories workspace", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders group memories with kind and sensitivity badges", () => {
    renderWorkspace()

    expect(screen.getByRole("heading", { name: "记忆" })).toBeInTheDocument()
    const rows = screen.getAllByTestId("memory-row")
    expect(rows).toHaveLength(2)
    expect(screen.getByText("打本指一起刷副本")).toBeInTheDocument()
    expect(screen.getByText("中性")).toBeInTheDocument()
    expect(screen.getByText("情绪")).toBeInTheDocument()
    expect(screen.getByText(/已排除抽取的群/)).toHaveTextContent("142904565")
  })

  it("extracts memories on demand and revalidates", async () => {
    mocks.extractPersonaMemories.mockResolvedValue({ saved: 3, failed: 0, results: [] })
    renderWorkspace()

    screen.getByRole("button", { name: /立即抽取/ }).click()

    await waitFor(() => expect(mocks.extractPersonaMemories).toHaveBeenCalledWith(""))
    await waitFor(() => expect(mocks.mutateMemories).toHaveBeenCalled())
  })

  it("shows purge action for member memories", () => {
    renderWorkspace(memberMemories)

    expect(screen.getByText("喜欢周末开黑")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /清除该成员全部记忆/ })).toBeInTheDocument()
  })

  it("does not offer purge on group memories", () => {
    cleanup()
    renderWorkspace(groupMemories)

    expect(screen.queryByRole("button", { name: /清除该成员全部记忆/ })).not.toBeInTheDocument()
  })

  it("shows empty state when nothing is remembered", () => {
    renderWorkspace({ scope: "group", items: [] })

    expect(screen.getByText("暂无记忆")).toBeInTheDocument()
  })
})
