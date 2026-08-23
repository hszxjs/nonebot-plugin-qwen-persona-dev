import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { BlacklistWorkspace } from "@/features/blacklist/blacklist-workspace"
import type { SettingsPayload } from "@/lib/types"

const mocks = vi.hoisted(() => ({
  useSettings: vi.fn(),
  saveSettingField: vi.fn(),
  testBlacklist: vi.fn(),
  mutateSettings: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    useSettings: mocks.useSettings,
    saveSettingField: mocks.saveSettingField,
    testBlacklist: mocks.testBlacklist,
  }
})

const settings: SettingsPayload = {
  override_file: "/tmp/ai-companion.toml",
  values: {
    ai_companion_blacklist_words: ["spam", "广告"],
    ai_companion_blacklist_regexes: ["^(?i)赌博$"],
  },
}

function cardByTitle(title: string): HTMLElement {
  const card = screen.getByText(title).closest('[data-slot="card"]')
  if (!(card instanceof HTMLElement)) throw new Error(`card not found: ${title}`)
  return card
}

function renderWorkspace(overrides: Partial<SettingsPayload> = {}) {
  mocks.useSettings.mockReturnValue({
    data: { ...settings, ...overrides },
    error: null,
    isLoading: false,
    mutate: mocks.mutateSettings,
  })
  mocks.saveSettingField.mockResolvedValue({})
  mocks.mutateSettings.mockResolvedValue(undefined)
  return render(<BlacklistWorkspace />)
}

describe("blacklist workspace", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders words and regexes from settings with counts", () => {
    renderWorkspace()

    expect(screen.getByRole("heading", { name: "黑名单词库" })).toBeInTheDocument()
    expect(screen.getByText("spam")).toBeInTheDocument()
    expect(screen.getByText("广告")).toBeInTheDocument()
    expect(screen.getByText("^(?i)赌博$")).toBeInTheDocument()
    expect(screen.getByText("词汇 2")).toBeInTheDocument()
    expect(screen.getByText("正则 1")).toBeInTheDocument()
    expect(screen.getByText(/ai_companion_blacklist_words/)).toBeInTheDocument()
  })

  it("adds a word optimistically and patches the whole array", async () => {
    renderWorkspace()
    const card = cardByTitle("黑名单词汇")

    fireEvent.change(within(card).getByLabelText("新词汇"), {
      target: { value: "外挂" },
    })
    fireEvent.click(within(card).getByRole("button", { name: "添加" }))

    expect(screen.getByText("外挂")).toBeInTheDocument()
    await waitFor(() =>
      expect(mocks.saveSettingField).toHaveBeenCalledWith(
        "ai_companion_blacklist_words",
        ["spam", "广告", "外挂"],
        true
      )
    )
    await waitFor(() => expect(mocks.mutateSettings).toHaveBeenCalled())
  })

  it("removes a word and patches the remaining array", async () => {
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "删除 spam" }))

    await waitFor(() =>
      expect(mocks.saveSettingField).toHaveBeenCalledWith(
        "ai_companion_blacklist_words",
        ["广告"],
        true
      )
    )
  })

  it("rejects invalid regexes before saving", async () => {
    renderWorkspace()
    const card = cardByTitle("黑名单正则")

    fireEvent.change(within(card).getByLabelText("新正则"), {
      target: { value: "([unclosed" },
    })
    fireEvent.click(within(card).getByRole("button", { name: "添加" }))

    await waitFor(() =>
      expect(within(card).getByLabelText("新正则")).toHaveValue("([unclosed")
    )
    expect(
      mocks.saveSettingField
    ).not.toHaveBeenCalledWith(
      "ai_companion_blacklist_regexes",
      expect.anything(),
      expect.anything()
    )
    expect(screen.queryByText("([unclosed")).not.toBeInTheDocument()
  })

  it("adds a valid regex and patches the whole array", async () => {
    renderWorkspace()
    const card = cardByTitle("黑名单正则")

    fireEvent.change(within(card).getByLabelText("新正则"), {
      target: { value: "^\\d{10}$" },
    })
    fireEvent.click(within(card).getByRole("button", { name: "添加" }))

    expect(screen.getByText("^\\d{10}$")).toBeInTheDocument()
    await waitFor(() =>
      expect(mocks.saveSettingField).toHaveBeenCalledWith(
        "ai_companion_blacklist_regexes",
        ["^(?i)赌博$", "^\\d{10}$"],
        true
      )
    )
  })

  it("rolls back the optimistic update when the save fails", async () => {
    renderWorkspace()
    mocks.saveSettingField.mockRejectedValue(new Error("配置写入失败"))
    const card = cardByTitle("黑名单词汇")

    fireEvent.change(within(card).getByLabelText("新词汇"), {
      target: { value: "临时词" },
    })
    fireEvent.click(within(card).getByRole("button", { name: "添加" }))

    await waitFor(() =>
      expect(mocks.saveSettingField).toHaveBeenCalled()
    )
    await waitFor(() => expect(screen.queryByText("临时词")).not.toBeInTheDocument())
    expect(screen.getByText("spam")).toBeInTheDocument()
  })

  it("runs a blacklist test and shows matched rules", async () => {
    renderWorkspace()
    mocks.testBlacklist.mockResolvedValue({
      text: "这是 spam 消息",
      matched: true,
      matches: [
        { kind: "word", value: "spam" },
        { kind: "regex", value: "^(?i)赌博$" },
      ],
      words: ["spam", "广告"],
      regexes: ["^(?i)赌博$"],
    })

    fireEvent.change(screen.getByLabelText("测试文本"), {
      target: { value: "这是 spam 消息" },
    })
    fireEvent.click(screen.getByRole("button", { name: /^测试$/ }))

    await waitFor(() =>
      expect(mocks.testBlacklist).toHaveBeenCalledWith("这是 spam 消息")
    )
    const result = await screen.findByTestId("blacklist-test-result")
    expect(within(result).getByText("已命中")).toBeInTheDocument()
    expect(within(result).getByText("该文本会被黑名单过滤")).toBeInTheDocument()
    expect(within(result).getAllByText("词汇").length).toBeGreaterThan(0)
    expect(within(result).getAllByText("正则").length).toBeGreaterThan(0)
    expect(within(result).getByText("spam")).toBeInTheDocument()
    expect(within(result).getByText("命中规则（2 条）：")).toBeInTheDocument()
  })

  it("shows unmatched state for clean text", async () => {
    renderWorkspace()
    mocks.testBlacklist.mockResolvedValue({
      text: "正常消息",
      matched: false,
      matches: [],
      words: ["spam"],
      regexes: [],
    })

    fireEvent.change(screen.getByLabelText("测试文本"), {
      target: { value: "正常消息" },
    })
    fireEvent.click(screen.getByRole("button", { name: /^测试$/ }))

    const result = await screen.findByTestId("blacklist-test-result")
    expect(within(result).getByText("未命中")).toBeInTheDocument()
    expect(within(result).getByText("该文本不会被黑名单过滤")).toBeInTheDocument()
    expect(within(result).queryByText(/命中规则/)).not.toBeInTheDocument()
  })
})
