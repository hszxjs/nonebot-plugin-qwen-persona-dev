import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { BackendsWorkspace } from "@/features/backends/backends-workspace"
import type { BackendsPayload, ProviderBindingsPayload } from "@/lib/types"

const mocks = vi.hoisted(() => ({
  useBackends: vi.fn(),
  useProviderBindings: vi.fn(),
  createBackend: vi.fn(),
  updateBackend: vi.fn(),
  deleteBackend: vi.fn(),
  healthCheckBackend: vi.fn(),
  setProviderBinding: vi.fn(),
}))

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>()
  return {
    ...actual,
    useBackends: mocks.useBackends,
    useProviderBindings: mocks.useProviderBindings,
    createBackend: mocks.createBackend,
    updateBackend: mocks.updateBackend,
    deleteBackend: mocks.deleteBackend,
    healthCheckBackend: mocks.healthCheckBackend,
    setProviderBinding: mocks.setProviderBinding,
  }
})

const payload: BackendsPayload = {
  items: [
    {
      id: "backend-1",
      backend_type: "mlx_lm",
      display_name: "本地 MLX",
      settings: {
        base_url: "http://127.0.0.1:8080/v1",
        model: "qwen3-4b",
        api_key: "sk-secret",
      },
      enabled: true,
      priority: 10,
      created_at: 1783795200,
      updated_at: 1783795300,
      spec: {
        name: "mlx_lm",
        provider_type: "local",
        display_name: "MLX LM",
        docs_url: "https://ml.example/docs",
      },
    },
    {
      id: "backend-2",
      backend_type: "glm",
      display_name: "GLM 在线",
      settings: { model: "glm-5.2", api_key: "sk-remote" },
      enabled: false,
      priority: 90,
      created_at: 1783795200,
      updated_at: 1783795300,
      spec: {
        name: "glm",
        provider_type: "remote",
        display_name: "GLM",
        docs_url: "https://glm.example/docs",
      },
    },
  ],
  total: 2,
  available_types: [
    {
      name: "mlx_lm",
      provider_type: "local",
      display_name: "MLX LM",
      docs_url: "https://ml.example/docs",
    },
    {
      name: "glm",
      provider_type: "remote",
      display_name: "GLM",
      docs_url: "https://glm.example/docs",
    },
  ],
}

const bindings: ProviderBindingsPayload = {
  items: [
    {
      role: "chat",
      provider_id: "backend-1",
      model: "qwen3-4b",
      fallback_provider_id: "backend-2",
      updated_at: 1783795400,
    },
  ],
  roles: ["chat", "vision", "data_cleaning"],
}

function renderWorkspace() {
  mocks.useBackends.mockReturnValue({
    data: payload,
    error: null,
    isLoading: false,
    mutate: vi.fn().mockResolvedValue(undefined),
  })
  mocks.useProviderBindings.mockReturnValue({
    data: bindings,
    error: null,
    isLoading: false,
    mutate: vi.fn().mockResolvedValue(undefined),
  })
  return render(<BackendsWorkspace />)
}

describe("backends workspace", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders backend instances with provider type badges and role bindings", () => {
    renderWorkspace()

    expect(screen.getByRole("heading", { name: "推理后端" })).toBeInTheDocument()
    // 卡片标题 + 角色绑定下拉框中都会出现实例名
    expect(screen.getAllByText("本地 MLX").length).toBeGreaterThan(0)
    expect(screen.getAllByText("GLM 在线").length).toBeGreaterThan(0)
    // 本地/在线 类型徽标 + backend_type 徽标
    expect(screen.getAllByText("本地").length).toBeGreaterThan(0)
    expect(screen.getAllByText("在线").length).toBeGreaterThan(0)
    expect(screen.getAllByText("mlx_lm").length).toBeGreaterThan(0)
    expect(screen.getAllByText("glm").length).toBeGreaterThan(0)
    // 停用实例标记
    expect(screen.getByText("停用")).toBeInTheDocument()
    // 设置摘要只展示 base_url / model，不泄露 api_key
    expect(screen.getByText("http://127.0.0.1:8080/v1 · qwen3-4b")).toBeInTheDocument()
    expect(screen.getByText("glm-5.2")).toBeInTheDocument()
    expect(screen.queryByText(/sk-secret|sk-remote/)).not.toBeInTheDocument()
    // 角色绑定三个角色行
    expect(screen.getByText("主对话")).toBeInTheDocument()
    expect(screen.getByText("视觉理解")).toBeInTheDocument()
    expect(screen.getByText("数据清洗")).toBeInTheDocument()
    expect(screen.getAllByRole("button", { name: "保存" })).toHaveLength(3)
    expect(screen.getByRole("button", { name: "新建后端" })).toBeInTheDocument()
  })

  it("patches enabled state through the instance switch", async () => {
    mocks.updateBackend.mockResolvedValue({ item: payload.items[0] })
    renderWorkspace()

    fireEvent.click(screen.getAllByRole("switch")[0])

    await waitFor(() =>
      expect(mocks.updateBackend).toHaveBeenCalledWith("backend-1", {
        backend_type: "mlx_lm",
        display_name: "本地 MLX",
        settings: payload.items[0].settings,
        enabled: false,
        priority: 10,
      })
    )
  })

  it("runs health check and surfaces unhealthy status with last_error", async () => {
    mocks.healthCheckBackend.mockResolvedValue({
      id: "backend-1",
      healthy: false,
      status: { healthy: false, model: "", last_error: "connection refused" },
    })
    renderWorkspace()

    fireEvent.click(screen.getAllByRole("button", { name: /健康检查/ })[0])

    expect(await screen.findByText("不可用")).toBeInTheDocument()
    expect(screen.getByText("connection refused")).toBeInTheDocument()
    await waitFor(() => expect(mocks.healthCheckBackend).toHaveBeenCalledWith("backend-1"))
  })

  it("saves a role binding row with provider, model and fallback", async () => {
    mocks.setProviderBinding.mockResolvedValue({ binding: bindings.items[0] })
    renderWorkspace()

    fireEvent.click(screen.getAllByRole("button", { name: "保存" })[0])

    await waitFor(() =>
      expect(mocks.setProviderBinding).toHaveBeenCalledWith("chat", {
        provider_id: "backend-1",
        model: "qwen3-4b",
        fallback_provider_id: "backend-2",
      })
    )
  })

  it("opens the create dialog with grouped type choices and connection fields", async () => {
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "新建后端" }))

    const dialog = await screen.findByRole("dialog", { name: "新建后端" })
    expect(within(dialog).getByLabelText("后端类型")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("显示名称")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("Base URL")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("模型")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("API Key")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("优先级")).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: /创建/ })).toBeDisabled()
  })
})
