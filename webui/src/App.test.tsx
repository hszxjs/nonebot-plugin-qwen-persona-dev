import { cleanup, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { OperationsDashboard } from "@/features/operations/operations-dashboard"
import type { OperationsOverview } from "@/lib/types"

const overview: OperationsOverview = {
  runtime: {
    runtime: "mlx_vlm",
    state: "loaded",
    status: "loaded",
    healthy: true,
    model: "target-model",
    loaded_model: "target-model",
    adapter_path: "adapter",
    loaded_adapter_path: "adapter",
    last_error: "",
    operation: {
      id: "op-1",
      type: "load",
      state: "running",
      phase: "downloading",
      message: "downloading model",
      percent: 42,
      downloaded_bytes: 42,
      total_bytes: 100,
      started_at: "2026-07-12T00:00:00Z",
      finished_at: null,
      error: "",
    },
    control: { can_load: true, can_unload: true, can_reload: true },
    metrics: { last: { tps: 30, latency_seconds: 1.2 }, history: [] },
    model_presets: [],
  },
  learning: {
    learned_messages: 10,
    trainable_samples: 8,
    pending_training_samples: 3,
    stickers: 2,
    today_samples: 5,
    last_24h_samples: 7,
    groups: 2,
    users: 4,
  },
  training: {
    summary: { last_success: null, last_run: null, pending_since: null },
    operation: {
      running: false,
      available: true,
      progress: {
        stage: "idle",
        status: "",
        reason: "",
        sample_count: 0,
        dataset_path: "",
        output_dir: "",
        log_tail: "",
        started_at: null,
        finished_at: null,
      },
    },
    config: {
      target_configured: true,
      helper_configured: false,
      min_train_samples: 20,
      dataset_limit: 4000,
      external_corpus_enabled: true,
      external_corpus_limit: 0,
      multimodal_training_enabled: true,
      mlx_lora_modules: ["language"],
    },
  },
  charts: { learning_trend: [], group_contribution: [], performance_trend: [] },
  inference: { client: { last: null, history: [] } },
  inference_events: {
    items: [
      {
        event_id: "evt-1",
        time: "2026-07-12T00:00:00Z",
        status: "success",
        trigger_reason: "mention",
        model: "target-model",
        adapter: "adapter",
        latency_seconds: 1.2,
        tps: 30,
        media_count: 1,
        response_chars: 6,
        error_summary: "",
        input_preview: "你好",
        output_preview: "你好呀",
        group_id: "100",
        user_id: "200",
      },
    ],
    total: 1,
    limit: 300,
    offset: 0,
  },
  errors: [],
}

describe("OperationsDashboard", () => {
  afterEach(() => cleanup())

  it("renders operations-first content with inference events and no sticker gallery", () => {
    render(<OperationsDashboard overview={overview} />)

    const runtime = screen.getByTestId("runtime-card")

    expect(screen.getByRole("heading", { name: "运维控制台" })).toBeInTheDocument()
    expect(within(runtime).getAllByText("target-model")).toHaveLength(2)
    expect(within(runtime).getByText("downloading model")).toBeInTheDocument()
    expect(within(runtime).getByText("42%")).toBeInTheDocument()
    expect(screen.getByText("推理事件")).toBeInTheDocument()
    expect(screen.getByText("mention")).toBeInTheDocument()
    expect(screen.getByText(/你好 \/ 你好呀/)).toBeInTheDocument()
    expect(screen.queryByText("表情包仓库")).not.toBeInTheDocument()
    expect(screen.queryByText("Project ready!")).not.toBeInTheDocument()
  })

  it("keeps runtime and training logs collapsed under their cards", () => {
    render(<OperationsDashboard overview={overview} />)

    const runtime = screen.getByTestId("runtime-card")
    const training = screen.getByTestId("training-card")

    expect(within(runtime).getByText("MLX Runtime Log")).toBeInTheDocument()
    expect(within(training).getByText("Training Log")).toBeInTheDocument()
  })

  it("renders completed runtime operations as complete instead of indeterminate", () => {
    render(
      <OperationsDashboard
        overview={{
          ...overview,
          runtime: {
            ...overview.runtime,
            operation: {
              ...overview.runtime.operation!,
              state: "succeeded",
              phase: "loaded",
              message: "loaded in 1.20s",
              percent: null,
            },
          },
        }}
      />
    )

    const runtime = screen.getByTestId("runtime-card")

    expect(within(runtime).getByText("loaded in 1.20s")).toBeInTheDocument()
    expect(within(runtime).getByText("完成")).toBeInTheDocument()
    expect(within(runtime).queryByText("进行中")).not.toBeInTheDocument()
  })
})
