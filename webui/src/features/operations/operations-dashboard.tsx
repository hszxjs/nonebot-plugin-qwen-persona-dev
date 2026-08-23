import { useMemo, useState } from "react"
import {
  IconAlertTriangle,
  IconActivity,
  IconBrain,
  IconCpu,
  IconDatabase,
  IconHeartbeat,
  IconMessageCircle,
  IconPlayerPause,
  IconPlayerPlay,
  IconStack2,
  IconTrash,
} from "@tabler/icons-react"
import { toast } from "sonner"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ButtonGroup } from "@/components/ui/button-group"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemTitle,
} from "@/components/ui/item"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  activateLoraAdapter,
  apiErrorMessage,
  deleteLoraAdapter,
  registerLoraAdapter,
  rollbackLoraAdapter,
  runRuntimeAction,
  runTrainingAction,
  useKpi,
  useLoraAdapters,
  usePluginLogs,
  useRoleCards,
  useRuntimeLogs,
  useTrainingLogs,
} from "@/lib/api"
import { saveSettingField } from "@/lib/api"
import { mutate } from "swr"
import type {
  InferenceEvent,
  KpiPayload,
  LogEntry,
  LoraAdapter,
  OperationsOverview,
  RuntimeAction,
  RuntimeOperation,
} from "@/lib/types"

function numberText(value: number | undefined) {
  return new Intl.NumberFormat("zh-CN").format(value || 0)
}

function secondsText(value: number | null) {
  if (value === null || value === undefined) return "-"
  return `${value.toFixed(2)}s`
}

function tpsText(value: number | null) {
  if (value === null || value === undefined) return "-"
  return `${value.toFixed(2)} tok/s`
}

function percentText(value: number | null | undefined) {
  if (value === null || value === undefined) return "进行中"
  return `${Math.round(value)}%`
}

function operationStatusText(operation: RuntimeOperation) {
  if (operation.state === "succeeded") return "完成"
  if (operation.state === "failed") return "失败"
  if (operation.state === "cancelled") return "已取消"
  return percentText(operation.percent)
}

function operationProgressValue(operation: RuntimeOperation) {
  if (typeof operation.percent === "number") return operation.percent
  if (operation.state === "succeeded") return 100
  if (operation.state === "running") return 35
  return 0
}

function bytesText(value: number | null | undefined) {
  if (value === null || value === undefined) return "-"
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MiB`
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GiB`
}

export function RuntimeLogPanelView({
  paused,
  logs,
  onTogglePaused,
}: {
  paused: boolean
  logs: LogEntry[]
  onTogglePaused: () => void
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline">历史日志</Badge>
        <Badge variant={paused ? "secondary" : "default"}>
          {paused ? "已暂停" : "自动刷新"}
        </Badge>
        <Button variant="outline" onClick={onTogglePaused}>
          {paused ? <IconPlayerPlay data-icon="inline-start" /> : <IconPlayerPause data-icon="inline-start" />}
          {paused ? "恢复" : "暂停"}
        </Button>
      </div>
      <ScrollArea className="h-56">
        {logs.length ? (
          <ItemGroup data-size="xs">
            {logs.map((entry, index) => (
              <Item key={`${entry.time}-${entry.stream}-${index}`}>
                <ItemContent>
                  <ItemTitle>
                    {entry.stream || "runtime"}
                    {entry.time ? <Badge variant="outline">{entry.time}</Badge> : null}
                  </ItemTitle>
                  <ItemDescription>{entry.message}</ItemDescription>
                </ItemContent>
              </Item>
            ))}
          </ItemGroup>
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>暂无日志</EmptyTitle>
              <EmptyDescription>等待运行时输出或手动刷新。</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        )}
      </ScrollArea>
    </div>
  )
}

function RuntimeLogPanel() {
  const [paused, setPaused] = useState(false)
  const { data } = useRuntimeLogs(paused)
  return (
    <RuntimeLogPanelView
      paused={paused}
      logs={data?.logs || []}
      onTogglePaused={() => setPaused((value) => !value)}
    />
  )
}

function RuntimeOperationProgress({ operation }: { operation: RuntimeOperation | null }) {
  if (!operation) return null
  const value = operationProgressValue(operation)
  return (
    <Alert>
      <IconActivity />
      <AlertTitle>运行任务 · {operation.phase || operation.type}</AlertTitle>
      <AlertDescription>
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>{operation.message || operation.phase || operation.type}</span>
            <Badge variant={operation.state === "failed" ? "destructive" : "secondary"}>
              {operationStatusText(operation)}
            </Badge>
          </div>
          <Progress value={value} />
          {operation.downloaded_bytes !== null || operation.total_bytes !== null ? (
            <span className="text-muted-foreground">
              {bytesText(operation.downloaded_bytes)} / {bytesText(operation.total_bytes)}
            </span>
          ) : null}
          {operation.error ? <span>{operation.error}</span> : null}
        </div>
      </AlertDescription>
    </Alert>
  )
}

function PluginLogPanel() {
  const [paused, setPaused] = useState(false)
  const { data } = usePluginLogs(paused)
  return (
    <RuntimeLogPanelView
      paused={paused}
      logs={data?.logs || []}
      onTogglePaused={() => setPaused((value) => !value)}
    />
  )
}

export function TrainingLogPanelView({
  paused,
  logs,
  onTogglePaused,
}: {
  paused: boolean
  logs: LogEntry[]
  onTogglePaused: () => void
}) {
  return (
    <RuntimeLogPanelView
      paused={paused}
      logs={logs}
      onTogglePaused={onTogglePaused}
    />
  )
}

function TrainingLogPanel({ fallbackLog }: { fallbackLog: string }) {
  const [paused, setPaused] = useState(false)
  const { data } = useTrainingLogs(paused)
  const fallbackLogs = fallbackLog
    ? fallbackLog.split("\n").filter(Boolean).map((message) => ({ time: "", stream: "train", message }))
    : []
  return (
    <TrainingLogPanelView
      paused={paused}
      logs={data?.logs?.length ? data.logs : fallbackLogs}
      onTogglePaused={() => setPaused((value) => !value)}
    />
  )
}

function RuntimeCard({ overview }: { overview: OperationsOverview }) {
  const runtime = overview.runtime
  const [pendingAction, setPendingAction] = useState<RuntimeAction | null>(null)
  const [confirmAction, setConfirmAction] = useState<RuntimeAction | null>(null)
  const [actionMessage, setActionMessage] = useState("")
  const isGlm = runtime.runtime === "glm_online"
  const [switchingBackend, setSwitchingBackend] = useState(false)

  async function switchBackend(value: string) {
    if (!value) return
    setSwitchingBackend(true)
    try {
      await saveSettingField("ai_companion_inference_backend", value)
      await mutate("operations/overview")
      toast.success(value === "glm" ? "已切换到 GLM 在线推理" : "已切换到本地 MLX 推理")
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      toast.error(`切换后端失败：${text}`)
    } finally {
      setSwitchingBackend(false)
    }
  }

  async function executeAction(action: RuntimeAction, confirmed = false) {
    setPendingAction(action)
    const message = `${action} model '${runtime.model || "unknown"}'`
    setActionMessage(message)
    try {
      const result = confirmed
        ? await runRuntimeAction(action, true)
        : await runRuntimeAction(action)
      setActionMessage(result.message || message)
      toast.success(result.message || message)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setActionMessage(text)
      toast.error(text)
    } finally {
      setPendingAction(null)
      setConfirmAction(null)
    }
  }

  return (
    <Card data-testid="runtime-card">
      <CardHeader>
        <CardTitle>模型运行</CardTitle>
        <CardDescription>
          <Badge variant={runtime.healthy ? "default" : "destructive"} className="mr-2">
            {runtime.state}
          </Badge>
          {runtime.model || "未配置模型"}
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={isGlm ? "glm" : "mlx"}
            onValueChange={(value) => switchBackend(value)}
            disabled={switchingBackend}
            spacing={1}
          >
            <ToggleGroupItem value="mlx">本地 MLX</ToggleGroupItem>
            <ToggleGroupItem value="glm">GLM 在线</ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {runtime.last_error ? (
          <Alert variant="destructive">
            <AlertTitle>运行错误</AlertTitle>
            <AlertDescription>{runtime.last_error}</AlertDescription>
          </Alert>
        ) : null}
        {actionMessage ? (
          <Alert>
            <IconAlertTriangle />
            <AlertTitle>运行操作</AlertTitle>
            <AlertDescription>{actionMessage}</AlertDescription>
          </Alert>
        ) : null}
        <RuntimeOperationProgress operation={runtime.operation} />
        <ItemGroup data-size="sm">
          <Item>
            <ItemMedia variant="icon">
              <IconCpu />
            </ItemMedia>
            <ItemContent>
              <ItemTitle>已加载模型</ItemTitle>
              <ItemDescription>{runtime.loaded_model || "-"}</ItemDescription>
            </ItemContent>
          </Item>
          <Item>
            <ItemMedia variant="icon">
              <IconDatabase />
            </ItemMedia>
            <ItemContent>
              <ItemTitle>Adapter</ItemTitle>
              <ItemDescription>{runtime.loaded_adapter_path || runtime.adapter_path || "-"}</ItemDescription>
            </ItemContent>
          </Item>
        </ItemGroup>
        {!isGlm ? (
          <>
            <ButtonGroup>
              <Button
                disabled={!runtime.control.can_load || Boolean(pendingAction)}
                onClick={() => executeAction("load")}
              >
                <IconPlayerPlay data-icon="inline-start" />
                {pendingAction === "load" ? "加载中" : "加载/检查"}
              </Button>
              <Button
                variant="outline"
                disabled={!runtime.control.can_reload || Boolean(pendingAction)}
                onClick={() => setConfirmAction("reload")}
              >
                {pendingAction === "reload" ? "重载中" : "重载"}
              </Button>
              <Button
                variant="destructive"
                disabled={!runtime.control.can_unload || Boolean(pendingAction)}
                onClick={() => setConfirmAction("unload")}
              >
                {pendingAction === "unload" ? "卸载中" : "卸载"}
              </Button>
            </ButtonGroup>
            <Collapsible defaultOpen>
              <CollapsibleTrigger asChild>
                <Button variant="ghost">MLX Runtime Log</Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <RuntimeLogPanel />
              </CollapsibleContent>
            </Collapsible>
          </>
        ) : null}
        <Collapsible>
          <CollapsibleTrigger asChild>
            <Button variant="ghost">Plugin Runtime Log</Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <PluginLogPanel />
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
      <AlertDialog open={Boolean(confirmAction)} onOpenChange={(open) => !open && setConfirmAction(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认{confirmAction === "unload" ? "卸载" : "重载"}模型？</AlertDialogTitle>
            <AlertDialogDescription>
              该操作会改变当前 MLX runtime 状态，可能中断正在进行的推理。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirmAction) {
                  executeAction(confirmAction, true)
                }
              }}
            >
              确认执行
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}

function InferenceEventItem({ event }: { event: InferenceEvent }) {
  return (
    <Item>
      <ItemMedia variant="icon">
        <IconMessageCircle />
      </ItemMedia>
      <ItemContent>
        <ItemTitle>
          {event.trigger_reason}
          <Badge variant={event.status === "success" ? "secondary" : "destructive"}>
            {event.status}
          </Badge>
        </ItemTitle>
        <ItemDescription>
          {event.input_preview || "-"} / {event.output_preview || "-"}
        </ItemDescription>
      </ItemContent>
      <ItemActions>
        <Badge variant="outline">{secondsText(event.latency_seconds)}</Badge>
        <Badge variant="outline">{tpsText(event.tps)}</Badge>
      </ItemActions>
    </Item>
  )
}

function InferenceEventsCard({ overview }: { overview: OperationsOverview }) {
  const events = overview.inference_events.items
  return (
    <Card>
      <CardHeader>
        <CardTitle>推理事件</CardTitle>
        <CardDescription>最近 {overview.inference_events.limit} 条主推理事件</CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <ItemGroup>
            {events.length ? (
              events.map((event) => <InferenceEventItem key={event.event_id} event={event} />)
            ) : (
              <Item>
                <ItemContent>
                  <ItemTitle>暂无推理事件</ItemTitle>
                  <ItemDescription>等待 bot 触发回复或检查回复开关。</ItemDescription>
                </ItemContent>
              </Item>
            )}
          </ItemGroup>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function TrainingCard({ overview }: { overview: OperationsOverview }) {
  const operation = overview.training.operation
  const progress = operation.progress
  const [pendingTraining, setPendingTraining] = useState<"run" | "full" | null>(null)
  const [confirmFullTraining, setConfirmFullTraining] = useState(false)
  const [trainingMessage, setTrainingMessage] = useState("")

  async function executeTraining(fullExternalCorpus = false) {
    const action = fullExternalCorpus ? "full" : "run"
    setPendingTraining(action)
    const message = fullExternalCorpus
      ? "start full external corpus training"
      : "start training"
    setTrainingMessage(message)
    try {
      const result = await runTrainingAction(fullExternalCorpus)
      setTrainingMessage(result.message || message)
      toast.success(result.message || message)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setTrainingMessage(text)
      toast.error(text)
    } finally {
      setPendingTraining(null)
      setConfirmFullTraining(false)
    }
  }

  return (
    <Card data-testid="training-card">
      <CardHeader>
        <CardTitle>训练状态</CardTitle>
        <CardDescription>{progress?.stage || "idle"}</CardDescription>
        <CardAction>
          <Badge variant={operation.running ? "default" : "secondary"}>
            {operation.running ? "running" : "idle"}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {trainingMessage ? (
          <Alert>
            <IconActivity />
            <AlertTitle>训练操作</AlertTitle>
            <AlertDescription>{trainingMessage}</AlertDescription>
          </Alert>
        ) : null}
        <ItemGroup data-size="sm">
          <Item>
            <ItemMedia variant="icon">
              <IconBrain />
            </ItemMedia>
            <ItemContent>
              <ItemTitle>样本</ItemTitle>
              <ItemDescription>{numberText(progress?.sample_count)} 条</ItemDescription>
            </ItemContent>
          </Item>
          <Item>
            <ItemMedia variant="icon">
              <IconActivity />
            </ItemMedia>
            <ItemContent>
              <ItemTitle>状态</ItemTitle>
              <ItemDescription>{progress?.status || "-"}</ItemDescription>
            </ItemContent>
          </Item>
        </ItemGroup>
        <div className="flex flex-wrap gap-2">
          <Button
            disabled={!operation.available || operation.running || Boolean(pendingTraining)}
            onClick={() => executeTraining(false)}
          >
            {pendingTraining === "run" ? "启动中" : "立即训练"}
          </Button>
          <Button
            variant="outline"
            disabled={!operation.available || operation.running || Boolean(pendingTraining)}
            onClick={() => setConfirmFullTraining(true)}
          >
            {pendingTraining === "full" ? "启动中" : "全量外部语料训练"}
          </Button>
        </div>
        <Collapsible defaultOpen>
          <CollapsibleTrigger asChild>
            <Button variant="ghost">Training Log</Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <TrainingLogPanel fallbackLog={progress?.log_tail || ""} />
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
      <AlertDialog open={confirmFullTraining} onOpenChange={setConfirmFullTraining}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认启动全量外部语料训练？</AlertDialogTitle>
            <AlertDialogDescription>
              该操作会忽略外部语料上限，把完整外部语料与本地样本一起导出并启动训练。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={() => executeTraining(true)}>
              确认执行
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}

function LearningSummaryCard({ overview }: { overview: OperationsOverview }) {
  const learning = overview.learning
  return (
    <Card>
      <CardHeader>
        <CardTitle>学习摘要</CardTitle>
        <CardDescription>样本、群组和用户覆盖</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ItemGroup data-size="sm">
          <Item>
            <ItemContent>
              <ItemTitle>已入库消息</ItemTitle>
              <ItemDescription>{numberText(learning.learned_messages)}</ItemDescription>
            </ItemContent>
            <ItemActions>
              <Badge variant="secondary">可训练 {numberText(learning.trainable_samples)}</Badge>
            </ItemActions>
          </Item>
          <Item>
            <ItemContent>
              <ItemTitle>待训练样本</ItemTitle>
              <ItemDescription>{numberText(learning.pending_training_samples)}</ItemDescription>
            </ItemContent>
          </Item>
          <Item>
            <ItemContent>
              <ItemTitle>覆盖范围</ItemTitle>
              <ItemDescription>
                {numberText(learning.groups)} 群 / {numberText(learning.users)} 用户
              </ItemDescription>
            </ItemContent>
          </Item>
        </ItemGroup>
      </CardContent>
    </Card>
  )
}

function percentValue(value: number | null | undefined) {
  if (value === null || value === undefined) return "-"
  return `${Math.round(value * 100)}%`
}

function formatTimestamp(value: number | undefined) {
  if (!value) return "-"
  const date = new Date(value > 1e12 ? value : value * 1000)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString("zh-CN", { hour12: false })
}

function KpiStatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card data-testid="kpi-stat" className="gap-2 py-4">
      <CardHeader className="px-4">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
      </CardHeader>
      {hint ? (
        <CardContent className="px-4 pt-0">
          <p className="text-xs text-muted-foreground">{hint}</p>
        </CardContent>
      ) : null}
    </Card>
  )
}

export function KpiSectionView({ kpi }: { kpi: KpiPayload }) {
  const maxReasonCount = Math.max(1, ...kpi.reply_reasons.map((item) => item.count))
  return (
    <section className="flex flex-col gap-3" data-testid="kpi-section">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2>首页 KPI</h2>
        <Badge variant="outline">近 {kpi.window_days} 天</Badge>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <KpiStatCard
          label="总回复数"
          value={numberText(kpi.total_replies)}
          hint={`触发原因 ${kpi.reply_reasons.length} 类`}
        />
        <KpiStatCard
          label="活跃会话"
          value={numberText(kpi.active_sessions)}
          hint="近 24h 有回复的去重群聊"
        />
        <KpiStatCard label="学习消息量" value={numberText(kpi.learned_messages)} />
        <KpiStatCard
          label="工具成功率"
          value={percentValue(kpi.tools.overall_success_rate)}
          hint={`共 ${numberText(kpi.tools.total_calls)} 次调用`}
        />
        <Card data-testid="kpi-stat" className="gap-2 py-4">
          <CardHeader className="px-4">
            <CardDescription>后端健康</CardDescription>
            <CardTitle className="flex flex-wrap items-center gap-2 text-base">
              <Badge
                variant={
                  kpi.backend.healthy === null
                    ? "secondary"
                    : kpi.backend.healthy
                      ? "default"
                      : "destructive"
                }
              >
                {kpi.backend.healthy === null ? "未知" : kpi.backend.healthy ? "健康" : "异常"}
              </Badge>
              <span className="truncate text-sm font-medium">{kpi.backend.model || "未配置模型"}</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pt-0">
            <p className="flex items-center gap-1 text-xs text-muted-foreground">
              <IconHeartbeat data-icon="inline-start" />
              {kpi.backend.backend || "-"}
            </p>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>触发原因分布</CardTitle>
            <CardDescription>按触发原因拆分的回复次数</CardDescription>
          </CardHeader>
          <CardContent data-testid="kpi-reasons">
            {kpi.reply_reasons.length ? (
              <div className="flex flex-col gap-2">
                {kpi.reply_reasons.map((item) => (
                  <div key={item.reason} className="flex items-center gap-2 text-sm">
                    <span className="w-28 shrink-0 truncate" title={item.reason}>
                      {item.reason}
                    </span>
                    <div className="h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${Math.round((item.count / maxReasonCount) * 100)}%` }}
                      />
                    </div>
                    <span className="w-12 shrink-0 text-right tabular-nums">{item.count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">窗口期内暂无回复。</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Provider 延迟 / TPS</CardTitle>
            <CardDescription>按模型聚合的推理次数、平均延迟与吞吐</CardDescription>
          </CardHeader>
          <CardContent data-testid="kpi-providers">
            {kpi.providers.length ? (
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 font-medium">模型</th>
                      <th className="px-3 py-2 font-medium">次数</th>
                      <th className="px-3 py-2 font-medium">平均延迟</th>
                      <th className="px-3 py-2 font-medium">平均 TPS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {kpi.providers.map((provider) => (
                      <tr key={provider.model} className="border-t">
                        <td className="max-w-[220px] truncate px-3 py-2 font-medium" title={provider.model}>
                          {provider.model}
                        </td>
                        <td className="px-3 py-2 tabular-nums">{numberText(provider.total)}</td>
                        <td className="px-3 py-2 tabular-nums">{secondsText(provider.avg_latency_seconds)}</td>
                        <td className="px-3 py-2 tabular-nums">{tpsText(provider.avg_tps)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">窗口期内暂无推理记录。</p>
            )}
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

export function KpiSection() {
  const { data, error, isLoading } = useKpi(7)
  if (isLoading) {
    return (
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, index) => (
          <Skeleton key={index} className="h-28" />
        ))}
      </div>
    )
  }
  // KPI 面板是辅助信息，读取失败时不阻塞运维控制台主体
  if (error || !data) return null
  return <KpiSectionView kpi={data} />
}

const ADAPTER_LANGUAGES = ["zh", "en", "zh-mixed-en"] as const

type AdapterGroup = {
  roleCardId: string
  sourceLanguage: string
  items: LoraAdapter[]
}

function groupAdapters(items: LoraAdapter[]): AdapterGroup[] {
  const groups = new Map<string, AdapterGroup>()
  for (const item of items) {
    const key = `${item.role_card_id}::${item.source_language}`
    const group = groups.get(key) ?? {
      roleCardId: item.role_card_id,
      sourceLanguage: item.source_language,
      items: [],
    }
    group.items.push(item)
    groups.set(key, group)
  }
  return [...groups.values()]
}

export function LoraAdaptersCardView({
  payload,
  roleCards,
  activatingId,
  rollingBackKey,
  deletingId,
  registering,
  onActivate,
  onRollback,
  onDelete,
  onRegister,
}: {
  payload: { items: LoraAdapter[]; total: number } | undefined
  roleCards: { id: string; name: string }[] | undefined
  activatingId: string | null
  rollingBackKey: string | null
  deletingId: string | null
  registering: boolean
  onActivate: (id: string) => void
  onRollback: (roleCardId: string, sourceLanguage: string) => void
  onDelete: (id: string) => void
  onRegister: (payload: {
    role_card_id: string
    source_language: string
    path: string
    note?: string
    activate?: boolean
  }) => void
}) {
  const [roleCardId, setRoleCardId] = useState("")
  const [language, setLanguage] = useState<string>("zh")
  const [path, setPath] = useState("")
  const [note, setNote] = useState("")
  const [activate, setActivate] = useState(false)
  const groups = useMemo(() => groupAdapters(payload?.items ?? []), [payload])
  const roleCardName = (id: string) =>
    roleCards?.find((card) => card.id === id)?.name ?? id

  return (
    <Card data-testid="adapters-card">
      <CardHeader>
        <CardTitle>Adapter 版本管理</CardTitle>
        <CardDescription>
          按角色卡与语言分组管理 LoRA adapter 版本，支持激活、回滚与手动登记。
        </CardDescription>
        <CardAction>
          <Badge variant="outline">共 {numberText(payload?.total ?? 0)} 个</Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {groups.length ? (
          groups.map((group) => {
            const groupKey = `${group.roleCardId}::${group.sourceLanguage}`
            const hasActive = group.items.some((item) => item.active)
            return (
              <div
                key={groupKey}
                className="rounded-lg border"
                data-testid="adapter-group"
                data-group={groupKey}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium">{roleCardName(group.roleCardId)}</span>
                    <Badge variant="outline">{group.sourceLanguage}</Badge>
                    <Badge variant={hasActive ? "default" : "secondary"}>
                      {hasActive ? "已激活" : "未激活"}
                    </Badge>
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={rollingBackKey === groupKey || group.items.length < 2}
                        title={
                          group.items.length < 2 ? "没有可回滚的历史版本" : "回滚到上一个版本"
                        }
                      >
                        回滚
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>回滚 {roleCardName(group.roleCardId)} 的 adapter？</AlertDialogTitle>
                        <AlertDialogDescription>
                          将把 {group.sourceLanguage} 语言的当前 active 版本切换为上一个历史版本。
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>取消</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => onRollback(group.roleCardId, group.sourceLanguage)}
                        >
                          确认回滚
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
                <div className="flex flex-col divide-y">
                  {group.items.map((adapter) => (
                    <div
                      key={adapter.id}
                      className="flex flex-wrap items-center gap-2 px-3 py-2 text-sm"
                      data-testid="adapter-version-row"
                    >
                      <Badge variant="outline">v{adapter.version}</Badge>
                      {adapter.active ? <Badge>使用中</Badge> : null}
                      <span
                        className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground"
                        title={adapter.path}
                      >
                        {adapter.path}
                      </span>
                      {adapter.note ? (
                        <span className="max-w-[180px] truncate text-xs" title={adapter.note}>
                          {adapter.note}
                        </span>
                      ) : null}
                      <span className="text-xs whitespace-nowrap text-muted-foreground">
                        {formatTimestamp(adapter.created_at)}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={adapter.active || activatingId === adapter.id}
                        onClick={() => onActivate(adapter.id)}
                      >
                        {activatingId === adapter.id ? "激活中" : "激活"}
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            variant="destructive"
                            size="sm"
                            disabled={deletingId === adapter.id}
                            aria-label={`删除 v${adapter.version}`}
                          >
                            <IconTrash data-icon="inline-start" />
                            删除
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>删除 adapter v{adapter.version}？</AlertDialogTitle>
                            <AlertDialogDescription>
                              仅移除版本登记记录（{adapter.path}），不会删除磁盘文件。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              variant="destructive"
                              onClick={() => onDelete(adapter.id)}
                            >
                              删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  ))}
                </div>
              </div>
            )
          })
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>暂无 adapter 登记</EmptyTitle>
              <EmptyDescription>训练产出或手动登记后，版本会按角色卡与语言分组显示。</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        )}
        <Separator />
        <div className="flex flex-col gap-3" data-testid="adapter-register-form">
          <p className="text-sm font-medium">手动登记</p>
          <FieldGroup className="grid gap-3 sm:grid-cols-2">
            <Field>
              <FieldLabel htmlFor="adapter-role-card">角色卡</FieldLabel>
              <Select value={roleCardId} onValueChange={setRoleCardId}>
                <SelectTrigger id="adapter-role-card" className="w-full">
                  <SelectValue placeholder="选择角色卡" />
                </SelectTrigger>
                <SelectContent>
                  {(roleCards ?? []).map((card) => (
                    <SelectItem key={card.id} value={card.id}>
                      {card.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel htmlFor="adapter-language">语言</FieldLabel>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger id="adapter-language" className="w-full">
                  <SelectValue placeholder="选择语言" />
                </SelectTrigger>
                <SelectContent>
                  {ADAPTER_LANGUAGES.map((lang) => (
                    <SelectItem key={lang} value={lang}>
                      {lang}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel htmlFor="adapter-path">Adapter 路径</FieldLabel>
              <Input
                id="adapter-path"
                value={path}
                onChange={(event) => setPath(event.target.value)}
                placeholder="/path/to/adapter"
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="adapter-note">备注（可选）</FieldLabel>
              <Input
                id="adapter-note"
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="训练来源、日期等"
              />
            </Field>
          </FieldGroup>
          <label className="flex w-fit cursor-pointer items-center gap-2 text-sm">
            <Checkbox
              checked={activate}
              onCheckedChange={(checked) => setActivate(checked === true)}
            />
            登记后立即激活
          </label>
          <Button
            disabled={!roleCardId || !path.trim() || registering}
            onClick={() =>
              onRegister({
                role_card_id: roleCardId,
                source_language: language,
                path: path.trim(),
                note: note.trim() || undefined,
                activate,
              })
            }
          >
            <IconStack2 data-icon="inline-start" />
            {registering ? "登记中…" : "登记 adapter"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export function LoraAdaptersCard() {
  const { data, error } = useLoraAdapters()
  const { data: roleCardsData } = useRoleCards()
  const [activatingId, setActivatingId] = useState<string | null>(null)
  const [rollingBackKey, setRollingBackKey] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [registering, setRegistering] = useState(false)

  async function handleActivate(id: string) {
    setActivatingId(id)
    try {
      const result = await activateLoraAdapter(id)
      toast.success(`已激活 adapter v${result.item.version}`)
      await mutate("lora-adapters")
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setActivatingId(null)
    }
  }

  async function handleRollback(roleCardId: string, sourceLanguage: string) {
    const key = `${roleCardId}::${sourceLanguage}`
    setRollingBackKey(key)
    try {
      const result = await rollbackLoraAdapter(roleCardId, sourceLanguage)
      toast.success(`已回滚到 adapter v${result.item.version}`)
      await mutate("lora-adapters")
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setRollingBackKey(null)
    }
  }

  async function handleDelete(id: string) {
    setDeletingId(id)
    try {
      await deleteLoraAdapter(id)
      toast.success("已删除 adapter 登记")
      await mutate("lora-adapters")
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setDeletingId(null)
    }
  }

  async function handleRegister(payload: {
    role_card_id: string
    source_language: string
    path: string
    note?: string
    activate?: boolean
  }) {
    setRegistering(true)
    try {
      const result = await registerLoraAdapter(payload)
      toast.success(`已登记 adapter v${result.item.version}`)
      await mutate("lora-adapters")
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setRegistering(false)
    }
  }

  if (error) {
    return (
      <Card data-testid="adapters-card">
        <CardHeader>
          <CardTitle>Adapter 版本管理</CardTitle>
          <CardDescription>无法读取 adapter 列表：{String(error)}</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <LoraAdaptersCardView
      payload={data}
      roleCards={roleCardsData?.items}
      activatingId={activatingId}
      rollingBackKey={rollingBackKey}
      deletingId={deletingId}
      registering={registering}
      onActivate={(id) => void handleActivate(id)}
      onRollback={(roleCardId, sourceLanguage) => void handleRollback(roleCardId, sourceLanguage)}
      onDelete={(id) => void handleDelete(id)}
      onRegister={(payload) => void handleRegister(payload)}
    />
  )
}

export function OperationsDashboard({ overview }: { overview: OperationsOverview }) {
  return (
    <div className="flex flex-col gap-4">
      <KpiSection />
      <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <section>
            <h1>运维控制台</h1>
            <p>模型运行、推理事件和训练状态。</p>
          </section>
          <RuntimeCard overview={overview} />
          <TrainingCard overview={overview} />
          <LoraAdaptersCard />
          <LearningSummaryCard overview={overview} />
        </div>
        <div className="flex min-w-0 flex-col gap-4">
          {overview.errors.length ? (
            <Alert variant="destructive">
              <AlertTitle>最近错误</AlertTitle>
              <AlertDescription>{overview.errors.join(" / ")}</AlertDescription>
            </Alert>
          ) : null}
          <InferenceEventsCard overview={overview} />
          <Separator />
        </div>
      </div>
    </div>
  )
}
