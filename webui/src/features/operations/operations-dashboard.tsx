import { useState } from "react"
import {
  IconAlertTriangle,
  IconActivity,
  IconBrain,
  IconCpu,
  IconDatabase,
  IconMessageCircle,
  IconPlayerPause,
  IconPlayerPlay,
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
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
import { Separator } from "@/components/ui/separator"
import { runRuntimeAction, runTrainingAction, usePluginLogs, useRuntimeLogs, useTrainingLogs } from "@/lib/api"
import type { InferenceEvent, LogEntry, OperationsOverview, RuntimeAction, RuntimeOperation } from "@/lib/types"

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
        <CardDescription>{runtime.model || "未配置模型"}</CardDescription>
        <CardAction>
          <Badge variant={runtime.healthy ? "default" : "destructive"}>
            {runtime.state}
          </Badge>
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

export function OperationsDashboard({ overview }: { overview: OperationsOverview }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
      <div className="flex flex-col gap-4">
        <section>
          <h1>运维控制台</h1>
          <p>模型运行、推理事件和训练状态。</p>
        </section>
        <RuntimeCard overview={overview} />
        <TrainingCard overview={overview} />
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
  )
}
