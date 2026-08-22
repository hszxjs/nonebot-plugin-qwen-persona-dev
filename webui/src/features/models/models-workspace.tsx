import { useCallback, useEffect, useRef, useState } from "react"
import { IconDatabase, IconDownload, IconPlayerStop, IconTrash } from "@tabler/icons-react"
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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  apiErrorMessage,
  cancelModelDownload,
  deleteLocalModel,
  modelDownloadEventsUrl,
  startModelDownload,
  useModelsManage,
} from "@/lib/api"
import type { ModelManageOperation, ModelsManagePayload } from "@/lib/types"

function bytesText(value: number | null | undefined) {
  if (value === null || value === undefined) return "-"
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MiB`
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GiB`
}

function operationStatusText(operation: ModelManageOperation) {
  if (operation.state === "success") return "成功"
  if (operation.state === "failed") return "失败"
  if (operation.state === "cancelled") return "已取消"
  if (operation.percent === null || operation.percent === undefined) return "进行中"
  return `${Math.round(operation.percent)}%`
}

function operationProgressValue(operation: ModelManageOperation) {
  if (typeof operation.percent === "number") return operation.percent
  if (operation.state === "success") return 100
  if (operation.state === "running") return 35
  return 0
}

function DownloadOperationProgress({ operation }: { operation: ModelManageOperation }) {
  return (
    <div className="flex flex-col gap-2" data-testid="download-progress">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span>{operation.message || operation.phase || operation.type}</span>
        <Badge variant={operation.state === "failed" ? "destructive" : "secondary"}>
          {operationStatusText(operation)}
        </Badge>
      </div>
      <Progress value={operationProgressValue(operation)} />
      {operation.downloaded_bytes !== null || operation.total_bytes !== null ? (
        <span className="text-muted-foreground">
          {bytesText(operation.downloaded_bytes)} / {bytesText(operation.total_bytes)}
        </span>
      ) : null}
      {operation.error ? <span className="text-sm text-destructive">{operation.error}</span> : null}
      {operation.local_path ? (
        <span className="text-xs text-muted-foreground">{operation.local_path}</span>
      ) : null}
    </div>
  )
}

function LocalModelsCard({
  payload,
  onDeleteModel,
}: {
  payload: ModelsManagePayload
  onDeleteModel: (modelRef: string) => Promise<void>
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>磁盘概览</CardTitle>
        <CardDescription>
          本地模型 {payload.local.length} 个，共占用 {bytesText(payload.total_size_bytes)}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {payload.local.length ? (
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">模型</th>
                  <th className="px-3 py-2 font-medium">大小</th>
                  <th className="px-3 py-2 font-medium">路径</th>
                  <th className="px-3 py-2 text-right font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {payload.local.map((model) => (
                  <tr key={model.path} className="border-t">
                    <td className="px-3 py-2 font-medium">{model.name}</td>
                    <td className="px-3 py-2 whitespace-nowrap">{model.size_display}</td>
                    <td
                      className="max-w-[280px] truncate px-3 py-2 text-muted-foreground"
                      title={model.path}
                    >
                      {model.path}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="destructive" size="sm">
                            <IconTrash data-icon="inline-start" />
                            删除
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>删除模型「{model.name}」？</AlertDialogTitle>
                            <AlertDialogDescription>
                              将从磁盘移除 {model.size_display}（{model.path}），删除后需要重新下载。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              variant="destructive"
                              onClick={() => void onDeleteModel(model.name)}
                            >
                              删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>暂无本地模型</EmptyTitle>
              <EmptyDescription>从下方下载卡片拉取一个模型到本地。</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        )}
      </CardContent>
    </Card>
  )
}

function DownloadCard({
  presets,
  operation,
  starting,
  cancelling,
  onStartDownload,
  onCancelDownload,
}: {
  presets: ModelsManagePayload["presets"]
  operation: ModelManageOperation | null
  starting: boolean
  cancelling: boolean
  onStartDownload: (modelRef: string) => Promise<void>
  onCancelDownload: () => Promise<void>
}) {
  const [modelRef, setModelRef] = useState("")
  const running = operation?.state === "running"

  async function handleStart() {
    const trimmed = modelRef.trim()
    if (!trimmed) return
    await onStartDownload(trimmed)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>模型下载</CardTitle>
        <CardDescription>从 Hugging Face 拉取模型到本地，支持实时进度与取消。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="model-ref">模型引用</FieldLabel>
            <Input
              id="model-ref"
              value={modelRef}
              onChange={(event) => setModelRef(event.target.value)}
              placeholder="例如：Qwen/Qwen3-4B-Instruct-2507"
              disabled={running}
            />
            <FieldDescription>填入 Hugging Face 仓库 ID，如 org/repo。</FieldDescription>
          </Field>
          {presets.length ? (
            <Field>
              <FieldLabel htmlFor="model-preset">快速选择</FieldLabel>
              <Select
                value={presets.some((preset) => preset.value === modelRef) ? modelRef : ""}
                onValueChange={(value) => setModelRef(value)}
                disabled={running}
              >
                <SelectTrigger id="model-preset" className="w-full">
                  <SelectValue placeholder="选择预设模型" />
                </SelectTrigger>
                <SelectContent>
                  {presets.map((preset) => (
                    <SelectItem key={preset.value} value={preset.value}>
                      <span className="flex flex-col items-start gap-0.5">
                        <span>{preset.label}</span>
                        {preset.description ? (
                          <span className="text-xs text-muted-foreground">
                            {preset.description}
                          </span>
                        ) : null}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          ) : null}
        </FieldGroup>
        <div className="flex flex-wrap gap-2">
          <Button
            disabled={!modelRef.trim() || running || starting}
            onClick={() => void handleStart()}
          >
            <IconDownload data-icon="inline-start" />
            {starting ? "启动中…" : "开始下载"}
          </Button>
          {running ? (
            <Button variant="destructive" disabled={cancelling} onClick={() => void onCancelDownload()}>
              <IconPlayerStop data-icon="inline-start" />
              {cancelling ? "取消中…" : "取消下载"}
            </Button>
          ) : null}
        </div>
        {operation ? <DownloadOperationProgress operation={operation} /> : null}
      </CardContent>
    </Card>
  )
}

export function ModelsWorkspaceView({
  payload,
  operation,
  starting,
  cancelling,
  onStartDownload,
  onCancelDownload,
  onDeleteModel,
}: {
  payload: ModelsManagePayload
  operation: ModelManageOperation | null
  starting: boolean
  cancelling: boolean
  onStartDownload: (modelRef: string) => Promise<void>
  onCancelDownload: () => Promise<void>
  onDeleteModel: (modelRef: string) => Promise<void>
}) {
  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>模型管理</h1>
        <p>查看本地模型占用，下载新模型并跟踪进度。</p>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="outline">
          <IconDatabase data-icon="inline-start" />
          {bytesText(payload.total_size_bytes)}
        </Badge>
        <Badge variant="outline">本地模型 {payload.local.length}</Badge>
      </div>
      <LocalModelsCard payload={payload} onDeleteModel={onDeleteModel} />
      <DownloadCard
        presets={payload.presets}
        operation={operation}
        starting={starting}
        cancelling={cancelling}
        onStartDownload={onStartDownload}
        onCancelDownload={onCancelDownload}
      />
    </div>
  )
}

export function ModelsWorkspace() {
  const { data, error, isLoading, mutate } = useModelsManage()
  const [liveOperation, setLiveOperation] = useState<ModelManageOperation | null>(null)
  const [starting, setStarting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)

  const closeStream = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  const attachStream = useCallback(() => {
    if (sourceRef.current) return
    const source = new EventSource(modelDownloadEventsUrl())
    sourceRef.current = source
    source.onmessage = (event: MessageEvent) => {
      try {
        const frame = JSON.parse(String(event.data)) as Partial<ModelManageOperation> & {
          state?: string
        }
        if (frame.state === "idle") {
          setLiveOperation(null)
          closeStream()
          void mutate()
          return
        }
        if (!frame.state) return
        setLiveOperation(frame as ModelManageOperation)
        if (frame.state !== "running") {
          closeStream()
          void mutate()
          if (frame.state === "success") {
            toast.success("模型下载完成")
          } else if (frame.state === "failed") {
            toast.error(frame.error || "模型下载失败")
          } else if (frame.state === "cancelled") {
            toast.info("模型下载已取消")
          }
        }
      } catch {
        // 忽略无法解析的 SSE 帧
      }
    }
    source.onerror = () => {
      closeStream()
      void mutate()
    }
  }, [closeStream, mutate])

  const runningInOverview = data?.operation?.state === "running"

  useEffect(() => {
    if (runningInOverview) {
      attachStream()
    }
  }, [runningInOverview, attachStream])

  useEffect(() => () => closeStream(), [closeStream])

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取模型管理数据</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return null
  }

  const operation = liveOperation ?? data.operation ?? null

  async function handleStartDownload(modelRef: string) {
    setStarting(true)
    try {
      const result = await startModelDownload(modelRef)
      setLiveOperation(result.operation)
      toast.success("下载已开始")
      attachStream()
      await mutate()
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setStarting(false)
    }
  }

  async function handleCancelDownload() {
    setCancelling(true)
    try {
      await cancelModelDownload()
      toast.success("已请求取消下载")
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setCancelling(false)
    }
  }

  async function handleDeleteModel(modelRef: string) {
    try {
      const result = await deleteLocalModel(modelRef)
      toast.success(`已删除模型，释放 ${result.freed_display}`)
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    }
    await mutate()
  }

  return (
    <ModelsWorkspaceView
      payload={data}
      operation={operation}
      starting={starting}
      cancelling={cancelling}
      onStartDownload={handleStartDownload}
      onCancelDownload={handleCancelDownload}
      onDeleteModel={handleDeleteModel}
    />
  )
}
