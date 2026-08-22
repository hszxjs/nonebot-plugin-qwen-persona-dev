import { useMemo, useState } from "react"
import {
  IconEdit,
  IconHeartbeat,
  IconPlus,
  IconServer2,
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
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import {
  apiErrorMessage,
  createBackend,
  deleteBackend,
  healthCheckBackend,
  setProviderBinding,
  updateBackend,
  useBackends,
  useProviderBindings,
} from "@/lib/api"
import type {
  BackendConfig,
  BackendHealthCheckResult,
  BackendPayload,
  BackendsPayload,
  ProviderBindingPayload,
  ProviderBindingsPayload,
  ProviderRole,
  ProviderSpec,
} from "@/lib/types"

const UNBOUND_PROVIDER = "__none__"

const PROVIDER_TYPE_LABELS: Record<string, string> = {
  local: "本地",
  remote: "在线",
  unknown: "未知",
}

const ROLE_LABELS: Record<string, string> = {
  chat: "主对话",
  vision: "视觉理解",
  data_cleaning: "数据清洗",
}

const DEFAULT_ROLES: ProviderRole[] = ["chat", "vision", "data_cleaning"]

function formatTimestamp(value: number) {
  const date = new Date(value > 1e12 ? value : value * 1000)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

function providerTypeLabel(spec: ProviderSpec | undefined) {
  if (!spec?.provider_type) return "未知"
  return PROVIDER_TYPE_LABELS[spec.provider_type] ?? "未知"
}

function settingsText(settings: Record<string, unknown>) {
  const parts: string[] = []
  if (typeof settings.base_url === "string" && settings.base_url) {
    parts.push(settings.base_url)
  }
  if (typeof settings.model === "string" && settings.model) {
    parts.push(settings.model)
  }
  return parts.join(" · ")
}

type BackendFormState = {
  backendType: string
  displayName: string
  baseUrl: string
  model: string
  apiKey: string
  priority: string
}

const EMPTY_FORM: BackendFormState = {
  backendType: "",
  displayName: "",
  baseUrl: "",
  model: "",
  apiKey: "",
  priority: "100",
}

function formFromBackend(backend: BackendConfig): BackendFormState {
  return {
    backendType: backend.backend_type,
    displayName: backend.display_name,
    baseUrl: typeof backend.settings.base_url === "string" ? backend.settings.base_url : "",
    model: typeof backend.settings.model === "string" ? backend.settings.model : "",
    apiKey: typeof backend.settings.api_key === "string" ? backend.settings.api_key : "",
    priority: String(backend.priority),
  }
}

function payloadFromForm(form: BackendFormState, enabled: boolean): BackendPayload {
  const settings: Record<string, string> = {}
  if (form.baseUrl.trim()) settings.base_url = form.baseUrl.trim()
  if (form.model.trim()) settings.model = form.model.trim()
  if (form.apiKey.trim()) settings.api_key = form.apiKey.trim()
  return {
    backend_type: form.backendType,
    display_name: form.displayName.trim(),
    settings,
    enabled,
    priority: Number.parseInt(form.priority, 10) || 100,
  }
}

function payloadFromBackend(backend: BackendConfig, enabled: boolean): BackendPayload {
  return {
    backend_type: backend.backend_type,
    display_name: backend.display_name,
    settings: backend.settings,
    enabled,
    priority: backend.priority,
  }
}

function SpecSelectItem({ spec }: { spec: ProviderSpec }) {
  return (
    <SelectItem value={spec.name}>
      <span className="flex flex-col items-start gap-0.5">
        <span>{spec.display_name || spec.name}</span>
        {spec.docs_url ? (
          <span className="text-xs text-muted-foreground">{spec.docs_url}</span>
        ) : null}
      </span>
    </SelectItem>
  )
}

function BackendTypeSelect({
  id,
  specs,
  value,
  onChange,
}: {
  id: string
  specs: ProviderSpec[]
  value: string
  onChange: (value: string) => void
}) {
  const { local, remote, others } = useMemo(
    () => ({
      local: specs.filter((spec) => spec.provider_type === "local"),
      remote: specs.filter((spec) => spec.provider_type === "remote"),
      others: specs.filter(
        (spec) => spec.provider_type !== "local" && spec.provider_type !== "remote"
      ),
    }),
    [specs]
  )
  const selectedSpec = specs.find((spec) => spec.name === value)
  return (
    <Field>
      <FieldLabel htmlFor={id}>后端类型</FieldLabel>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={id} className="w-full">
          <SelectValue placeholder="选择后端类型" />
        </SelectTrigger>
        <SelectContent>
          {local.length ? (
            <SelectGroup>
              <SelectLabel>本地</SelectLabel>
              {local.map((spec) => (
                <SpecSelectItem key={spec.name} spec={spec} />
              ))}
            </SelectGroup>
          ) : null}
          {remote.length ? (
            <SelectGroup>
              <SelectLabel>在线</SelectLabel>
              {remote.map((spec) => (
                <SpecSelectItem key={spec.name} spec={spec} />
              ))}
            </SelectGroup>
          ) : null}
          {others.length ? (
            <SelectGroup>
              <SelectLabel>其他</SelectLabel>
              {others.map((spec) => (
                <SpecSelectItem key={spec.name} spec={spec} />
              ))}
            </SelectGroup>
          ) : null}
        </SelectContent>
      </Select>
      {selectedSpec?.docs_url ? (
        <FieldDescription>
          {selectedSpec.provider_type === "local"
            ? "本地后端需要先自行部署："
            : "接入文档："}
          <a
            className="ml-1 underline underline-offset-2"
            href={selectedSpec.docs_url}
            target="_blank"
            rel="noreferrer"
          >
            {selectedSpec.docs_url}
          </a>
        </FieldDescription>
      ) : null}
    </Field>
  )
}

function BackendFormFields({
  id,
  form,
  specs,
  onChange,
}: {
  id: string
  form: BackendFormState
  specs: ProviderSpec[]
  onChange: (patch: Partial<BackendFormState>) => void
}) {
  return (
    <FieldGroup>
      <BackendTypeSelect
        id={`${id}-type`}
        specs={specs}
        value={form.backendType}
        onChange={(value) => onChange({ backendType: value })}
      />
      <Field>
        <FieldLabel htmlFor={`${id}-display-name`}>显示名称</FieldLabel>
        <Input
          id={`${id}-display-name`}
          value={form.displayName}
          onChange={(event) => onChange({ displayName: event.target.value })}
          placeholder="留空使用后端类型默认名称"
        />
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-base-url`}>Base URL</FieldLabel>
        <Input
          id={`${id}-base-url`}
          value={form.baseUrl}
          onChange={(event) => onChange({ baseUrl: event.target.value })}
          placeholder="例如：http://127.0.0.1:11434/v1"
        />
        <FieldDescription>本地后端填服务地址，远程后端填 API 网关地址。</FieldDescription>
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-model`}>模型</FieldLabel>
        <Input
          id={`${id}-model`}
          value={form.model}
          onChange={(event) => onChange({ model: event.target.value })}
          placeholder="例如：qwen3-4b-instruct"
        />
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-api-key`}>API Key</FieldLabel>
        <Input
          id={`${id}-api-key`}
          type="password"
          value={form.apiKey}
          onChange={(event) => onChange({ apiKey: event.target.value })}
          placeholder="远程后端鉴权密钥"
        />
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-priority`}>优先级</FieldLabel>
        <Input
          id={`${id}-priority`}
          type="number"
          value={form.priority}
          onChange={(event) => onChange({ priority: event.target.value })}
        />
        <FieldDescription>同角色存在多个可用后端时按优先级挑选实例。</FieldDescription>
      </Field>
    </FieldGroup>
  )
}

function CreateBackendDialog({
  open,
  onOpenChange,
  specs,
  onCreate,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  specs: ProviderSpec[]
  onCreate: (payload: BackendPayload) => Promise<void>
}) {
  const [form, setForm] = useState<BackendFormState>(EMPTY_FORM)
  const [submitting, setSubmitting] = useState(false)
  const selectedSpec = specs.find((spec) => spec.name === form.backendType)

  async function handleSubmit() {
    if (!form.backendType) return
    setSubmitting(true)
    try {
      const displayName =
        form.displayName.trim() || selectedSpec?.display_name || form.backendType
      await onCreate({ ...payloadFromForm({ ...form, displayName }, true) })
      onOpenChange(false)
      setForm(EMPTY_FORM)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>新建后端</DialogTitle>
          <DialogDescription>
            从注册表中选择后端类型，并填入连接参数。
          </DialogDescription>
        </DialogHeader>
        <BackendFormFields
          id="backend-new"
          form={form}
          specs={specs}
          onChange={(patch) => setForm((current) => ({ ...current, ...patch }))}
        />
        <DialogFooter>
          <Button variant="outline" disabled={submitting} onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button disabled={!form.backendType || submitting} onClick={() => void handleSubmit()}>
            <IconPlus data-icon="inline-start" />
            {submitting ? "创建中…" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function EditBackendSheet({
  open,
  onOpenChange,
  backend,
  specs,
  onUpdate,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  backend: BackendConfig
  specs: ProviderSpec[]
  onUpdate: (id: string, payload: BackendPayload) => Promise<void>
}) {
  const [form, setForm] = useState<BackendFormState>(() => formFromBackend(backend))
  const [enabled, setEnabled] = useState(backend.enabled)
  const [submitting, setSubmitting] = useState(false)

  async function handleSave() {
    setSubmitting(true)
    try {
      await onUpdate(backend.id, payloadFromForm(form, enabled))
      onOpenChange(false)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-lg">
        <SheetHeader>
          <SheetTitle>编辑后端 · {backend.display_name}</SheetTitle>
          <SheetDescription>调整连接参数、优先级与启用状态。</SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto px-4">
          <FieldGroup>
            <BackendFormFields
              id={`backend-${backend.id}`}
              form={form}
              specs={specs}
              onChange={(patch) => setForm((current) => ({ ...current, ...patch }))}
            />
            <Field orientation="horizontal">
              <Switch
                id={`backend-${backend.id}-enabled`}
                checked={enabled}
                onCheckedChange={setEnabled}
              />
              <FieldLabel htmlFor={`backend-${backend.id}-enabled`}>启用该后端</FieldLabel>
            </Field>
          </FieldGroup>
        </div>
        <SheetFooter>
          <Button variant="outline" disabled={submitting} onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button disabled={submitting} onClick={() => void handleSave()}>
            保存
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

function BackendEntry({
  backend,
  specs,
  onUpdate,
  onDelete,
  onHealthCheck,
}: {
  backend: BackendConfig
  specs: ProviderSpec[]
  onUpdate: (id: string, payload: BackendPayload) => Promise<void>
  onDelete: (id: string) => Promise<void>
  onHealthCheck: (id: string) => Promise<BackendHealthCheckResult | null>
}) {
  const [editOpen, setEditOpen] = useState(false)
  const [checking, setChecking] = useState(false)
  const [health, setHealth] = useState<BackendHealthCheckResult | null>(null)
  const isLocal = backend.spec?.provider_type === "local"

  async function handleHealthCheck() {
    setChecking(true)
    try {
      setHealth(await onHealthCheck(backend.id))
    } finally {
      setChecking(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          <span>{backend.display_name}</span>
          <Badge variant={isLocal ? "secondary" : "outline"}>{providerTypeLabel(backend.spec)}</Badge>
          <Badge variant="outline">{backend.backend_type}</Badge>
          {!backend.enabled ? <Badge variant="destructive">停用</Badge> : null}
        </CardTitle>
        <CardDescription>
          优先级 {backend.priority} · 更新于 {formatTimestamp(backend.updated_at)}
        </CardDescription>
        <CardAction>
          <Switch
            aria-label={`启用 ${backend.display_name}`}
            checked={backend.enabled}
            onCheckedChange={(value) =>
              void onUpdate(backend.id, payloadFromBackend(backend, value))
            }
          />
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="text-sm text-muted-foreground">
          {settingsText(backend.settings) || "未配置 base_url / model"}
        </div>
        {health ? (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={health.healthy ? "default" : "destructive"}>
              {health.healthy ? "健康" : "不可用"}
            </Badge>
            {!health.healthy && health.status?.last_error ? (
              <span className="text-sm text-destructive">{health.status.last_error}</span>
            ) : null}
            {health.healthy && health.status?.model ? (
              <span className="text-xs text-muted-foreground">{health.status.model}</span>
            ) : null}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={checking}
            onClick={() => void handleHealthCheck()}
          >
            <IconHeartbeat data-icon="inline-start" />
            {checking ? "检查中…" : "健康检查"}
          </Button>
          <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>
            <IconEdit data-icon="inline-start" />
            编辑
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm">
                <IconTrash data-icon="inline-start" />
                删除
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>删除后端「{backend.display_name}」？</AlertDialogTitle>
                <AlertDialogDescription>
                  删除后不可恢复，绑定到该实例的角色会回退到默认后端。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction variant="destructive" onClick={() => void onDelete(backend.id)}>
                  删除
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
        {editOpen ? (
          <EditBackendSheet
            key={backend.id}
            open={editOpen}
            onOpenChange={setEditOpen}
            backend={backend}
            specs={specs}
            onUpdate={onUpdate}
          />
        ) : null}
      </CardContent>
    </Card>
  )
}

function BindingRow({
  role,
  binding,
  backends,
  onSave,
}: {
  role: string
  binding: { provider_id: string; model: string; fallback_provider_id: string | null; updated_at: number } | null
  backends: BackendConfig[]
  onSave: (role: string, payload: ProviderBindingPayload) => Promise<void>
}) {
  const [providerId, setProviderId] = useState(binding?.provider_id ?? "")
  const [model, setModel] = useState(binding?.model ?? "")
  const [fallbackId, setFallbackId] = useState(binding?.fallback_provider_id ?? "")
  const [saving, setSaving] = useState(false)

  async function handleSave() {
    setSaving(true)
    try {
      await onSave(role, {
        provider_id: providerId,
        model: model.trim(),
        fallback_provider_id: fallbackId ? fallbackId : null,
      })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">{role}</Badge>
        <span className="text-sm font-medium">{ROLE_LABELS[role] ?? role}</span>
        <span className="text-xs text-muted-foreground">
          {binding ? `更新于 ${formatTimestamp(binding.updated_at)}` : "尚未绑定"}
        </span>
      </div>
      <div className="grid gap-2 md:grid-cols-[minmax(170px,1fr)_minmax(150px,1fr)_minmax(170px,1fr)_auto] md:items-start">
        <Field>
          <FieldLabel htmlFor={`binding-${role}-provider`} className="text-xs">
            推理后端
          </FieldLabel>
          <Select
            value={providerId || UNBOUND_PROVIDER}
            onValueChange={(value) => setProviderId(value === UNBOUND_PROVIDER ? "" : value)}
          >
            <SelectTrigger id={`binding-${role}-provider`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={UNBOUND_PROVIDER}>未绑定（默认）</SelectItem>
              {backends.map((backend) => (
                <SelectItem key={backend.id} value={backend.id}>
                  {backend.display_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field>
          <FieldLabel htmlFor={`binding-${role}-model`} className="text-xs">
            模型
          </FieldLabel>
          <Input
            id={`binding-${role}-model`}
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder="留空使用后端默认模型"
          />
        </Field>
        <Field>
          <FieldLabel htmlFor={`binding-${role}-fallback`} className="text-xs">
            降级后端
          </FieldLabel>
          <Select
            value={fallbackId || UNBOUND_PROVIDER}
            onValueChange={(value) => setFallbackId(value === UNBOUND_PROVIDER ? "" : value)}
          >
            <SelectTrigger id={`binding-${role}-fallback`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={UNBOUND_PROVIDER}>无降级</SelectItem>
              {backends.map((backend) => (
                <SelectItem key={backend.id} value={backend.id}>
                  {backend.display_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <div className="flex items-end">
          <Button variant="outline" disabled={saving} onClick={() => void handleSave()}>
            {saving ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </div>
  )
}

function RoleBindingsCard({
  backends,
  bindingsPayload,
  onSaveBinding,
}: {
  backends: BackendConfig[]
  bindingsPayload: ProviderBindingsPayload
  onSaveBinding: (role: string, payload: ProviderBindingPayload) => Promise<void>
}) {
  const byRole = new Map(bindingsPayload.items.map((binding) => [binding.role, binding]))
  const roles = bindingsPayload.roles.length ? bindingsPayload.roles : DEFAULT_ROLES
  return (
    <Card>
      <CardHeader>
        <CardTitle>角色绑定</CardTitle>
        <CardDescription>
          为对话、视觉与数据清洗任务分别指定推理后端，可配置失败时的降级实例。
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {roles.map((role, index) => (
          <div key={role} className="flex flex-col gap-4">
            {index > 0 ? <Separator /> : null}
            <BindingRow
              role={role}
              binding={byRole.get(role) ?? null}
              backends={backends}
              onSave={onSaveBinding}
            />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

export function BackendsWorkspaceView({
  payload,
  bindingsPayload,
  onCreate,
  onUpdate,
  onDelete,
  onHealthCheck,
  onSaveBinding,
}: {
  payload: BackendsPayload
  bindingsPayload: ProviderBindingsPayload
  onCreate: (payload: BackendPayload) => Promise<void>
  onUpdate: (id: string, payload: BackendPayload) => Promise<void>
  onDelete: (id: string) => Promise<void>
  onHealthCheck: (id: string) => Promise<BackendHealthCheckResult | null>
  onSaveBinding: (role: string, payload: ProviderBindingPayload) => Promise<void>
}) {
  const [createOpen, setCreateOpen] = useState(false)

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>推理后端</h1>
        <p>管理本地与在线推理后端实例，并把角色任务绑定到具体实例。</p>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="outline">实例 {payload.total}</Badge>
        <Badge variant="outline">可用类型 {payload.available_types.length}</Badge>
        <Button onClick={() => setCreateOpen(true)}>
          <IconPlus data-icon="inline-start" />
          新建后端
        </Button>
      </div>
      {payload.items.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {payload.items.map((backend) => (
            <BackendEntry
              key={backend.id}
              backend={backend}
              specs={payload.available_types}
              onUpdate={onUpdate}
              onDelete={onDelete}
              onHealthCheck={onHealthCheck}
            />
          ))}
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <IconServer2 />
            </EmptyMedia>
            <EmptyTitle>暂无推理后端</EmptyTitle>
            <EmptyDescription>先创建一个本地或在线后端实例。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
      <RoleBindingsCard
        backends={payload.items}
        bindingsPayload={bindingsPayload}
        onSaveBinding={onSaveBinding}
      />
      <CreateBackendDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        specs={payload.available_types}
        onCreate={onCreate}
      />
    </div>
  )
}

export function BackendsWorkspace() {
  const { data, error, isLoading, mutate } = useBackends()
  const bindingsQuery = useProviderBindings()

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取推理后端</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return null
  }

  const bindingsPayload: ProviderBindingsPayload = bindingsQuery.data ?? {
    items: [],
    roles: DEFAULT_ROLES,
  }

  async function refreshAll() {
    await mutate()
    await bindingsQuery.mutate()
  }

  async function runMutation(action: () => Promise<unknown>, successMessage: string) {
    try {
      await action()
      toast.success(successMessage)
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    }
    await refreshAll()
  }

  return (
    <BackendsWorkspaceView
      payload={data}
      bindingsPayload={bindingsPayload}
      onCreate={(payload) => runMutation(() => createBackend(payload), "后端已创建")}
      onUpdate={(id, payload) => runMutation(() => updateBackend(id, payload), "后端已更新")}
      onDelete={(id) => runMutation(() => deleteBackend(id), "后端已删除")}
      onHealthCheck={async (id) => {
        try {
          return await healthCheckBackend(id)
        } catch (caught) {
          toast.error(await apiErrorMessage(caught))
          return null
        }
      }}
      onSaveBinding={(role, payload) =>
        runMutation(() => setProviderBinding(role, payload), `「${ROLE_LABELS[role] ?? role}」绑定已保存`)
      }
    />
  )
}
