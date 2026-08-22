import { useState } from "react"
import {
  IconCopy,
  IconEdit,
  IconIdBadge2,
  IconLink,
  IconPlus,
  IconRefresh,
  IconTrash,
  IconUnlink,
  IconUserScan,
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
import { Checkbox } from "@/components/ui/checkbox"
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
  FieldError,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Item,
  ItemActions,
  ItemContent,
  ItemTitle,
} from "@/components/ui/item"
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
import { Textarea } from "@/components/ui/textarea"
import {
  apiErrorMessage,
  bindRoleCard,
  cloneRoleCard,
  createRoleCard,
  deleteRoleCard,
  generateMemberMimic,
  refreshMemberMimicStyle,
  unbindRoleCard,
  updateRoleCard,
  useRoleCards,
  useTools,
} from "@/lib/api"
import type { RoleCard, RoleCardPayload, RoleCardsPayload, ToolInfo } from "@/lib/types"

const PROMPT_VARIABLE_HINT =
  "可用模板变量：{persona_name} {user_nickname} {time} {memory_summary}"

function formatTimestamp(value: number) {
  const date = new Date(value > 1e12 ? value : value * 1000)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

function splitLines(text: string) {
  return text
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean)
}

function unbindBodyFromSession(sessionKey: string): { group_id?: string; user_id?: string } {
  const separator = sessionKey.indexOf(":")
  if (separator === -1) return {}
  const scope = sessionKey.slice(0, separator)
  const id = sessionKey.slice(separator + 1)
  if (!id) return {}
  if (scope.includes("group")) return { group_id: id }
  if (scope.includes("user") || scope.includes("private")) return { user_id: id }
  return {}
}

function ToolPermissionMatrix({
  tools,
  selected,
  onToggle,
}: {
  tools: ToolInfo[]
  selected: Set<string>
  onToggle: (name: string, checked: boolean) => void
}) {
  return (
    <FieldGroup data-slot="checkbox-group">
      {tools.length ? (
        tools.map((tool) => (
          <Field key={tool.name} orientation="horizontal">
            <Checkbox
              id={`tool-${tool.name}`}
              checked={selected.has(tool.name)}
              onCheckedChange={(checked) => onToggle(tool.name, checked === true)}
            />
            <FieldLabel htmlFor={`tool-${tool.name}`}>{tool.name}</FieldLabel>
            {tool.description ? (
              <FieldDescription>{tool.description}</FieldDescription>
            ) : null}
          </Field>
        ))
      ) : (
        <FieldDescription>后台没有可用的工具注册。</FieldDescription>
      )}
    </FieldGroup>
  )
}

function RoleCardFormSheet({
  open,
  onOpenChange,
  card,
  tools,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  card: RoleCard | null
  tools: ToolInfo[]
  onSave: (payload: RoleCardPayload) => Promise<void>
}) {
  const isMimic = card?.role_type === "member_mimic"
  const [name, setName] = useState(card?.name ?? "")
  const [systemPrompt, setSystemPrompt] = useState(card?.system_prompt ?? "")
  const [toneExamples, setToneExamples] = useState((card?.tone_examples ?? []).join("\n"))
  const [styleTags, setStyleTags] = useState((card?.style_tags ?? []).join("\n"))
  const [allowedTools, setAllowedTools] = useState<Set<string>>(
    new Set(card?.allowed_tools ?? [])
  )
  const [enabled, setEnabled] = useState(card?.enabled ?? true)
  const fieldId = card ? `role-card-${card.id}` : "role-card-new"

  function toggleTool(toolName: string, checked: boolean) {
    setAllowedTools((current) => {
      const next = new Set(current)
      if (checked) {
        next.add(toolName)
      } else {
        next.delete(toolName)
      }
      return next
    })
  }

  async function handleSave() {
    const payload: RoleCardPayload = isMimic
      ? { name, allowed_tools: [...allowedTools], enabled }
      : {
          name,
          system_prompt: systemPrompt,
          tone_examples: splitLines(toneExamples),
          style_tags: splitLines(styleTags),
          allowed_tools: [...allowedTools],
          enabled,
        }
    await onSave(payload)
    onOpenChange(false)
    if (!card) {
      setName("")
      setSystemPrompt("")
      setToneExamples("")
      setStyleTags("")
      setAllowedTools(new Set())
      setEnabled(true)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-lg">
        <SheetHeader>
          <SheetTitle>{card ? "编辑角色卡" : "新建人格"}</SheetTitle>
          <SheetDescription>
            {isMimic
              ? "模仿卡的提示词与语言风格由分析生成，只能调整名称、工具权限和启用状态。"
              : "配置人格的系统提示词、语言风格和工具权限。"}
          </SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor={`${fieldId}-name`}>名称</FieldLabel>
              <Input
                id={`${fieldId}-name`}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="例如：毒舌管家"
              />
            </Field>
            <Field>
              <FieldLabel htmlFor={`${fieldId}-system-prompt`}>
                系统提示词{isMimic ? "（分析生成，只读）" : ""}
              </FieldLabel>
              <Textarea
                id={`${fieldId}-system-prompt`}
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                readOnly={isMimic}
                rows={6}
                placeholder="描述人格的身份、性格与回复约束"
              />
              <FieldDescription>
                {isMimic ? "由群聊样本分析生成，可通过「刷新语言风格」重新分析。" : PROMPT_VARIABLE_HINT}
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel htmlFor={`${fieldId}-tone-examples`}>
                语气示例{isMimic ? "（分析生成，只读）" : "（每行一条）"}
              </FieldLabel>
              <Textarea
                id={`${fieldId}-tone-examples`}
                value={toneExamples}
                onChange={(event) => setToneExamples(event.target.value)}
                readOnly={isMimic}
                rows={4}
                placeholder={"例如：这么简单还要问？\n例如：笑死，看看你。"}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor={`${fieldId}-style-tags`}>
                风格标签{isMimic ? "（分析生成，只读）" : "（每行一个）"}
              </FieldLabel>
              <Textarea
                id={`${fieldId}-style-tags`}
                value={styleTags}
                onChange={(event) => setStyleTags(event.target.value)}
                readOnly={isMimic}
                rows={3}
                placeholder={"例如：毒舌\n例如：短句"}
              />
            </Field>
            <Field>
              <FieldLabel>允许使用的工具（工具权限矩阵）</FieldLabel>
              <ToolPermissionMatrix
                tools={tools}
                selected={allowedTools}
                onToggle={toggleTool}
              />
            </Field>
            <Field orientation="horizontal">
              <Switch
                id={`${fieldId}-enabled`}
                checked={enabled}
                onCheckedChange={setEnabled}
              />
              <FieldLabel htmlFor={`${fieldId}-enabled`}>启用该角色卡</FieldLabel>
            </Field>
          </FieldGroup>
        </div>
        <SheetFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button disabled={!name.trim()} onClick={() => void handleSave()}>
            保存
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

function BindRoleCardDialog({
  open,
  onOpenChange,
  card,
  onBind,
  onUnbind,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  card: RoleCard
  onBind: (id: string, body: { group_id?: string }) => Promise<void>
  onUnbind: (body: { group_id?: string; user_id?: string }) => Promise<void>
}) {
  const [groupId, setGroupId] = useState("")

  async function handleBind() {
    const trimmed = groupId.trim()
    if (!trimmed) return
    await onBind(card.id, { group_id: trimmed })
    setGroupId("")
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>绑定群 · {card.name}</DialogTitle>
          <DialogDescription>
            将角色卡绑定到群会话，每个会话同一时间只使用一张卡。
          </DialogDescription>
        </DialogHeader>
        {card.bindings.length ? (
          <FieldGroup data-slot="checkbox-group">
            {card.bindings.map((sessionKey) => {
              const body = unbindBodyFromSession(sessionKey)
              return (
                <Item key={sessionKey} variant="outline" size="sm">
                  <ItemContent>
                    <ItemTitle>{sessionKey}</ItemTitle>
                  </ItemContent>
                  <ItemActions>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!body.group_id && !body.user_id}
                      onClick={() => void onUnbind(body)}
                    >
                      <IconUnlink data-icon="inline-start" />
                      解绑
                    </Button>
                  </ItemActions>
                </Item>
              )
            })}
          </FieldGroup>
        ) : (
          <FieldDescription>当前没有绑定任何会话。</FieldDescription>
        )}
        <Field>
          <FieldLabel htmlFor={`bind-${card.id}`}>群号</FieldLabel>
          <div className="flex gap-2">
            <Input
              id={`bind-${card.id}`}
              value={groupId}
              onChange={(event) => setGroupId(event.target.value)}
              placeholder="例如：888"
            />
            <Button variant="outline" disabled={!groupId.trim()} onClick={() => void handleBind()}>
              <IconLink data-icon="inline-start" />
              绑定
            </Button>
          </div>
          <FieldDescription>绑定后该群会使用此角色卡的提示词与工具权限。</FieldDescription>
        </Field>
      </DialogContent>
    </Dialog>
  )
}

function MemberMimicDialog({
  open,
  onOpenChange,
  onGenerate,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onGenerate: (userId: string, groupId?: string) => Promise<void>
}) {
  const [userId, setUserId] = useState("")
  const [groupId, setGroupId] = useState("")
  const [error, setError] = useState("")
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit() {
    const trimmedUser = userId.trim()
    if (!trimmedUser) return
    const trimmedGroup = groupId.trim()
    setSubmitting(true)
    setError("")
    try {
      await onGenerate(trimmedUser, trimmedGroup || undefined)
      onOpenChange(false)
      setUserId("")
      setGroupId("")
    } catch (caught) {
      setError(await apiErrorMessage(caught))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>生成群友模仿卡</DialogTitle>
          <DialogDescription>
            基于群聊记录分析成员的语言风格，生成 member_mimic 角色卡。
          </DialogDescription>
        </DialogHeader>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="mimic-user">QQ 号</FieldLabel>
            <Input
              id="mimic-user"
              value={userId}
              onChange={(event) => setUserId(event.target.value)}
              placeholder="被模仿成员的 QQ 号"
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="mimic-group">群号（可选）</FieldLabel>
            <Input
              id="mimic-group"
              value={groupId}
              onChange={(event) => setGroupId(event.target.value)}
              placeholder="限定取样群，留空使用全部样本"
            />
          </Field>
        </FieldGroup>
        {error ? <FieldError>{error}</FieldError> : null}
        <DialogFooter>
          <Button variant="outline" disabled={submitting} onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button disabled={!userId.trim() || submitting} onClick={() => void handleSubmit()}>
            <IconUserScan data-icon="inline-start" />
            {submitting ? "分析中…" : "开始分析"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function RoleCardEntry({
  card,
  tools,
  onUpdate,
  onClone,
  onBind,
  onUnbind,
  onDelete,
  onRefreshStyle,
}: {
  card: RoleCard
  tools: ToolInfo[]
  onUpdate: (id: string, payload: Partial<RoleCardPayload>) => Promise<void>
  onClone: (id: string, name: string) => Promise<void>
  onBind: (id: string, body: { group_id?: string }) => Promise<void>
  onUnbind: (body: { group_id?: string; user_id?: string }) => Promise<void>
  onDelete: (id: string) => Promise<void>
  onRefreshStyle: (id: string) => Promise<void>
}) {
  const [editOpen, setEditOpen] = useState(false)
  const [bindOpen, setBindOpen] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const isMimic = card.role_type === "member_mimic"

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          <span>{card.name}</span>
          {isMimic ? (
            <>
              <Badge variant="secondary">模仿</Badge>
              {card.target_user_id ? (
                <Badge variant="outline">QQ {card.target_user_id}</Badge>
              ) : null}
            </>
          ) : (
            <Badge>人格</Badge>
          )}
          {!card.enabled ? <Badge variant="destructive">停用</Badge> : null}
        </CardTitle>
        <CardDescription>更新于 {formatTimestamp(card.updated_at)}</CardDescription>
        <CardAction>
          <Switch
            aria-label={`启用 ${card.name}`}
            checked={card.enabled}
            onCheckedChange={(value) => void onUpdate(card.id, { enabled: value })}
          />
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {card.bindings.length ? (
          <div className="flex flex-wrap gap-2">
            {card.bindings.map((sessionKey) => (
              <Badge key={sessionKey} variant="outline">
                {sessionKey}
              </Badge>
            ))}
          </div>
        ) : (
          <CardDescription>未绑定会话</CardDescription>
        )}
        <Separator />
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>
            <IconEdit data-icon="inline-start" />
            编辑
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void onClone(card.id, `${card.name} 副本`)}
          >
            <IconCopy data-icon="inline-start" />
            克隆
          </Button>
          <Button variant="outline" size="sm" onClick={() => setBindOpen(true)}>
            <IconLink data-icon="inline-start" />
            绑定群
          </Button>
          {isMimic ? (
            <Button
              variant="outline"
              size="sm"
              disabled={refreshing}
              onClick={() => {
                setRefreshing(true)
                void onRefreshStyle(card.id).finally(() => setRefreshing(false))
              }}
            >
              <IconRefresh data-icon="inline-start" />
              {refreshing ? "刷新中…" : "刷新语言风格"}
            </Button>
          ) : null}
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm">
                <IconTrash data-icon="inline-start" />
                删除
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>删除角色卡「{card.name}」？</AlertDialogTitle>
                <AlertDialogDescription>
                  删除后不可恢复，已绑定的会话将回退到默认人格。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction variant="destructive" onClick={() => void onDelete(card.id)}>
                  删除
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
        {editOpen ? (
          <RoleCardFormSheet
            key={card.id}
            open={editOpen}
            onOpenChange={setEditOpen}
            card={card}
            tools={tools}
            onSave={(payload) => onUpdate(card.id, payload)}
          />
        ) : null}
        <BindRoleCardDialog
          open={bindOpen}
          onOpenChange={setBindOpen}
          card={card}
          onBind={onBind}
          onUnbind={onUnbind}
        />
      </CardContent>
    </Card>
  )
}

export function RoleCardsWorkspaceView({
  payload,
  tools,
  onCreate,
  onUpdate,
  onDelete,
  onClone,
  onBind,
  onUnbind,
  onGenerateMimic,
  onRefreshStyle,
}: {
  payload: RoleCardsPayload
  tools: ToolInfo[]
  onCreate: (payload: RoleCardPayload) => Promise<void>
  onUpdate: (id: string, payload: Partial<RoleCardPayload>) => Promise<void>
  onDelete: (id: string) => Promise<void>
  onClone: (id: string, name: string) => Promise<void>
  onBind: (id: string, body: { group_id?: string }) => Promise<void>
  onUnbind: (body: { group_id?: string; user_id?: string }) => Promise<void>
  onGenerateMimic: (userId: string, groupId?: string) => Promise<void>
  onRefreshStyle: (id: string) => Promise<void>
}) {
  const [createOpen, setCreateOpen] = useState(false)
  const [mimicOpen, setMimicOpen] = useState(false)

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>角色卡</h1>
        <p>维护 bot 人格与群友模仿卡，并把卡片绑定到具体群会话。</p>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="outline">总计 {payload.total}</Badge>
        <Button onClick={() => setCreateOpen(true)}>
          <IconPlus data-icon="inline-start" />
          新建人格
        </Button>
        <Button variant="outline" onClick={() => setMimicOpen(true)}>
          <IconUserScan data-icon="inline-start" />
          生成群友模仿卡
        </Button>
      </div>
      {payload.items.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {payload.items.map((card) => (
            <RoleCardEntry
              key={card.id}
              card={card}
              tools={tools}
              onUpdate={onUpdate}
              onClone={onClone}
              onBind={onBind}
              onUnbind={onUnbind}
              onDelete={onDelete}
              onRefreshStyle={onRefreshStyle}
            />
          ))}
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <IconIdBadge2 />
            </EmptyMedia>
            <EmptyTitle>暂无角色卡</EmptyTitle>
            <EmptyDescription>先新建一个人格，或从群成员的聊天记录生成模仿卡。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
      {createOpen ? (
        <RoleCardFormSheet
          open={createOpen}
          onOpenChange={setCreateOpen}
          card={null}
          tools={tools}
          onSave={onCreate}
        />
      ) : null}
      <MemberMimicDialog
        open={mimicOpen}
        onOpenChange={setMimicOpen}
        onGenerate={onGenerateMimic}
      />
    </div>
  )
}

export function RoleCardsWorkspace() {
  const { data, error, isLoading, mutate } = useRoleCards()
  const toolsQuery = useTools()

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取角色卡</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return null
  }

  const tools = toolsQuery.data?.items ?? []

  async function runMutation(action: () => Promise<unknown>, successMessage: string) {
    try {
      await action()
      toast.success(successMessage)
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    }
    await mutate()
  }

  return (
    <RoleCardsWorkspaceView
      payload={data}
      tools={tools}
      onCreate={(payload) => runMutation(() => createRoleCard(payload), "角色卡已创建")}
      onUpdate={(id, payload) => runMutation(() => updateRoleCard(id, payload), "角色卡已更新")}
      onDelete={(id) => runMutation(() => deleteRoleCard(id), "角色卡已删除")}
      onClone={(id, name) => runMutation(() => cloneRoleCard(id, name), `已克隆为「${name}」`)}
      onBind={(id, body) => runMutation(() => bindRoleCard(id, body), "已绑定群会话")}
      onUnbind={(body) => runMutation(() => unbindRoleCard(body), "已解绑会话")}
      onRefreshStyle={(id) => runMutation(() => refreshMemberMimicStyle(id), "语言风格已刷新")}
      onGenerateMimic={async (userId, groupId) => {
        await generateMemberMimic(userId, groupId)
        toast.success("群友模仿卡已生成")
        await mutate()
      }}
    />
  )
}
