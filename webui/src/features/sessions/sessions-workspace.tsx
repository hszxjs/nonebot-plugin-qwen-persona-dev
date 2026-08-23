import { useMemo, useState } from "react"
import {
  IconChevronRight,
  IconClock,
  IconSend,
  IconSparkles,
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Item, ItemActions, ItemContent } from "@/components/ui/item"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"
import {
  apiErrorMessage,
  decayMemories,
  deleteMemory,
  extractMemories,
  pokeSession,
  useMemories,
  useSessionLogs,
  useSessions,
} from "@/lib/api"
import { mutate } from "swr"
import type {
  LongTermMemory,
  MemoriesPayload,
  MessageLogEntry,
  SessionSummary,
  SessionsPayload,
} from "@/lib/types"

const SUCCESS_STATUSES = new Set(["success", "sent", "ok", "completed"])
const FAILED_STATUSES = new Set(["failed", "error", "rejected", "timeout"])

function secondsText(value: number | null | undefined) {
  if (value === null || value === undefined) return "-"
  return `${value.toFixed(2)}s`
}

function formatTimestamp(value: number | undefined) {
  if (!value) return "-"
  const date = new Date(value > 1e12 ? value : value * 1000)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString("zh-CN", { hour12: false })
}

function clockText(entry: MessageLogEntry) {
  let date: Date | null = null
  if (entry.time) {
    date = new Date(entry.time)
  } else if (entry.created_at) {
    date = new Date(entry.created_at > 1e12 ? entry.created_at : entry.created_at * 1000)
  }
  if (!date || Number.isNaN(date.getTime())) return entry.time || "-"
  return date.toLocaleString("zh-CN", { hour12: false })
}

function statusDotClass(status?: string) {
  if (!status) return "bg-muted-foreground/40"
  if (SUCCESS_STATUSES.has(status)) return "bg-emerald-500"
  if (FAILED_STATUSES.has(status)) return "bg-destructive"
  return "bg-amber-500"
}

function sessionKindText(sessionKey: string) {
  return sessionKey.startsWith("private:") ? "私聊" : "群聊"
}

function groupIdOf(sessionKey: string) {
  return sessionKey.startsWith("private:") ? null : sessionKey
}

function userIdOf(sessionKey: string) {
  return sessionKey.startsWith("private:") ? sessionKey.slice("private:".length) : null
}

function SessionLogRow({ entry }: { entry: MessageLogEntry }) {
  const providerModel = [entry.provider, entry.model].filter(Boolean).join("/")
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border px-2 py-1.5 text-sm" data-testid="session-log-row">
      <span className="font-mono text-xs whitespace-nowrap text-muted-foreground">
        {clockText(entry)}
      </span>
      <span
        title={entry.status || ""}
        className={cn("size-2 shrink-0 rounded-full", statusDotClass(entry.status))}
      />
      <span className="shrink-0">{entry.trigger_reason || "-"}</span>
      <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
        {[entry.input_preview, entry.output_preview].filter(Boolean).join(" → ") || "-"}
      </span>
      <Badge variant="outline" className="shrink-0">
        耗时 {secondsText(entry.latency_seconds)}
      </Badge>
      {providerModel ? (
        <Badge variant="outline" className="shrink-0">
          {providerModel}
        </Badge>
      ) : null}
    </div>
  )
}

function SessionCard({
  session,
  onPoke,
  poking,
}: {
  session: SessionSummary
  onPoke: (groupId: string) => void
  poking: boolean
}) {
  const [open, setOpen] = useState(false)
  const groupId = groupIdOf(session.session_key)
  const userId = userIdOf(session.session_key)
  const { data, isLoading } = useSessionLogs(session.session_key, open)
  const entries = useMemo(() => {
    const items = (data?.items ?? []).filter((entry): entry is MessageLogEntry => entry.source === "message")
    // 私聊会话没有 group_id，下钻日志按 user_id 过滤
    return userId ? items.filter((entry) => entry.user_id === userId) : items
  }, [data, userId])
  const latest = session.latest

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Item variant="outline" data-testid="session-card">
        <ItemContent>
          <div className="flex flex-wrap items-center gap-2">
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left outline-none"
                aria-label={`展开会话 ${session.session_key} 日志`}
              >
                <span className="font-mono text-sm font-medium">{session.session_key}</span>
                <Badge variant="secondary">{sessionKindText(session.session_key)}</Badge>
                <Badge variant="outline">{session.count} 条</Badge>
                <span
                  title={latest.status || ""}
                  className={cn("size-2 shrink-0 rounded-full", statusDotClass(latest.status))}
                />
                <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">
                  {[latest.trigger_reason, latest.role_card_name].filter(Boolean).join(" · ") || "-"}
                </span>
                <span className="text-xs whitespace-nowrap text-muted-foreground">
                  {clockText(latest)}
                </span>
                <Badge variant="outline">耗时 {secondsText(latest.latency_seconds)}</Badge>
                <IconChevronRight
                  data-icon="inline-end"
                  className={cn(
                    "size-4 shrink-0 text-muted-foreground transition-transform",
                    open && "rotate-90"
                  )}
                />
              </button>
            </CollapsibleTrigger>
            {groupId ? (
              <ItemActions>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="outline" size="sm" disabled={poking}>
                      <IconSend data-icon="inline-start" />
                      {poking ? "发送中" : "手动触发接话"}
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>在会话 {groupId} 触发一次主动接话？</AlertDialogTitle>
                      <AlertDialogDescription>
                        需要有 bot 在线，bot 会主动向该群发送一条接话消息。
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>取消</AlertDialogCancel>
                      <AlertDialogAction onClick={() => onPoke(groupId)}>确认触发</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </ItemActions>
            ) : null}
          </div>
          <CollapsibleContent>
            <div className="mt-2 flex flex-col gap-1.5" data-testid="session-log-list">
              {isLoading ? (
                <Skeleton className="h-20" />
              ) : entries.length ? (
                entries.map((entry, index) => (
                  <SessionLogRow key={`${entry.created_at}-${entry.reply_message_id ?? index}`} entry={entry} />
                ))
              ) : (
                <p className="text-sm text-muted-foreground">该会话暂无 message 日志。</p>
              )}
            </div>
          </CollapsibleContent>
        </ItemContent>
      </Item>
    </Collapsible>
  )
}

export function SessionsTabView({
  payload,
  pokingKey,
  onPoke,
}: {
  payload: SessionsPayload | undefined
  pokingKey: string | null
  onPoke: (groupId: string) => void
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline">会话 {payload?.total ?? 0} 个</Badge>
        <span className="text-xs text-muted-foreground">
          按最近回复聚合的群聊 / 私聊会话，展开查看该会话的回复日志。
        </span>
      </div>
      {payload?.items.length ? (
        <div className="flex flex-col gap-2" data-testid="session-list">
          {payload.items.map((session) => (
            <SessionCard
              key={session.session_key}
              session={session}
              poking={pokingKey === session.session_key}
              onPoke={onPoke}
            />
          ))}
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>暂无会话记录</EmptyTitle>
            <EmptyDescription>等待 bot 产生回复日志后，会话会按群聊/私聊聚合在这里。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
    </div>
  )
}

function MemoryKindBadge({ kind }: { kind: string }) {
  const variant =
    kind === "profile" ? "default" : kind === "preference" ? "secondary" : "outline"
  return <Badge variant={variant}>{kind}</Badge>
}

export function MemoriesTabView({
  payload,
  userId,
  onUserIdChange,
  includeDecayed,
  onIncludeDecayedChange,
  extracting,
  decaying,
  deletingId,
  onExtract,
  onDecay,
  onDelete,
}: {
  payload: MemoriesPayload | undefined
  userId: string
  onUserIdChange: (value: string) => void
  includeDecayed: boolean
  onIncludeDecayedChange: (value: boolean) => void
  extracting: boolean
  decaying: boolean
  deletingId: number | null
  onExtract: (userId: string) => void
  onDecay: () => void
  onDelete: (id: number) => void
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label="用户 ID 过滤"
          placeholder="按用户 ID 过滤"
          className="w-44"
          value={userId}
          onChange={(event) => onUserIdChange(event.target.value)}
        />
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="size-4 accent-primary"
            checked={includeDecayed}
            onChange={(event) => onIncludeDecayedChange(event.target.checked)}
          />
          显示已衰减
        </label>
        <Button size="sm" disabled={!userId.trim() || extracting} onClick={() => onExtract(userId.trim())}>
          <IconSparkles data-icon="inline-start" />
          {extracting ? "提取中…" : "手动提取记忆"}
        </Button>
        <Button variant="outline" size="sm" disabled={decaying} onClick={onDecay}>
          <IconClock data-icon="inline-start" />
          {decaying ? "执行中…" : "执行衰减"}
        </Button>
        <Badge variant="outline">共 {payload?.total ?? 0} 条</Badge>
      </div>
      <p className="text-xs text-muted-foreground">
        手动提取会为过滤框中的用户 ID 重新抽取长期记忆；执行衰减按半衰期更新重要度并标记遗忘。
      </p>
      {payload?.items.length ? (
        <div className="overflow-x-auto rounded-lg border" data-testid="memories-table">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
              <tr>
                <th className="px-3 py-2 font-medium">用户</th>
                <th className="px-3 py-2 font-medium">类型</th>
                <th className="px-3 py-2 font-medium">内容</th>
                <th className="px-3 py-2 font-medium">重要度</th>
                <th className="px-3 py-2 font-medium">最近见到</th>
                <th className="px-3 py-2 font-medium">状态</th>
                <th className="px-3 py-2 text-right font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {payload.items.map((memory: LongTermMemory) => (
                <tr key={memory.id} data-testid="memory-row">
                  <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">{memory.user_id}</td>
                  <td className="px-3 py-2">
                    <MemoryKindBadge kind={memory.kind} />
                  </td>
                  <td className="max-w-[320px] px-3 py-2">
                    <span className="block truncate" title={memory.content}>
                      {memory.content}
                    </span>
                  </td>
                  <td className="px-3 py-2 tabular-nums">{memory.importance.toFixed(2)}</td>
                  <td className="px-3 py-2 whitespace-nowrap text-muted-foreground">
                    {formatTimestamp(memory.last_seen_at)}
                  </td>
                  <td className="px-3 py-2">
                    {memory.decayed ? <Badge variant="destructive">已衰减</Badge> : <Badge variant="secondary">生效中</Badge>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="destructive"
                          size="sm"
                          disabled={deletingId === memory.id}
                          aria-label={`删除记忆 ${memory.id}`}
                        >
                          <IconTrash data-icon="inline-start" />
                          删除
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>删除这条记忆？</AlertDialogTitle>
                          <AlertDialogDescription>{memory.content}</AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>取消</AlertDialogCancel>
                          <AlertDialogAction variant="destructive" onClick={() => onDelete(memory.id)}>
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
            <EmptyTitle>暂无长期记忆</EmptyTitle>
            <EmptyDescription>手动提取或等待后台自动抽取后，记忆会按重要度排序显示。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
    </div>
  )
}

export function SessionsWorkspace() {
  const { data: sessions, error: sessionsError, isLoading: sessionsLoading } = useSessions(50)
  const [userId, setUserId] = useState("")
  const [includeDecayed, setIncludeDecayed] = useState(false)
  const memories = useMemories(userId, includeDecayed)
  const [pokingKey, setPokingKey] = useState<string | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [decaying, setDecaying] = useState(false)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  async function handlePoke(groupId: string) {
    setPokingKey(groupId)
    try {
      const result = await pokeSession(groupId)
      toast.success(`已触发接话，消息 ID ${result.reply_message_id}`)
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setPokingKey(null)
    }
  }

  async function handleExtract(targetUserId: string) {
    setExtracting(true)
    try {
      const result = await extractMemories(targetUserId)
      toast.success(`已提取 ${result.count} 条记忆`)
      await mutate((key) => typeof key === "string" && key.startsWith("memories?"))
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setExtracting(false)
    }
  }

  async function handleDecay() {
    setDecaying(true)
    try {
      const result = await decayMemories()
      toast.success(`衰减完成：${result.decayed} 条被标记遗忘 / 共 ${result.total} 条`)
      await mutate((key) => typeof key === "string" && key.startsWith("memories?"))
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setDecaying(false)
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id)
    try {
      await deleteMemory(id)
      toast.success("已删除记忆")
      await mutate((key) => typeof key === "string" && key.startsWith("memories?"))
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>会话监控</h1>
        <p>按会话聚合的回复监控、手动接话与长期记忆管理。</p>
      </section>
      {sessionsError ? (
        <Alert variant="destructive">
          <AlertTitle>无法读取会话数据</AlertTitle>
          <AlertDescription>{String(sessionsError)}</AlertDescription>
        </Alert>
      ) : null}
      {memories.error ? (
        <Alert variant="destructive">
          <AlertTitle>无法读取记忆数据</AlertTitle>
          <AlertDescription>{String(memories.error)}</AlertDescription>
        </Alert>
      ) : null}
      <Tabs defaultValue="sessions">
        <TabsList>
          <TabsTrigger value="sessions">会话</TabsTrigger>
          <TabsTrigger value="memories">记忆</TabsTrigger>
        </TabsList>
        <TabsContent value="sessions">
          {sessionsLoading ? (
            <Skeleton className="h-64" />
          ) : (
            <SessionsTabView
              payload={sessions}
              pokingKey={pokingKey}
              onPoke={(groupId) => void handlePoke(groupId)}
            />
          )}
        </TabsContent>
        <TabsContent value="memories">
          {memories.isLoading ? (
            <Skeleton className="h-64" />
          ) : (
            <MemoriesTabView
              payload={memories.data}
              userId={userId}
              onUserIdChange={setUserId}
              includeDecayed={includeDecayed}
              onIncludeDecayedChange={setIncludeDecayed}
              extracting={extracting}
              decaying={decaying}
              deletingId={deletingId}
              onExtract={(target) => void handleExtract(target)}
              onDecay={() => void handleDecay()}
              onDelete={(id) => void handleDelete(id)}
            />
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
