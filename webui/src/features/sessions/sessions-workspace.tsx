import { useMemo, useState } from "react"
import {
  IconChevronRight,
  IconSend,
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
import { Item, ItemActions, ItemContent } from "@/components/ui/item"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"
import {
  apiErrorMessage,
  pokeSession,
  useSessionLogs,
  useSessions,
} from "@/lib/api"
import type {
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


export function SessionsWorkspace() {
  const { data: sessions, error: sessionsError, isLoading: sessionsLoading } = useSessions(50)
  const [pokingKey, setPokingKey] = useState<string | null>(null)

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

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>会话监控</h1>
        <p>
          按会话聚合的回复监控与手动接话。群/成员记忆已独立成页，见左侧「记忆」。
        </p>
      </section>
      {sessionsError ? (
        <Alert variant="destructive">
          <AlertTitle>无法读取会话数据</AlertTitle>
          <AlertDescription>{String(sessionsError)}</AlertDescription>
        </Alert>
      ) : null}
      <Tabs defaultValue="sessions">
        <TabsList>
          <TabsTrigger value="sessions">会话</TabsTrigger>
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
      </Tabs>
    </div>
  )
}
