import { useState } from "react"
import { IconBrain, IconSparkles, IconRefresh, IconTrash, IconUserOff } from "@tabler/icons-react"
import { toast } from "sonner"

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
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  apiErrorMessage,
  decayPersonaMemories,
  deletePersonaMemory,
  extractPersonaMemories,
  purgeMemberMemories,
  refreshConversationNames,
  updatePersonaMemory,
  usePersonaMemories,
  usePersonaMemoryGroups,
} from "@/lib/api"
import type { PersonaMemory } from "@/lib/types"

const ALL_GROUPS = "__all__"

const SENSITIVITY_LABELS: Record<string, string> = {
  neutral: "中性",
  relation: "关系",
  emotion: "情绪",
  private: "隐私",
}

function sensitivityVariant(sensitivity: string) {
  if (sensitivity === "private") return "destructive" as const
  if (sensitivity === "emotion") return "default" as const
  if (sensitivity === "relation") return "secondary" as const
  return "outline" as const
}

function formatTime(value: number) {
  if (!value) return "-"
  return new Date(value > 1e12 ? value : value * 1000).toLocaleString()
}

export function MemoriesWorkspace() {
  const [scope, setScope] = useState<"group" | "member">("group")
  const [groupId, setGroupId] = useState<string>(ALL_GROUPS)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<PersonaMemory | null>(null)
  const [editContent, setEditContent] = useState("")

  const groups = usePersonaMemoryGroups()
  const effectiveGroup = groupId === ALL_GROUPS ? "" : groupId
  const memories = usePersonaMemories(scope, effectiveGroup)
  const items = memories.data?.items ?? []

  async function runAction(action: () => Promise<string>) {
    setBusy(true)
    try {
      toast.success(await action())
      await memories.mutate()
    } catch (error) {
      toast.error(await apiErrorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>记忆</h1>
        <p className="text-sm text-muted-foreground">
          群记忆与成员记忆分开维护：群聊只会用到「中性 / 关系」类，情绪与隐私类只在私聊生效。
        </p>
      </section>

      <Card data-testid="memories-toolbar">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <IconBrain className="size-4" /> 记忆库
          </CardTitle>
          <CardDescription>批量增量抽取会自动进行；这里可以手动补抽、衰减与清理。</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2">
          <Select value={groupId} onValueChange={setGroupId}>
            <SelectTrigger className="w-[220px]" aria-label="选择群">
              <SelectValue placeholder="全部群" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_GROUPS}>全部群</SelectItem>
              {(groups.data?.groups ?? []).map((item) => (
                <SelectItem key={item} value={item}>
                  {item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={busy}
            onClick={() =>
              void runAction(async () => {
                const result = await extractPersonaMemories(effectiveGroup)
                return `抽取完成：新增 ${result.saved} 条${result.failed ? `，失败 ${result.failed} 个会话` : ""}`
              })
            }
          >
            <IconRefresh data-icon="inline-start" />
            立即抽取
          </Button>
          <Button
            variant="outline"
            disabled={busy}
            onClick={() =>
              void runAction(async () => {
                const result = await decayPersonaMemories()
                return `衰减完成：更新 ${result.updated} 条，标记过期 ${result.decayed} 条`
              })
            }
          >
            <IconSparkles data-icon="inline-start" />
            执行衰减
          </Button>
          <Button
            variant="outline"
            disabled={busy}
            onClick={() =>
              void runAction(async () => {
                const result = await refreshConversationNames()
                return `昵称回填完成：${result.groups} 个群，共 ${result.names} 个昵称`
              })
            }
          >
            <IconUserOff data-icon="inline-start" />
            回填群昵称
          </Button>
          {(groups.data?.disabled ?? []).length > 0 ? (
            <span className="text-xs text-muted-foreground">
              已排除抽取的群：{groups.data?.disabled.join("、")}
            </span>
          ) : null}
        </CardContent>
      </Card>

      <Tabs value={scope} onValueChange={(value) => setScope(value as "group" | "member")}>
        <TabsList>
          <TabsTrigger value="group">群记忆</TabsTrigger>
          <TabsTrigger value="member">成员记忆</TabsTrigger>
        </TabsList>
      </Tabs>

      {memories.error ? (
        <Alert variant="destructive">
          <AlertTitle>读取失败</AlertTitle>
          <AlertDescription>{String(memories.error)}</AlertDescription>
        </Alert>
      ) : null}

      {memories.isLoading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-14 w-full" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>暂无记忆</EmptyTitle>
            <EmptyDescription>等群里有新消息积累后会自动抽取，也可以点上面的「立即抽取」。</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <div className="flex flex-col gap-2" data-testid="memory-list">
          {items.map((item) => (
            <Card key={`${item.scope}-${item.id}`} data-testid="memory-row">
              <CardContent className="flex flex-col gap-2 py-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">{item.kind}</Badge>
                  <Badge variant={sensitivityVariant(item.sensitivity)}>
                    {SENSITIVITY_LABELS[item.sensitivity] ?? item.sensitivity}
                  </Badge>
                  <span className="text-xs text-muted-foreground">重要度 {item.importance.toFixed(1)}/5</span>
                  <span className="text-xs text-muted-foreground">群 {item.group_id}</span>
                  {item.user_id ? <span className="text-xs text-muted-foreground">QQ {item.user_id}</span> : null}
                  {item.decayed ? <Badge variant="secondary">已过期</Badge> : null}
                  <span className="ml-auto text-xs text-muted-foreground">
                    更新于 {formatTime(item.last_seen_at)}
                  </span>
                </div>
                {editing?.id === item.id && editing.scope === item.scope ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <Input
                      value={editContent}
                      aria-label="编辑记忆内容"
                      onChange={(event) => setEditContent(event.target.value)}
                    />
                    <Button
                      size="sm"
                      disabled={busy || !editContent.trim()}
                      onClick={() =>
                        void runAction(async () => {
                          await updatePersonaMemory(item.scope, item.id, { content: editContent.trim() })
                          setEditing(null)
                          return "记忆已更新"
                        })
                      }
                    >
                      保存
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>
                      取消
                    </Button>
                  </div>
                ) : (
                  <p className="text-sm">{item.content}</p>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setEditing(item)
                      setEditContent(item.content)
                    }}
                  >
                    编辑
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={busy}
                    onClick={() =>
                      void runAction(async () => {
                        await deletePersonaMemory(item.scope, item.id)
                        return "记忆已删除"
                      })
                    }
                  >
                    <IconTrash data-icon="inline-start" />
                    删除
                  </Button>
                  {item.scope === "member" && item.user_id ? (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={busy}
                      onClick={() =>
                        void runAction(async () => {
                          const result = await purgeMemberMemories(item.group_id, item.user_id)
                          return `已清除该成员 ${result.deleted} 条记忆`
                        })
                      }
                    >
                      清除该成员全部记忆
                    </Button>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
