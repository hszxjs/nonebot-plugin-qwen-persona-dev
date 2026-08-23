import { useState, type ReactNode } from "react"
import {
  IconFlask,
  IconPlus,
  IconRegex,
  IconShieldOff,
  IconX,
} from "@tabler/icons-react"
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
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { apiErrorMessage, saveSettingField, testBlacklist, useSettings } from "@/lib/api"
import type { BlacklistTestResult } from "@/lib/types"

const WORDS_FIELD = "ai_companion_blacklist_words"
const REGEXES_FIELD = "ai_companion_blacklist_regexes"

function isValidRegex(value: string) {
  try {
    new RegExp(value)
    return true
  } catch {
    return false
  }
}

function ChipList({
  values,
  onRemove,
  emptyText,
  mono,
}: {
  values: string[]
  onRemove: (value: string) => void
  emptyText: string
  mono?: boolean
}) {
  if (!values.length) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>{emptyText}</EmptyTitle>
          <EmptyDescription>通过上方输入框添加条目。</EmptyDescription>
        </EmptyHeader>
        <EmptyContent />
      </Empty>
    )
  }
  return (
    <div className="flex flex-wrap gap-2">
      {values.map((value) => (
        <span
          key={value}
          className="inline-flex max-w-full items-center gap-1 rounded-full border bg-muted/40 py-0.5 pr-1 pl-2.5 text-sm"
        >
          <span className={mono ? "truncate font-mono text-xs" : "truncate"} title={value}>
            {value}
          </span>
          <button
            type="button"
            aria-label={`删除 ${value}`}
            title={`删除 ${value}`}
            className="rounded-full p-0.5 text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
            onClick={() => onRemove(value)}
          >
            <IconX className="size-3.5" />
          </button>
        </span>
      ))}
    </div>
  )
}

function BlacklistListCard({
  title,
  description,
  icon,
  values,
  inputLabel,
  placeholder,
  mono,
  validate,
  onAdd,
  onRemove,
}: {
  title: string
  description: string
  icon: ReactNode
  values: string[]
  inputLabel: string
  placeholder: string
  mono?: boolean
  validate?: (value: string) => boolean
  onAdd: (value: string) => void
  onRemove: (value: string) => void
}) {
  const [input, setInput] = useState("")

  function handleAdd() {
    const trimmed = input.trim()
    if (!trimmed) return
    if (values.includes(trimmed)) {
      toast.info("该条目已存在")
      return
    }
    if (validate && !validate(trimmed)) {
      toast.error("无效的正则表达式，请检查语法")
      return
    }
    onAdd(trimmed)
    setInput("")
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {icon}
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex gap-2">
          <Input
            aria-label={inputLabel}
            placeholder={placeholder}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") handleAdd()
            }}
          />
          <Button disabled={!input.trim()} onClick={handleAdd}>
            <IconPlus data-icon="inline-start" />
            添加
          </Button>
        </div>
        <ChipList values={values} onRemove={onRemove} emptyText="暂无条目" mono={mono} />
      </CardContent>
    </Card>
  )
}

function TestCard({
  onTest,
}: {
  onTest: (text: string) => Promise<BlacklistTestResult>
}) {
  const [text, setText] = useState("")
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<BlacklistTestResult | null>(null)

  async function handleTest() {
    const trimmed = text.trim()
    if (!trimmed) return
    setTesting(true)
    try {
      setResult(await onTest(trimmed))
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
    } finally {
      setTesting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <IconFlask className="size-4" />
          规则测试
        </CardTitle>
        <CardDescription>输入样本文本，检查是否命中当前黑名单词库。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Textarea
          aria-label="测试文本"
          rows={4}
          placeholder="粘贴一条群消息样本，点击测试检查命中情况…"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div>
          <Button disabled={!text.trim() || testing} onClick={() => void handleTest()}>
            <IconFlask data-icon="inline-start" />
            {testing ? "测试中…" : "测试"}
          </Button>
        </div>
        {result ? (
          <div
            className="flex flex-col gap-3 rounded-lg border p-3"
            data-testid="blacklist-test-result"
          >
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={result.matched ? "destructive" : "secondary"}>
                {result.matched ? "已命中" : "未命中"}
              </Badge>
              <span
                className={
                  result.matched ? "text-sm text-destructive" : "text-sm text-muted-foreground"
                }
              >
                {result.matched ? "该文本会被黑名单过滤" : "该文本不会被黑名单过滤"}
              </span>
            </div>
            {result.matches.length ? (
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted-foreground">命中规则（{result.matches.length} 条）：</span>
                <div className="flex flex-wrap gap-2">
                  {result.matches.map((match, index) => (
                    <span
                      key={`${match.kind}-${match.value}-${index}`}
                      className="inline-flex max-w-full items-center gap-1.5 rounded-full border bg-muted/40 py-0.5 pr-2.5 pl-1 text-xs"
                    >
                      <Badge variant="secondary">{match.kind === "word" ? "词汇" : "正则"}</Badge>
                      <span className="truncate font-mono" title={match.value}>
                        {match.value}
                      </span>
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              <Badge variant="outline">词汇 {result.words.length}</Badge>
              <Badge variant="outline">正则 {result.regexes.length}</Badge>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

export function BlacklistWorkspaceView({
  words,
  regexes,
  onAddWord,
  onRemoveWord,
  onAddRegex,
  onRemoveRegex,
  onTest,
}: {
  words: string[]
  regexes: string[]
  onAddWord: (value: string) => void
  onRemoveWord: (value: string) => void
  onAddRegex: (value: string) => void
  onRemoveRegex: (value: string) => void
  onTest: (text: string) => Promise<BlacklistTestResult>
}) {
  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>黑名单词库</h1>
        <p>维护消息黑名单词汇与正则，并即时测试命中效果。</p>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="outline">
          <IconShieldOff data-icon="inline-start" />
          词汇 {words.length}
        </Badge>
        <Badge variant="outline">
          <IconRegex data-icon="inline-start" />
          正则 {regexes.length}
        </Badge>
      </div>
      <Alert>
        <AlertTitle>保存即生效</AlertTitle>
        <AlertDescription>
          每次添加或删除都会把完整列表写回配置（{WORDS_FIELD} / {REGEXES_FIELD}）。
        </AlertDescription>
      </Alert>
      <div className="grid gap-4 xl:grid-cols-2">
        <BlacklistListCard
          title="黑名单词汇"
          description="命中任一词汇的消息将被过滤。"
          icon={<IconShieldOff className="size-4" />}
          values={words}
          inputLabel="新词汇"
          placeholder="输入词汇后回车或点击添加"
          onAdd={onAddWord}
          onRemove={onRemoveWord}
        />
        <BlacklistListCard
          title="黑名单正则"
          description="按正则表达式匹配消息内容，添加前会校验语法。"
          icon={<IconRegex className="size-4" />}
          values={regexes}
          inputLabel="新正则"
          placeholder="例如：^(?i)(spam|广告)$"
          mono
          validate={isValidRegex}
          onAdd={onAddRegex}
          onRemove={onRemoveRegex}
        />
      </div>
      <TestCard onTest={onTest} />
    </div>
  )
}

export function BlacklistWorkspace() {
  const { data, error, isLoading, mutate } = useSettings()
  const [wordsOverride, setWordsOverride] = useState<string[] | null>(null)
  const [regexesOverride, setRegexesOverride] = useState<string[] | null>(null)

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取黑名单配置</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return null
  }

  const serverWords = Array.isArray(data.values[WORDS_FIELD])
    ? data.values[WORDS_FIELD].map(String)
    : []
  const serverRegexes = Array.isArray(data.values[REGEXES_FIELD])
    ? data.values[REGEXES_FIELD].map(String)
    : []
  const words = wordsOverride ?? serverWords
  const regexes = regexesOverride ?? serverRegexes

  async function persistList(field: string, next: string[], rollback: () => void, action: string) {
    try {
      await saveSettingField(field, next, true)
      toast.success(`${action}，黑名单已保存`)
      await mutate()
    } catch (caught) {
      toast.error(await apiErrorMessage(caught))
      rollback()
    }
  }

  function commitChange(
    current: string[],
    value: string,
    mode: "add" | "remove",
    field: string,
    setOverride: (next: string[] | null) => void
  ) {
    const next = mode === "add" ? [...current, value] : current.filter((item) => item !== value)
    // 乐观更新：先更新本地展示，再整表写回配置，失败时回退
    setOverride(next)
    void persistList(field, next, () => setOverride(null), mode === "add" ? `已添加：${value}` : `已删除：${value}`)
  }

  return (
    <BlacklistWorkspaceView
      words={words}
      regexes={regexes}
      onAddWord={(value) => commitChange(words, value, "add", WORDS_FIELD, setWordsOverride)}
      onRemoveWord={(value) => commitChange(words, value, "remove", WORDS_FIELD, setWordsOverride)}
      onAddRegex={(value) => commitChange(regexes, value, "add", REGEXES_FIELD, setRegexesOverride)}
      onRemoveRegex={(value) => commitChange(regexes, value, "remove", REGEXES_FIELD, setRegexesOverride)}
      onTest={(text) => testBlacklist(text)}
    />
  )
}
