import { useState } from "react"
import { IconEdit, IconHistory, IconPhoto, IconTrash } from "@tabler/icons-react"
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
import { AspectRatio } from "@/components/ui/aspect-ratio"
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
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { InputGroup, InputGroupInput } from "@/components/ui/input-group"
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemTitle,
} from "@/components/ui/item"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  deleteStickers,
  updateStickerSummary,
  useStickerEvents,
  useStickers,
} from "@/lib/api"
import type { StickerEvent, StickerGalleryPayload, StickerItem } from "@/lib/types"

function imageUrl(item: StickerItem) {
  return String(item.preview_url || item.data.url || item.data.preview || item.data.thumb || "")
}

function StickerHistoryItem({ event }: { event: StickerEvent }) {
  return (
    <Item>
      <ItemMedia variant="icon">
        <IconHistory />
      </ItemMedia>
      <ItemContent>
        <ItemTitle>
          <span>{event.event_type}</span>
          <Badge variant="secondary">{event.status}</Badge>
        </ItemTitle>
        <ItemDescription>
          {event.message || event.key || event.time}
        </ItemDescription>
        <ItemDescription>
          {event.key || "-"} / {event.group_id || "-"} / {event.user_id || "-"}
        </ItemDescription>
      </ItemContent>
    </Item>
  )
}

function StickerHistoryDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { data, error, isLoading } = useStickerEvents(open)
  const items = data?.items ?? []

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline">
          <IconHistory data-icon="inline-start" />
          历史
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>表情事件历史</DialogTitle>
          <DialogDescription>classifier 和 selector 的最近观测事件。</DialogDescription>
        </DialogHeader>
        {isLoading ? (
          <div className="flex flex-col gap-3">
            <Skeleton className="h-16" />
            <Skeleton className="h-16" />
            <Skeleton className="h-16" />
          </div>
        ) : error ? (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>无法读取历史</EmptyTitle>
              <EmptyDescription>{String(error)}</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        ) : items.length ? (
          <ScrollArea className="h-[min(60vh,32rem)]">
            <ItemGroup>
              {items.map((event) => (
                <StickerHistoryItem key={event.event_id} event={event} />
              ))}
            </ItemGroup>
          </ScrollArea>
        ) : (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>暂无历史</EmptyTitle>
              <EmptyDescription>还没有 classifier 或 selector 事件。</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        )}
      </DialogContent>
    </Dialog>
  )
}

function StickerEntry({
  item,
  selected,
  onToggleSelected,
  onEditSummary,
}: {
  item: StickerItem
  selected: boolean
  onToggleSelected: (key: string, selected: boolean) => void
  onEditSummary: (item: StickerItem, summary: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [summary, setSummary] = useState(item.summary)
  const src = imageUrl(item)

  return (
    <Card>
      <AspectRatio ratio={1}>
        {src ? (
          <img src={src} alt={item.summary || item.key} className="size-full object-cover" />
        ) : (
          <div className="grid size-full place-items-center">
            <IconPhoto />
          </div>
        )}
      </AspectRatio>
      <CardHeader>
        <CardTitle>{item.summary || item.key}</CardTitle>
        <CardDescription>{item.uri || item.key}</CardDescription>
        <CardAction>
          <Checkbox
            aria-label={`选择 ${item.summary || item.key}`}
            checked={selected}
            onCheckedChange={(value) => onToggleSelected(item.key, value === true)}
          />
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ItemGroup data-size="sm">
          <Item>
            <ItemMedia variant="icon">
              <IconPhoto />
            </ItemMedia>
            <ItemContent>
              <ItemTitle>{item.type}</ItemTitle>
              <ItemDescription>
                {item.group_id || "-"} / {item.user_id || "-"}
              </ItemDescription>
            </ItemContent>
            <ItemActions>
              <Badge variant="secondary">{item.count}</Badge>
            </ItemActions>
          </Item>
        </ItemGroup>
        <Sheet open={open} onOpenChange={setOpen}>
          <Button variant="outline" onClick={() => setOpen(true)}>
            <IconEdit data-icon="inline-start" />
            编辑提示词
          </Button>
          <SheetContent>
            <SheetHeader>
              <SheetTitle>编辑表情提示词</SheetTitle>
              <SheetDescription>{item.key}</SheetDescription>
            </SheetHeader>
            <div className="px-4">
              <FieldGroup>
                <Field>
                  <FieldLabel htmlFor={`summary-${item.key}`}>summary</FieldLabel>
                  <Textarea
                    id={`summary-${item.key}`}
                    value={summary}
                    onChange={(event) => setSummary(event.target.value)}
                  />
                </Field>
              </FieldGroup>
            </div>
            <SheetFooter>
              <Button
                onClick={() => {
                  onEditSummary(item, summary)
                  setOpen(false)
                }}
              >
                保存
              </Button>
            </SheetFooter>
          </SheetContent>
        </Sheet>
      </CardContent>
    </Card>
  )
}

export function StickersWorkspaceView({
  payload,
  selectedKeys,
  onToggleSelected,
  onEditSummary,
  onDeleteSelected,
}: {
  payload: StickerGalleryPayload
  selectedKeys: string[]
  onToggleSelected: (key: string, selected: boolean) => void
  onEditSummary: (item: StickerItem, summary: string) => void
  onDeleteSelected: () => void
}) {
  const typeEntries = Object.entries(payload.type_counts)
  const [historyOpen, setHistoryOpen] = useState(false)

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>表情仓库</h1>
        <p>卡片式表情 entry、summary 编辑和批量管理。</p>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <InputGroup className="max-w-sm">
          <InputGroupInput placeholder="搜索 summary、来源或协议字段" defaultValue={payload.query} />
        </InputGroup>
        <ToggleGroup type="single" defaultValue={payload.type || "all"} spacing={2}>
          <ToggleGroupItem value="all">全部</ToggleGroupItem>
          {typeEntries.slice(0, 5).map(([type, count]) => (
            <ToggleGroupItem key={type} value={type}>
              {type}
              <Badge variant="secondary">{count}</Badge>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <Badge variant="outline">总计 {payload.total}</Badge>
        <StickerHistoryDialog open={historyOpen} onOpenChange={setHistoryOpen} />
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="destructive" disabled={!selectedKeys.length}>
              <IconTrash data-icon="inline-start" />
              删除选中
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>删除选中的表情？</AlertDialogTitle>
              <AlertDialogDescription>
                将从表情仓库删除 {selectedKeys.length} 个 entry，可能影响后续表情选择。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction variant="destructive" onClick={onDeleteSelected}>
                删除
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
      {payload.items.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {payload.items.map((item) => (
            <StickerEntry
              key={item.key}
              item={item}
              selected={selectedKeys.includes(item.key)}
              onToggleSelected={onToggleSelected}
              onEditSummary={onEditSummary}
            />
          ))}
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>暂无表情</EmptyTitle>
            <EmptyDescription>只会收录符合表情学习边界的表情段。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
    </div>
  )
}

export function StickersWorkspace() {
  const { data, error, isLoading, mutate } = useStickers()
  const [selectedKeys, setSelectedKeys] = useState<string[]>([])

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return <Empty><EmptyHeader><EmptyTitle>无法读取表情仓库</EmptyTitle><EmptyDescription>{String(error)}</EmptyDescription></EmptyHeader></Empty>
  }
  if (!data) {
    return null
  }

  return (
    <StickersWorkspaceView
      payload={data}
      selectedKeys={selectedKeys}
      onToggleSelected={(key, selected) => {
        setSelectedKeys((current) =>
          selected ? [...new Set([...current, key])] : current.filter((item) => item !== key)
        )
      }}
      onEditSummary={async (item, summary) => {
        await updateStickerSummary(item.key, summary)
        toast.success("表情提示词已保存")
        await mutate()
      }}
      onDeleteSelected={async () => {
        await deleteStickers(selectedKeys)
        toast.success("已删除选中的表情")
        setSelectedKeys([])
        await mutate()
      }}
    />
  )
}
