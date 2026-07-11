import { IconFileCode, IconServer, IconSettings } from "@tabler/icons-react"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import {
  Item,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemTitle,
} from "@/components/ui/item"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { useDiagnostics } from "@/lib/api"
import type { DiagnosticsPayload } from "@/lib/types"

function JsonBlock({ value }: { value: unknown }) {
  return (
    <ScrollArea className="h-72">
      <Textarea readOnly value={JSON.stringify(value, null, 2)} />
    </ScrollArea>
  )
}

export function DiagnosticsWorkspaceView({ payload }: { payload: DiagnosticsPayload }) {
  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>诊断</h1>
        <p>原始 JSON、内部日志和构建元数据。</p>
      </section>
      <div className="grid gap-4 xl:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>静态资源</CardTitle>
            <CardDescription>{payload.static.root}</CardDescription>
            <CardAction>
              <Badge variant={payload.static.built ? "default" : "destructive"}>
                {payload.static.built ? "built" : "missing"}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            <ItemGroup>
              <Item>
                <ItemMedia variant="icon">
                  <IconFileCode />
                </ItemMedia>
                <ItemContent>
                  <ItemTitle>index</ItemTitle>
                  <ItemDescription>{payload.static.index}</ItemDescription>
                </ItemContent>
              </Item>
            </ItemGroup>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Backend</CardTitle>
            <CardDescription>{String(payload.build.admin_server || "-")}</CardDescription>
          </CardHeader>
          <CardContent>
            <ItemGroup>
              <Item>
                <ItemMedia variant="icon">
                  <IconServer />
                </ItemMedia>
                <ItemContent>
                  <ItemTitle>admin server</ItemTitle>
                  <ItemDescription>{String(payload.build.admin_server || "-")}</ItemDescription>
                </ItemContent>
              </Item>
            </ItemGroup>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>配置覆盖</CardTitle>
            <CardDescription>{payload.settings.override_file}</CardDescription>
          </CardHeader>
          <CardContent>
            <ItemGroup>
              <Item>
                <ItemMedia variant="icon">
                  <IconSettings />
                </ItemMedia>
                <ItemContent>
                  <ItemTitle>sections</ItemTitle>
                  <ItemDescription>{payload.settings.sections.length}</ItemDescription>
                </ItemContent>
              </Item>
            </ItemGroup>
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Raw Diagnostics JSON</CardTitle>
          <CardDescription>调试数据只在诊断页展开。</CardDescription>
        </CardHeader>
        <CardContent>
          <JsonBlock value={payload} />
        </CardContent>
      </Card>
    </div>
  )
}

export function DiagnosticsWorkspace() {
  const { data, error, isLoading } = useDiagnostics()

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>无法读取诊断数据</EmptyTitle>
          <EmptyDescription>{String(error)}</EmptyDescription>
        </EmptyHeader>
        <EmptyContent />
      </Empty>
    )
  }
  if (!data) {
    return null
  }
  return <DiagnosticsWorkspaceView payload={data} />
}
