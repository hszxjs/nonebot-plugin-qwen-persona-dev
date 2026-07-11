import { IconRegex, IconShieldCheck } from "@tabler/icons-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemTitle,
} from "@/components/ui/item"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { saveSettingField, useSettingsSections } from "@/lib/api"
import type { SettingsField, SettingsSectionsPayload } from "@/lib/types"

function ruleFields(payload: SettingsSectionsPayload) {
  return payload.sections
    .filter((section) => section.key === "rules")
    .flatMap((section) => section.fields)
}

function fieldText(field: SettingsField) {
  if (Array.isArray(field.value)) return field.value.join("\n")
  if (field.value && typeof field.value === "object") return JSON.stringify(field.value, null, 2)
  return String(field.value ?? "")
}

export function RulesWorkspaceView({
  payload,
  onSaveField,
}: {
  payload: SettingsSectionsPayload
  onSaveField: (field: SettingsField, value: string[]) => void
}) {
  const fields = ruleFields(payload)
  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>规则过滤</h1>
        <p>正则、黑名单和命令过滤规则管理。</p>
      </section>
      {fields.length ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {fields.map((field) => (
            <Card key={field.name}>
              <CardHeader>
                <CardTitle>{field.name}</CardTitle>
                <CardDescription>每行一条规则或 token。</CardDescription>
              </CardHeader>
              <CardContent>
                <FieldGroup>
                  <Field>
                    <FieldLabel htmlFor={`rules-${field.name}`}>规则内容</FieldLabel>
                    <Textarea
                      id={`rules-${field.name}`}
                      defaultValue={fieldText(field)}
                      onBlur={(event) => {
                        onSaveField(
                          field,
                          event.target.value
                            .split("\n")
                            .map((item) => item.trim())
                            .filter(Boolean)
                        )
                      }}
                    />
                  </Field>
                </FieldGroup>
                <ItemGroup data-size="sm">
                  <Item>
                    <ItemMedia variant="icon">
                      {field.name.includes("regex") ? <IconRegex /> : <IconShieldCheck />}
                    </ItemMedia>
                    <ItemContent>
                      <ItemTitle>{field.kind}</ItemTitle>
                      <ItemDescription>失焦后自动保存。</ItemDescription>
                    </ItemContent>
                    <ItemActions>
                      <Badge variant={field.danger ? "destructive" : "secondary"}>
                        {field.danger ? "危险" : "规则"}
                      </Badge>
                    </ItemActions>
                  </Item>
                </ItemGroup>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>暂无规则字段</EmptyTitle>
            <EmptyDescription>后端没有返回规则过滤分区。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
    </div>
  )
}

export function RulesWorkspace() {
  const { data, error, isLoading, mutate } = useSettingsSections()

  if (isLoading) return <Skeleton className="h-96" />
  if (error) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>无法读取规则</EmptyTitle>
          <EmptyDescription>{String(error)}</EmptyDescription>
        </EmptyHeader>
        <EmptyContent />
      </Empty>
    )
  }
  if (!data) return null

  return (
    <RulesWorkspaceView
      payload={data}
      onSaveField={async (field, value) => {
        await saveSettingField(field.name, value, field.reload_sensitive || field.danger)
        toast.success("规则已保存")
        await mutate()
      }}
    />
  )
}
