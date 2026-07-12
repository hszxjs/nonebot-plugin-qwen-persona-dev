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
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { saveSettingField, useSettings } from "@/lib/api"
import type { ConfigValue, SettingsPayload } from "@/lib/types"
import { bindRuleFields, type BoundField } from "@/features/settings/settings-schema"

function fieldText(field: BoundField) {
  if (Array.isArray(field.value)) return field.value.join("\n")
  if (field.value && typeof field.value === "object") return JSON.stringify(field.value, null, 2)
  return String(field.value ?? "")
}

function ruleTextValue(text: string) {
  return text.split("\n").map((item) => item.trim()).filter(Boolean)
}

export function RulesWorkspaceView({
  payload,
  onSaveField,
}: {
  payload: SettingsPayload
  onSaveField: (field: BoundField, value: ConfigValue) => void
}) {
  const fields = bindRuleFields(payload.values)
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
                <CardTitle>{field.label}</CardTitle>
                <CardDescription>{field.description || field.name}</CardDescription>
              </CardHeader>
              <CardContent>
                <FieldGroup>
                  <Field>
                    <FieldLabel htmlFor={`rules-${field.name}`}>规则内容</FieldLabel>
                    {field.control === "switch" ? (
                      <Switch
                        id={`rules-${field.name}`}
                        aria-label={field.name}
                        checked={Boolean(field.value)}
                        onCheckedChange={(value) => onSaveField(field, value)}
                      />
                    ) : (
                      <Textarea
                        id={`rules-${field.name}`}
                        defaultValue={fieldText(field)}
                        onBlur={(event) => onSaveField(field, ruleTextValue(event.target.value))}
                      />
                    )}
                  </Field>
                </FieldGroup>
                <ItemGroup data-size="sm">
                  <Item>
                    <ItemMedia variant="icon">
                      {field.name.includes("regex") ? <IconRegex /> : <IconShieldCheck />}
                    </ItemMedia>
                    <ItemContent>
                      <ItemTitle>{field.name.includes("regex") ? "regex" : "rule"}</ItemTitle>
                      <ItemDescription>{field.name}</ItemDescription>
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
            <EmptyDescription>当前配置值中没有规则过滤字段。</EmptyDescription>
          </EmptyHeader>
          <EmptyContent />
        </Empty>
      )}
    </div>
  )
}

export function RulesWorkspace() {
  const { data, error, isLoading, mutate } = useSettings()

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
        try {
          await saveSettingField(field.name, value, Boolean(field.reloadSensitive || field.danger))
          toast.success("规则已保存")
          await mutate()
        } catch (error) {
          toast.error(error instanceof Error ? error.message : String(error))
        }
      }}
    />
  )
}
