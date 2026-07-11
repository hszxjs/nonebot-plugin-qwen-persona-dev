import { useState } from "react"
import { IconDeviceFloppy, IconRefreshAlert } from "@tabler/icons-react"
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
} from "@/components/ui/alert-dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldTitle,
} from "@/components/ui/field"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { saveSettingField, useSettingsSections } from "@/lib/api"
import type {
  ConfigValue,
  SettingsField,
  SettingsSectionsPayload,
} from "@/lib/types"

type PendingSave = {
  field: SettingsField
  value: ConfigValue
} | null

function fieldDescription(field: SettingsField) {
  if (field.reload_sensitive) {
    return "保存后可能触发 reload 或需要重启。"
  }
  if (field.danger) {
    return "该字段会改变敏感命令或凭据。"
  }
  return "变更后自动保存。"
}

function numericBounds(field: SettingsField) {
  if (field.name.includes("probability") || field.name.endsWith("top_p") || field.name.includes("dropout")) {
    return { min: 0, max: 1, step: 0.01 }
  }
  if (field.name.includes("temperature")) {
    return { min: 0, max: 2, step: 0.05 }
  }
  if (field.name.includes("tokens")) {
    return { min: 1, max: 4096, step: 1 }
  }
  return { min: 0, max: 100, step: 1 }
}

function fieldValueText(value: ConfigValue) {
  if (Array.isArray(value)) return value.join(", ")
  if (value && typeof value === "object") return JSON.stringify(value, null, 2)
  return String(value ?? "")
}

function SettingControl({
  field,
  onChange,
}: {
  field: SettingsField
  onChange: (field: SettingsField, value: ConfigValue) => void
}) {
  if (field.kind === "bool") {
    return (
      <Switch
        id={field.name}
        aria-label={field.name}
        checked={Boolean(field.value)}
        onCheckedChange={(value) => onChange(field, value)}
      />
    )
  }

  if (field.name === "qwen_persona_context_scope") {
    return (
      <ToggleGroup
        type="single"
        value={String(field.value || "group")}
        onValueChange={(value) => value && onChange(field, value)}
        spacing={2}
      >
        <ToggleGroupItem value="group">group</ToggleGroupItem>
        <ToggleGroupItem value="global">global</ToggleGroupItem>
        <ToggleGroupItem value="mixed">mixed</ToggleGroupItem>
      </ToggleGroup>
    )
  }

  if (field.kind === "int" || field.kind === "float") {
    const value = Number(field.value || 0)
    const bounds = numericBounds(field)
    return (
      <div className="grid gap-2 md:grid-cols-[minmax(160px,1fr)_120px]">
        <Slider
          min={bounds.min}
          max={bounds.max}
          step={bounds.step}
          value={[value]}
          onValueCommit={(next) => onChange(field, next[0] ?? value)}
        />
        <InputGroup>
          <InputGroupInput
            type="number"
            value={value}
            onChange={(event) => onChange(field, Number(event.target.value))}
          />
          <InputGroupAddon align="inline-end">{field.kind}</InputGroupAddon>
        </InputGroup>
      </div>
    )
  }

  if (field.kind === "json" || field.name.includes("prompt") || field.name.includes("command")) {
    return (
      <Textarea
        value={fieldValueText(field.value)}
        onChange={(event) => {
          const text = event.target.value
          if (field.kind === "json") {
            onChange(field, text.split(",").map((item) => item.trim()).filter(Boolean))
            return
          }
          onChange(field, text)
        }}
      />
    )
  }

  return (
    <InputGroup>
      <InputGroupInput
        value={fieldValueText(field.value)}
        type={field.danger ? "password" : "text"}
        onChange={(event) => onChange(field, event.target.value)}
      />
    </InputGroup>
  )
}

export function SettingsWorkspaceView({
  payload,
  onSaveField,
}: {
  payload: SettingsSectionsPayload
  onSaveField: (field: SettingsField, value: ConfigValue, confirmed: boolean) => void
}) {
  const [pending, setPending] = useState<PendingSave>(null)

  function requestSave(field: SettingsField, value: ConfigValue) {
    if (field.reload_sensitive || field.danger) {
      setPending({ field, value })
      return
    }
    onSaveField(field, value, false)
  }

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>配置管理</h1>
        <p>语义分区、自动保存和 modal 确认。</p>
      </section>
      <Alert>
        <IconDeviceFloppy />
        <AlertTitle>自动保存</AlertTitle>
        <AlertDescription>普通字段变更后立即保存；危险和 reload 字段需要确认。</AlertDescription>
      </Alert>
      <div className="grid gap-4 xl:grid-cols-2">
        {payload.sections.map((section) => (
          <Card key={section.key}>
            <CardHeader>
              <CardTitle>{section.title}</CardTitle>
              <CardDescription>{section.fields.length} 个字段</CardDescription>
            </CardHeader>
            <CardContent>
              <FieldGroup>
                {section.fields.map((field) => (
                  <Field key={field.name} orientation="responsive">
                    <FieldContent>
                      <FieldTitle>{field.name}</FieldTitle>
                      <FieldDescription>{fieldDescription(field)}</FieldDescription>
                      <div className="flex flex-wrap gap-2">
                        {field.reload_sensitive ? <Badge variant="secondary">需要确认</Badge> : null}
                        {field.danger ? <Badge variant="destructive">危险</Badge> : null}
                      </div>
                    </FieldContent>
                    <FieldLabel htmlFor={field.name} className="sr-only">
                      {field.name}
                    </FieldLabel>
                    <SettingControl field={field} onChange={requestSave} />
                  </Field>
                ))}
              </FieldGroup>
            </CardContent>
          </Card>
        ))}
      </div>
      <AlertDialog open={Boolean(pending)} onOpenChange={(open) => !open && setPending(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认保存该配置？</AlertDialogTitle>
            <AlertDialogDescription>
              {pending?.field.name} 会改变运行行为，可能触发 reload、重启或中断当前任务。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pending) {
                  onSaveField(pending.field, pending.value, true)
                }
                setPending(null)
              }}
            >
              <IconRefreshAlert data-icon="inline-start" />
              确认保存
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

export function SettingsWorkspace() {
  const { data, error, isLoading, mutate } = useSettingsSections()

  if (isLoading) {
    return <Skeleton className="h-96" />
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取配置</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return null
  }

  return (
    <SettingsWorkspaceView
      payload={data}
      onSaveField={async (field, value, confirmed) => {
        await saveSettingField(field.name, value, confirmed)
        toast.success("配置已保存")
        await mutate()
      }}
    />
  )
}
