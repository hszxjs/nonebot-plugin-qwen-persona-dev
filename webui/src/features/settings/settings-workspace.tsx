import { useState } from "react"
import { IconRefreshAlert } from "@tabler/icons-react"
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
import { Checkbox } from "@/components/ui/checkbox"
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldTitle,
} from "@/components/ui/field"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { saveSettingField, useModels, useSettings } from "@/lib/api"
import type { ConfigValue, SettingsPayload } from "@/lib/types"
import {
  bindSettingsSections,
  rawNameBadge,
  type BoundField,
} from "./settings-schema"

type PendingSave = {
  field: BoundField
  value: ConfigValue
} | null

function stringArray(value: ConfigValue) {
  if (Array.isArray(value)) return value.map(String)
  if (typeof value === "string" && value.trim()) {
    return value.split(/\n|,/).map((item) => item.trim()).filter(Boolean)
  }
  return []
}

function fieldValueText(value: ConfigValue) {
  if (Array.isArray(value)) return value.join("\n")
  if (value && typeof value === "object") return JSON.stringify(value, null, 2)
  return String(value ?? "")
}

function parseTextValue(field: BoundField, text: string): ConfigValue {
  if (field.control === "tokens") {
    return text.split("\n").map((item) => item.trim()).filter(Boolean)
  }
  if (field.value && typeof field.value === "object" && !Array.isArray(field.value)) {
    try {
      return text.trim() ? JSON.parse(text) : {}
    } catch {
      return text
    }
  }
  return text
}

function NumberControl({
  field,
  onChange,
}: {
  field: BoundField
  onChange: (field: BoundField, value: ConfigValue) => void
}) {
  const value = Number(field.value || 0)
  const min = field.min ?? 0
  const max = field.max ?? 100
  const step = field.step ?? 1
  return (
    <div className="grid gap-2 md:grid-cols-[minmax(160px,1fr)_120px]">
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueCommit={(next) => onChange(field, next[0] ?? value)}
      />
      <InputGroup>
        <InputGroupInput
          id={field.name}
          type="number"
          defaultValue={value}
          step={step}
          min={min}
          max={max}
          onBlur={(event) => onChange(field, Number(event.target.value))}
        />
        <InputGroupAddon align="inline-end">num</InputGroupAddon>
      </InputGroup>
    </div>
  )
}

function CheckboxGroupControl({
  field,
  onChange,
}: {
  field: BoundField
  onChange: (field: BoundField, value: ConfigValue) => void
}) {
  const selected = new Set(stringArray(field.value))
  return (
    <FieldGroup data-slot="checkbox-group">
      {(field.options || []).map((option) => (
        <Field key={option.value} orientation="horizontal">
          <Checkbox
            id={`${field.name}-${option.value}`}
            checked={selected.has(option.value)}
            onCheckedChange={(checked) => {
              const next = new Set(selected)
              if (checked) {
                next.add(option.value)
              } else {
                next.delete(option.value)
              }
              onChange(field, [...next])
            }}
          />
          <FieldLabel htmlFor={`${field.name}-${option.value}`}>
            {option.label}
          </FieldLabel>
          {option.description ? (
            <FieldDescription>{option.description}</FieldDescription>
          ) : null}
        </Field>
      ))}
    </FieldGroup>
  )
}

function SelectControl({
  field,
  onChange,
}: {
  field: BoundField
  onChange: (field: BoundField, value: ConfigValue) => void
}) {
  const { data } = useModels()
  const models = data?.models ?? []
  const currentValue = String(field.value ?? "")
  return (
    <Select value={currentValue} onValueChange={(value) => onChange(field, value)}>
      <SelectTrigger id={field.name} aria-label={field.name}>
        <SelectValue placeholder="选择模型" />
      </SelectTrigger>
      <SelectContent>
        {models.length === 0 ? (
          <SelectItem value={currentValue || "glm-5.2"}>
            {currentValue || "glm-5.2"}
          </SelectItem>
        ) : (
          models.map((model) => (
            <SelectItem key={model} value={model}>
              {model}
            </SelectItem>
          ))
        )}
      </SelectContent>
    </Select>
  )
}

function SettingControl({
  field,
  onChange,
}: {
  field: BoundField
  onChange: (field: BoundField, value: ConfigValue) => void
}) {
  if (field.control === "switch") {
    return (
      <Switch
        id={field.name}
        aria-label={field.name}
        checked={Boolean(field.value)}
        onCheckedChange={(value) => onChange(field, value)}
      />
    )
  }

  if (field.control === "toggle") {
    return (
      <ToggleGroup
        type="single"
        value={String(field.value || "")}
        onValueChange={(value) => value && onChange(field, value)}
        spacing={2}
      >
        {(field.options || []).map((option) => (
          <ToggleGroupItem key={option.value} value={option.value}>
            {option.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    )
  }

  if (field.control === "select") {
    return <SelectControl field={field} onChange={onChange} />
  }

  if (field.control === "checkboxes") {
    return <CheckboxGroupControl field={field} onChange={onChange} />
  }

  if (field.control === "number") {
    return <NumberControl field={field} onChange={onChange} />
  }

  if (field.control === "textarea" || field.control === "tokens") {
    return (
      <Textarea
        id={field.name}
        defaultValue={fieldValueText(field.value)}
        onBlur={(event) => onChange(field, parseTextValue(field, event.target.value))}
      />
    )
  }

  return (
    <InputGroup>
      <InputGroupInput
        id={field.name}
        defaultValue={fieldValueText(field.value)}
        type={field.control === "password" ? "password" : "text"}
        onBlur={(event) => onChange(field, event.target.value)}
      />
    </InputGroup>
  )
}

export function SettingsWorkspaceView({
  payload,
  onSaveField,
}: {
  payload: SettingsPayload
  onSaveField: (field: BoundField, value: ConfigValue, confirmed: boolean) => void
}) {
  const [pending, setPending] = useState<PendingSave>(null)
  const sections = bindSettingsSections(payload.values)

  function requestSave(field: BoundField, value: ConfigValue) {
    if (field.reloadSensitive || field.danger) {
      setPending({ field, value })
      return
    }
    onSaveField(field, value, false)
  }

  return (
    <div className="flex flex-col gap-4">
      <section>
        <h1>配置管理</h1>
        <p>按运行领域管理插件参数，高风险项保存前确认。</p>
      </section>
      <Alert>
        <AlertTitle>配置文件</AlertTitle>
        <AlertDescription>{payload.override_file}</AlertDescription>
      </Alert>
      <Tabs defaultValue={sections[0]?.key} className="min-w-0">
        <TabsList className="max-w-full flex-wrap">
          {sections.map((section) => (
            <TabsTrigger key={section.key} value={section.key}>
              {section.title}
            </TabsTrigger>
          ))}
        </TabsList>
        {sections.map((section) => (
          <TabsContent key={section.key} value={section.key}>
            <Card>
              <CardHeader>
                <CardTitle>{section.title}</CardTitle>
                <CardDescription>{section.fields.length} 个字段</CardDescription>
              </CardHeader>
              <CardContent>
                <FieldGroup>
                  {section.fields.map((field) => (
                    <Field key={field.name} orientation="responsive">
                      <FieldContent>
                        <FieldTitle>
                          {field.label}
                          <Badge variant="outline" title={field.name}>
                            {rawNameBadge(field.name)}
                          </Badge>
                          {field.reloadSensitive ? <Badge variant="secondary">需要确认</Badge> : null}
                          {field.danger ? <Badge variant="destructive">危险</Badge> : null}
                        </FieldTitle>
                        {field.description ? (
                          <FieldDescription>{field.description}</FieldDescription>
                        ) : null}
                      </FieldContent>
                      <FieldLabel htmlFor={field.name} className="sr-only">
                        {field.name}
                      </FieldLabel>
                      <div className="w-full md:max-w-md">
                        <SettingControl field={field} onChange={requestSave} />
                      </div>
                    </Field>
                  ))}
                </FieldGroup>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
      <AlertDialog open={Boolean(pending)} onOpenChange={(open) => !open && setPending(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认保存该配置？</AlertDialogTitle>
            <AlertDialogDescription>
              {pending?.field.label || pending?.field.name} 会改变运行行为，可能触发 reload、重启或中断当前任务。
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
  const { data, error, isLoading, mutate } = useSettings()

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
        try {
          await saveSettingField(field.name, value, confirmed)
          toast.success("配置已保存")
          await mutate()
        } catch (error) {
          toast.error(error instanceof Error ? error.message : String(error))
        }
      }}
    />
  )
}
