import {
  IconAdjustments,
  IconHomeStats,
  IconIdBadge2,
  IconMoodSmile,
  IconShieldCheck,
  IconTool,
} from "@tabler/icons-react"
import { NavLink, Route, Routes } from "react-router-dom"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { DiagnosticsWorkspace } from "@/features/diagnostics/diagnostics-workspace"
import { OperationsDashboard } from "@/features/operations/operations-dashboard"
import { RoleCardsWorkspace } from "@/features/role-cards/role-cards-workspace"
import { RulesWorkspace } from "@/features/rules/rules-workspace"
import { SettingsWorkspace } from "@/features/settings/settings-workspace"
import { StickersWorkspace } from "@/features/stickers/stickers-workspace"
import { useOperationsOverview } from "@/lib/api"

const navItems = [
  { to: "/", label: "运维控制台", icon: IconHomeStats },
  { to: "/stickers", label: "表情仓库", icon: IconMoodSmile },
  { to: "/role-cards", label: "角色卡", icon: IconIdBadge2 },
  { to: "/settings", label: "配置管理", icon: IconAdjustments },
  { to: "/rules", label: "规则过滤", icon: IconShieldCheck },
  { to: "/diagnostics", label: "诊断", icon: IconTool },
]

function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>Qwen Persona</SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Admin</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <SidebarMenuButton asChild>
                    <NavLink to={item.to}>
                      <item.icon />
                      <span>{item.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}

function OperationsPage() {
  const { data, error, isLoading } = useOperationsOverview()
  if (isLoading) {
    return (
      <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
        <Skeleton className="h-96" />
        <Skeleton className="h-96" />
      </div>
    )
  }
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>无法读取运维数据</AlertTitle>
        <AlertDescription>{String(error)}</AlertDescription>
      </Alert>
    )
  }
  if (!data) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>暂无运维数据</EmptyTitle>
          <EmptyDescription>等待后台 API 返回状态。</EmptyDescription>
        </EmptyHeader>
        <EmptyContent />
      </Empty>
    )
  }
  return <OperationsDashboard overview={data} />
}

export function App() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <main className="p-4 md:p-6">
          <Routes>
            <Route path="/" element={<OperationsPage />} />
            <Route path="/stickers" element={<StickersWorkspace />} />
            <Route path="/role-cards" element={<RoleCardsWorkspace />} />
            <Route path="/settings" element={<SettingsWorkspace />} />
            <Route path="/rules" element={<RulesWorkspace />} />
            <Route path="/diagnostics" element={<DiagnosticsWorkspace />} />
          </Routes>
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default App
