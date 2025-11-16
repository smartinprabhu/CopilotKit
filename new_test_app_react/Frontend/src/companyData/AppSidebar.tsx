
import React from 'react';
import { Building2, Briefcase, Users, MapPin, Settings, Calendar, Database, Code, Upload, HelpCircle, LogOut } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
} from "@/components/ui/sidebar";

const primaryEntities = [
  {
    title: "Business Units",
    icon: Building2,
    id: "business-units",
  },
  {
    title: "Line of Business",
    icon: Briefcase,
    id: "line-of-business",
  },
  {
    title: "Channel",
    icon: Users,
    id: "channel",
  },
];

const supportEntities = [
  {
    title: "Feed Parameters",
    icon: Database,
    id: "feed-parameters",
  },
  {
    title: "Calendar Weeks",
    icon: Calendar,
    id: "calendar-weeks",
  },
  {
    title: "System Settings",
    icon: Settings,
    id: "system-settings",
  },
];

export function AppSidebar({ activeTab }) {
  return (
    <div className="border-r border-slate-700">
      <div className="bg-sidebar" style={{ height: '100vh', overflow: 'auto' }}>
        <SidebarGroup>
          <SidebarGroupLabel className="text-foreground text-xs font-medium uppercase tracking-wide px-3 py-2">
            Primary Entities
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {primaryEntities.map((item) => (
                <SidebarMenuItem key={item.id}>
<SidebarMenuButton 
  asChild 
  className={`text-foreground hover:text-white hover:bg-slate-800 data-[active=true]:bg-blue-600 data-[active=true]:text-white ${activeTab === item.title ? 'bg-primary text-primary-foreground font-bold' : 'hover:bg-accent hover:text-accent-foreground'}`}
>
                    <button data-entity={item.id} className="w-full flex items-center gap-3 px-3 py-2 text-left">
                      <item.icon className="w-4 h-4" />
                      <span className="text-sm">{item.title}</span>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-foreground text-xs font-medium uppercase tracking-wide px-3 py-2">
            Support Entities
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {supportEntities.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton 
                    asChild 
                    className={`text-foreground hover:text-white hover:bg-slate-800 data-[active=true]:bg-blue-600 data-[active=true]:text-white ${activeTab === item.title ? 'bg-primary text-primary-foreground font-bold' : 'hover:bg-accent hover:text-accent-foreground'}`}
                  >
                    <button data-entity={item.id} className="w-full flex items-center gap-3 px-3 py-2 text-left">
                      <item.icon className="w-4 h-4" />
                      <span className="text-sm">{item.title}</span>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

      </div>
    </div>
  );
}
