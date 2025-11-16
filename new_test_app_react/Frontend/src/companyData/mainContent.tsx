
import React, { useState } from 'react';
import { SidebarTrigger } from "@/components/ui/sidebar";
import { EntityDataTable } from "./EntityDataTable";
import { EntityForm } from "./EntityForm";
import { Button } from "@/components/ui/button";
import { Plus, RefreshCw, Download, Filter } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useNavigate } from "react-router-dom";

import CustomSidebar from "@/components/Sidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import AuthService from "@/auth/utils/authService";

import { AppSidebar } from './AppSidebar';

export function MainContent() {
  const [selectedEntity, setSelectedEntity] = useState("business-units");
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingItem, setEditingItem] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [reload, setReload] = useState("");
  const [activeTab, setActiveTab] = useState("businessPerformance");
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

     
  const handleLogout = () => {
    AuthService.clearToken();
    window.location.href = "/";
  };

   const onRelaod = () => {
    setReload(Math.random().toString());
  };


  React.useEffect(() => {
    const handleEntityClick = (e: Event) => {
      const target = e.target as HTMLElement;
      const button = target.closest('[data-entity]');
      if (button) {
        const entityId = button.getAttribute('data-entity');
        if (entityId) {
          setSelectedEntity(entityId);
          setSearchTerm("");
        }
      }
    };

    document.addEventListener('click', handleEntityClick);
    return () => document.removeEventListener('click', handleEntityClick);
  }, []);

  const getEntityTitle = (entityId: string) => {
    const entityTitles: Record<string, string> = {
      "business-units": "Business Units",
      "line-of-business": "Line of Business",
      "departments": "Departments",
      "locations": "Locations",
      "feed-parameters": "Feed Parameters",
      "calendar-weeks": "Calendar Weeks",
      "system-settings": "System Settings",
      "status-codes": "Status Codes",
    };
    return entityTitles[entityId] || "Entity Management";
  };

  const handleEdit = (item: any) => {
    setEditingItem(item);
    setIsFormOpen(true);
  };

  const handleCreate = () => {
    setEditingItem(null);
    setIsFormOpen(true);
  };

  const handleFormClose = () => {
    setIsFormOpen(false);
    setEditingItem(null);
  };

  return (
     <SidebarProvider>
      <div className="flex h-screen w-full overflow-hidden">
        <CustomSidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          setOpenModal={setIsDrawerOpen}
          handleLogout={handleLogout}
        />

         <div className="flex flex-col flex-1 h-screen overflow-hidden bg-background text-foreground transition-all duration-300"
          style={{ zIndex: 0, overflowY: 'auto' }} >

              <div className="border-b border-slate-700 ">
            <div className="flex items-center justify-between pad-6">
              <div className="flex items-center gap-4">
                {/* <SidebarTrigger className="text-foreground hover:text-white" /> */}
                <div>
                  <h1 className="text-2xl font-semibold">Company Settings</h1>
                  <p className="text-sm text-slate-400">Manage your primary and support data</p>
                </div>
              </div>
            </div>
          </div>
        
       <div className="flex flex-1 overflow-hidden">
    {/* Left Sidebar */}
    <div className="w-64 border-r border-slate-700">
      <AppSidebar activeTab={getEntityTitle(selectedEntity)} />
    </div>
        
         <div className="flex flex-col flex-1 h-screen overflow-hidden bg-background text-foreground transition-all duration-300">
          <div className="border-b border-slate-700 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <h1 className="text-2xl font-semibold">{getEntityTitle(selectedEntity)}</h1>
                  <p className="text-sm text-slate-400">Manage and configure {getEntityTitle(selectedEntity).toLowerCase()}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button onClick={() => onRelaod()} variant="outline" size="sm" className="border-slate-600 text-foreground hover:text-white">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
                <Button variant="outline" size="sm" className="border-slate-600 text-foreground hover:text-white">
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Input
                    placeholder="Search entities..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-80 bg-sidebar border-slate-600 text-foreground placeholder:text-slate-400"
                  />
                </div>
              </div>
              
              <Dialog open={isFormOpen} onOpenChange={setIsFormOpen}>
                <DialogTrigger asChild>
                  <Button onClick={handleCreate} className="bg-primaryhover:bg-blue-700">
                    <Plus className="w-4 h-4 mr-2" />
                    Add New
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-slate-800 border-slate-600 text-white max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>
                      {editingItem ? 'Edit' : 'Create'} {getEntityTitle(selectedEntity).slice(0, -1)}
                    </DialogTitle>
                  </DialogHeader>
                  <EntityForm 
                    entityType={selectedEntity} 
                    editingItem={editingItem}
                    onClose={handleFormClose}
                  />
                </DialogContent>
              </Dialog>
            </div>

            <EntityDataTable 
              entityType={selectedEntity}
              searchTerm={searchTerm}
              reload={reload}
              onEdit={handleEdit}
            />
          </div>
          </div> 
         </div>
        </div>
    </div>
    </SidebarProvider>
  );
}
