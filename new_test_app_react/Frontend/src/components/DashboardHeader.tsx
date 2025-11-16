import {
  RefreshCw, Menu,
  UserCircle,
  Sun,
  Moon,
  Settings,
  HelpCircle,
  LogOut,
  Grid,
  Monitor,
  Zap, CalendarCheck, LineChart, MonitorSmartphone, BarChart2, LayoutDashboard,
  Bot,
  LucideComputer,
  Grid3x3,
  Search,
  X,
} from "lucide-react";
import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { Input } from "@/components/ui/input";
import axios from 'axios';
import { useToast } from "@/hooks/use-toast";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator
} from "@/components/ui/dropdown-menu";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { UnitSelection } from './UnitSelection';
import { Chip, IconButton, TextField } from "@mui/material"
import ThemeSelector from "./ThemeSelector";
import { ChevronDown, Check, Building2, House } from "lucide-react";
import AuthService from "@/auth/utils/authService";
import { cn } from "@/lib/utils";
import { useTheme } from "../components/ThemeContext";

import AppConfig from '../auth/config.js';

interface DashboardHeaderProps {
  title: string;
  description: string | unknown
  onLogout?: () => void;
  businessData?: any[];
  businessDataList: Array<{
    id?: number;
    display_name: string;
    code: string;
    sequence?: number;
    description?: string;
    [key: string]: unknown;
  }>;
  lastUpdated: string;
  forecastPeriod?: string | number;
  defaultConfigId?: boolean | number;
  onRefresh?: () => void;
  userData?: {
    sub?: number;
    [key: string]: unknown;
  };
  setBusinessUnit?: (business: {
    id?: number;
    display_name: string;
    code: string;
    [key: string]: unknown;
  }) => void;
  activeTab?: string;
  handleLogout?: () => void;
  options?: any[];
  setActiveMainTab?: () => void;
}

export const DashboardHeader = ({
  title = "Walmart Fulfillment Services",
  description,
  lastUpdated,
  businessDataList = [],
  forecastPeriod,
  onRefresh,
  setBusinessUnit,
  userData,
  businessData,
  handleLogout,
  defaultConfigId,
  activeTab,
  options,
  setActiveMainTab,
}: DashboardHeaderProps) => {
  const { toast } = useToast();
  const { toggleSidebar, state } = useSidebar();
  const { themeMode } = useTheme();
  const navigate = useNavigate();

  const themeAnchorRef = useRef<HTMLDivElement | null>(null);

  const [open, setOpen] = useState(false);
  const [appOpen, setAppOpen] = useState(false);

  const features = [
    {
      title: "Forecasting",
      icon: <LineChart className="h-7 w-7 text-blue-600" />,
      description: "Predict trends, stay ahead, act with confidence.",
      roadmap: false,
    },
    {
      title: "Planning",
      icon: <CalendarCheck className="h-7 w-7 text-green-600" />,
      description: "Strategize resources, align goals, execute flawlessly.",
      roadmap: false,
    },
    {
      title: "Scheduling",
      icon: <Zap className="h-7 w-7 text-yellow-500" />,
      description: "Optimize shifts, balance workload, maximize efficiency.",
      roadmap: true,
    },
    {
      title: "Realtime Ops Support",
      icon: <MonitorSmartphone className="h-7 w-7 text-rose-600" />,
      description: "Monitor live, resolve fast, keep operations smooth.",
      roadmap: true,
    },
    {
      title: "Dashboards & Analytics",
      icon: <LayoutDashboard className="h-7 w-7 text-indigo-600" />,
      description: "Visualize performance, identify patterns, and unlock insights.",
      roadmap: true,
    },
    {
      title: "Agent Gauri (Chat bot)",
      icon: <Bot className="h-7 w-7 text-cyan-600" />,
      description: "AI-powered assistant to guide agents and answer questions.",
      roadmap: true,
    },
    {
      title: "AI Agent",
      icon: <LucideComputer className="h-7 w-7 text-teal-600" />,
      description: "Manage and monitor operations on the go.",
      roadmap: false,
    },
    {
      title: "CXO Insights",
      icon: <BarChart2 className="h-7 w-7 text-purple-600" />,
      description: "Executive dashboards for data-driven support decisions.",
      roadmap: false,
    },
    {
      title: "Settings",
      icon: <Settings className="h-7 w-7 text-gray-600" />,
      description: "Customize and control your system settings.",
      roadmap: false,
    },
  ];


  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState(null);

  const handleCardClick = (feature) => {
    if (feature.title === "Settings") {
      navigate("/company");
    } else if (feature.title === "AI Agent") {
      navigate("/new-agent");
    } else {
      setSelectedFeature(feature);
      setIsModalOpen(true);
    }
  };

  const handleRefresh = () => {
    toast({
      title: "Refreshing Dashboard",
      description: "Updating with the latest data...",
    });
  };

  const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;

  async function createForm(model: string, values: Record<string, unknown>) {
    const formData = new FormData();

    formData.append("model", model);
    formData.append("values", JSON.stringify(values)); // nested object as JSON string

    const url = `${WEBAPPAPIURL}create`;

    const response = await axios.post(url, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        Authorization: `Bearer ${AuthService.getAccessToken()}`
      },
    });

    return response.data;
  }

  const onCompanyClick = () => {
    navigate("/company");
  };


  async function updateForm(id: number, model: string, values: Record<string, unknown>) {
    const formData = new FormData();

    formData.append("ids", `[${id}]`);
    formData.append("model", model);
    formData.append("values", JSON.stringify(values)); // nested object as JSON string

    const url = `${WEBAPPAPIURL}write`;

    const response = await axios.put(url, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        Authorization: `Bearer ${AuthService.getAccessToken()}`
      },
    });

    return response.data;
  }

  interface BusinessItem {
    id?: number;
    display_name: string;
    code: string;
    [key: string]: unknown;
  }

  const onSwitchBusiness = async (item: BusinessItem) => {
    setBusinessUnit?.(item);
    if (defaultConfigId) {
      const response = await updateForm(Number(defaultConfigId), 'user.default_preferences', {
        value: item.code,
      });
    } else {
      const response = await createForm('user.default_preferences', {
        value: item.code,
        name: 'default_business_unit',
        user_id: userData?.sub || 0,
      });
    }
  };

  const [search, setSearch] = useState("");
  const [showSearch, setShowSearch] = useState(false);

  const filteredFeatures = features.filter((feature) =>
    feature.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="w-full sticky top-0 z-1200 bg-background border-b border-border/30 dark:border-border/50">
      <div
        className="flex flex-col sm:flex-row justify-between items-center gap-4 w-full"
        style={{ paddingBottom: "7px", paddingTop: "7px" }}
      >
        {/* Left Section */}
        <div
          className={`flex flex-row items-center gap-4 w-full ${activeTab === "House" ? "marginbt" : ""
            }`}
        >
          <div className="flex flex-row items-center gap-2">
            {activeTab === "House" ? (
              <House
                style={{ width: "20px", height: "20px", color: "gray" }}
                className="w-6 h-6"
              />
            ) : (
              <Building2
                style={{ width: "20px", height: "20px", color: "gray" }}
                className="w-6 h-6"
              />
            )}

            {activeTab !== "House" ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <div
                    className="flex flex-row items-center gap-2 cursor-pointer"
                    role="button"
                    tabIndex={0}
                  >
                    <div className="flex flex-col justify-center">
                      <h1 className="text-2xl font-bold tracking-tight text-foreground">
                        {title}
                      </h1>
                      {typeof description === "string" &&
                        description.trim() !== "" && (
                          <div
                            className="text-sm font-medium text-muted-foreground"
                            style={{ marginTop: "-5px" }}
                          >
                            {description}
                          </div>
                        )}
                    </div>
                    <ChevronDown className="w-5 h-5 mt-1 text-muted-foreground" />
                  </div>
                </DropdownMenuTrigger>

                <DropdownMenuContent align="start" className="w-[280px] mt-1">
                  <p className="px-3 pt-2 pb-1 text-sm font-semibold text-foreground">
                    Select a Business Unit
                  </p>
                  <DropdownMenuSeparator />
                  {businessDataList
                    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
                    .map((business) => {
                      const isSelected = title === business.display_name;
                      return (
                        <DropdownMenuItem
                          key={business.id || business.display_name}
                          onClick={() => onSwitchBusiness(business)}
                          disabled={isSelected}
                          className={cn(
                            "flex flex-col items-start px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-800",
                            isSelected && "font-semibold"
                          )}
                        >
                          <div className="flex items-center justify-between w-full">
                            <div className="flex items-center gap-2">
                              <Building2 className="w-4 h-4 text-muted-foreground" />
                              <span>{business.display_name}</span>
                            </div>
                            {isSelected && (
                              <Check className="w-5 h-5 font-bold text-primary" />
                            )}
                          </div>
                          {business.description && (
                            <div className="text-xs text-muted-foreground pl-6">
                              {business.description}
                            </div>
                          )}
                        </DropdownMenuItem>
                      );
                    })}
                  <DropdownMenuSeparator />
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <div className="flex flex-col justify-center">
                <h1 className="text-2xl font-bold tracking-tight text-foreground">
                  Home
                </h1>
              </div>
            )}
          </div>
        </div>

        {/* Right Section */}
        <div className="flex flex-row items-center gap-2">
          {/* Apps Icon */}


          <Popover open={appOpen} onOpenChange={setAppOpen}>
            <PopoverTrigger asChild>
              <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition">
                <Grid className="w-5 h-5 text-muted-foreground" />
              </button>
            </PopoverTrigger>

            <PopoverContent
              className="w-[600px] max-h-[500px] overflow-y-auto p-4 bg-popover text-popover-foreground border-border"
              side="bottom"
              align="start"
            >
              <TooltipProvider delayDuration={200}>
                <div className="">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <h2 className="text-base font-semibold">Apps</h2>
                    <IconButton sx={{
                      color: "var(--foreground)", // icon adapts to theme
                    }} size="small" onClick={() => setShowSearch((s) => !s)}>
                      <Search size={18} />
                    </IconButton>
                  </div>

                  <div className="w-100 mb-3">
                    {showSearch && (
                      <div className="flex items-center gap-3">
                        <TextField
                          variant="outlined"
                          size="medium"
                          placeholder="Search apps..."
                          value={search}
                          fullWidth
                          onChange={(e) => setSearch(e.target.value)}
                          sx={{
                            "& .MuiOutlinedInput-root": {
                              borderRadius: "0px",
                              backgroundColor: "var(--background)",
                              color: "var(--foreground)",
                              "& fieldset": {
                                borderColor: "var(--foreground)", // subtle border
                              },
                              "&:hover fieldset": {
                                borderColor: "var(--foreground)",
                                color: "var(--foreground)",
                              },
                            },
                            "& input": {
                              padding: "6px 10px",
                              fontSize: "0.9rem",
                              color: "var(--foreground)", // input text
                              "&::placeholder": {
                                color: "var(--foreground)", // placeholder adapts
                              },
                            },
                          }}
                        />
                        {search && (
                          <IconButton sx={{
                            color: "var(--foreground)", // icon adapts to theme
                          }} size="small" onClick={() => { setSearch(""); setShowSearch(false); }}>
                            <X size={16} />
                          </IconButton>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {filteredFeatures.map((feature) => (
                    <UITooltip key={feature.title}>
                      <TooltipTrigger asChild>
                        <div
                          onClick={() => handleCardClick(feature)}
                          className={cn(
                            "flex cursor-pointer flex-col items-center justify-center",
                            "p-6 bg-card border shadow-sm",
                            "hover:bg-accent transition-all duration-200"
                          )}
                        >
                          <div className="mb-3 text-4xl">{feature.icon}</div>
                          <h3 className="text-base font-semibold text-center">
                            {feature.title}
                            {feature.roadmap && (
                              <Chip
                                label="R"
                                size="small"
                                className="colour ml-2"
                                sx={{ color: 'var(--primary-foreground)' }}
                              />
                            )}
                          </h3>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-xs p-4 rounded-lg shadow-md">
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-semibold">{feature.title}</span>
                            {feature.roadmap && (
                              <Chip
                                label="R"
                                size="small"
                                className="colour ml-2"
                                sx={{ color: 'var(--primary-foreground)' }}
                              />
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">{feature.description}</p>
                        </div>
                      </TooltipContent>
                    </UITooltip>
                  ))}
                </div>
              </TooltipProvider>

              {/* Your modal for unit selection */}
              <UnitSelection
                open={isModalOpen}
                feature={selectedFeature}
                options={options}
                businessData={businessData}
                setBusinessData={setBusinessUnit}
                setActiveTab={setActiveMainTab}
                onClose={() => { setIsModalOpen(false); setAppOpen(false); setSearch(""); }}
              />
            </PopoverContent>
          </Popover>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="p-2 mr-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition">
                <UserCircle className="w-5 h-5 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <div ref={themeAnchorRef}>
                <DropdownMenuItem
                  className="cursor-pointer flex items-center gap-2"
                  onSelect={(e) => {
                    e.preventDefault(); // prevent default menu action
                    setOpen(true); // open popover after dropdown closes
                  }}
                >
                  {themeMode === "light" && <Sun className="w-4 h-4" />}
                  {themeMode === "dark" && <Moon className="w-4 h-4" />}
                  {themeMode === "system" && <Monitor className="w-4 h-4" />}
                  Themes
                </DropdownMenuItem>
              </div>
              <DropdownMenuItem className="cursor-pointer" onClick={onCompanyClick}>
                <Settings className="w-4 h-4 mr-2" />
                Company Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <HelpCircle className="w-4 h-4 mr-2" />
                Help
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout} className="cursor-pointer text-red-500 hover:text-red-600">
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Popover open={open} >
            <PopoverTrigger asChild>
              {/* use div anchor */}
              <div ref={themeAnchorRef} />
            </PopoverTrigger>
            <PopoverContent
              className="w-80 p-4 bg-popover text-popover-foreground border-border"
              side="right"
              align="start"
              onEscapeKeyDown={() => setOpen(false)}
              onPointerDownOutside={() => setOpen(false)}
            >
              <ThemeSelector />
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </div >
  );
};

export default DashboardHeader;
