"use client";

import React, { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui2/input";
import { Textarea } from "@/components/ui2/textarea";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import axios from 'axios';
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Building2, Briefcase, ChevronDown, Plus, Download, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import type { HeaderSectionProps, BusinessUnitName, LineOfBusinessName, TimeInterval } from "./types";
import { DateRangePicker } from "./date-range-picker";
import { AiGroupingDialog } from "./ai-grouping-dialog";
import { AVAILABLE_MODELS } from "../../models/shared/constants";
import AuthService from "@/auth/utils/authService";

import AppConfig from '../../../auth/config.js';



export function HeaderSection({
  allBusinessUnits,
  actualLobsForCurrentBu,
  selectedBusinessUnit,
  selectedModel,
  onSelectModel,
  navigateSimulator,
  whatIfDetails,
  businessId,
  onSelectBusinessUnit,
  selectedLineOfBusiness,
  onSelectLineOfBusiness,
  selectedTimeInterval,
  onSelectTimeInterval,
  selectedDateRange,
  onSelectDateRange,
  allAvailablePeriods,
  // For merged header
  displayedPeriodHeaders,
  activeHierarchyContext,
  headerPeriodScrollerRef,
}: HeaderSectionProps) {
  const [isAiDialogOpen, setIsAiDialogOpen] = useState(false);

  const [isAddOpen, setAddOpen] = useState(false);
  const [formLoading, setFormLoading] = useState(false);
  const [isWhatIfNavigate, setWhatIfNavigate] = useState(false);

  const { toast } = useToast();

  const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;


  const [formData, setFormData] = useState<Record<string, any>>({ name: "", description: "", end_date: "", start_date: "" });

  const handleFormInputChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const fields = [
    { name: "name", label: "Name", placeholder: "Name your what-if scenario..", type: "text", required: true },
    { name: "start_date", label: "Start Date", type: "date", required: true },
    { name: "end_date", label: "End Date", type: "date", required: false },
    { name: "description", placeholder: "Enter the description", label: "Description", type: "textarea", required: false },
  ];


  type ForecastFormData = {
    name: string;
    description: string;
    start_date: string;
    end_date: string;
    business_unit: number;
    client_id: number;
  };


  async function submitForecastForm(model: string, values: ForecastFormData) {
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormLoading(true);
    console.log('isWhatIfNavigate', isWhatIfNavigate)
    try {
      const response = await submitForecastForm("what_if_simulator", {
        name: formData.name,
        description: formData.description,
        start_date: formData.start_date,
        end_date: formData.end_date,
        business_unit: businessId,
        client_id: AuthService.getCompanyId()
      });
      toast({
        title: "Record Created",
        description: `What-if Simulator has been created successfully.`,
      });
      setFormLoading(false);
      setAddOpen(false);
      if (isWhatIfNavigate && navigateSimulator) {
        setTimeout(() => {
          navigateSimulator(response?.[0] || '');
        }, 2000);
      }
    } catch (err) {
      setFormLoading(false);
      console.error("API Error:", err);
    }

  };

  const handleLobSelectionChange = (lob: string, checked: boolean) => {
    const newSelectedLOBs = checked
      ? [...selectedLineOfBusiness, lob]
      : selectedLineOfBusiness.filter((item) => item !== lob);
    onSelectLineOfBusiness(newSelectedLOBs);
  };

  let lobDropdownLabel = "Select LOBs";
  if (selectedLineOfBusiness.length === 1) {
    lobDropdownLabel = selectedLineOfBusiness[0];
  } else if (actualLobsForCurrentBu.length > 0 && selectedLineOfBusiness.length === actualLobsForCurrentBu.length) {
    lobDropdownLabel = `All ${actualLobsForCurrentBu.length} LOBs`;
  } else if (selectedLineOfBusiness.length > 1) {
    lobDropdownLabel = `${selectedLineOfBusiness.length} LOBs Selected`;
  } else if (actualLobsForCurrentBu.length === 0) {
    lobDropdownLabel = "No LOBs for " + selectedBusinessUnit;
  }

  return (
    <TooltipProvider>
      <header className="sticky top-0 z-50 bg-background p-4 border-b border-border">
        <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
          {!(whatIfDetails && whatIfDetails?.name) && (
            <h1 className="text-xl font-semibold text-foreground">Tactical Capacity Insights</h1>
          )}
          {whatIfDetails && whatIfDetails?.name && (
            <h1 className="text-xl font-semibold text-foreground">
              <span
                className="ml-1 cursor-pointer"
                onClick={() => navigateSimulator(whatIfDetails?.id)}
              >
                What-If Simulators
              </span>
              <span className="ml-2 opacity-7" style={{ opacity: 0.7 }}>
                /
                {' '}
                {`${whatIfDetails?.name} (${whatIfDetails.create_uid && whatIfDetails.create_uid?.[1] || ''})`}
              </span>
            </h1>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {!(whatIfDetails && whatIfDetails?.name) && (
              <Button variant="outline" size="sm" onClick={() => setAddOpen(true)}>
                <Plus className="mr-2" /> What-If Simulate
              </Button>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" /> Export CSV
                </Button>
              </TooltipTrigger>
              <TooltipContent><p>Export current view as CSV (not implemented)</p></TooltipContent>
            </Tooltip>
            <Button variant="default" size="sm" onClick={() => setIsAiDialogOpen(true)}>
              <Zap className="mr-2 h-4 w-4" /> Assumptions Assister
            </Button>
            {whatIfDetails && whatIfDetails?.name && (
              <div className="flex items-center ml-auto">
                <div className="border rounded mr-2">
                  <Button variant="ghost" className="" size="sm" onClick={() => navigateSimulator(whatIfDetails?.id)}>
                    Reset
                  </Button>
                </div>
                <Button size="sm" onClick={() => navigateSimulator(whatIfDetails?.id)}>
                  Save
                </Button>
              </div>
            )}
          </div>
        </div>

        <Sheet open={isAddOpen} onOpenChange={setAddOpen}>
          <SheetContent
            side="right"
            className="w-[1000px] h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]"
          >
            <SheetHeader>
              <SheetTitle>Create What-if Simulator</SheetTitle>
            </SheetHeader>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-4 mt-3">
                {/* Forecast Period */}
                <div className="p-2">

                  <div className="">
                    {fields.map((field) => (
                      <div key={field.name} className={field.type === 'textarea' ? 'md:col-span-2 mb-2' : 'mb-3'}>
                        <Label
                          htmlFor={field.name}
                          className="text-gray-800 dark:text-slate-300"
                        >
                          {field.label}
                        </Label>

                        {field.type === 'select' ? (
                          <Select
                            value={formData[field.name] || ''}
                            onValueChange={(value) => handleFormInputChange(field.name, value)}
                          >
                            <SelectTrigger className="bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400">
                              <SelectValue placeholder={`Select ${field.label}`} />
                            </SelectTrigger>
                            <SelectContent className="bg-white border-gray-300 text-black
                                                                                        dark:bg-slate-700 dark:border-slate-600 dark:text-white">
                              {field.options?.map((option) => (
                                <SelectItem
                                  key={option}
                                  value={option}
                                  className="text-black hover:bg-gray-100
                                                                                dark:text-white dark:hover:bg-slate-600"
                                >
                                  {option}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : field.type === 'textarea' ? (
                          <Textarea
                            id={field.name}
                            value={formData[field.name] || ''}
                            onChange={(e) => handleFormInputChange(field.name, e.target.value)}
                            className="w-full bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400"
                            rows={3}
                            placeholder={field.placeholder}
                          />
                        ) : (
                          <Input
                            id={field.name}
                            type={field.type}
                            value={formData[field.name] || ''}
                            required={field.required}
                            placeholder={field.placeholder}
                            onChange={(e) => handleFormInputChange(field.name, e.target.value)}
                            className="w-full bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
                {/* Apply Changes Button */}
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <Button
                    size="sm"
                    className="w-full mb-2"
                    type="submit"
                    disabled={formLoading}
                  >
                    {formLoading ? 'Saving' : 'Save'}
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    className="w-full"
                    type="submit"
                    onClick={() => setWhatIfNavigate(true)}
                    disabled={formLoading}
                  >
                    {formLoading ? 'Saving' : 'Save and show Simulator'}
                  </Button>
                </div>
              </div>
            </form>
          </SheetContent>
        </Sheet>

        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:flex xl:flex-wrap xl:items-end gap-x-4 gap-y-4">

          <div className="flex flex-col gap-2">
            <Label className="text-sm font-medium text-foreground">Model</Label>
            <Select value={selectedModel} onValueChange={onSelectModel}>
              <SelectTrigger className="w-full xl:w-[250px] text-sm h-10 border-2 transition-colors hover:border-primary/50 focus:border-primary">
                <Building2 className="mr-2 h-4 w-4 text-primary" />
                <SelectValue placeholder="Select Model" />
              </SelectTrigger>
              <SelectContent>
                {AVAILABLE_MODELS.map((bu) => (
                  <SelectItem key={bu.id} value={bu.id} className="cursor-pointer">
                    <div className="flex items-center gap-2">
                      <span>{bu.name}</span>
                    </div>
                    {bu.description && (
                      <div className="text-xs text-muted-foreground">
                        {bu.description}
                      </div>
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <Label className="text-sm font-medium text-foreground">Line of Business</Label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="w-full xl:w-[280px] text-sm h-10 justify-between border-2 transition-colors hover:border-primary/50 focus:border-primary">
                  <div className="flex items-center truncate">
                    <Briefcase className="mr-2 h-4 w-4 text-primary flex-shrink-0" />
                    <span className="truncate" title={lobDropdownLabel}>{lobDropdownLabel}</span>
                  </div>
                  <ChevronDown className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[280px]">
                <DropdownMenuLabel className="text-sm font-semibold">Select Lines of Business</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {actualLobsForCurrentBu.length > 0 ? (
                  actualLobsForCurrentBu.map((lob) => (
                    <DropdownMenuCheckboxItem
                      key={lob}
                      checked={selectedLineOfBusiness.includes(lob)}
                      onCheckedChange={(checked) => handleLobSelectionChange(lob, Boolean(checked))}
                      onSelect={(e) => e.preventDefault()}
                      className="cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <Briefcase className="h-4 w-4 text-muted-foreground" />
                        <span>{lob}</span>
                      </div>
                    </DropdownMenuCheckboxItem>
                  ))
                ) : (
                  <DropdownMenuItem disabled>No LOBs available for {selectedBusinessUnit}</DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="flex flex-col gap-2">
            <Label className="text-sm font-medium text-foreground">Interval</Label>
            <div className="flex items-center gap-1 border-2 rounded-lg p-1 bg-muted/30 h-10 transition-colors hover:border-primary/50">
              <Button
                variant={selectedTimeInterval === "Week" ? "default" : "ghost"}
                size="sm"
                onClick={() => onSelectTimeInterval("Week")}
                className={cn(
                  "h-8 px-4 flex-1 text-sm font-medium transition-all",
                  selectedTimeInterval === "Week" && "shadow-sm"
                )}
              >
                Week
              </Button>
              <Button
                variant={selectedTimeInterval === "Month" ? "default" : "ghost"}
                size="sm"
                onClick={() => onSelectTimeInterval("Month")}
                className={cn(
                  "h-8 px-4 flex-1 text-sm font-medium transition-all",
                  selectedTimeInterval === "Month" && "shadow-sm"
                )}
              >
                Month
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <Label className="text-sm font-medium text-foreground">Date Range</Label>
            <div className="[&_button]:h-10 [&_button]:border-2 [&_button]:transition-colors [&_button:hover]:border-primary/50">
              <DateRangePicker date={selectedDateRange} onDateChange={onSelectDateRange} allAvailablePeriods={allAvailablePeriods} />
            </div>
          </div>
        </div>

        {/* Merged Table Header Row */}
        <div className="mt-4 flex items-stretch border-b border-border bg-card">
          <div className="sticky left-0 z-55 bg-card min-w-[320px] px-4 py-2 flex items-center border-r border-border/50 h-12">
            <span className="text-sm font-medium text-muted-foreground truncate">{activeHierarchyContext}</span>
          </div>
          <div
            ref={headerPeriodScrollerRef}
            className="flex-grow overflow-x-auto scrollbar-thin whitespace-nowrap flex items-stretch h-12"
          >
            {displayedPeriodHeaders.map((period) => {
              const parts = period.split(': ');
              const weekLabelPart = parts[0].replace("FWk", "W");
              let dateRangePart = "";
              if (parts.length > 1) {
                const dateAndYearPart = parts[1];
                // Regex to match DD/MM-DD/MM, ignoring the year part for display here
                const dateMatch = dateAndYearPart.match(/^(\d{2}\/\d{2}-\d{2}\/\d{2})/);
                if (dateMatch) {
                  dateRangePart = dateMatch[1];
                }
              }
              return (
                <div
                  key={period}
                  className="min-w-[100px] px-2 py-2 flex flex-col items-end justify-center text-right border-l border-border/50 h-full"
                >
                  <span className="text-sm font-medium">{weekLabelPart}</span>
                  {dateRangePart && (
                    <span className="text-xs text-muted-foreground">
                      ({dateRangePart})
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        <AiGroupingDialog open={isAiDialogOpen} onOpenChange={setIsAiDialogOpen} />
      </header>
    </TooltipProvider>
  );
}
