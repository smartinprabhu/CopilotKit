import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { CalendarIcon, ChevronDown, Building2, Briefcase, Download, Zap } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import AiGroupingDialog from "./AiGroupingDialog";
import DateRangePicker from "./DateRangePicker";
// import { BusinessUnitName, TimeInterval, DateRange } from "./types"; // Original, remove if redundant

import { 
  BusinessUnitName, 
  TimeInterval, 
  DateRange, 
  CapacityDataRow,
  TeamPeriodicMetrics, // Added
  AggregatedPeriodicMetrics, // Added
  TEAM_METRIC_ROW_DEFINITIONS, // Added
  AGGREGATED_METRIC_ROW_DEFINITIONS // Added
} from "./types";

interface HeaderSectionProps {
  filterOptions: { businessUnits: BusinessUnitName[]; linesOfBusiness: string[] };
  selectedBusinessUnit: BusinessUnitName;
  onSelectBusinessUnit: (value: BusinessUnitName) => void;
  selectedLineOfBusiness: string[];
  onSelectLineOfBusiness: (value: string[]) => void;
  selectedTimeInterval: TimeInterval;
  onSelectTimeInterval: (value: TimeInterval) => void;
  selectedDateRange: DateRange | undefined;
  onSelectDateRange: (value: DateRange | undefined) => void;
  allAvailablePeriods: string[];
  displayedPeriodHeaders: string[]; // This was already here
  activeHierarchyContext: string;
  headerPeriodScrollerRef: React.RefObject<HTMLDivElement>;
  displayableCapacityData: CapacityDataRow[]; // Add this
}

export default function HeaderSection({
  filterOptions,
  selectedBusinessUnit,
  onSelectBusinessUnit,
  selectedLineOfBusiness,
  onSelectLineOfBusiness,
  selectedTimeInterval,
  onSelectTimeInterval,
  selectedDateRange,
  onSelectDateRange,
  allAvailablePeriods,
  displayedPeriodHeaders, // This was already here
  activeHierarchyContext,
  headerPeriodScrollerRef,
  displayableCapacityData, // Add this
}: HeaderSectionProps) {
  const [isAiDialogOpen, setIsAiDialogOpen] = useState(false);

  const handleExportCsv = () => {
    if (!displayableCapacityData || !displayedPeriodHeaders) {
      console.error("CSV Export: Data not available");
      // Optionally, show a toast to the user
      return;
    }

    const dataForExport: string[][] = [];

    // Helper to format period headers for CSV (e.g., "W1", "W2")
    const formatPeriodHeaderForCsv = (header: string): string => {
      const parts = header.split(': ');
      return parts[0].replace("FWk", "W");
    };
    const csvHeaders = ["Hierarchy / Metric", ...displayedPeriodHeaders.map(formatPeriodHeaderForCsv)];
    dataForExport.push(csvHeaders);

    // Recursive helper to flatten data and collect metric rows
    const processRow = (row: CapacityDataRow, indentLevel: number) => {
      const indent = "  ".repeat(indentLevel); // 2 spaces per indent level
      const namePrefix = indent + row.name;

      // Add primary HC rows for BU/LOB/Team first
      const primaryHcKeys: Array<keyof (TeamPeriodicMetrics & AggregatedPeriodicMetrics)> = ["requiredHC", "actualHC", "overUnderHC"];
      
      primaryHcKeys.forEach(metricKey => {
        const metricDef = AGGREGATED_METRIC_ROW_DEFINITIONS.find(def => def.key === metricKey) || TEAM_METRIC_ROW_DEFINITIONS.find(def => def.key === metricKey);
        if (!metricDef) return;

        // Skip LOB specific metrics if item is a BU
        if (row.itemType === 'BU' && (metricDef.key === 'lobVolumeForecast' || metricDef.key === 'lobAverageAHT' || metricDef.key === 'lobTotalBaseRequiredMinutes')) {
          return;
        }

        const values = displayedPeriodHeaders.map(period => {
          const periodData = row.periodicData[period];
          const val = periodData ? (periodData as any)[metricKey] : null;
          if (val === null || val === undefined) return "";
          return typeof val === 'number' && (metricDef.isHC || ['requiredHC', 'actualHC', 'overUnderHC'].includes(metricKey as string)) ? Math.round(val) : val;
        });
        dataForExport.push([namePrefix + (indentLevel === 0 ? "" : ` - ${metricDef.label}`), ...values.map(String)]);
      });
      
      // For LOBs, add LOB-specific aggregated metrics
      if (row.itemType === 'LOB') {
          const lobSpecificKeys: Array<keyof AggregatedPeriodicMetrics> = ["lobVolumeForecast", "lobAverageAHT", "lobTotalBaseRequiredMinutes"];
          lobSpecificKeys.forEach(metricKey => {
              const metricDef = AGGREGATED_METRIC_ROW_DEFINITIONS.find(def => def.key === metricKey);
              if (!metricDef) return;
              const values = displayedPeriodHeaders.map(period => {
                  const periodData = row.periodicData[period];
                  const val = periodData ? (periodData as any)[metricKey] : null;
                  return (val === null || val === undefined) ? "" : String(val);
              });
              dataForExport.push([namePrefix + ` - ${metricDef.label}`, ...values]);
          });
      }

      // For Teams, add detailed metrics based on TEAM_METRIC_ROW_DEFINITIONS
      if (row.itemType === 'Team') {
        const teamMetricCategories = ['Assumption', 'HCAdjustment']; // Categories to iterate over
        teamMetricCategories.forEach(category => {
          // Add a sub-header for the category if desired, or just list metrics
          // dataForExport.push([namePrefix + ` - ${category}`]); // Optional category row
          TEAM_METRIC_ROW_DEFINITIONS.forEach(metricDef => {
            if (metricDef.category === category && !primaryHcKeys.includes(metricDef.key as any) && metricDef.key !== '_calculatedActualProductiveAgentMinutes' && metricDef.key !== '_lobTotalBaseReqMinutesForCalc' && metricDef.key !== '_calculatedRequiredAgentMinutes') { // Exclude already handled primary HC and internal calcs
              const values = displayedPeriodHeaders.map(period => {
                const periodData = row.periodicData[period] as TeamPeriodicMetrics;
                const val = periodData ? periodData[metricDef.key as keyof TeamPeriodicMetrics] : null;
                if (val === null || val === undefined) return "";
                return metricDef.isHC ? Math.round(val as number) : val;
              });
              dataForExport.push([namePrefix + ` - ${metricDef.label}`, ...values.map(String)]);
            }
          });
        });
      }

      // Process children
      if (row.children) {
        row.children.forEach(child => processRow(child, indentLevel + 1));
      }
    };

    displayableCapacityData.forEach(buRow => processRow(buRow, 0));

    // Convert array of arrays to CSV string
    const csvString = dataForExport.map(e => e.join(",")).join("\n");

    // Trigger download
    const filename = "capacity_insights_export.csv";
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    if (link.download !== undefined) { // feature detection
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } else {
        console.error("CSV Export: Download not supported by browser.");
        // Fallback or message to user
    }
  };

  const handleLobSelectionChange = (lob: string, checked: boolean) => {
    const newSelectedLOBs = checked
      ? [...selectedLineOfBusiness, lob]
      : selectedLineOfBusiness.filter((item) => item !== lob);
    onSelectLineOfBusiness(newSelectedLOBs);
  };

  const actualLobsForCurrentBu = filterOptions.linesOfBusiness;
  let lobDropdownLabel = "Select LOBs";
  if (selectedLineOfBusiness.length === 1) {
    lobDropdownLabel = selectedLineOfBusiness[0];
  } else if (actualLobsForCurrentBu.length > 0 && selectedLineOfBusiness.length === actualLobsForCurrentBu.length) {
    lobDropdownLabel = `All ${actualLobsForCurrentBu.length} LOBs`;
  } else if (selectedLineOfBusiness.length > 1) {
    lobDropdownLabel = `${selectedLineOfBusiness.length} LOBs Selected`;
  } else if (actualLobsForCurrentBu.length === 0) {
    lobDropdownLabel = "No LOBs";
  }

  return (
    <TooltipProvider>
      <header className="sticky top-0 z-50 bg-background p-4 border-b border-border">
        <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
          <h1 className="text-2xl font-semibold text-foreground">Tactical Capacity Insights</h1>
          <div className="flex flex-wrap items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm" onClick={handleExportCsv}>
                  <Download className="mr-2" /> Export CSV
                </Button>
              </TooltipTrigger>
              <TooltipContent><p>Export current view as CSV</p></TooltipContent>
            </Tooltip>
            <Button variant="default" size="sm" onClick={() => setIsAiDialogOpen(true)}>
              <Zap className="mr-2" /> Assumptions Assister
            </Button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:flex lg:flex-wrap items-center gap-x-4 gap-y-2">
          <Select value={selectedBusinessUnit} onValueChange={onSelectBusinessUnit}>
            <SelectTrigger className="w-full lg:w-[180px] text-sm h-9">
              <Building2 className="mr-2 opacity-70" />
              <SelectValue placeholder="Business Unit" />
            </SelectTrigger>
            <SelectContent>
              {filterOptions.businessUnits.map((bu) => (
                <SelectItem key={bu} value={bu}>
                  {bu}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="w-full lg:w-[240px] text-sm h-9 justify-between">
                <div className="flex items-center truncate">
                  <Briefcase className="mr-2 opacity-70 flex-shrink-0" />
                  <span className="truncate" title={lobDropdownLabel}>{lobDropdownLabel}</span>
                </div>
                <ChevronDown className="h-4 w-4 opacity-50 flex-shrink-0" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-full md:w-[240px]">
              <DropdownMenuLabel>Select Lines of Business</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {actualLobsForCurrentBu.length > 0 ? (
                actualLobsForCurrentBu.map((lob) => (
                  <DropdownMenuCheckboxItem
                    key={lob}
                    checked={selectedLineOfBusiness.includes(lob)}
                    onCheckedChange={(checkedValue) => handleLobSelectionChange(lob, Boolean(checkedValue))}
                    onSelect={(e) => e.preventDefault()}
                  >
                    {lob}
                  </DropdownMenuCheckboxItem>
                ))
              ) : (
                <DropdownMenuItem disabled>No LOBs available for {selectedBusinessUnit}</DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>

          <div className="flex items-center gap-2 border rounded-md p-1 bg-muted">
            <Button
              variant={selectedTimeInterval === "Week" ? "default" : "ghost"}
              size="sm"
              onClick={() => onSelectTimeInterval("Week")}
              className="h-7 px-3"
            >
              Week
            </Button>
            <Button
              variant={selectedTimeInterval === "Month" ? "default" : "ghost"}
              size="sm"
              onClick={() => onSelectTimeInterval("Month")}
              className="h-7 px-3"
            >
              Month
            </Button>
          </div>
          <DateRangePicker date={selectedDateRange} onDateChange={onSelectDateRange} />
        </div>

        <div className="flex items-center border-b border-border bg-card px-4 h-12 mt-4">
          <div className="sticky left-0 z-55 bg-card min-w-[300px] whitespace-nowrap px-4 py-2 text-sm font-semibold text-foreground h-full flex items-center">
            {activeHierarchyContext}
          </div>
          <div ref={headerPeriodScrollerRef} className="flex-grow overflow-x-auto scrollbar-hide whitespace-nowrap h-full">
            <div className="flex h-full">
              {displayedPeriodHeaders.map((period) => {
                const parts = period.split(': ');
                const weekLabelPart = parts.length > 0 ? parts[0].replace("FWk", "W") : period;
                let dateRangePart = "";
                if (parts.length > 1) {
                  const dateAndYearPart = parts[1];
                  const dateMatch = dateAndYearPart.match(/^(\d{2}\/\d{2}-\d{2}\/\d{2})/);
                  if (dateMatch) {
                    dateRangePart = dateMatch[1];
                  }
                }
                return (
                  <div
                    key={period}
                    className="text-right min-w-[100px] px-2 py-2 border-l border-border/50 h-full flex flex-col justify-center items-end"
                  >
                    <span className="text-sm font-medium text-foreground">{weekLabelPart}</span>
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
        </div>
        <AiGroupingDialog open={isAiDialogOpen} onOpenChange={setIsAiDialogOpen} />
      </header>
    </TooltipProvider>
  );
}
