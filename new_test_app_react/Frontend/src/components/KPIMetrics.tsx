import React, { useState, useEffect } from "react";
import { KPIMetricsCard } from "./KPIMetricsCard";
import LoadingSkeleton from "./LoadingSkeleton";
import axios from "axios";
import { format } from 'date-fns';
import ExcelJS from 'exceljs';
import { toZonedTime } from 'date-fns-tz';
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from '@/components/ui/checkbox'; // Your checkbox component
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import DateRangePicker from "./DateRangePicker";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"; // Added imports
import NoDataFullPage from "@/components/noDataPage";
import { formatWithPrecision } from "@/lib/utils";

import { getComparisonRange } from './ComparisonUtil';
import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';


interface KPIMetricsProps {
  kpiData: any[];
  parametersList: any[];
  loading: boolean;
  kpiOldData: any[]; // Assuming this is used for future enhancements
  children?: React.ReactNode;
  businessId?: number;
  selectedRange?: any[];
  businessName?: string;
  weekStartsOn?: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  defaultDecimalPrecisions?: number;
}

const compareValueMap = {
  previous: "Previous Period",
  sameYear: "Same Period Previous Year",
  selectYear: "Same Period: Select Year",
  custom: "Custom",
};

function formatDateToUTCString(date: Date, formats: string): string {
  const utcDate = toZonedTime(date, 'UTC');
  return format(utcDate, formats);
}

const KPIMetrics: React.FC<KPIMetricsProps> = ({ businessId, businessName, selectedRange, weekStartsOn = 1 as 0 | 1 | 2 | 3 | 4 | 5 | 6, kpiData, defaultDecimalPrecisions, kpiOldData, parametersList, loading, children }) => {

  const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;
  const { toast } = useToast();

  const [compareEnabled, setCompareEnabled] = useState(true);
  const [loadingKpi, setLoading] = useState(false);
  const [kpiCompareData, setKpiCompareData] = useState([]);
  const [kpiCompareTempData, setKpiCompareTempData] = useState([]);
  const [compareOption, setCompareOption] = useState<'previous' | 'sameYear' | 'selectYear' | 'custom'>('previous');
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear() - 1);
  const [customRange, setCustomRange] = useState<{ from: Date | null; to: Date | null }>({ from: null, to: null });

  const fetchData = async (id: number, formattedFrom: string, formattedTo: string) => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[["business_unit_id","=",${id}],["parameter_id.type","=","Volume"],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`,
      fields: '["date", "calendar_week_id", "business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
      model: 'data_feeds',
    });
    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    const runFetch = async () => {
      if (
        businessId &&
        compareEnabled &&
        selectedRange?.from &&
        selectedRange?.to
      ) {
        const comparisonRange = getComparisonRange(
          selectedRange,
          compareOption,
          selectedYear,
          customRange,
          weekStartsOn,
        );

        if (!comparisonRange) return;

        setLoading(true);

        try {
          const formattedFrom = format(comparisonRange.from, 'yyyy-MM-dd');
          const formattedTo = format(comparisonRange.to, 'yyyy-MM-dd');

          const result = await fetchData(businessId, formattedFrom, formattedTo);
          setKpiCompareTempData(result);
          setLoading(false);
        } catch (err) {
          setKpiCompareTempData([]);
          setLoading(false);
          console.error("Failed to fetch:", err);
        } finally {
          // setKpiCompareTempData([]);
        }
      }
    };

    runFetch();
  }, [
    compareEnabled,
    businessId,
    selectedYear,
    customRange,
    JSON.stringify(selectedRange),
    compareOption,
  ]);

  useEffect(() => {
    if (!kpiCompareTempData?.length) {
      setKpiCompareData([]);
      return;
    }

    const parameterOldGroups: Record<string, { title: string; value: number; parameterName?: string; parameterId?: number }> = {};
    const lobOldGroups: Record<string, { title: string; value: number; parameterName?: string; lobId?: number; parameterId?: number }> = {};

    kpiCompareTempData.forEach((item) => {
      const lobName = item.lob_id?.[1]?.trim();
      const lobId = item.lob_id?.[0];
      const paramName = item.parameter_id?.[1]?.trim();
      const paramId = item.parameter_id?.[0];
      const value = item.value || 0;

      if (!lobName && paramName) {
        if (!parameterOldGroups[paramName]) {
          parameterOldGroups[paramName] = {
            title: paramName,
            value: 0,
            parameterId: paramId,
          };
        }
        parameterOldGroups[paramName].value += value;
      } else if (lobName) {
        if (!lobOldGroups[lobName]) {
          lobOldGroups[lobName] = {
            title: lobName,
            value: 0,
            lobId,
            parameterName: paramName,
            parameterId: paramId,
          };
        }
        lobOldGroups[lobName].value += value;
      }
    });

    const groupedOldData = [
      ...Object.values(parameterOldGroups),
      ...Object.values(lobOldGroups),
    ];

    setKpiCompareData(groupedOldData);
  }, [kpiCompareTempData]);


  const exportKpiToCSV = () => {
    if (!kpiData?.length) return;

    const csvHeader = [
      ['Business Performance Metrics'], // title
      [`Date Range: ${format(selectedRange?.from, 'yyyy-MM-dd') ?? ''} to ${format(selectedRange?.to, 'yyyy-MM-dd') ?? ''}`],
      compareEnabled ? [`Compare Period: ${compareValueMap?.[compareOption] ?? 'N/A'} ${compareOption === 'selectYear' ? `(${selectedYear.toString()})` : ''} ${compareOption === 'custom' ? `(${format(customRange?.from, 'yyyy-MM-dd') ?? ''} to ${format(customRange?.to, 'yyyy-MM-dd') ?? ''})` : ''}`] : [],
      [],
      ['Business Unit', 'LOB', 'Value', 'Comparison Value', 'Difference (%)']
    ];

    const rows = kpiData.map((kpi) => {
      const oldKpi = compareEnabled
        ? kpiCompareData.find((old) => old.title === kpi.title)
        : kpiOldData.find((old) => old.title === kpi.title);

      const businessUnit = businessName;
      const lob = kpi.title;

      const oldValue = Number(oldKpi?.value ?? 0);
      const currentValue = Number(kpi.value ?? 0);

      const changePercent = oldValue > 0 ? (((currentValue - oldValue) / oldValue) * 100).toFixed(2) : 'N/A';

      return [
        businessUnit,
        lob,
        currentValue.toLocaleString(),
        oldValue.toLocaleString(),
        changePercent !== 'N/A' ? `${changePercent}%` : changePercent
      ];
    });

    const csvContent = [...csvHeader, ...rows]
      .map(row => row.map(field => `"${String(field).replace(/"/g, '""')}"`).join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'business_performance_metrics.csv';
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Export Successful",
      description: "Business Performance Metrics has been exported as CSV.",
      duration: 4000,
    });
  };


  const exportKpiToXLSX = async () => {
  if (!kpiData?.length) return;

  const workbook = new ExcelJS.Workbook();
  const worksheet = workbook.addWorksheet('Business Metrics');

  // Title
  worksheet.addRow(['Business Performance Metrics']);
  worksheet.addRow([
    `Date Range: ${selectedRange?.from ? format(selectedRange.from, 'yyyy-MM-dd') : ''} to ${selectedRange?.to ? format(selectedRange.to, 'yyyy-MM-dd') : ''}`
  ]);

  if (compareEnabled) {
    const compareLabel =
      compareOption === 'selectYear'
        ? `(${selectedYear.toString()})`
        : compareOption === 'custom'
          ? `(${format(customRange?.from, 'yyyy-MM-dd')} to ${format(customRange?.to, 'yyyy-MM-dd')})`
          : '';

    worksheet.addRow([
      `Compare Period: ${compareValueMap?.[compareOption] ?? 'N/A'} ${compareLabel}`
    ]);
  }

  worksheet.addRow([]); // empty row for spacing

  // Header Row
  const headerRow = worksheet.addRow(['Business Unit', 'LOB', 'Value', 'Comparison Value', 'Difference (%)']);
  headerRow.font = { bold: true };

  // Data Rows
  kpiData.forEach((kpi) => {
    const oldKpi = compareEnabled
      ? kpiCompareData.find((old) => old.title === kpi.title)
      : kpiOldData.find((old) => old.title === kpi.title);

    const businessUnit = businessName;
    const lob = kpi.title;

    const oldValue = Number(oldKpi?.value ?? 0);
    const currentValue = Number(kpi.value ?? 0);

    // Compute percent difference as decimal for Excel formatting
    const changePercent = oldValue > 0 ? (currentValue - oldValue) / oldValue : null;

    worksheet.addRow([
      businessUnit,
      lob,
      currentValue,
      oldValue,
      changePercent !== null ? changePercent : null // push raw decimal
    ]);
  });

  // Format number columns
  worksheet.getColumn(3).numFmt = '#,##0.00';   // Value
  worksheet.getColumn(4).numFmt = '#,##0.00';   // Comparison Value
  worksheet.getColumn(5).numFmt = '0.00%';      // Difference as percent

  // Adjust column widths
  worksheet.columns.forEach(col => {
    col.width = 20;
  });

  // Export as XLSX without file-saver
  const buffer = await workbook.xlsx.writeBuffer();
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'Business_Performance_Metrics.xlsx';
  link.click();
  URL.revokeObjectURL(url);

  toast({
    title: "Export Successful",
    description: "Business Performance Metrics has been exported as XLSX.",
    duration: 4000,
  });
};



  const onDateDefaultChange = () => {
    setCustomRange({
      from: null,
      to: null,
    })
  };

  const currentYear = new Date().getFullYear();
  const yearOptions = Array.from({ length: 5 }, (_, i) => currentYear - i - 1); // Past 5 years

  if (loading) {
    return (
      <Card> {/* Loading state might also benefit from Card structure */}
        <CardHeader>
          <CardTitle>Business Performance Metrics</CardTitle>
          {/* Optional: Skeleton for date */}
        </CardHeader>
        <div className="space-y-4">
          <LoadingSkeleton />
        </div>
      </Card>
    );
  }

  return (
    <Card> {/* Root element is now Card */}
      <CardHeader>
        <div className="flex items-center"> {/* To keep title and date on same line if desired */}
          <CardTitle className="text-xl font-semibold"> {/* Applied original h2 classes for consistency, though CardTitle has its own */}
            Business Performance Metrics
          </CardTitle>
          <div className="text-sm px-3 py-1 ml-auto">
            {children}
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Checkbox
                checked={compareEnabled}
                disabled
                onCheckedChange={(checked) => setCompareEnabled(!!checked)}
              />
              Compare
            </label>

            {compareEnabled && (
              <div className="flex flex-wrap items-center gap-3">
                <Select value={compareOption} onValueChange={(val) => setCompareOption(val as typeof compareOption)}>
                  <SelectTrigger className="w-[220px]">
                    <SelectValue placeholder="Select comparison" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="previous">Previous Period</SelectItem>
                    <SelectItem value="sameYear">Same Period Previous Year</SelectItem>
                    <SelectItem value="selectYear">Same Period: Select Year</SelectItem>
                    <SelectItem value="custom">Custom Range</SelectItem>
                  </SelectContent>
                </Select>

                {compareOption === 'selectYear' && (
                  <Select value={selectedYear.toString()} onValueChange={(val) => setSelectedYear(Number(val))}>
                    <SelectTrigger className="w-[140px]">
                      <SelectValue placeholder="Select year" />
                    </SelectTrigger>
                    <SelectContent>
                      {yearOptions.map((year) => (
                        <SelectItem key={year} value={year.toString()}>
                          {year}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                {compareOption === 'custom' && (
                  <div>
                    <DateRangePicker
                      date={customRange}
                      onDateChange={setCustomRange}
                      onDateDefaultChange={onDateDefaultChange}
                      className=""
                      comparePicker
                      maxRangeToDate={new Date(format(selectedRange?.from, 'yyyy-MM-dd'))}
                      weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                    />
                  </div>
                )}
              </div>
            )}



            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger asChild>
                  <div>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button disabled={!(kpiData?.length > 0)} variant="outline" size="icon">
                          <Download className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[200px] p-0" align="end">
                        <div className="p-2">
                          <p className="text-sm font-medium">Select Export Type</p>
                        </div>
                        <Separator />
                        <div className="p-1">
                          <button
                            onClick={exportKpiToCSV}
                            disabled={!(kpiData?.length > 0)}
                            className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                          >
                            CSV
                          </button>
                          <button
                            onClick={exportKpiToXLSX}
                            disabled={!(kpiData?.length > 0)}
                            className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                          >
                            XLSX
                          </button>
                        </div>
                      </PopoverContent>
                    </Popover>
                  </div>
                </TooltipTrigger>
              </UITooltip>
            </TooltipProvider>
          </div>
        </div>
      </CardHeader>
      <CardContent style={{ overflow: 'auto' }}> {/* Added padding for CardContent */}
        {!loading && kpiData.length === 0 ? (
          <NoDataFullPage message="No KPIs available." />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 mb-4">
            {kpiData.map((kpi, index) => {
              const oldKpi = compareEnabled ? kpiCompareData.find(old => old.title === kpi.title) : kpiOldData.find(old => old.title === kpi.title);
              const parameterdata = parametersList.find(old => old.name === kpi.title);
              const kpiDirection = parameterdata?.direction ?? 'Low is Good';
              const oldValue = oldKpi?.value ?? 0;
              const currentValue = kpi.value;

              let change = 0;
              let changeText = '';
              let invertChange = false;

              if (oldValue > 0) {
                change = ((currentValue - oldValue) / oldValue) * 100;
                changeText = `${change > 0 ? 'increase' : 'decrease'}`;
                invertChange = change > 0; // Red if increase
              } else if (currentValue > 0) {
                changeText = "↑ New";
                invertChange = true;
                change = 100;
              }

              return (
                <KPIMetricsCard
                  key={index}
                  title={kpi.title}
                  value={formatWithPrecision(currentValue, defaultDecimalPrecisions)}
                  kpiDirection={kpiDirection}
                  compareEnabled={compareEnabled}
                  compareOption={compareOption}
                  selectedYear={selectedYear}
                  subtitle={`Total ${kpi.title}`}
                  oldValue={formatWithPrecision(oldValue, defaultDecimalPrecisions)}
                  changeValue={loadingKpi ? 0 : formatWithPrecision(change, defaultDecimalPrecisions)}
                  changeText={loadingKpi ? 'Loading..' : changeText}
                  defaultDecimalPrecisions={defaultDecimalPrecisions}
                  invertChange// true = red ↑, false = green ↓
                />
              );
            })}

          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default KPIMetrics;
