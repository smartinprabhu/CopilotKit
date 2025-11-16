import React, { useState, useMemo, useEffect } from "react";
import {
  Area,
  AreaChart,
  Bar,
  ComposedChart,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ExcelJS from 'exceljs';
import { toZonedTime, format } from 'date-fns-tz';
import { startOfWeek, differenceInCalendarWeeks, getISOWeek, endOfWeek, subWeeks } from 'date-fns';
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import axios from 'axios';
import NoDataFullPage from "@/components/noDataPage";
import { Button } from "@/components/ui/button";
import { ChevronDown, Download, Settings, HelpCircle, Info, BarChart2 } from "lucide-react";
import AreaChartOutlinedIcon from '@mui/icons-material/AreaChartOutlined';
import SettingsIcon from '@mui/icons-material/Settings';
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { LightbulbIcon, TrendingUpIcon, AlertTriangleIcon, InfoIcon } from "lucide-react";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CircularProgress from "@mui/material/CircularProgress";
import AuthService from "@/auth/utils/authService";
import LoadingSkeleton from "./LoadingSkeleton";
import AppConfig from '../auth/config.js';
import AskAIButton from './AskAIButton';

interface ForecastTabProps {
  aggregationType: string;
  modelType: string;
  forecastPeriod: number;
  data: {
    dz_df: any[];
    future_df: any[];
  };
  insights: any;
  metricKeys?: any[];
  algorithmModels?: any[];
  metricData?: any[];
  loading?: boolean;
  lobData?: any[];
  businessData?: any[];
  setTriger?: (type: string) => void;
  children?: React.ReactNode;
  setAggregationType?: (type: string) => void;
  plotAggregationType?: string;
  setPlotAggregationType?: (type: string) => void;
  setIsDrawerOpen?: (type: boolen) => void;
  defaultDecimalPrecisions?: number;
  selectedRange?: { from?: Date; to?: Date } | undefined;
  weekStartsOn?: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  setSelectedMetrics?: (type: string) => void;
  selectedParentMetrics: any[];
}

const getIconForInsightType = (type: string) => {
  switch (type) {
    case "trend":
      return <TrendingUpIcon className="h-5 w-5 text-blue-500" />;
    case "anomaly":
      return <AlertTriangleIcon className="h-5 w-5 text-amber-500" />;
    case "improvement":
      return <TrendingUpIcon className="h-5 w-5 text-green-500" />;
    case "correlation":
      return <InfoIcon className="h-5 w-5 text-purple-500" />;
    case "pattern":
      return <LightbulbIcon className="h-5 w-5 text-orange-500" />;
    default:
      return <LightbulbIcon className="h-5 w-5 text-blue-500" />;
  }
};

const WEBAPPAPIURL =  `${AppConfig.API_URL}/api/v2/`;

function pivotActualData(raw: any[]): any[] {
  const grouped: Record<string, any> = {};

  raw.forEach((item) => {
    const date = item.date;
    const value = item.value;

    // Use lob_id if present, else fallback to parameter_id
    const metric = item.lob_id?.[1]?.trim() || item.parameter_id?.[1]?.trim();
    if (!metric) return;

    if (!grouped[date]) {
      grouped[date] = { Week: date };
    }

    grouped[date][metric] = (grouped[date][metric] || 0) + value;
  });

  return Object.values(grouped);
}

function pivotForecastData(raw: any[]): any[] {
  const grouped: Record<string, any> = {};

  raw.forEach((item) => {
    const date = item.date;
    const value = item.value;
     const lowerBound = item.lower_bound;
      const upperBound = item.upper_bound;

    // Use lob_id if present, else fallback to parameter_id
    const metric = item.lob_id?.[1]?.trim() || item.parameter_id?.[1]?.trim();
    if (!metric) return;

    if (!grouped[date]) {
      grouped[date] = { Week: date };
    }

    grouped[date][metric] = (grouped[date][metric] || 0) + value;

      // Add bounds (if present)
    if (typeof lowerBound === "number") {
      grouped[date][`${metric}_lower`] = lowerBound;
    }

    if (typeof upperBound === "number") {
      grouped[date][`${metric}_upper`] = upperBound;
    }
  });

  return Object.values(grouped);
}

function transformGroupData(data: any[], aggregationType: string): any[] {
  if (aggregationType === 'Monthly') {
    const monthlyData: Record<string, any> = {};

    data.forEach(item => {
      const date = new Date(item.Week);
      const year = date.getFullYear();
      const month = date.getMonth(); // 0-11
      const key = `${year}-${month}`; // Unique per month

      if (!monthlyData[key]) {
        monthlyData[key] = {
          Week: new Date(year, month, 1).toISOString().split('T')[0], // e.g., '2025-05-01'
          isForecast: item.isForecast, // Keep forecast flag (optional: resolve if mixed)
        };
      }

      Object.entries(item).forEach(([k, v]) => {
        if (k === 'Week' || k === 'isForecast') return;
        monthlyData[key][k] = (monthlyData[key][k] || 0) + (typeof v === 'number' ? v : 0);
      });
    });

    // Optional: sort by real date
    return Object.values(monthlyData).sort(
      (a, b) => new Date(a.Week).getTime() - new Date(b.Week).getTime()
    );
  }

  return data;
}


const transformData = (data: any[], metricsMaps: any[]): any[] => {
  // Flatten keys from metricsMapped into one array of metric base names
 const metricKeys = Array.from(
    new Set(
     metricsMaps.map((key) =>
        key.replace(/_lower$|_upper$/, '')
      )
    )
  );

  return data.map((item) => {
    const result: any = {
      Week: item["Week"],
      isForecast: false,
    };

    metricKeys.forEach((metric) => {
      result[metric] = item[metric] || 0;
      result[`${metric}_lower`] = item[`${metric}_lower`] || 0;
      result[`${metric}_upper`] = item[`${metric}_upper`] || 0;
    });

    return result;
  });
};

function getISOWeekNumber(date: Date): number {
  const tempDate = new Date(date.getTime());
  tempDate.setHours(0, 0, 0, 0);
  tempDate.setDate(tempDate.getDate() + 4 - (tempDate.getDay() || 7));
  const yearStart = new Date(tempDate.getFullYear(), 0, 1);
  const weekNo = Math.ceil(
    ((tempDate.getTime() - yearStart.getTime()) / 86400000 + 1) / 7
  );
  return weekNo;
}

export const ForecastTab = ({
  aggregationType = "Daily",
  modelType,
  forecastPeriod,
  data,
  insights,
  loading = false,
  metricKeys,
  selectedRange,
  lobData,
  metricData,
  weekStartsOn = 1 as 0 | 1 | 2 | 3 | 4 | 5 | 6,
  children,
  algorithmModels,
  businessData,
  setIsDrawerOpen,
  setTriger,
  defaultDecimalPrecisions,
  setSelectedMetrics,
  selectedParentMetrics,
  plotAggregationType: propPlotAggregationType,
  setPlotAggregationType: propSetPlotAggregationType,
}: ForecastTabProps) => {

  const timeZone = 'Asia/Kolkata';
  
  function getLocalDateString(utcDateStr: string, formatStr = 'yyyy-MM-dd', tz = timeZone): string {
    const fixedUtcString = utcDateStr.replace(' ', 'T') + 'Z';
    const zonedDate = toZonedTime(new Date(fixedUtcString), tz);
    return format(zonedDate, formatStr, { timeZone: tz });
  }
  

  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` :  `${Math.round(value / 1_000_000)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(1)}K` : `${Math.round(value / 1_000)}K`;
    return value.toLocaleString();
  };

  const [metrics, setMetrics] = useState([]);
  const [metricsMapped, setMetricsMapped] = useState([]);
  const [isMetricForecastOpen, setMetricForecastOpen] = useState(false);
   const [updateLoading, setUpdateLoading] = useState(false);
  const [currentMetricForecast, setCurrentMetricForeCast] = useState({});
  const [selectedModel, setSelectedModel] = useState(0);

 const selectedMetrics = selectedParentMetrics;

     type DefaultAlgorithm = { id: number; name: string } | null;

          interface LobItem {
            id: number;
            preferred_algorithm?: [number, string];
          }

          interface SelectedItem {
            lob_id?: number;
            parameter_id?: number;
            source: string;
          }

        function getDefaultAlgorithmByLob(
            lobList: LobItem[],
            selected: SelectedItem
          ): DefaultAlgorithm {
            // Case 1: Match by lob_id
            if (selected?.lob_id && selected?.source === 'lob') {
              const match = lobList.find((lob) => lob.id === selected.lob_id);

              if (
                match &&
                Array.isArray(match.preferred_algorithm) &&
                match.preferred_algorithm.length === 2
              ) {
                return {
                  id: match.preferred_algorithm[0],
                  name: match.preferred_algorithm[1],
                };
              }
            }

            // Case 2: Fallback to parameter_id and use businessData
            if (
              selected?.parameter_id &&
              selected?.source === 'parameter' &&
              businessData &&
              Array.isArray(businessData.preferred_algorithm) &&
              businessData.preferred_algorithm.length === 2
            ) {
              return {
                id: businessData.preferred_algorithm[0],
                name: businessData.preferred_algorithm[1],
              };
            }

            // Case 3: No match found
            return null;
        }

      useEffect(() => {
           setSelectedModel(getDefaultAlgorithmByLob(lobData, currentMetricForecast)?.id || 0) 
       }, [lobData, businessData, currentMetricForecast]);

   const [localPlotAggregationType, setLocalPlotAggregationType] = useState('Weekly');
    const plotAggregationType = propPlotAggregationType ?? localPlotAggregationType;
    const setPlotAggregationType = propSetPlotAggregationType ?? setLocalPlotAggregationType;
    const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);

   const [chartType, setChartType] = useState('area');

   const numberOfWeeks =
    selectedRange && selectedRange.from && selectedRange.to
      ? differenceInCalendarWeeks(selectedRange.to, selectedRange.from) + 1 // +1 to include the start week
      : 0;


  const { toast } = useToast();

   const defaultColorPalette = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C",
    "#98DF8A", "#D62728", "#FF9896", "#9467BD", "#C5B0D5",
    "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", "#7F7F7F",
    "#C7C7C7", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5",
    "#393B79", "#5254A3", "#6B6ECF", "#9C9EDE", "#637939",
    "#8CA252", "#B5CF6B", "#CEDB9C", "#8C6D31", "#BD9E39",
    "#E7BA52", "#E7CB94", "#843C39", "#AD494A", "#D6616B",
    "#E7969C", "#7B4173", "#A55194", "#CE6DBD", "#DE9ED6"
  ];

  const getColorByIndex = (index: number) =>
  defaultColorPalette[index];

   useMemo(() => {
    const newData = metricData.map((cat, index) => ({
        id: cat.title,
        name: cat.title,
        color: getColorByIndex(index),
        source: cat.source,
        parameter_id: cat?.parameterId || false,
        lob_id: cat?.lobId || false,
      }));
      setMetrics(newData)
        const baseNames = newData.map((m) => m.name);

        const allFields: string[] = [];

        baseNames.forEach((name) => {
          allFields.push(name);
          allFields.push(`${name}_lower`);
          allFields.push(`${name}_upper`);
        });
      setMetricsMapped(allFields)
      if(!(selectedMetrics && selectedMetrics.length > 0) || (selectedMetrics && selectedMetrics.length && selectedMetrics.filter((m) => m !== '').length === 0)) {
        setSelectedMetrics([newData?.length && newData[0].name ? newData[0].name : ''])
      }
  }, [metricData]);



  const dzData = useMemo(() => {
    return data.dz_df && metricsMapped?.length
      ? transformData(pivotActualData(data.dz_df), metricsMapped).map((d) => ({ ...d, isForecast: false }))
      : [];
  }, [data, metricsMapped]);


const futureData = useMemo(() => {
  if (!data.future_df) return [];
  if (!metricsMapped.length) return [];

  const rawData = pivotForecastData(data.future_df);
  const transformed = transformData(rawData, metricsMapped);

  return transformed.map((d) => ({ ...d, isForecast: true }))
}, [data, metricsMapped]);


const groupedModels = useMemo(() => {
  if (!algorithmModels) return [];

  return algorithmModels.reduce((acc: Record<string, any[]>, model: any) => {
    const category = model.category || "Uncategorized";
    acc[category] = acc[category] || [];
    acc[category].push(model);
    return acc;
  }, {});
}, [algorithmModels]);


const combinedDataNoGroup = useMemo(() => {
  const allWeeks = new Set([
    ...dzData.map((d) => getLocalDateString(d.Week)),
    ...futureData.map((d) => getLocalDateString(d.Week)),
  ]);

  const sortedWeeks = Array.from(allWeeks).sort(
    (a, b) => new Date(a).getTime() - new Date(b).getTime()
  );

  const metricNames = metrics.map((m) => m.name);
  const today = new Date();

      // Get current day of week (0 = Sunday, ..., 6 = Saturday)
    const dayOfWeek = today.getDay();

    // Days since last week’s end (Sunday)
    const daysSinceLastWeekEnd = dayOfWeek + 1;

    // Get last week's Sunday (end)
    const endOfLastWeek = new Date(today);
    endOfLastWeek.setDate(today.getDate() - daysSinceLastWeekEnd);

    // Get last week's Monday (start = 6 days before Sunday)
    const startOfLastWeek = new Date(endOfLastWeek);
    startOfLastWeek.setDate(endOfLastWeek.getDate() - 0);

  return sortedWeeks.map((week) => {
    const actualRow = dzData.find((d) => getLocalDateString(d.Week) === week);
    const forecastRow = futureData.find((d) => getLocalDateString(d.Week) === week);
    const weekDate = new Date(week);
    const stripTime = (date) => new Date(date.setHours(0, 0, 0, 0));
    const isFutureForecast = forecastRow && stripTime(weekDate) >= stripTime(endOfLastWeek) ? true : false;

    const row: any = {
      Week: week,
      isForecast: isFutureForecast,
    };

    metricNames.forEach((metric) => {
      row[`${metric}_actual`] =
        actualRow && actualRow[metric] !== undefined ? actualRow[metric] : null;

      row[`${metric}_forecast`] =
         forecastRow && forecastRow[metric] !== undefined
          ? forecastRow[metric]
          : null;

      row[`${metric}_forecast_upper`] =
        isFutureForecast && forecastRow && forecastRow[`${metric}_upper`] !== undefined
          ? forecastRow[`${metric}_upper`]
          : null;

      row[`${metric}_forecast_lower`] =
        isFutureForecast && forecastRow && forecastRow[`${metric}_lower`] !== undefined
          ? forecastRow[`${metric}_lower`]
          : null;
    });

    return row;
  });
}, [metrics, dzData, futureData]);

 const combinedData = useMemo(() => {
    return transformGroupData(combinedDataNoGroup, plotAggregationType);
  }, [combinedDataNoGroup, plotAggregationType]);




  const handleMetricToggle = (metricId: string) => {
    setSelectedMetrics((current) =>
      current.includes(metricId)
        ? current.filter((id) => id !== metricId)
        : [...current, metricId]
    );
  };

    const onMetricForecastChange = (metric: any) => {
      setCurrentMetricForeCast(metric);
      setMetricForecastOpen(true)
    };

const exportToCSV = () => {
  const headerRows = ["Week"];

  selectedMetrics.forEach((metricId) => {
    const metricName = metrics.find((m) => m.id === metricId)?.name || metricId;
    headerRows.push(
      `${metricName} Actual`,
      `${metricName} Forecast`,
      `${metricName} Forecast Lower`,
      `${metricName} Forecast Upper`,
      `${metricName} MAPE %` // 👈 Add MAPE column
    );
  });

  const csvHeader = headerRows.join(",");

  const csvRows = combinedData.map((item) => {
    const row = [getLocalDateString(item.Week, 'yyyy-MM-dd')];
    selectedMetrics.forEach((metricId) => {
      const actual = item[`${metricId}_actual`] ?? "";
      const forecast = item[`${metricId}_forecast`] ?? "";
      const lower = item[`${metricId}_forecast_lower`] ?? "";
      const upper = item[`${metricId}_forecast_upper`] ?? "";

      let mape = "";
      if (typeof actual === "number" && typeof forecast === "number" && actual !== 0) {
        mape = `${((Math.abs(actual - forecast) / actual) * 100).toFixed(0)} %`;
      }

      row.push(actual ? actual.toFixed(0) : actual, forecast ? forecast.toFixed(0) : forecast, lower ? lower.toFixed(0) : lower, upper ? upper.toFixed(0) : upper, mape);
    });
    return row.join(",");
  });

  const csvContent = [csvHeader, ...csvRows].join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute(
    "download",
    `Forecast_${aggregationType.toLowerCase()}_${new Date()
      .toISOString()
      .split("T")[0]}.csv`
  );
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  toast({
    title: "Export successful",
    description: `${numberOfWeeks} weeks history + ${forecastPeriod} weeks forecast (${aggregationType.toLowerCase()}) exported to CSV`,
  });
};

const exportToXLSX = async () => {
  if (!combinedData?.length) return;

  const workbook = new ExcelJS.Workbook();
  const worksheet = workbook.addWorksheet('Forecast');

  // Step 1: Build header row
  const headerRow = ['Week'];

  selectedMetrics.forEach((metricId) => {
    const metricName = metrics.find((m) => m.id === metricId)?.name || metricId;
    headerRow.push(
      `${metricName} Actual`,
      `${metricName} Forecast`,
      `${metricName} Forecast Lower`,
      `${metricName} Forecast Upper`,
      `${metricName} MAPE %`
    );
  });

  worksheet.addRow(headerRow).font = { bold: true };

  // Step 2: Populate data rows
  combinedData.forEach((item) => {
    const row = [];

    // Format date
    const date = new Date(getLocalDateString(item.Week, 'yyyy-MM-dd'));
    row.push(isNaN(date.getTime()) ? null : date);

    selectedMetrics.forEach((metricId) => {
      const actual = item[`${metricId}_actual`];
      const forecast = item[`${metricId}_forecast`];
      const lower = item[`${metricId}_forecast_lower`];
      const upper = item[`${metricId}_forecast_upper`];

      const mape =
        typeof actual === 'number' && typeof forecast === 'number' && actual !== 0
          ? Math.round(Math.abs(actual - forecast) / actual * 100)
          : null;

      row.push(
        typeof actual === 'number' ? actual : null,
        typeof forecast === 'number' ? forecast : null,
        typeof lower === 'number' ? lower : null,
        typeof upper === 'number' ? upper : null,
        mape
      );
    });

    worksheet.addRow(row);
  });

  // Step 3: Format columns
  worksheet.columns.forEach((col, index) => {
    if (index === 0) {
      col.numFmt = 'yyyy-mm-dd';
      col.width = 15;
    } else {
      col.numFmt = '0'; // integer format
      col.width = 20;
    }
  });

  // Step 4: Export
  const buffer = await workbook.xlsx.writeBuffer();
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `Forecast_${aggregationType.toLowerCase()}_${new Date().toISOString().split('T')[0]}.xlsx`;
  link.click();
  URL.revokeObjectURL(url);

  toast({
    title: 'Export successful',
    description: `${numberOfWeeks} weeks history + ${forecastPeriod} weeks forecast (${aggregationType.toLowerCase()}) exported to Excel`,
  });
};




  const tickInterval = useMemo(() => {
    const len = combinedData.length;
    if (len <= 20) return 0; // show all
    if (len <= 50) return 2;
    if (len <= 100) return 5;
    return Math.floor(len / 20); // roughly 20 ticks max
  }, [combinedData]);

    async function updateForm(id: number, model: string, values: Record<string, unknown>) {
      const formData = new FormData();
    
      formData.append("ids", `[${id}]`);
      formData.append("model", model);
      formData.append("values", JSON.stringify(values)); // nested object as JSON string
    
      const url = `${WEBAPPAPIURL}write`;
      setUpdateLoading(true);
      try{
        const response = await axios.put(url, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${AuthService.getAccessToken()}`
          },
        });
        setUpdateLoading(false);
        setMetricForecastOpen(false);
        setTriger(Math.random())
                              toast({
                                title: "Settings Applied",
                                description: "Forecast settings have been updated successfully.",
                              });
        return response.data;
      } catch(e){
        setUpdateLoading(false);
        setMetricForecastOpen(false)
      }
    }

   const onSaveModelChanges = async () => {
    if(!selectedModel) return;
      if (currentMetricForecast && currentMetricForecast?.lob_id) { 
        const response = await updateForm(Number(currentMetricForecast?.lob_id), 'line_business_lob', {
          preferred_algorithm: selectedModel,
        });
      } else if(currentMetricForecast && currentMetricForecast?.parameter_id && businessData && businessData?.id){
          const response = await updateForm(Number(businessData?.id), 'business.unit', {
          preferred_algorithm: selectedModel,
        });
      }
  };

  return (
    <TooltipProvider>
      <Card className="">
        <CardContent className="pt-2 mt-[-10px]">
          <div className="flex flex-col space-y-4 sm:flex-row sm:justify-between sm:space-y-0 mb-6">
            <div>
              <div className="flex items-center gap-2 ">
                <h3 className="text-xl font-semibold">Trends & Forecast</h3>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6">
                      <HelpCircle className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="w-80 p-4">
                    <p className="font-medium mb-201">About this chart</p>
                    <p className="text-sm text-muted-foreground">
                      This chart shows actual and forecasted data for various metrics.
                    </p>
                  </TooltipContent>
                </UITooltip>
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                {numberOfWeeks} weeks history + {forecastPeriod} weeks forecast
              </p>
            </div>
           
            <div className="flex flex-col sm:flex-row gap-2 flex items-center">
              <div className="text-sm">
                {children}
              </div>
      
                     <div className="flex items-center gap-2 border rounded-md p-1  h-10">
                            <Button
                              variant={plotAggregationType === "Weekly" ? "default" : "ghost"}
                              size="sm"
                              onClick={() => setPlotAggregationType("Weekly")}
                              disabled={!(data?.dz_df?.length > 0)}
                              className="h-7 px-3"
                            >
                              Weekly
                            </Button>
                            <Button
                              variant={plotAggregationType === "Monthly" ? "default" : "ghost"}
                              size="sm"
                               disabled={!(data?.dz_df?.length > 0)}
                              onClick={() => setPlotAggregationType("Monthly")}
                              className="h-7 px-3"
                            >
                              Month
                            </Button>
                        </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="w-[180px] justify-between">
                      {selectedMetrics.length === 1
                        ? metrics.find((m) => m.id === selectedMetrics[0])?.name
                        : `${selectedMetrics.length} metrics selected`}
                      <ChevronDown className="ml-2 h-4 w-4 opacity-50" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-[300px]" style={{ maxHeight: "450px", overflowY: "auto" }}>
                    <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                      <DropdownMenuCheckboxItem
                        key="select-all"
                        checked={selectedMetrics.length === metrics.length}
                        disabled={!(data?.dz_df?.length > 0)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedMetrics(metrics.map((m) => m.id));
                          } else {
                            setSelectedMetrics([]);
                          }
                        }}
                        onSelect={(e) => e.preventDefault()}
                      >
                        Select All
                      </DropdownMenuCheckboxItem>
                    <DropdownMenuSeparator />
                     {metrics.map((metric) => (
                      <DropdownMenuCheckboxItem
                        key={metric.id}
                        checked={selectedMetrics.includes(metric.id)}
                        onCheckedChange={() => handleMetricToggle(metric.id)}
                        onSelect={(e) => e.preventDefault()}
                        className="group flex items-center justify-between" 
                      >
                        <span>{metric.name}</span>
                        <UITooltip>
                          <TooltipTrigger asChild>
                            <div
                              className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted cursor-pointer"
                              onClick={(e) => {
                                e.stopPropagation(); // Prevent dropdown from closing
                                onMetricForecastChange(metric)
                              }}
                            >
                              <Settings fontSize="small" className="w-4 h-4 text-muted-foreground hover:text-primary" />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Settings</p>
                          </TooltipContent>
                        </UITooltip>
                      </DropdownMenuCheckboxItem>
                    ))}
                    <DropdownMenuSeparator />

                  </DropdownMenuContent>
                </DropdownMenu>

                {/* Forecast Settings Drawer */}
             
                  <Sheet open={isMetricForecastOpen} onOpenChange={setMetricForecastOpen}>
                    <SheetContent
                      side="right"
                      className="w-[1000px] h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]"
                    >
                      <SheetHeader>
                        <SheetTitle>Forecast Settings</SheetTitle>
                      </SheetHeader>
                      {/* You may want to add loading state/logic here if needed */}
                      <div className="space-y-4 mt-3">
                        {/* Forecast Period */}
                         <div className="p-2">
                                        <div className="space-y-4">
                                          <div>
                                              <h4 className="text-sm font-semibold text-muted-foreground">Metric</h4>
                                              <p className="text-base">{currentMetricForecast?.name || '—'}</p>
                                            </div>
                                            <div>
                                              <label className="block font-medium mb-2">Select a Forecasting Model</label>
                                            </div>  
                                          </div>
                              <div className="space-y-2 pl-3">
                              
                                {Object.entries(groupedModels).map(([category, models]) => (
                                  <div key={category} className="mb-3">
                                    <div className="text-sm font-semibold opacity-70 mb-1">{category}</div>
                                    <div className="space-y-1 ml-2">
                                      {models.map((model: any) => {
                                        const isDefault =
                                          getDefaultAlgorithmByLob(lobData, currentMetricForecast)?.id === Number(model.id);

                                        return (
                                          <label
                                            key={model.id}
                                            className="flex items-start space-x-2 p-1 rounded-md hover:bg-accent transition-colors"
                                          >
                                            <input
                                              type="radio"
                                              value={model.id}
                                              checked={Number(selectedModel) === Number(model.id)}
                                              onChange={(e) => setSelectedModel(Number(e.target.value))}
                                              className="mt-1 dark:bg-gray-700 dark:text-white"
                                            />

                                            <div className="flex flex-col space-y-0.5">
                                              <div className="flex items-center space-x-2">
                                                <span className="font-medium">{model.name}</span>
                                                {isDefault && (
                                                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                                                    Default
                                                  </span>
                                                )}
                                              </div>
                                            </div>

                                            <TooltipProvider>
                                              <UITooltip>
                                                <TooltipTrigger asChild>
                                                  <div className="ml-auto p-1 rounded hover:bg-muted transition-opacity cursor-pointer">
                                                    <Info className="w-4 h-4 text-muted-foreground hover:text-primary" />
                                                  </div>
                                                </TooltipTrigger>
                                                <TooltipContent side="top" className="text-sm max-w-[300px]">
                                                  <p>{model.description || "No description available."}</p>
                                                </TooltipContent>
                                              </UITooltip>
                                            </TooltipProvider>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                        {/* Apply Changes Button */}
                        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                          <Button
                            size="sm"
                            className="w-full"
                            disabled={updateLoading}
                            onClick={() => onSaveModelChanges()}
                          >
                            {updateLoading ? 'Saving...' : 'Save Changes' }
                          </Button>
                        </div>
                      </div>
                    </SheetContent>
                  </Sheet>
           
                <AskAIButton mainData={data?.dz_df || []} aiParams={{ graph: futureData?.length ? futureData.slice(-50) : [] }} parametersList={[]} />

            <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Button onClick={() => setIsDrawerOpen(true)} disabled={!(data?.dz_df?.length > 0)} variant="outline" size="icon">
                            <SettingsIcon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                      </Button>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Forecast Seetings</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>

              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button disabled={!(data?.dz_df?.length > 0)} variant="outline" size="icon">
                            <AreaChartOutlinedIcon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[200px] p-0" align="end">
                          <div className="p-2">
                            <p className="text-sm font-medium">Select Chart</p>
                          </div>
                          <Separator />
                          <div className="p-1">
                            <button
                              onClick={() => setChartType('area')}
                              disabled={chartType === 'area' || !(data?.dz_df?.length > 0)}
                              className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                            >
                              Area Chart
                            </button>
                            <button
                              onClick={() => setChartType('line')}
                              disabled={chartType === 'line' || !(data?.dz_df?.length > 0)}
                              className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                            >
                              Line Chart
                            </button>
                             <button
                              onClick={() => setChartType('bar')}
                              disabled={chartType === 'bar' || !(data?.dz_df?.length > 0)}
                              className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                            >
                              Bar Chart
                            </button>
                          </div>
                        </PopoverContent>
                      </Popover>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Select Chart</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
              
            

                           <TooltipProvider>
                                  <UITooltip>
                                    <TooltipTrigger asChild>
                                      <div>
                                        <Popover>
                                          <PopoverTrigger asChild>
                                            <Button disabled={!(data?.dz_df?.length > 0)} variant="outline" size="icon">
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
                                                onClick={exportToCSV}
                                                disabled={!(data?.dz_df?.length > 0)}
                                                className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                              >
                                                CSV
                                              </button>
                                              <button
                                                onClick={exportToXLSX}
                                                disabled={!(data?.dz_df?.length > 0)}
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

          <div className="w-full h-[400px]  mb-6">
            {loading ? (
              <div className="space-y-4 h-full flex flex-col justify-center items-center">
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="h-4 bg-gray-300 rounded w-3/4 animate-pulse"
                  ></div>
                ))}
                <div className="flex justify-center">
                  <CircularProgress />
                </div>
              </div>
            ) : (
             <>
              {!loading && (!data?.dz_df?.length) && (
                 <NoDataFullPage message="No data found." />
              )}
               {(data?.dz_df?.length > 0) && (
                <>
                {chartType === "area" && (
                   <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={combinedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                    <defs>
                      {metrics.map((metric) => (
                        <React.Fragment key={`gradient-${metric.id}`}>
                          <linearGradient id={`color${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop
                              offset="5%"
                              stopColor={metric.color}
                              stopOpacity={0.8}
                            />
                            <stop
                              offset="95%"
                              stopColor={metric.color}
                              stopOpacity={0.1}
                            />
                          </linearGradient>
                          <linearGradient id={`color${metric.id}Forecast`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={metric.color} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={metric.color} stopOpacity={0.05} />
                          </linearGradient>
                        </React.Fragment>
                      ))}
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" vertical={false} className="opacity-30" />
                    <XAxis
                      dataKey="Week"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      interval={tickInterval}
                      angle={0}
                      textAnchor="end"
                      minTickGap={20}
                      tickFormatter={(tick) => {
                            const date = new Date(tick);
                            if (isNaN(date.getTime())) return tick;
                            let label = '';

                            if (plotAggregationType === 'Weekly') {
                              const weekNumber = getISOWeek(date);
                              label = `WK${weekNumber.toString().padStart(2, '0')}`;
                            } else {
                              const month = date.toLocaleString('default', { month: 'short' });
                              const year = date.getFullYear();
                              label = `${month}\n${year}`;
                            }
                            return label;
                      }}
                    />
                    <YAxis
                      yAxisId="left"
                      orientation="left"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload || payload.length === 0) return null;

                        const date = new Date(label);
                        const day = date.getDate().toString().padStart(2, "0");
                        const month = date.toLocaleString("default", { month: "short" });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        const dateStr = `${day} ${month} ${year} (W${week})`;

                        const isForecast = payload.some((p: any) => p.payload?.isForecast);

                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                              <span>{dateStr}</span>
                              <span
                                style={{
                                  backgroundColor: "#f1f5f9",
                                  color: "#334155",
                                  fontSize: 10,
                                  padding: "2px 6px",
                                  borderRadius: 12,
                                  fontWeight: 500,
                                }}
                              >
                                {(() => {
                                  const weekDate = new Date(payload?.[0]?.payload?.Week);
                                  const today = new Date();
                                  today.setHours(0, 0, 0, 0);
                                  const isFuture = weekDate > today;

                                  const hasActual = selectedMetrics.some((metricId) => {
                                    const actual = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                    return actual?.value != null;
                                  });

                                  const hasForecast = selectedMetrics.some((metricId) => {
                                    const forecast = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                    return forecast?.value != null;
                                  });

                                  if (isFuture && hasForecast) return "Forecast";
                                  if (!isFuture && hasActual && hasForecast) return "Actual & Forecast";
                                  if (!isFuture && hasForecast) return "Forecast";
                                  return "Actual";
                                })()}
                              </span>
                            </div>
                              {selectedMetrics.map((metricId) => {
                                const metricInfo = metrics.find((m) => m.id === metricId);

                                const actualPoint = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                const forecastPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                const upperPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_upper`);
                                const lowerPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_lower`);

                                const weekDate = new Date(payload?.[0]?.payload?.Week);
                                const today = new Date();
                                today.setHours(0, 0, 0, 0);

                                const isFuture = weekDate > today;

                                const hasActual = actualPoint?.value != null;
                                const hasForecast = forecastPoint?.value != null;

                                // Don't render if there's nothing to show
                                if (!hasActual && !hasForecast) return null;

                                return (
                                  <div key={metricId} style={{ marginBottom: 8 }}>
                                    {/* Metric name and legend color */}
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                      <div
                                        style={{
                                          width: 10,
                                          height: 10,
                                          backgroundColor: metricInfo?.color,
                                          borderRadius: 2,
                                        }}
                                      ></div>
                                      <span style={{ color: "var(--metric-label-color)", fontWeight: 700 }}>{metricInfo?.name}</span>
                                    </div>

                                    {/* Past week: show both if available */}
                                      {!isFuture && hasActual && (
                                        <div style={{ fontSize: 13, color: "var(--actual-color)", marginLeft: 16 }}>
                                          • Actual: {formatNumber(Number(actualPoint.value))}
                                        </div>
                                      )}

                                      {hasForecast && (
                                        <>
                                        <div style={{ fontSize: 13, color: "var(--forecast-color)", marginLeft: 16 }}>
                                          • Forecast: {formatNumber(Number(forecastPoint.value))}
                                        </div>
                                        {getDefaultAlgorithmByLob(lobData, metricInfo) && (
                                        <div style={{ fontSize: 13, color: "var(--forecast-color)", marginLeft: 16 }}>
                                           • ({getDefaultAlgorithmByLob(lobData, metricInfo).name})
                                        </div>
                                        )}
                                        </>
                                      )}

                                      {/* Bounds only in future */}
                                      {isFuture && hasForecast && upperPoint?.value != null && lowerPoint?.value != null && (
                                        <div
                                          style={{
                                            fontSize: 11,
                                            color: "var(--bounds-color)",
                                            marginLeft: 16,
                                            marginTop: 2,
                                          }}
                                        >
                                          Confidence bounds: {formatNumber(Number(lowerPoint.value))} - {formatNumber(Number(upperPoint.value))}
                                        </div>
                                      )}
                                  </div>
                                );
                              })}
                          </div>
                        );
                      }}
                    />
                    <Legend
                          payload={selectedMetrics.map((metricId) => {
                            const metricInfo = metrics.find((m) => m.id === metricId);
                            return {
                              id: metricId,
                              value: metricInfo?.name || metricId,
                              type: 'square',
                              color: metricInfo?.color,
                            };
                          })}
                          onMouseEnter={(e) => {
                            if (e && e.id) setHoveredLegend(e.id);
                          }}
                          onMouseLeave={() => setHoveredLegend(null)}
                          formatter={(value, entry) => {
                                                const parts = value.split(" (");
                                                return <span  style={{ fontWeight: hoveredLegend === entry.id ? 'bold' : 'normal', opacity: hoveredLegend && hoveredLegend !== entry.id ? 0.4 : 1 }} className="text-sm">{parts[0]}</span>;
                                              }}
                        />

                    {selectedMetrics.map((metricId) => {
                      const metricInfo = metrics.find((m) => m.id === metricId);
                      const source = metricInfo?.source || "Unknown Source";

                      return (
                        <React.Fragment key={metricId}>

                            {/* Forecast lower bound */}
                          <Area
                            key={`forecast-lower-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_lower`}
                            name=""
                            stroke="none"
                            fill={metricInfo?.color}
                            fillOpacity={hoveredLegend === null || hoveredLegend === metricId ? 0.5 : 0.05}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            connectNulls
                          />
                          <Area
                            key={`forecast-upper-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_upper`}
                            name=""
                            stroke="none"
                            fill={metricInfo?.color}
                            fillOpacity={hoveredLegend === null || hoveredLegend === metricId ? 0.25 : 0.02}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            connectNulls
                          />

                          {/* Historical Data */}
                          <Area
                            key={`actual-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_actual`}
                            name={metricInfo?.name}
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            fill={metricInfo?.color}
                            fillOpacity={
                              hoveredLegend === null || hoveredLegend === metricId
                                ? source === 'parameter' ? 0.4 : 1
                                : 0.05
                            }
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            dot={{ r: 3, strokeWidth: 1 }}
                            activeDot={{ r: 5 }}
                            connectNulls
                          />
                          {/* Forecast data */}
                          <Area
                            key={`forecast-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast`}
                            name=""
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            fill={metricInfo?.color}
                            fillOpacity={hoveredLegend === null || hoveredLegend === metricId ? 0.3 : 0.03}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            dot={{ r: 3, strokeWidth: 1 }}
                            activeDot={{ r: 5 }}
                            legendType="none"
                            connectNulls
                          />
                        
                        </React.Fragment>
                      );
                    })}
                    </AreaChart>
                  </ResponsiveContainer>
                )}
                 {chartType === "line" && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={combinedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                    <defs>
                      {metrics.map((metric) => (
                        <React.Fragment key={`gradient-${metric.id}`}>
                          <linearGradient id={`color${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop
                              offset="5%"
                              stopColor={metric.color}
                              stopOpacity={0.8}
                            />
                            <stop
                              offset="95%"
                              stopColor={metric.color}
                              stopOpacity={0.1}
                            />
                          </linearGradient>
                          <linearGradient id={`color${metric.id}Forecast`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={metric.color} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={metric.color} stopOpacity={0.05} />
                          </linearGradient>
                        </React.Fragment>
                      ))}
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" vertical={false} className="opacity-30" />
                    <XAxis
                      dataKey="Week"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      interval={tickInterval}
                      angle={0}
                      textAnchor="end"
                      minTickGap={20}
                      tickFormatter={(tick) => {
                            const date = new Date(tick);
                            if (isNaN(date.getTime())) return tick;

                            let label = '';

                            if (plotAggregationType === 'Weekly') {
                              const weekNumber = getISOWeek(date);
                              label = `WK${weekNumber.toString().padStart(2, '0')}`;
                            } else {
                              const month = date.toLocaleString('default', { month: 'short' });
                              const year = date.getFullYear();
                              label = `${month}\n${year}`;
                            }
                            return label;
                      }}
                    />
                    <YAxis
                      yAxisId="left"
                      orientation="left"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                  <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload || payload.length === 0) return null;

                        const date = new Date(label);
                        const day = date.getDate().toString().padStart(2, "0");
                        const month = date.toLocaleString("default", { month: "short" });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        const dateStr = `${day} ${month} ${year} (W${week})`;

                        const isForecast = payload.some((p: any) => p.payload?.isForecast);

                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                              <span>{dateStr}</span>
                              <span
                                style={{
                                  backgroundColor: "#f1f5f9",
                                  color: "#334155",
                                  fontSize: 10,
                                  padding: "2px 6px",
                                  borderRadius: 12,
                                  fontWeight: 500,
                                }}
                              >
                                {(() => {
                                  const weekDate = new Date(payload?.[0]?.payload?.Week);
                                  const today = new Date();
                                  today.setHours(0, 0, 0, 0);
                                  const isFuture = weekDate > today;

                                  const hasActual = selectedMetrics.some((metricId) => {
                                    const actual = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                    return actual?.value != null;
                                  });

                                  const hasForecast = selectedMetrics.some((metricId) => {
                                    const forecast = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                    return forecast?.value != null;
                                  });

                                  if (isFuture && hasForecast) return "Forecast";
                                  if (!isFuture && hasActual && hasForecast) return "Actual & Forecast";
                                  if (!isFuture && hasForecast) return "Forecast";
                                  return "Actual";
                                })()}
                              </span>
                            </div>
                              {selectedMetrics.map((metricId) => {
                                const metricInfo = metrics.find((m) => m.id === metricId);

                                const actualPoint = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                const forecastPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                const upperPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_upper`);
                                const lowerPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_lower`);

                                const weekDate = new Date(payload?.[0]?.payload?.Week);
                                const today = new Date();
                                today.setHours(0, 0, 0, 0);

                                const isFuture = weekDate > today;

                                const hasActual = actualPoint?.value != null;
                                const hasForecast = forecastPoint?.value != null;

                                // Don't render if there's nothing to show
                                if (!hasActual && !hasForecast) return null;

                                return (
                                  <div key={metricId} style={{ marginBottom: 8 }}>
                                    {/* Metric name and legend color */}
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                      <div
                                        style={{
                                          width: 10,
                                          height: 10,
                                          backgroundColor: metricInfo?.color,
                                          borderRadius: 2,
                                        }}
                                      ></div>
                                      <span style={{ color: "var(--metric-label-color)", fontWeight: 700 }}>{metricInfo?.name}</span>
                                    </div>

                                    {/* Past week: show both if available */}
                                      {!isFuture && hasActual && (
                                        <div style={{ fontSize: 13, color: "var(--actual-color)", marginLeft: 16 }}>
                                          • Actual: {formatNumber(Number(actualPoint.value))}
                                        </div>
                                      )}

                                      {hasForecast && (
                                        <div style={{ fontSize: 13, color: "var(--forecast-color)", marginLeft: 16 }}>
                                          • Forecast: {formatNumber(Number(forecastPoint.value))}
                                        </div>
                                      )}

                                      {/* Bounds only in future */}
                                      {isFuture && hasForecast && upperPoint?.value != null && lowerPoint?.value != null && (
                                        <div
                                          style={{
                                            fontSize: 11,
                                            color: "var(--bounds-color)",
                                            marginLeft: 16,
                                            marginTop: 2,
                                          }}
                                        >
                                          Confidence bounds: {formatNumber(Number(lowerPoint.value))} - {formatNumber(Number(upperPoint.value))}
                                        </div>
                                      )}
                                  </div>
                                );
                              })}
                          </div>
                        );
                      }}
                    />
                    <Legend
                      formatter={(value) => {
                        const parts = value.split(" (");
                        return <span className="text-sm">{parts[0]}</span>;
                      }}
                    />

                    {selectedMetrics.map((metricId) => {
                      const metricInfo = metrics.find((m) => m.id === metricId);
                      const source = metricInfo?.source || "Unknown Source";

                      return (
                        <React.Fragment key={metricId}>

                            {/* Forecast lower bound */}
                          <Line
                            key={`forecast-lower-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_lower`}
                            name=""
                            stroke="none"
                            fill={metricInfo?.color}
                            fillOpacity={0.5}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            connectNulls
                          />
                          <Line
                            key={`forecast-upper-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_upper`}
                            name=""
                            stroke="none"
                            fill={metricInfo?.color}
                            fillOpacity={0.25}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            connectNulls
                          />

                          {/* Historical Data */}
                          <Line
                            key={`actual-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_actual`}
                            name={metricInfo?.name}
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            fill={metricInfo?.color}
                            fillOpacity={source === 'parameter' ? 0.4 : 1}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            dot={{ r: 3, strokeWidth: 1 }}
                            activeDot={{ r: 5 }}
                            connectNulls
                          />
                          {/* Forecast data */}
                          <Line
                            key={`forecast-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast`}
                            name=""
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            fill={metricInfo?.color}
                            fillOpacity={0.3}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            dot={{ r: 3, strokeWidth: 1 }}
                            activeDot={{ r: 5 }}
                            legendType="none"
                            connectNulls
                          />
                        
                        </React.Fragment>
                      );
                    })}
                    </LineChart>
                 </ResponsiveContainer>
              )}
                {chartType === "bar" && (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={combinedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                    <defs>
                      {metrics.map((metric) => (
                        <React.Fragment key={`gradient-${metric.id}`}>
                          <linearGradient id={`color${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop
                              offset="5%"
                              stopColor={metric.color}
                              stopOpacity={0.8}
                            />
                            <stop
                              offset="95%"
                              stopColor={metric.color}
                              stopOpacity={0.1}
                            />
                          </linearGradient>
                          <linearGradient id={`color${metric.id}Forecast`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={metric.color} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={metric.color} stopOpacity={0.05} />
                          </linearGradient>
                        </React.Fragment>
                      ))}
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" vertical={false} className="opacity-30" />
                    <XAxis
                      dataKey="Week"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      interval={tickInterval}
                      angle={0}
                      textAnchor="end"
                      minTickGap={20}
                      tickFormatter={(tick) => {
                            const date = new Date(tick);
                            if (isNaN(date.getTime())) return tick;

                            let label = '';

                            if (plotAggregationType === 'Weekly') {
                              const weekNumber = getISOWeek(date);
                              label = `WK${weekNumber.toString().padStart(2, '0')}`;
                            } else {
                              const month = date.toLocaleString('default', { month: 'short' });
                              const year = date.getFullYear();
                              label = `${month}\n${year}`;
                            }
                            return label;
                      }}
                    />
                    <YAxis
                      yAxisId="left"
                      orientation="left"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      stroke="currentColor"
                      className="text-muted-foreground text-opacity-70"
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                  <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload || payload.length === 0) return null;

                        const date = new Date(label);
                        const day = date.getDate().toString().padStart(2, "0");
                        const month = date.toLocaleString("default", { month: "short" });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        const dateStr = `${day} ${month} ${year} (W${week})`;

                        const isForecast = payload.some((p: any) => p.payload?.isForecast);

                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                              <span>{dateStr}</span>
                              <span
                                style={{
                                  backgroundColor: "#f1f5f9",
                                  color: "#334155",
                                  fontSize: 10,
                                  padding: "2px 6px",
                                  borderRadius: 12,
                                  fontWeight: 500,
                                }}
                              >
                                {(() => {
                                  const weekDate = new Date(payload?.[0]?.payload?.Week);
                                  const today = new Date();
                                  today.setHours(0, 0, 0, 0);
                                  const isFuture = weekDate > today;

                                  const hasActual = selectedMetrics.some((metricId) => {
                                    const actual = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                    return actual?.value != null;
                                  });

                                  const hasForecast = selectedMetrics.some((metricId) => {
                                    const forecast = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                    return forecast?.value != null;
                                  });

                                  if (isFuture && hasForecast) return "Forecast";
                                  if (!isFuture && hasActual && hasForecast) return "Actual & Forecast";
                                  if (!isFuture && hasForecast) return "Forecast";
                                  return "Actual";
                                })()}
                              </span>
                            </div>
                              {selectedMetrics.map((metricId) => {
                                const metricInfo = metrics.find((m) => m.id === metricId);

                                const actualPoint = payload.find((p: any) => p.dataKey === `${metricId}_actual`);
                                const forecastPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast`);
                                const upperPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_upper`);
                                const lowerPoint = payload.find((p: any) => p.dataKey === `${metricId}_forecast_lower`);

                                const weekDate = new Date(payload?.[0]?.payload?.Week);
                                const today = new Date();
                                today.setHours(0, 0, 0, 0);

                                const isFuture = weekDate > today;

                                const hasActual = actualPoint?.value != null;
                                const hasForecast = forecastPoint?.value != null;

                                // Don't render if there's nothing to show
                                if (!hasActual && !hasForecast) return null;

                                return (
                                  <div key={metricId} style={{ marginBottom: 8 }}>
                                    {/* Metric name and legend color */}
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                      <div
                                        style={{
                                          width: 10,
                                          height: 10,
                                          backgroundColor: metricInfo?.color,
                                          borderRadius: 2,
                                        }}
                                      ></div>
                                      <span style={{ color: "var(--metric-label-color)", fontWeight: 700 }}>{metricInfo?.name}</span>
                                    </div>

                                    {/* Past week: show both if available */}
                                      {!isFuture && hasActual && (
                                        <div style={{ fontSize: 13, color: "var(--actual-color)", marginLeft: 16 }}>
                                          • Actual: {formatNumber(Number(actualPoint.value))}
                                        </div>
                                      )}

                                      {hasForecast && (
                                        <div style={{ fontSize: 13, color: "var(--forecast-color)", marginLeft: 16 }}>
                                          • Forecast: {formatNumber(Number(forecastPoint.value))}
                                        </div>
                                      )}

                                      {/* Bounds only in future */}
                                      {isFuture && hasForecast && upperPoint?.value != null && lowerPoint?.value != null && (
                                        <div
                                          style={{
                                            fontSize: 11,
                                            color: "var(--bounds-color)",
                                            marginLeft: 16,
                                            marginTop: 2,
                                          }}
                                        >
                                          Confidence bounds: {formatNumber(Number(lowerPoint.value))} - {formatNumber(Number(upperPoint.value))}
                                        </div>
                                      )}
                                  </div>
                                );
                              })}
                          </div>
                        );
                      }}
                    />
                    <Legend
                      formatter={(value) => {
                        const parts = value.split(" (");
                        return <span className="text-sm">{parts[0]}</span>;
                      }}
                    />

                    {selectedMetrics.map((metricId) => {
                      const metricInfo = metrics.find((m) => m.id === metricId);
                      const source = metricInfo?.source || "Unknown Source";

                      return (
                        <React.Fragment key={metricId}>

                            {/* Forecast lower bound */}
                          <Line
                            key={`forecast-lower-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_lower`}
                            name=""
                            stroke={metricInfo?.color}
                            fill={metricInfo?.color}
                            fillOpacity={0.5}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            
                          />
                          <Line
                            key={`forecast-upper-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast_upper`}
                            name=""
                            stroke={metricInfo?.color}
                            fill={metricInfo?.color}
                            fillOpacity={0.25}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                            legendType="none"
                            
                          />

                          {/* Historical Data */}
                          <Bar
                            key={`actual-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_actual`}
                            name={metricInfo?.name}
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            fill={metricInfo?.color}
                            fillOpacity={source === 'parameter' ? 0.4 : 1}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                           // dot={{ r: 3, strokeWidth: 1 }}
                            // activeDot={{ r: 5 }}
                            
                          />
                          {/* Forecast data */}
                          <Bar
                            key={`forecast-${metricId}`}
                            type="monotone"
                            dataKey={`${metricId}_forecast`}
                            name=""
                            stroke={metricInfo?.color}
                            strokeWidth={2}
                            strokeDasharray="2 2"
                            fill={metricInfo?.color}
                            fillOpacity={0.3}
                            yAxisId={source === 'parameter' ? 'left' : 'right'}
                           // dot={{ r: 3, strokeWidth: 1 }}
                           // activeDot={{ r: 5 }}
                            legendType="none"
                            
                          />
                        
                        </React.Fragment>
                      );
                    })}
                    </ComposedChart>
                 </ResponsiveContainer>
              )}
              </>
               )}
            </>
            )}
          </div>

          {(loading || (!insights && !data.dz_df)) ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Automated Insights</h3>
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 animate-pulse"
                > <CircularProgress />
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      <div className="h-5 w-5 bg-gray-300 rounded-full"></div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                        <div className="h-4 bg-gray-300 rounded w-1/4"></div>
                      </div>
                      <div className="mt-1 h-4 bg-gray-300 rounded w-full"></div>
                    </div>
                  </div>
                </div>
              ))}

            </div>
          ) : insights && Object.keys(insights).length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Automated Insights</h3>
              
              {Object.keys(insights).map((key: string, index: number) => (
                <div
                  key={index}
                  className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-100 dark:border-blue-800"
                >

               

                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {getIconForInsightType(
                        key.toLowerCase().includes("return") ? "anomaly" : "trend"
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <h4 className="font-medium text-base sm:text-lg text-gray-900 dark:text-gray-100">
                          {metrics.find(m => m.id === key)?.name?.replace(/_/g, ' ') || key}
                        </h4>
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            key.toLowerCase().includes("exception") ||
                            key.toLowerCase().includes("return")
                              ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200"
                              : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200"
                          }`}
                        >
                          {key.toLowerCase().includes("exception") ||
                          key.toLowerCase().includes("return")
                            ? "Critical"
                            : "Positive"}
                        </span>
                      </div>
                      <p className="mt-1 text-sm sm:text-base text-gray-600 dark:text-gray-300">
                        {insights[key]}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-10">No insights available</div>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};
export default ForecastTab;
