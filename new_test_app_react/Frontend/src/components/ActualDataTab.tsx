import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui2/button";
import { Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ExcelJS from 'exceljs';
import {
  AreaChart, BarChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  Bar, ResponsiveContainer,
  LineChart, Line
} from 'recharts';
import { useEffect, useMemo, useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { getISOWeek, format } from 'date-fns';
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import AreaChartOutlinedIcon from '@mui/icons-material/AreaChartOutlined';
import { Separator } from "@/components/ui/separator";
import { formatWithPrecision } from "@/lib/utils";
import AskAIButton from './AskAIButton';

import { useTheme } from "../components/ThemeContext";
import NoDataFullPage from "@/components/noDataPage";

interface ActualDataTabProps {
  data?: {
    dz_df?: any[];
  };
  aggregationType?: string;
  metricKeys?: any[];
  metricData?: any[];
  setAggregationType?: (type: string) => void;
  plotAggregationType?: string;
  defaultDecimalPrecisions?: number;
  setPlotAggregationType?: (type: string) => void;
  children?: React.ReactNode;
}

// Pivot raw API data to chart-friendly format
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

type StatSummary = {
  metric: string;
  avg: number;
  min: number;
  max: number;
  median: number;
  stdDev: number;
};

function calculateStats(data: any[], metrics: any[]): StatSummary[] {
  return metrics.map((metric) => {
    const values = data.map((d) => Number(d[metric])).filter((v) => !isNaN(v));
    if (values.length === 0) {
      return { metric, avg: 0, min: 0, max: 0, median: 0, stdDev: 0 };
    }

    const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median =
      sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
    const stdDev = Math.sqrt(
      values.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / values.length
    );

    return { metric, avg, min, max, median, stdDev };
  });
}



// Optional: aggregate Monthly
function transformData(data: any[], aggregationType: string): any[] {
  if (aggregationType === 'Monthly') {
    const monthlyData: Record<string, any> = {};
    data.forEach(item => {
      const date = new Date(item.Week);
      const month = date.toLocaleString('default', { month: 'long' });
      const year = date.getFullYear();
      const key = `${month} ${year}`;

      if (!monthlyData[key]) monthlyData[key] = { Week: key };

      Object.entries(item).forEach(([k, v]) => {
        if (k === 'Week') return;
        monthlyData[key][k] = (monthlyData[key][k] || 0) + (v as number);
      });
    });
    return Object.values(monthlyData);
  } else {
    return data;
  }
}



function getISOWeekNumber(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
}

export const ActualDataTab = ({
  data,
  plotAggregationType: propPlotAggregationType,
  metricKeys,
  metricData,
  defaultDecimalPrecisions,
  setPlotAggregationType: propSetPlotAggregationType,
  children
}: ActualDataTabProps) => {
  const { toast } = useToast();

  const formatNumber = (value: number) => {
    if (value >= 1_000_000) return defaultDecimalPrecisions > 0 ? `${(value / 1_000_000).toFixed(defaultDecimalPrecisions)}M` : `${Math.round(value / 1_000_000)}M`;
    if (value >= 1_000) return defaultDecimalPrecisions > 0 ? `${(value / 1_000).toFixed(1)}K` : `${Math.round(value / 1_000)}K`;
    return value.toLocaleString();
  };

  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);
  const [localPlotAggregationType, setLocalPlotAggregationType] = useState('Weekly');
  const plotAggregationType = propPlotAggregationType ?? localPlotAggregationType;
  const setPlotAggregationType = propSetPlotAggregationType ?? setLocalPlotAggregationType;
  const [chartType, setChartType] = useState('area');

  const rawPivoted = useMemo(() => {
    return data?.dz_df ? pivotActualData(data.dz_df) : [];
  }, [data?.dz_df]);

  const parameterMetrics = useMemo(() =>
    metricData
      .filter(item => item.source === 'parameter')
      .map(item => item.title)
    , [metricData]);

  const lobMetrics = useMemo(() =>
    metricData
      .filter(item => item.source === 'lob')
      .map(item => item.title)
    , [metricData]);

  const { themeMode } = useTheme();
  const transformedData = useMemo(() => {
    return transformData(rawPivoted, plotAggregationType);
  }, [rawPivoted, plotAggregationType]);

  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (metricKeys.length) {
      const initial: Record<string, boolean> = {};
      metricKeys.forEach(key => { initial[key] = true; });
      setSelectedMetrics(initial);
    }
  }, [metricKeys.join(",")]);

  const defaultColorPalette = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#ffffff", "#000000", "#ff4500", "#00ced1", "#7fff00",
    "#dc143c", "#00fa9a", "#1e90ff", "#f0e68c", "#dda0dd",
    "#b0c4de", "#ff69b4", "#a0522d", "#6a5acd", "#2e8b57",
    "#ff6347", "#8b0000", "#20b2aa", "#ffb6c1", "#32cd32",
    "#483d8b", "#ff1493", "#5f9ea0", "#8fbc8f", "#8a2be2",
    "#6495ed", "#deb887", "#00ff7f", "#4b0082", "#b22222"
  ];

  const getColorByIndex = (index: number) =>
    defaultColorPalette[index];


  const metricColors = useMemo(() => {
    const colorMap: Record<string, string> = {};
    metricKeys.forEach((metric, index) => {
      colorMap[metric] = getColorByIndex(index);
    });
    return colorMap;
  }, [metricKeys.join(',')]);

  const CustomLegend = ({ payload }: any) => {
    if (!payload) return null;

    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 0 }}>
        {payload.map((entry: any) => (
          <div
            key={entry.value}
            onMouseEnter={() => setHoveredMetric(entry.value)}
            onMouseLeave={() => setHoveredMetric(null)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: 4,
              backgroundColor: hoveredMetric === entry.value ? 'rgba(0,0,0,0.05)' : 'transparent',
              transition: 'all 0.2s',
            }}
          >
            <div
              style={{
                width: 12,
                height: 12,
                backgroundColor: entry.color,
                opacity: hoveredMetric && hoveredMetric !== entry.value ? 0.3 : 1,
                borderRadius: 2,
              }}
            />
            <span
              style={{
                fontSize: 12,
                fontWeight: hoveredMetric === entry.value ? 'bold' : 'normal',
                opacity: hoveredMetric && hoveredMetric !== entry.value ? 0.5 : 1,
              }}
            >
              {entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  };


  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics(prev => ({ ...prev, [metric]: !prev[metric] }));
  };

  const filteredData = useMemo(() => {
    if (!transformedData.length) return [];

    // Get the keys of selected metrics
    const enabledMetrics = Object.entries(selectedMetrics)
      .filter(([, selected]) => selected)
      .map(([metric]) => metric);

    return transformedData.map(row => {
      const filteredRow: Record<string, any> = { Week: row.Week };
      enabledMetrics.forEach(metric => {
        filteredRow[metric] = row[metric] ?? 0;
      });
      return filteredRow;
    });
  }, [transformedData, selectedMetrics]);

  const stats = useMemo(() => {
    const enabledMetrics = Object.entries(selectedMetrics)
      .filter(([, selected]) => selected)
      .map(([metric]) => metric);
    return calculateStats(filteredData, enabledMetrics);
  }, [filteredData, selectedMetrics]);

  const [sortConfig, setSortConfig] = useState({ key: 'metric', direction: 'asc' });

  const sortedStats = useMemo(() => {
    const sorted = [...stats];
    sorted.sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      if (typeof aValue === 'string') {
        return sortConfig.direction === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      } else {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }
    });
    return sorted;
  }, [stats, sortConfig]);

  const handleSort = (key) => {
    setSortConfig((prev) => {
      if (prev.key === key) {
        return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key, direction: 'asc' };
    });
  };


  const overallSummary = useMemo(() => {
    const count = stats.length;

    const sum = (arr: number[]) => arr.reduce((a, b) => a + b, 0);
    const avg = (arr: number[]) => (arr.length ? sum(arr) / arr.length : 0);

    return {
      avg: avg(stats.map((s) => s.avg)),
      min: Math.min(...stats.map((s) => s.min)),
      max: Math.max(...stats.map((s) => s.max)),
      median: avg(stats.map((s) => s.median)),
      stdDev: avg(stats.map((s) => s.stdDev))
    };
  }, [stats, selectedMetrics]);

  const tickInterval = (() => {
    const len = filteredData.length;
    if (len <= 20) return 0;
    if (len <= 50) return 2;
    if (len <= 100) return 5;
    return Math.floor(len / 20);
  })();

  const exportToCSV = () => {
    const header = ['Week', ...metricKeys].join(',');
    const rows = filteredData.map(row =>
      [format(row.Week, 'yyyy-MM-dd'), ...metricKeys.map(key => row[key] ?? '')].join(',')
    );
    const csv = [header, ...rows].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'Historical_Data.csv';
    link.click();

    toast({
      title: "Data Exported",
      description: "Historical Data has been downloaded as CSV",
    });
  };

  const exportToXLSX = async () => {
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Historical Data');

    // Add header
    const headerRow = ['Week', ...metricKeys];
    worksheet.addRow(headerRow);

    // Add data rows
    filteredData.forEach(row => {
      const weekDate = format(row.Week, 'yyyy-MM-dd');
      const dataRow = [weekDate, ...metricKeys.map(key => row[key] ?? '')];
      worksheet.addRow(dataRow);
    });

    // Format columns
    worksheet.columns.forEach((column, index) => {
      column.width = index === 0 ? 15 : 12;
      if (index > 0) {
        column.numFmt = '0.00'; // Keep number format (you can change this)
      }
    });

    // Generate and download file without file-saver
    const buffer = await workbook.xlsx.writeBuffer();
    const blob = new Blob([buffer], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'Historical_Data.xlsx';
    document.body.appendChild(link); // Optional but safe
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    toast({
      title: 'Data Exported',
      description: 'Historical Data downloaded as XLSX',
    });
  };


  return (
    <Card className="text-gray-900 dark:text-gray-100">
      <CardContent className="pt-2 mt-[-14px]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Historical Data Analysis</h2>
          <div className="flex flex-col sm:flex-row gap-2 flex items-center">

            <div className="flex items-center border rounded-md p-1  h-10">
              <Button
                variant={Object.values(selectedMetrics).every(v => v) ? "default" : "ghost"}
                size="sm"
                onClick={() => {
                  const allSelected: Record<string, boolean> = {};
                  metricKeys.forEach(key => { allSelected[key] = true; });
                  setSelectedMetrics(allSelected);
                }}
                disabled={!(data?.dz_df?.length > 0)}
                className="h-7 px-3"
              >
                Select All
              </Button>
              <Button
                variant={Object.values(selectedMetrics).every(v => !v) ? "default" : "ghost"}
                size="sm"
                disabled={!(data?.dz_df?.length > 0)}
                onClick={() => {
                  const noneSelected: Record<string, boolean> = {};
                  metricKeys.forEach(key => { noneSelected[key] = false; });
                  setSelectedMetrics(noneSelected);
                }}
                className="h-7 px-3"
              >
                Deselect All
              </Button>
            </div>


            <div className="text-sm">
              {children}
            </div>

            <div className="flex items-center border rounded-md p-1  h-10">
              <Button
                variant={plotAggregationType === "Weekly" ? "default" : "ghost"}
                size="sm"
                disabled={!(data?.dz_df?.length > 0)}
                onClick={() => setPlotAggregationType("Weekly")}
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
            <AskAIButton mainData={data?.dz_df || []} aiParams={{ graph: filteredData }} parametersList={[]} />
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
                          <Download className="h-5 w-5 text-muted-foreground hover:text-foreground" />
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
        {data?.dz_df?.length > 0 && (
          <div className="flex gap-4 mb-4 flex-wrap">
            {metricKeys.map(metric => (
              <div key={metric} className="flex items-center space-x-2">
                <Checkbox
                  id={metric}
                  checked={!!selectedMetrics[metric]}
                  onCheckedChange={() => handleMetricToggle(metric)}
                />
                <label
                  style={{
                    color: metricColors[metric] // label matches chart
                  }}
                  onMouseEnter={() => setHoveredMetric(selectedMetrics[metric] ? metric : null)}
                  onMouseLeave={() => setHoveredMetric(null)}
                  htmlFor={metric}
                  className="text-sm font-medium">
                  {metric}
                </label>
              </div>
            ))}
          </div>
        )}
        {!data?.dz_df?.length && (
          <NoDataFullPage message="No data found." />
        )}
        {data?.dz_df?.length > 0 && (
          <>
            <div className="h-[430px]">
              <ResponsiveContainer width="100%" height="100%">
                {chartType === 'area' && (
                  <AreaChart data={filteredData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="Week"
                      stroke="currentColor"
                      tick={({ x, y, payload }) => {
                        const date = new Date(payload.value);
                        if (isNaN(date.getTime())) return null;

                        let label = '';

                        if (plotAggregationType === 'Weekly') {
                          const weekNumber = getISOWeek(date);
                          label = `WK${weekNumber.toString().padStart(2, '0')}`;
                        } else {
                          const month = date.toLocaleString('default', { month: 'short' });
                          const year = date.getFullYear();
                          label = `${month}\n${year}`;
                        }

                        return (
                          <text
                            x={x}
                            y={y + 15}
                            textAnchor="middle"
                            fontSize={10}
                            style={{ whiteSpace: 'pre', fill: 'currentColor' }}
                          >
                            {label}
                          </text>
                        );
                      }}
                      interval={tickInterval}
                      height={100}
                    />
                    <YAxis yAxisId="left" tickFormatter={formatNumber} />
                    <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />

                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const date = new Date(label);
                        if (isNaN(date.getTime())) return null;
                        const day = date.getDate().toString().padStart(2, '0');
                        const month = date.toLocaleString('default', { month: 'short' });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div className="font-medium mb-1">{`${day} ${month} ${year} (W${week})`}</div>
                            {payload.map(entry => (
                              <div key={entry.name} className="flex items-center gap-2">
                                <div style={{ width: 10, height: 10, backgroundColor: entry.color, borderRadius: 2 }}></div>
                                <span>{entry.name}</span>
                                <span className="font-bold">{formatNumber(entry.value)}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />

                    { /* <Legend content={CustomLegend} /> */}

                    {parameterMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Area
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 2 : 3}
                          fillOpacity={hoveredMetric === metric ? 0.9 : 0.4}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          yAxisId="left"
                        />
                      )
                    ))}

                    {lobMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Area
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 2 : 3}
                          fillOpacity={hoveredMetric === metric ? 0.9 : 0.4}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          yAxisId="right"
                        />
                      )
                    ))}

                  </AreaChart>
                )}
                {chartType === 'line' && (
                  <LineChart data={filteredData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="Week"
                      stroke="currentColor"
                      tick={({ x, y, payload }) => {
                        const date = new Date(payload.value);
                        if (isNaN(date.getTime())) return null;

                        let label = '';

                        if (plotAggregationType === 'Weekly') {
                          const weekNumber = getISOWeek(date);
                          label = `WK${weekNumber.toString().padStart(2, '0')}`;
                        } else {
                          const month = date.toLocaleString('default', { month: 'short' });
                          const year = date.getFullYear();
                          label = `${month}\n${year}`;
                        }

                        return (
                          <text
                            x={x}
                            y={y + 15}
                            textAnchor="middle"
                            fontSize={10}
                            style={{ whiteSpace: 'pre', fill: 'currentColor' }}
                          >
                            {label}
                          </text>
                        );
                      }}
                      interval={tickInterval}
                      height={100}
                    />
                    <YAxis yAxisId="left" tickFormatter={formatNumber} />
                    <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />

                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const date = new Date(label);
                        if (isNaN(date.getTime())) return null;
                        const day = date.getDate().toString().padStart(2, '0');
                        const month = date.toLocaleString('default', { month: 'short' });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div className="font-medium mb-1">{`${day} ${month} ${year} (W${week})`}</div>
                            {payload.map(entry => (
                              <div key={entry.name} className="flex items-center gap-2">
                                <div style={{ width: 10, height: 10, backgroundColor: entry.color, borderRadius: 2 }}></div>
                                <span>{entry.name}</span>
                                <span className="font-bold">{formatNumber(entry.value)}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />

                    { /* <Legend content={CustomLegend} /> */}

                    {parameterMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Line
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 3 : 1.5}
                          fillOpacity={hoveredMetric === metric ? 0.25 : 0.7}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          yAxisId="left"
                        />
                      )
                    ))}

                    {lobMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Line
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 3 : 1.5}
                          fillOpacity={hoveredMetric === metric ? 0.25 : 0.7}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          yAxisId="right"
                        />
                      )
                    ))}

                  </LineChart>
                )}
                {chartType === 'bar' && (
                  <BarChart data={filteredData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="Week"
                      stroke="currentColor"
                      tick={({ x, y, payload }) => {
                        const date = new Date(payload.value);
                        if (isNaN(date.getTime())) return null;

                        let label = '';

                        if (plotAggregationType === 'Weekly') {
                          const weekNumber = getISOWeek(date);
                          label = `WK${weekNumber.toString().padStart(2, '0')}`;
                        } else {
                          const month = date.toLocaleString('default', { month: 'short' });
                          const year = date.getFullYear();
                          label = `${month}\n${year}`;
                        }

                        return (
                          <text
                            x={x}
                            y={y + 15}
                            textAnchor="middle"
                            fontSize={10}
                            style={{ whiteSpace: 'pre', fill: 'currentColor' }}
                          >
                            {label}
                          </text>
                        );
                      }}
                      interval={tickInterval}
                      height={100}
                    />
                    <YAxis yAxisId="left" tickFormatter={formatNumber} />
                    <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />

                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const date = new Date(label);
                        if (isNaN(date.getTime())) return null;
                        const day = date.getDate().toString().padStart(2, '0');
                        const month = date.toLocaleString('default', { month: 'short' });
                        const year = date.getFullYear();
                        const week = getISOWeekNumber(date);
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <div className="font-medium mb-1">{`${day} ${month} ${year} (W${week})`}</div>
                            {payload.map(entry => (
                              <div key={entry.name} className="flex items-center gap-2">
                                <div style={{ width: 10, height: 10, backgroundColor: entry.color, borderRadius: 2 }}></div>
                                <span>{entry.name}</span>
                                <span className="font-bold">{formatNumber(entry.value)}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />

                    { /* <Legend content={CustomLegend} /> */}

                    {parameterMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Bar
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 3 : 1.5}
                          fillOpacity={hoveredMetric === metric ? 0.25 : 0.7}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          activeBar={false}
                          yAxisId="left"
                        />
                      )
                    ))}

                    {lobMetrics.map(metric => (
                      selectedMetrics[metric] && (
                        <Bar
                          type="monotone"
                          key={metric}
                          dataKey={metric}
                          strokeWidth={hoveredMetric === metric ? 3 : 1.5}
                          fillOpacity={hoveredMetric === metric ? 0.25 : 0.7}
                          opacity={hoveredMetric && hoveredMetric !== metric ? 0.25 : 1}
                          stroke={metricColors[metric]}
                          fill={metricColors[metric]}
                          activeBar={false}
                          yAxisId="right"
                        />
                      )
                    ))}

                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>
            <div className="mt-2 mb-8 w-full">
              <Table className="min-w-full h-[200px]">
                <TableHeader>
                  <TableRow className="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200">
                    <TableHead onClick={() => handleSort('metric')} className="font-medium cursor-pointer">Metric {sortConfig.key === 'metric' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                    <TableHead onClick={() => handleSort('avg')} className="text-right cursor-pointer">Average {sortConfig.key === 'avg' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                    <TableHead onClick={() => handleSort('min')} className="text-right cursor-pointer">Min {sortConfig.key === 'min' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                    <TableHead onClick={() => handleSort('max')} className="text-right cursor-pointer">Max {sortConfig.key === 'max' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                    <TableHead onClick={() => handleSort('median')} className="text-right cursor-pointer">Median {sortConfig.key === 'median' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                    <TableHead onClick={() => handleSort('stdDev')} className="text-right cursor-pointer">Std Dev {sortConfig.key === 'stdDev' && (sortConfig.direction === 'asc' ? '↑' : '↓')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedStats.map((s) => (
                    <TableRow key={s.metric}>
                      <TableCell>{s.metric}</TableCell>
                      <TableCell className="text-right">{formatWithPrecision(s.avg, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                      <TableCell className="text-right">{formatWithPrecision(s.min, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                      <TableCell className="text-right">{formatWithPrecision(s.max, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                      <TableCell className="text-right">{formatWithPrecision(s.median, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                      <TableCell className="text-right">{formatWithPrecision(s.stdDev, defaultDecimalPrecisions).toLocaleString(undefined, { maximumFractionDigits: 2 })}</TableCell>
                    </TableRow>
                  ))}

                  {/* Overall Summary Row */}
                  <TableRow className="font-semibold bg-muted/30 dark:bg-muted/20">
                    <TableCell>Overall</TableCell>
                    <TableCell className="text-right">{formatWithPrecision(overallSummary.avg, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{formatWithPrecision(overallSummary.min, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{formatWithPrecision(overallSummary.max, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{formatWithPrecision(overallSummary.median, defaultDecimalPrecisions).toLocaleString()}</TableCell>
                    <TableCell className="text-right">{formatWithPrecision(overallSummary.stdDev, defaultDecimalPrecisions).toLocaleString(undefined, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};
