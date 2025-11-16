import React, { useState, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ChevronDown, Download, HelpCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { LightbulbIcon, TrendingUpIcon, AlertTriangleIcon, InfoIcon } from "lucide-react";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CircularProgress from "@mui/material/CircularProgress";
import LoadingSkeleton from "./LoadingSkeleton";

interface ForecastTabProps {
  aggregationType: string;
  modelType: string;
  forecastPeriod: number;
  data: {
    dz_df: any[];
    future_df: any[];
  };
  insights: any;
  loading?: boolean;
  externalFactors?: {
    majorEvents: boolean;
    dynamicTarget: boolean;
    dynamicTargetStartDate: string;
    dynamicTargetEndDate: string;
    dynamicTargetDecreasePercentage: string;
  };
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

const transformData = (data: any[], isForecast: boolean = false): any[] => {
  return data.map((item) => ({
    Week: item["Week"],
    "Total IB Units": item["Total IB Units"] || 0,
    "Total IB Units_lower": item["Total IB Units_lower"] || 0,
    "Total IB Units_upper": item["Total IB Units_upper"] || 0,
    exceptions: item.exceptions || 0,
    exceptions_lower: item.exceptions_lower || 0,
    exceptions_upper: item.exceptions_upper || 0,
    inventory: item.inventory || 0,
    inventory_lower: item.inventory_lower || 0,
    inventory_upper: item.inventory_upper || 0,
    returns: item.returns || 0,
    returns_lower: item.returns_lower || 0,
    returns_upper: item.returns_upper || 0,
    wfs_china: item.wfs_china || 0,
    wfs_china_lower: item.wfs_china_lower || 0,
    wfs_china_upper: item.wfs_china_upper || 0,
    isForecast,
  }));
};

const formatNumber = (value: number) => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toLocaleString();
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
  externalFactors = {
    majorEvents: false,
    dynamicTarget: false,
    dynamicTargetStartDate: "",
    dynamicTargetEndDate: "",
    dynamicTargetDecreasePercentage: "",
  },
}: ForecastTabProps) => {
  const metrics = [
    { id: "Total IB Units", name: "IB Units", color: "#8884d8" },
    { id: "exceptions", name: "IB Exceptions", color: "#82ca9d" },
    { id: "inventory", name: "Inventory", color: "#ffc658" },
    { id: "returns", name: "Customer Returns", color: "#ff7300" },
    { id: "wfs_china", name: "WFS_China", color: "#a05195" },
  ];

  const [selectedMetrics, setSelectedMetrics] = useState(["Total IB Units"]);
  const { toast } = useToast();

  const dzData = useMemo(() => {
    return data.dz_df
      ? transformData(data.dz_df, false).slice(-5) // Last 5 weeks of Historical Data
      : [];
  }, [data.dz_df]);

  const futureData = useMemo(() => {
    if (!data.future_df || !dzData.length) return [];

    // Calculate trend from the last 3 data points of dzData
    const trendWindow = dzData.slice(-3);
    const metricTrends: { [key: string]: number } = {};

    metrics.forEach((metric) => {
      const values = trendWindow.map((item) => item[metric.id] || 0);
      if (values.length >= 2) {
        const differences = values.slice(1).map((v, i) => v - values[i]);
        const averageChange = differences.reduce((sum, diff) => sum + diff, 0) / differences.length;
        metricTrends[metric.id] = averageChange / values[values.length - 1]; // Relative change rate
      } else {
        metricTrends[metric.id] = 0; // No trend if insufficient data
      }
    });

    return data.future_df.map((item, index) => {
      const lastActual = dzData[dzData.length - 1];
      const weekDate = new Date(item.Week || new Date());
      const isInDynamicTargetRange =
        externalFactors.dynamicTarget &&
        externalFactors.dynamicTargetStartDate &&
        externalFactors.dynamicTargetEndDate &&
        weekDate >= new Date(externalFactors.dynamicTargetStartDate) &&
        weekDate <= new Date(externalFactors.dynamicTargetEndDate);
      const decreaseFactor =
        isInDynamicTargetRange && externalFactors.dynamicTargetDecreasePercentage
          ? 1 - parseFloat(externalFactors.dynamicTargetDecreasePercentage) / 100
          : 1;

      const majorEventFactor = externalFactors.majorEvents && index === 0 ? 1.1 : 1; // 10% spike for major events on first forecast week

      const transformed = transformData([item], true)[0];
      const adjusted: any = { ...transformed };

      metrics.forEach((metric) => {
        const lastValue = lastActual[metric.id] || 0;
        const trendRate = metricTrends[metric.id] || 0;
        const forecastValue =
          lastValue * (1 + trendRate * (index + 1)) * decreaseFactor * majorEventFactor;

        adjusted[metric.id] = Math.max(0, forecastValue);
        adjusted[`${metric.id}_lower`] = Math.max(0, forecastValue * 0.9); // 10% lower bound
        adjusted[`${metric.id}_upper`] = forecastValue * 1.1; // 10% upper bound
      });

      return adjusted;
    });
  }, [data.future_df, dzData, externalFactors]);

  const combinedData = useMemo(() => {
    const allData = [...dzData, ...futureData];
    const sorted = allData.sort(
      (a, b) => new Date(a.Week).getTime() - new Date(b.Week).getTime()
    );

    return sorted.map((d) => {
      const newEntry: any = { ...d };
      ["Total IB Units", "exceptions", "inventory", "returns", "wfs_china"].forEach(
        (metric) => {
          if (d.isForecast) {
            newEntry[`${metric}_forecast`] = d[metric];
            newEntry[`${metric}_forecast_upper`] = d[`${metric}_upper`] ?? null;
            newEntry[`${metric}_forecast_lower`] = d[`${metric}_lower`] ?? null;
            newEntry[`${metric}_actual`] = null;
          } else {
            newEntry[`${metric}_actual`] = d[metric];
            newEntry[`${metric}_forecast`] = null;
            newEntry[`${metric}_forecast_upper`] = null;
            newEntry[`${metric}_forecast_lower`] = null;
          }
        }
      );
      return newEntry;
    });
  }, [dzData, futureData]);
  
  const handleMetricToggle = (metricId: string) => {
    setSelectedMetrics((current) =>
      current.includes(metricId)
        ? current.filter((id) => id !== metricId)
        : [...current, metricId]
    );
  };

  const exportToCSV = () => {
    const headerRows = ["Week"];
    selectedMetrics.forEach((metricId) => {
      const metricName = metrics.find((m) => m.id === metricId)?.name || metricId;
      headerRows.push(metricName, `${metricName}_lower`, `${metricName}_upper`);
    });

    const csvHeader = headerRows.join(",");

    const csvRows = [...dzData, ...futureData].map((item) => {
      const row = [item.Week];
      selectedMetrics.forEach((metricId) => {
        row.push(item[metricId], item[`${metricId}_lower`], item[`${metricId}_upper`]);
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
      `combined_data_${aggregationType.toLowerCase()}_${new Date()
        .toISOString()
        .split("T")[0]}.csv`
    );
    document.body.appendChild(link);

    link.click();
    document.body.removeChild(link);

    toast({
      title: "Export successful",
      description: `5+ forecast (${aggregationType.toLowerCase()}) exported to CSV`,
    });
  };

  const tickInterval = useMemo(() => {
    const len = combinedData.length;
    if (len <= 20) return 0; // show all
    if (len <= 50) return 2;
    if (len <= 100) return 5;
    return Math.floor(len / 20); // roughly 20 ticks max
  }, [combinedData]);

  return (
    <TooltipProvider>
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="flex flex-col space-y-4 sm:flex-row sm:justify-between sm:space-y-0 mb-6">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-semibold">Data Analysis</h3>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6">
                      <HelpCircle className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent className="w-80 p-4">
                    <p className="font-medium mb-2">About this chart</p>
                    <p className="text-sm text-muted-foreground">
                      This chart shows actual and forecasted data for various metrics, with forecasts adjusted for trends and external factors.
                    </p>
                  </TooltipContent>
                </UITooltip>
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                {dzData.length} weeks history + {forecastPeriod} weeks forecast ({modelType})
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="w-[200px] justify-between">
                    {selectedMetrics.length === 1
                      ? metrics.find((m) => m.id === selectedMetrics[0])?.name
                      : `${selectedMetrics.length} metrics selected`}
                    <ChevronDown className="ml-2 h-4 w-4 opacity-50" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-[200px]">
                  <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuCheckboxItem
                    key="select-all"
                    checked={selectedMetrics.length === metrics.length}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedMetrics(metrics.map((m) => m.id));
                      } else {
                        setSelectedMetrics([]);
                      }
                    }}
                  >
                    Select All
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSeparator />
                  {metrics.map((metric) => (
                    <DropdownMenuCheckboxItem
                      key={metric.id}
                      checked={selectedMetrics.includes(metric.id)}
                      onCheckedChange={() => handleMetricToggle(metric.id)}
                    >
                      {metric.name}
                    </DropdownMenuCheckboxItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              <Button
                variant="outline"
                size="icon"
                onClick={exportToCSV}
                className="text-muted-foreground hover:text-foreground"
              >
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="w-full h-[400px] bg-gradient-to-b from-card/40 to-background/10 p-4 rounded-lg border border-border/10 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={combinedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <defs>
                  {metrics.map((metric) => (
                    <React.Fragment key={`gradient-${metric.id}`}>
                      <linearGradient id={`color${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="5%"
                          stopColor={metric.id === "Total IB Units" ? "#928EDA" : metric.color}
                          stopOpacity={0.8}
                        />
                        <stop
                          offset="95%"
                          stopColor={metric.id === "Total IB Units" ? "#928EDA" : metric.color}
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
                    const day = date.getDate().toString().padStart(2, "0");
                    const month = date.toLocaleString("default", { month: "short" });
                    const year = date.getFullYear().toString().slice(-2);
                    return `${month} ${year}`;
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
                            {isForecast ? "Forecast" : "Actual"}
                          </span>
                        </div>
                        {selectedMetrics.map((metricId) => {
                          const metricInfo = metrics.find((m) => m.id === metricId);
                          const actualPoint = payload.find(
                            (p: any) => p.dataKey === `${metricId}_actual`
                          );
                          const forecastPoint = payload.find(
                            (p: any) => p.dataKey === `${metricId}_forecast`
                          );
                          const upperPoint = payload.find(
                            (p: any) => p.dataKey === `${metricId}_forecast_upper`
                          );
                          const lowerPoint = payload.find(
                            (p: any) => p.dataKey === `${metricId}_forecast_lower`
                          );

                          return (
                            <div key={metricId} style={{ marginBottom: 8 }}>
                              {(actualPoint && actualPoint.value != null) ||
                              (forecastPoint && forecastPoint.value != null) ? (
                                <div
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    fontSize: 13,
                                    fontWeight: 500,
                                  }}
                                >
                                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    <div
                                      style={{
                                        width: 10,
                                        height: 10,
                                        backgroundColor: metricInfo?.color,
                                        borderRadius: 2,
                                      }}
                                    ></div>
                                    <span style={{ color: "#334155" }}>{metricInfo?.name}</span>
                                  </div>
                                  <div style={{ color: "#334155" }}>
                                    {formatNumber(
                                      (actualPoint?.value as number) ??
                                        (forecastPoint?.value as number)
                                    )}
                                  </div>
                                </div>
                              ) : null}

                              {upperPoint &&
                                lowerPoint &&
                                upperPoint.value != null &&
                                lowerPoint.value != null && (
                                  <div
                                    style={{
                                      fontSize: 11,
                                      color: "#94a3b8",
                                      marginLeft: 16,
                                      marginTop: 2,
                                    }}
                                  >
                                    Confidence bounds:{" "}
                                    {formatNumber(lowerPoint.value as number)} -{" "}
                                    {formatNumber(upperPoint.value as number)}
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

                  return (
                    <React.Fragment key={metricId}>
                      {/* Historical Data */}
                      <Area
                        key={`actual-${metricId}`}
                        type="monotone"
                        dataKey={`${metricId}_actual`}
                        name={metricInfo?.name}
                        stroke={metricInfo?.color}
                        strokeWidth={2}
                        fill={
                          metricId === "Total IB Units"
                            ? "#928EDA"
                            : `url(#color${metricInfo?.id})`
                        }
                        fillOpacity={metricId === "Total IB Units" ? 0.4 : 1}
                        yAxisId={metricId === "Total IB Units" ? "right" : "left"}
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
                        fill={`url(#color${metricInfo?.id}Forecast)`}
                        fillOpacity={0.3}
                        yAxisId={metricId === "Total IB Units" ? "right" : "left"}
                        dot={{ r: 3, strokeWidth: 1 }}
                        activeDot={{ r: 5 }}
                        legendType="none"
                        connectNulls
                      />
                      {/* Forecast upper bound */}
                      <Area
                        key={`forecast-upper-${metricId}`}
                        type="monotone"
                        dataKey={`${metricId}_forecast_upper`}
                        name=""
                        stroke="none"
                        fill={metricId === "Total IB Units" ? "#928EDA" : metricInfo?.color}
                        fillOpacity={0.25}
                        yAxisId={metricId === "Total IB Units" ? "right" : "left"}
                        legendType="none"
                        connectNulls
                      />
                      {/* Forecast lower bound */}
                      <Area
                        key={`forecast-lower-${metricId}`}
                        type="monotone"
                        dataKey={`${metricId}_forecast_lower`}
                        name=""
                        stroke="none"
                        fill={metricId === "Total IB Units" ? "#928EDA" : metricInfo?.color}
                        fillOpacity={0.25}
                        yAxisId={metricId === "Total IB Units" ? "right" : "left"}
                        legendType="none"
                        connectNulls
                      />
                    </React.Fragment>
                  );
                })}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {loading ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Automated Insights</h3>
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 animate-pulse"
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      <div className="h-5 w-5 bg-gray-300 rounded-full"></div>
                    </div>
                    <div className="flex justify-center">
                      <CircularProgress />
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
                          {metrics.find((m) => m.id === key)?.name?.replace(/_/g, " ") || key}
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
