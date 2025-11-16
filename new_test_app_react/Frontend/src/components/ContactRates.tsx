import React, { useState } from "react";
import {
  PieChart,
  BarChart,
  Bar,
  AreaChart,
  Area,
  Pie,
  Cell,
  Tooltip as RechartsTooltip,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Button } from "@/components/ui2/button";
import { TrendingDown, TrendingUp } from "lucide-react";
import { formatWithPrecision } from "@/lib/utils";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import AreaChartOutlinedIcon from '@mui/icons-material/AreaChartOutlined';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { getISOWeek } from 'date-fns';
import { Separator } from "@/components/ui/separator";
import axios from "axios";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';
import AskAIButton from './AskAIButton'


type RawItem = {
  business_unit_id: [number, string];
  lob_id?: [number, string]; // Optional or may be invalid
  parameter_id?: [number, string]; // Optional or may be invalid
  value: number;
};

  const WEBAPPAPIURL =  `${AppConfig.API_URL}/`;

function summarizeLobByBusinessUnit(data: RawItem[], targetBusinessUnitId: number) {
  const summaryMap = new Map<string, number>();

  data.forEach((item) => {
    const [buId] = item.business_unit_id;
    const lobInfo = item.lob_id;

    // ✅ Skip if business unit doesn't match OR lob_id is missing or malformed
    if (buId !== targetBusinessUnitId || !Array.isArray(lobInfo) || !lobInfo[1]) return;

    const lobName = lobInfo[1];
    const value = Number(item.value) || 0;

    summaryMap.set(lobName, (summaryMap.get(lobName) || 0) + value);
  });

  return Array.from(summaryMap.entries()).map(([name, value]) => ({ name, value }));
}


type RawItem1 = {
  business_unit_id: [number, string];
  lob_id: [number, string] | null;
  parameter_id: [number, string];
  value: number | string;
  date: string;
};

function getISOWeekNumber(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
}

function summarizeByLobAndParameterByDate(data: RawItem1[], targetBusinessUnitId: number) {
  const summaryMap = new Map<string, Record<string, number>>();
  const seriesSourceMap = new Map<string, 'lob' | 'param'>();

  data.forEach((item) => {
    const [buId] = item.business_unit_id;
    if (buId !== targetBusinessUnitId) return;

    const dateKey = item.date;
    const value = Number(item.value) || 0;

    const lobInfo = item.lob_id;
    const paramInfo = item.parameter_id;

    if (!Array.isArray(paramInfo) || !paramInfo[1]) return;

    const seriesName = paramInfo[1];

    // Decide source: if lobInfo present, it's lob-based
    const isLob = Array.isArray(lobInfo) && lobInfo[1];

    // Track source
    seriesSourceMap.set(seriesName, isLob ? 'lob' : 'param');

    if (!summaryMap.has(dateKey)) {
      summaryMap.set(dateKey, {});
    }

    const entry = summaryMap.get(dateKey)!;
    entry[seriesName] = (entry[seriesName] || 0) + value;
  });

  // Format data for recharts
  const chartData = Array.from(summaryMap.entries())
    .sort(([a], [b]) => new Date(a).getTime() - new Date(b).getTime())
    .map(([date, metrics]) => ({ date, ...metrics }));

  // Separate keys
  const paramKeys: string[] = [];
  const lobKeys: string[] = [];

  seriesSourceMap.forEach((type, key) => {
    if (type === 'param') paramKeys.push(key);
    else lobKeys.push(key);
  });

  return {
    chartData,
    paramKeys,
    lobKeys,
  };
}




  const COLORS = [
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
  COLORS[index];

const lineData = [
  { month: "Jan", value: 400 },
  { month: "Feb", value: 300 },
  { month: "Mar", value: 500 },
  { month: "Apr", value: 600 },
  { month: "May", value: 700 },
  { month: "Jun", value: 650 },
];

export default function ContactRates({ options, parametersList, defaultDecimalPrecisions, children, overallOldData, overallData, overallLoading }) {

   
  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` :  `${Math.round(value / 1_000_000)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(1)}K` : `${Math.round(value / 1_000)}K`;
    return value.toLocaleString();
  };


    const [activeIndex, setActiveIndex] = useState<number | null>(null);

      const [chartType, setChartType] = useState('area');
      const [selectedId, setSelectedId] = useState(0);

    const [hoveredKey, setHoveredKey] = useState<string | null>(null);

      const handleMouseEnter = (o: any) => {
        setHoveredKey(o.value);
      };

      const handleMouseLeave = () => {
        setHoveredKey(null);
      };


     const renderActiveLabel = ({
      cx,
      cy,
      midAngle,
      outerRadius,
      value,
      index,
    }: any) => {
    if (activeIndex === index) return null;

    const RADIAN = Math.PI / 180;
    const radius = outerRadius + 10;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text
        x={x}
        y={y}
        fill="currentColor"
        textAnchor={x > cx ? "start" : "end"}
        dominantBaseline="central"
        fontSize={12}
      >
        {value.toLocaleString()}
      </text>
    );
  };


  return (
    <div className="p-2 space-y-3 mb-8">
      <div className="float-right">
        {children}
      </div>
         {options.map((opt) => {
             // Filter + summarize data for this business unit
            const summarized = summarizeLobByBusinessUnit(overallData, opt.id);
           
            // Sort by value descending
            const sortedLobs = summarized.sort((a, b) => b.value - a.value);
            
            const trendData = summarizeByLobAndParameterByDate(overallData, opt.id);
          
              const calculateStats = (key: string) => {
                const values = trendData.chartData
                  .map((row) => Number(row[key]) || 0)
                  .filter((v) => !isNaN(v));

                const total = values.reduce((sum, v) => sum + v, 0);
                const avg = values.length ? total / values.length : 0;
                const min = Math.min(...values);
                const max = Math.max(...values);

                return { key, avg, min, max, total };
              };

              const paramStats = trendData.paramKeys.map(calculateStats);
              const lobStats = trendData.lobKeys.map(calculateStats)

              const trendDataOld = summarizeByLobAndParameterByDate(overallOldData, opt.id);
          
              const calculateStatsOld = (key: string) => {
                const values = trendDataOld.chartData
                  .map((row) => Number(row[key]) || 0)
                  .filter((v) => !isNaN(v));

                const total = values.reduce((sum, v) => sum + v, 0);
                const avg = values.length ? total / values.length : 0;
                const min = Math.min(...values);
                const max = Math.max(...values);

                return { key, avg, min, max, total };
              };

              const paramStatsOld = trendDataOld.paramKeys.map(calculateStatsOld);
              const lobStatsOld = trendDataOld.lobKeys.map(calculateStatsOld)

            return (
                <>
                <h2 className="text-xl font-semibold">{opt.display_name}</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg executive-card shadow-md">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-medium">{paramStats?.[0]?.key || 'Units'} vs {lobStats?.[0]?.key || 'Cases'}</h3>
                            <div className="flex items-center p-1 gap-2 h-10">
                
                                <AskAIButton mainData={trendData?.chartData} aiParams={{ graph: trendData.chartData, cards: { stats: [...paramStats, ...lobStats], oldStats: [...paramStatsOld, ...lobStatsOld] } }} parametersList={parametersList} />
                                <TooltipProvider>
                                          <UITooltip>
                                            <TooltipTrigger asChild>
                                              <div>
                                                <Popover>
                                                  <PopoverTrigger asChild>
                                                    <Button disabled={!(trendData?.chartData?.length > 0)} variant="outline" size="icon">
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
                                                        onClick={() => { setChartType('area'); setSelectedId(opt.id); }}
                                                        disabled={(chartType === 'area' && (selectedId === opt.id || !selectedId)) || !(trendData?.chartData?.length > 0)}
                                                        className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                                      >
                                                        Area Chart
                                                      </button>
                                                      <button
                                                        onClick={() => { setChartType('line'); setSelectedId(opt.id); }}
                                                        disabled={(chartType === 'line' && (selectedId === opt.id || !selectedId)) || !(trendData?.chartData?.length > 0)}
                                                        className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                                      >
                                                        Line Chart
                                                      </button>
                                                      <button
                                                        onClick={() => { setChartType('bar'); setSelectedId(opt.id); }}
                                                        disabled={(chartType === 'bar' && (selectedId === opt.id || !selectedId)) || !(trendData?.chartData?.length > 0)}
                                                        className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                                      >
                                                        Bar Chart
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

          


                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-4 xl:grid-cols-2 gap-2 mb-8">
                             {paramStats.length > 0 && (
                               <>
                           
                                    {paramStats.map((stat) => {

                                      const oldKpi = paramStatsOld.find(old => old.key === stat.key);
                                        const parameterdata = parametersList.find(old => old.name === stat.key);
                                        const kpiDirection = parameterdata?.direction ?? 'Low is Good';
                                        const oldValue = oldKpi?.total ?? 0;
                                        const currentValue = stat.total;

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

                                         const isPositive = kpiDirection === "Low is Good"
                                            ? change <= 0
                                            : change >= 0;

                                            const Icon = () => {
                                            if (isPositive) {
                                              return kpiDirection === "Low is Good"
                                                ? <TrendingDown className="w-4 h-4 text-green-400" />
                                                : <TrendingUp className="w-4 h-4 text-green-400" />;
                                            } else {
                                              return kpiDirection === "Low is Good"
                                                ? <TrendingUp className="w-4 h-4 text-red-400" />
                                                : <TrendingDown className="w-4 h-4 text-red-400" />;
                                            }
                                          };

                                       return (
                                        <div key={stat.key} className="bg-white dark:bg-gray-700 p-3 rounded-lg shadow">
                                          {/* Header */}
                                          <h4 className="text-lg font-medium text-gray-600 dark:text-gray-300 mb-1">
                                            {stat.key}
                                          </h4>

                                          {/* Total - emphasized */}
                                          <div className="text-xl font-semibold text-gray-800 dark:text-white flex items-center gap-2 mb-1">
                                            {stat.total.toLocaleString()}
                                            {changeText && (
                                              <div className="flex items-center">
                                                        <Icon />
                                                        <span className={`text-sm ml-2 font-medium ${isPositive ? 'text-green-400' : 'text-red-400'} truncate`}>
                                                          {change >= 0 ? '+' : '-'} {formatWithPrecision(Math.abs(change), 0)}% {changeText}
                                                        </span>
                                                </div>
                                            )}
                                          </div>

                                          {/* Grid for Avg, Min, Max */}
                                          <div className="grid grid-cols-3 gap-2 text-xs text-gray-500 dark:text-gray-400">
                                            <div className="flex flex-col">
                                              <span className="font-semibold text-gray-800 dark:text-white">Avg</span>
                                              <span>{stat.avg.toLocaleString()}</span>
                                            </div>
                                            <div className="flex flex-col">
                                              <span className="font-semibold text-gray-800 dark:text-white">Min</span>
                                              <span>{stat.min.toLocaleString()}</span>
                                            </div>
                                            <div className="flex flex-col">
                                              <span className="font-semibold text-gray-800 dark:text-white">Max</span>
                                              <span>{stat.max.toLocaleString()}</span>
                                            </div>
                                          </div>
                                        </div>
                                       )
                                   })}
                              
                                </>
                              )}

                              {/* 🧾 LOB-based KPI cards */}
                              {lobStats.length > 0 && (
                                <>
                                    {lobStats.map((stat) => {
                                      
                                      const oldKpi = lobStatsOld.find(old => old.key === stat.key);
                                        const parameterdata = parametersList.find(old => old.name === stat.key);
                                        const kpiDirection = parameterdata?.direction ?? 'Low is Good';
                                        const oldValue = oldKpi?.total ?? 0;
                                        const currentValue = stat.total;
                                        
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

                                         const isPositive = kpiDirection === "Low is Good"
                                            ? change <= 0
                                            : change >= 0;

                                            const Icon = () => {
                                            if (isPositive) {
                                              return kpiDirection === "Low is Good"
                                                ? <TrendingDown className="w-4 h-4 text-green-400" />
                                                : <TrendingUp className="w-4 h-4 text-green-400" />;
                                            } else {
                                              return kpiDirection === "Low is Good"
                                                ? <TrendingUp className="w-4 h-4 text-red-400" />
                                                : <TrendingDown className="w-4 h-4 text-red-400" />;
                                            }
                                          };

                                       return (
                                       <div key={stat.key} className="bg-white dark:bg-gray-700 p-3 rounded-lg shadow">
                                        {/* Header */}
                                        <h4 className="text-lg font-medium text-gray-600 dark:text-gray-300 mb-1">
                                          {stat.key}
                                        </h4>

                                        {/* Total - emphasized */}
                                        <div className="text-xl font-semibold text-gray-800 dark:text-white mb-1 flex items-center gap-2">
                                         {stat.total.toLocaleString()}
                                          {changeText && (
                                              <div className="flex items-center">
                                                        <Icon />
                                                        <span className={`text-sm ml-2 font-medium ${isPositive ? 'text-green-400' : 'text-red-400'} truncate`}>
                                                          {change >= 0 ? '+' : '-'} {formatWithPrecision(Math.abs(change), 0)}% {changeText}
                                                        </span>
                                                </div>
                                            )}
                                        </div>

                                        {/* Grid for Avg, Min, Max */}
                                        <div className="grid grid-cols-3 gap-2 text-xs text-gray-500 dark:text-gray-400">
                                          <div className="flex flex-col">
                                            <span className="font-semibold text-gray-800 dark:text-white">Avg</span>
                                            <span>{stat.avg.toLocaleString()}</span>
                                          </div>
                                          <div className="flex flex-col">
                                            <span className="font-semibold text-gray-800 dark:text-white">Min</span>
                                            <span>{stat.min.toLocaleString()}</span>
                                          </div>
                                          <div className="flex flex-col">
                                            <span className="font-semibold text-gray-800 dark:text-white">Max</span>
                                            <span>{stat.max.toLocaleString()}</span>
                                          </div>
                                        </div>
                                      </div>
                                          )
                                   })}
                                  </>
                              )}
                        </div>

                        <ResponsiveContainer width="100%" height={250}>
                          {chartType && (!selectedId || selectedId !== opt.id || chartType === 'area') && (
                            <AreaChart data={trendData.chartData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                              <XAxis
                                dataKey="date"
                                stroke="currentColor" 
                                tick={({ x, y, payload }) => {
                                                        const date = new Date(payload.value);
                                                        if (isNaN(date.getTime())) return null;
                                
                                                        const weekNumber = getISOWeek(date);
                                                          const label = `WK${weekNumber.toString().padStart(2, '0')}`;
                                
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
                              />
                              <YAxis yAxisId="left" tickFormatter={formatNumber} />
                              <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />
                              <RechartsTooltip
                                formatter={(value: number, name: string) => [`${value.toLocaleString()}`, name]}
                                contentStyle={{
                                  backgroundColor: "#fff",
                                  border: "1px solid #ccc",
                                  borderRadius: "0.5rem",
                                  fontSize: "0.875rem",
                                }}
                              />
                              <Legend onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} />

                              {trendData.paramKeys.map((key, index) => (
                                <Area
                                  key={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index]}
                                  fill={COLORS[index]}
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.4}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="left"
                                />
                              ))}
                              {trendData.lobKeys.map((key, index) => (
                                <Area
                                  key={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index + 1]}
                                  fill={COLORS[index + 1]}
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.4}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="right"
                                />
                              ))}
                            </AreaChart>
                          )}
                          {chartType === 'bar' && (selectedId === opt.id) && (
                            <BarChart data={trendData.chartData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                              <XAxis
                                dataKey="date"
                                stroke="currentColor" 
                                tick={({ x, y, payload }) => {
                                                        const date = new Date(payload.value);
                                                        if (isNaN(date.getTime())) return null;
                                
                                                        const weekNumber = getISOWeek(date);
                                                          const label = `WK${weekNumber.toString().padStart(2, '0')}`;
                                
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
                              />
                              <YAxis yAxisId="left" tickFormatter={formatNumber} />
                              <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />
                              <RechartsTooltip
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
                                                  

                              <Legend
                                onMouseEnter={handleMouseEnter}
                                onMouseLeave={handleMouseLeave}
                                formatter={(value, entry) => (
                                  <span style={{ fontSize: "0.75rem", color: entry.color }}>
                                    {value}
                                  </span>
                                )}
                              />


                              {trendData.paramKeys.map((key, index) => (
                                <Bar
                                  key={key}
                                  name={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index]}
                                  fill={COLORS[index]} 
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.7}
                                  strokeOpacity={hoveredKey === key ? 0.9 : 0.7}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="left"
                                  legendType="square"
                                />
                              ))}
                              {trendData.lobKeys.map((key, index) => (
                                <Bar
                                  key={key}
                                  name={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index + 1]}
                                  fill={COLORS[index + 1]} 
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.7}
                                  strokeOpacity={hoveredKey === key ? 0.9 : 0.7}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="right"
                                  legendType="square"
                                />
                              ))}
                            </BarChart>
                          )}
                          {chartType === 'line' && (selectedId === opt.id) && (
                            <LineChart data={trendData.chartData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                              <XAxis
                                dataKey="date"
                                stroke="currentColor" 
                                tick={({ x, y, payload }) => {
                                                        const date = new Date(payload.value);
                                                        if (isNaN(date.getTime())) return null;
                                
                                                        const weekNumber = getISOWeek(date);
                                                          const label = `WK${weekNumber.toString().padStart(2, '0')}`;
                                
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
                              />
                              <YAxis yAxisId="left" tickFormatter={formatNumber} />
                              <YAxis yAxisId="right" orientation="right" tickFormatter={formatNumber} />
                              <RechartsTooltip
                                formatter={(value: number, name: string) => [`${value.toLocaleString()}`, name]}
                                contentStyle={{
                                  backgroundColor: "#fff",
                                  border: "1px solid #ccc",
                                  borderRadius: "0.5rem",
                                  fontSize: "0.875rem",
                                }}
                              />
                              <Legend onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} />

                              {trendData.paramKeys.map((key, index) => (
                                <Line
                                  key={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index]}
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.4}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="left"
                                />
                              ))}
                              {trendData.lobKeys.map((key, index) => (
                                <Line
                                  key={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={COLORS[index + 1]}
                                  strokeWidth={hoveredKey === key ? 2 : 3}
                                  fillOpacity={hoveredKey === key ? 0.9 : 0.4}
                                  opacity={hoveredKey && hoveredKey !== key ? 0.25 : 1}
                                  yAxisId="right"
                                />
                              ))}
                            </LineChart>
                          )}
                        </ResponsiveContainer>
                    </div>
                    {/* Pie Chart Section */}
                    

                    {/* Line Chart Section */}
                  <div className="p-4 rounded-lg executive-card shadow-md">
                     <h3 className="text-lg font-medium mb-4 flex items-center">
                      Cases by LOB

                      <AskAIButton mainData={trendData?.chartData} aiParams={{ graph: sortedLobs }} parametersList={parametersList} />

                  
                     </h3>
                     <div className="flex flex-col md:flex-row gap-4">
                        <div className="flex-1 flex items-center min-h-[300px]">
                            <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                            <Pie
                                data={sortedLobs}
                                dataKey="value"
                                cx="50%"
                                cy="50%"
                                outerRadius={100}
                                onMouseEnter={(_, index) => setActiveIndex(index)}
                                onMouseLeave={() => setActiveIndex(null)}
                                >
                                {sortedLobs.map((entry, index) => (
                                    <Cell
                                      key={`cell-${index}`}
                                      fill={getColorByIndex(index)}
                                      fillOpacity={activeIndex === index ? 1 : 0.7}
                                      stroke={activeIndex === index ? "#000" : undefined}
                                      strokeWidth={activeIndex === index ? 2 : 1}
                                      />
                                ))}
                                </Pie>

                                <RechartsTooltip
                                 content={({ active, payload }) => {
                                    if (!active || !payload?.length || (activeIndex === null)) return null;
                                    return (
                                    <div className="rounded-md border shadow-sm p-2 text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border-gray-300 dark:border-gray-700">
                                        {payload.map((entry, index) => (
                                        <div
                                            key={`item-${index}`}
                                            className="flex items-center gap-2"
                                        >
                                            <div
                                            className="w-2.5 h-2.5 rounded-sm"
                                            style={{ backgroundColor: entry.color }}
                                            />
                                            <span style={{ color: entry.color }}>
                                            {entry.name}:{" "}
                                            <strong>{entry.value.toLocaleString()}</strong>
                                            </span>
                                        </div>
                                        ))}
                                    </div>
                                    );
                                }}
                                wrapperStyle={{ zIndex: 50 }}
                                />
                            </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Legend Section */}
                        <div className="w-full md:w-64 overflow-y-auto max-h-[300px] px-2">
                            
                            <ul className="space-y-2">
                            {sortedLobs.map((entry, index) => (
                                <li 
                                  key={entry.name} 
                                  className="flex items-center gap-2 cursor-pointer"
                                  onMouseEnter={() => setActiveIndex(index)}
                                  onMouseLeave={() => setActiveIndex(null)}
                                  >
                                <div
                                    className="w-3 h-3 rounded-sm"
                                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                                />
                                <span className={`text-sm text-gray-800 dark:text-gray-200 ${activeIndex === index ? "font-semibold scale-[1.02]" : ''}`}>
                                     {entry.name} –{" "}
                                    <span className="font-semibold">
                                      {formatNumber(entry.value)}
                                    </span>
                                </span>
                                </li>
                            ))}
                            </ul>
                        </div>
                    </div>
                    </div>
                </div>
                </>
            );
        })}
    </div>
  );
}
