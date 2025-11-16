import {
  BarChart,
  Bar,
  CartesianGrid,
  LineChart,
  Line,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Label,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectLabel, SelectTrigger, SelectValue, SelectGroup, SelectScrollDownButton } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import React, { useState, useEffect, useMemo } from "react";
import { LightbulbIcon, TrendingUpIcon, AlertTriangleIcon, InfoIcon, ChevronDown, Download } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from"@/components/ui/table";
import { Button } from "@/components/ui/button";
import NoDataFullPage from "@/components/noDataPage";
import { formatWithPrecision } from "@/lib/utils";
import AskAIButton from './AskAIButton';


// Sample data for insights
interface InsightsTabProps {
  data: any;
  metricData: any;
  children?: React.ReactNode;
  insights: any;
  defaultDecimalPrecisions?: number;
  loading: boolean;
  metricKeys: string[];
}

const dataA = [
  { lob: "LOB1", current: "4.20%", avg: "5.50%", target: "5%" },
  { lob: "LOB2", current: "-", avg: "-", target: "-" },
];

const dataC = [
  {
    function: "Voice",
    current: "3.80%",
    avg: "4.50%",
    notes: "High Accuracy, Steady",
  },
  {
    function: "Chat",
    current: "4.80%",
    avg: "5.80%",
    notes: "Improved 1% post ML model tuning",
  },
  { function: "Email", current: "-", avg: "-", notes: "" },
];

const dataD = [
  {
    month: "May-25",
    cause: "Unexpected volume surge",
    action: "Added historical anomaly tag",
  },
];

const insights = [
  {
    id: 1,
    type: "trend",
    title: "Inventory management expected to spike in next 30 days",
    description: "Based on historical patterns, we predict a 23% increase in inventory values over the next 30 days.",
    impact: "high",
  },
  {
    id: 2,
    type: "pattern",
    title: "Customer returns pattern shows seasonal variation",
    description: "Returns increase by 18% during holiday seasons, particularly in December and January.",
    impact: "medium",
  },
  {
    id: 3,
    type: "improvement",
    title: "WSF China trend improving with new supplier",
    description: "The new supplier relationship has reduced variance by 15% and improved predictability.",
    impact: "high",
  },
  {
    id: 4,
    type: "correlation",
    title: "Inbound exceptions correlate strongly with customer returns",
    description: "For every 5% increase in inbound exceptions, customer returns increase by approximately 8% within 14 days.",
    impact: "high",
  },
  {
    id: 5,
    type: "anomaly",
    title: "Unusual spike detected in IB Units on weekends",
    description: "Weekend processing is 27% higher than the historical average for the past 3 weeks.",
    impact: "medium",
  },
];

const mapeTrend = [
  { month: "Jun-24", value: 5.3 },
  { month: "Jul-24", value: 4.8 },
  { month: "Aug-24", value: 4.9 },
  { month: "Sep-24", value: 4.5 },
  { month: "Oct-24", value: 4.3 },
  { month: "Nov-24", value: 5.0, event: "Model Update" },
  { month: "Dec-24", value: 5.6 },
  { month: "Jan-25", value: 5.1 },
  { month: "Feb-25", value: 4.7 },
  { month: "Mar-25", value: 4.2 },
  { month: "Apr-25", value: 4.4 },
  { month: "May-25", value: 5.8, event: "Volume Surge" },
];

function computeCorrelationMatrix(data: any[], categories: string[]): number[][] {
  const matrix: number[][] = [];
  const n = data.length;

  // Precompute all category arrays
  const series: Record<string, number[]> = {};
  categories.forEach((cat) => {
    series[cat] = data.map(row => Number(row[cat]) || 0);
  });

  for (let i = 0; i < categories.length; i++) {
    matrix[i] = [];
    const xi = series[categories[i]];
    const meanX = xi.reduce((a, b) => a + b, 0) / n;
    const denominatorX = Math.sqrt(xi.reduce((sum, xk) => sum + Math.pow(xk - meanX, 2), 0));

    for (let j = 0; j < categories.length; j++) {
      const yi = series[categories[j]];
      const meanY = yi.reduce((a, b) => a + b, 0) / n;
      const denominatorY = Math.sqrt(yi.reduce((sum, yk) => sum + Math.pow(yk - meanY, 2), 0));

      const numerator = xi.reduce((sum, xk, idx) => sum + (xk - meanX) * (yi[idx] - meanY), 0);
      const corr = denominatorX && denominatorY ? numerator / (denominatorX * denominatorY) : 0;
      matrix[i][j] = Math.max(-1, Math.min(1, corr)); // clamp to [-1, 1]
    }
  }

  return matrix;
}


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

const processMonthlyData = (dz_df: any[], heatmapCategories: string[]) => {
  const monthlyData: Record<string, any> = {};

  dz_df.forEach((row) => {
    const dateStr = row.Week;
    if (!dateStr) return;

    const date = new Date(dateStr);
    if (isNaN(date.getTime())) {
      console.error("Invalid date:", dateStr);
      return;
    }

    const monthKey = `${date.getFullYear()}-${date.getMonth() + 1}`; // e.g., 2025-5

    if (!monthlyData[monthKey]) {
      monthlyData[monthKey] = {
        name: date.toLocaleString('default', { month: 'short', year: 'numeric' }),
      };
      heatmapCategories.forEach((category) => {
        monthlyData[monthKey][category] = 0;
      });
    }

    heatmapCategories.forEach((category) => {
      monthlyData[monthKey][category] += Number(row[category]) || 0;
    });
  });

  return Object.values(monthlyData);
};

export const InsightsTab = ({ data, defaultDecimalPrecisions, loading, metricKeys, metricData, children }: InsightsTabProps) => {
  const [activeTab, setActiveTab] = useState<"correlations" | "seasonality" | "mape">("correlations");
  const [selectedTraces, setSelectedTraces] = useState<string[]>(["inventory"]);
  const [traces, setTraces] = useState<any[]>([]);

  const categories = metricKeys;
  const labels = metricKeys;
    
const correlationMatrix = useMemo(() => {
  const rawCombinedData = pivotActualData(data || []);

  console.log("Raw Combined Data:", rawCombinedData);

  // Ensure rawCombinedData is array of objects with numerical values for each category
  if (!rawCombinedData || rawCombinedData.length < 2) return null;

  return computeCorrelationMatrix(rawCombinedData, selectedTraces);
}, [data, selectedTraces]);
    
  const heatmapCategories = labels;

  console.log("Correlation Matrix:", correlationMatrix);
    console.log("heatmapCategories", heatmapCategories);

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


   useEffect(() => {
        const newData = metricData.map((cat, index) => ({
            value: cat.title,
            label: cat.title,
            color: getColorByIndex(index),
            source: cat.source,
          }));
          setTraces(newData)
          setSelectedTraces(newData.length > 0 ? newData.map(trace => trace.value) : []);
        //  setParentSelectedMetric(newData?.length && newData[0].name ? newData[0].name : '')
      }, [metricData]);

  const handleTraceChange = (value: string[]) => {
    setSelectedTraces(value);
  };

  const exportToCSV = () => {
    if (!data) return;

    const monthlyData = processMonthlyData(pivotActualData(data), selectedTraces);
    const csvHeader = ["Month", ...selectedTraces.map(trace => traces.find(t => t.value === trace)?.label || trace)].join(",");
    const csvRows = monthlyData.map(row => [row.name, ...selectedTraces.map(trace => row[trace])].join(","));

    const csvContent = [csvHeader, ...csvRows].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `monthly_data_${new Date().toISOString().split("T")[0]}.csv`);
    document.body.appendChild(link);

    link.click();
    document.body.removeChild(link);
  };

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

  const getBadgeColorForImpact = (impact: string) => {
    switch (impact) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

 function getCorrelationColor(value: number): string {
  const intensity = Math.abs(value);
  const hue = value < 0 ? 0 : 210; // red for negative, blue for positive
  const saturation = 90;
  const lightness = 50 + (1 - intensity) * 25; // keep readable even at low intensity

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

const camelCase = (str: string) =>
  str
    .replace(/[^a-zA-Z0-9 ]/g, '')
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) =>
      index === 0 ? word.toLowerCase() : word.toUpperCase()
    )
    .replace(/\s+/g, '');

  return (
    <Card>
      <CardContent className="pt-2 mt-[-10px]">
        <div className="mb-6">
           <div className="flex flex-col space-y-4 sm:flex-row sm:justify-between sm:space-y-0 mb-6">
              <div className="flex items-center gap-2 ">
                <h3 className="text-xl font-semibold">Insights & Analysis</h3>
              </div>
                <div className="flex flex-col sm:flex-row gap-2 flex items-center">
                  {children}

                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-[200px] justify-between">
                          {selectedTraces.length === 1
                            ? traces.find((t) => t.value === selectedTraces[0])?.label
                            : selectedTraces.length === traces.length
                              ? "All Metrics"
                              : `${selectedTraces.length} traces selected`}
                          <ChevronDown className="ml-2 h-4 w-4 opacity-50" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-[200px]" style={{ maxHeight: "400px", overflowY: "auto" }} onCloseAutoFocus={(e) => e.preventDefault()}>
                        <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                          <DropdownMenuCheckboxItem
                            key="select-all"
                            checked={selectedTraces.length === traces.length}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                handleTraceChange(traces.map((t) => t.value));
                              } else {
                                handleTraceChange([]);
                              }
                            }}
                            onSelect={(e) => e.preventDefault()}
                          >
                            Select All
                          </DropdownMenuCheckboxItem>
                        <DropdownMenuSeparator />
                        {traces.map((trace) => (
                          <DropdownMenuCheckboxItem
                            key={trace.value}
                            checked={selectedTraces.includes(trace.value)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                handleTraceChange([...selectedTraces, trace.value]);
                              } else {
                                handleTraceChange(selectedTraces.filter((t) => t !== trace.value));
                              }
                            }}
                            onSelect={(e) => e.preventDefault()}
                          >
                            {trace.label}
                          </DropdownMenuCheckboxItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>

                    <AskAIButton mainData={data || []} aiParams={activeTab === "correlations" ? { heatmap: correlationMatrix } : { graph: processMonthlyData(pivotActualData(data), selectedTraces) }} parametersList={[]} />

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
          <div className="flex mt-4 border-b">
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "correlations" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("correlations")}
            >
              Correlation Heatmap
            </button>
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "seasonality" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("seasonality")}
            >
              Seasonality Analysis
            </button>
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "mape" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("mape")}
            >
              MAPE Analysis
            </button>
          </div>
        </div>

                  {!loading && !data?.length && (
                          <NoDataFullPage message="No data found." />
                        )}

    {data?.length > 0 && (
      <>
        {activeTab === "correlations" && (
          <div className="rounded-lg p-4">
            <p className="text-sm text-muted-foreground mb-4">
              This heatmap shows the correlation between different metrics. Darker colors indicate stronger correlations.
            </p>
          {correlationMatrix ? (
              <div className="w-full overflow-x-auto rounded-lg bg-background p-4">
                <div className="flex min-w-fit">
                  <div
                    className="grid gap-px border bg-border"
                    style={{
                      gridTemplateColumns: `160px repeat(${selectedTraces.length}, 80px)`,
                      gridAutoRows: '80px',
                    }}
                  >
                    {/* Top-left empty cell */}
                    <div className="bg-background" />

                    {/* Column headers */}
                    {selectedTraces.map((category, i) => (
                      <div
                        key={`header-${i}`}
                        className="flex items-center justify-center bg-background text-sm font-medium text-foreground px-2 text-center"
                      >
                        {category}
                      </div>
                    ))}

                    {/* Row labels + correlation cells */}
                    {selectedTraces.map((category, rowIndex) => (
                      <React.Fragment key={`row-${rowIndex}`}>
                        {/* Row label */}
                        <div className="flex items-center bg-background text-sm font-semibold text-foreground px-2 whitespace-nowrap">
                          {category}
                        </div>

                        {/* Row cells */}
                        {correlationMatrix[rowIndex].map((value, colIndex) => (
                          <div
                            key={`cell-${rowIndex}-${colIndex}`}
                            title={`${category} ↔ ${selectedTraces[colIndex]}: ${value.toFixed(2)}`}
                            className="flex items-center justify-center text-xs font-medium text-foreground"
                            style={{
                              backgroundColor: getCorrelationColor(value),
                            }}
                          >
                            {value.toFixed(2)}
                          </div>
                        ))}
                      </React.Fragment>
                    ))}
                  </div>

                  {/* Color Legend */}
                  <div className="ml-4 flex flex-col items-center justify-between py-2">
                    <span className="text-xs text-muted-foreground">Positive</span>
                    <div
                      style={{
                        width: 12,
                        height: '100%',
                        background:
                          'linear-gradient(to top, rgba(239,68,68,1), white, rgba(59,130,246,1))',
                        borderRadius: 4,
                        margin: '4px 0',
                        flexGrow: 1,
                      }}
                    ></div>
                    <span className="text-xs text-muted-foreground">Negative</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-10">
                No correlation data available
              </div>
            )}
          </div>
        )}

        {activeTab === "seasonality" && (
          <div className="rounded-lg p-4">
            <p className="text-sm text-muted-foreground mb-4">
              This chart shows the monthly aggregated data for different metrics throughout the year.
            </p>
            <div className="w-full h-[400px] bg-background">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={data ? processMonthlyData(pivotActualData(data), selectedTraces) : []}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  barCategoryGap="15%"
                  barGap="10%"
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tickLine={false}
                    stroke="currentColor"
                    tick={({ x, y, payload }) => {
                      const label = payload.value;
                      const parts = label.split(' ');
                      const month = parts[0];
                      const year = parts[1];

                      return (
                        <g transform={`translate(${x},${y + 10})`}>
                          <text x={0} y={0} textAnchor="middle" fontSize={10} fill="hsl(var(--foreground))">
                            {month}
                          </text>
                          {month.toLowerCase() === 'jan' && (
                            <text x={0} y={15} textAnchor="middle" fontSize={8} fill="hsl(var(--foreground))">
                              {year}
                            </text>
                          )}
                        </g>
                      );
                    }}
                    interval={0}
                  />
                  <YAxis
                    yAxisId="left"
                    orientation="left"
                    tick={{ fill: "hsl(var(--foreground))" }}
                    axisLine={{ stroke: "hsl(var(--foreground))" }}
                    tickFormatter={(value: number) => {
                      if (value >= 1e6) return `${formatWithPrecision((value / 1e6), defaultDecimalPrecisions)}M`;
                      if (value >= 1e3) return `${formatWithPrecision((value / 1e3), defaultDecimalPrecisions)}K`;
                      return formatWithPrecision(value, defaultDecimalPrecisions);
                    }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="hsl(var(--destructive))"
                    tick={{ fill: "hsl(var(--destructive))" }}
                    axisLine={{ stroke: "hsl(var(--destructive))" }}
                    tickFormatter={(value: number) => {
                      if (value >= 1e6) return `${formatWithPrecision((value / 1e6), defaultDecimalPrecisions)}M`;
                      if (value >= 1e3) return `${formatWithPrecision((value / 1e3), defaultDecimalPrecisions)}K`;
                      return formatWithPrecision(value, defaultDecimalPrecisions);
                    }}
                  />
                 <Tooltip
                    cursor={false}
                    content={({ active, payload, label }) => {
                      if (!active || !payload || payload.length === 0) return null;

                      return (
                        <div className="p-2 rounded-lg border text-sm bg-background text-foreground border-border">
                          <div className="font-medium mb-1">{label}</div>
                          {payload.map((entry: any) => (
                            <div key={entry.name} className="flex items-center gap-2">
                              <div
                                style={{
                                  width: 10,
                                  height: 10,
                                  backgroundColor: entry.fill,
                                  borderRadius: 2,
                                }}
                              />
                              <span>{entry.name}</span>
                              <span className="font-bold">
                                {entry.value >= 1e6
                                  ? `${formatWithPrecision((entry.value / 1e6), defaultDecimalPrecisions)}M`
                                  : entry.value >= 1e3
                                  ? `${formatWithPrecision((entry.value / 1e3), defaultDecimalPrecisions)}K`
                                  : formatWithPrecision(entry.value, defaultDecimalPrecisions)}
                              </span>
                            </div>
                          ))}
                        </div>
                      );
                    }}
                  />
                 <Legend
                    wrapperStyle={{ color: "hsl(var(--foreground))", marginBottom: -10 }}
                    formatter={(value: string) => {
                      const trace = traces.find(t => t.value === value);
                      return <span style={{ color: trace?.color || "#888" }}>{value}</span>;
                    }}
                  />

                  {traces.map(({ value, color, source }) =>
                    selectedTraces.includes(value) ? (
                      <Bar
                        key={value}
                        dataKey={value}
                        name={value}
                        fill={color}
                        activeBar={false}
                        yAxisId={source === "parameter" ? "left" : "right"} // optional axis condition
                      />
                    ) : null
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {activeTab === "mape" && (
          <div className="p-4">
             <div className="space-y-6">
              <h2 className="text-xl font-semibold mt-0">Automated MAPE Insight</h2>

              {/* Section A */}
              <Card className="p-2">
                <h3 className="text-sm mb-2">
                  A. MAPE Insight Dashboard (Last 12 months)
                </h3>
                <Table>
                  <TableHeader>
                    <TableRow className="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200">
                      <TableHead>LOB</TableHead>
                      <TableHead>Current Month</TableHead>
                      <TableHead>12 Mths Avg.</TableHead>
                      <TableHead>Target (+/-)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataA.map((row) => (
                      <TableRow key={row.lob}>
                        <TableCell>{row.lob}</TableCell>
                        <TableCell>{row.current}</TableCell>
                        <TableCell>{row.avg}</TableCell>
                        <TableCell>{row.target}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Card>

              {/* Section B */}
              <Card className="p-2">
                <h3 className="text-sm mb-3">
                  B. MAPE Trend Visualization (Line Chart)
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={mapeTrend} margin={{ top: 80, right: 30, left: 20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" stroke="currentColor" />
                    <YAxis stroke="currentColor" domain={[0, 'dataMax + 1']}>
                      <Label
                        angle={-90}
                        position="insideLeft"
                        style={{ textAnchor: 'middle' }}
                      >
                        MAPE (%)
                      </Label>
                    </YAxis>
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Legend />
                    <ReferenceLine y={5} stroke="red" strokeDasharray="5 5" label="Target" />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#4f46e5"
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                      name="MAPE"
                    />
                    {mapeTrend.map((d, i) =>
                      d.event ? (
                        <ReferenceLine
                          key={i}
                          x={d.month}
                          stroke="#8884d8"
                          strokeDasharray="3 3"
                          label={{ value: d.event, angle: -90, position: 'top', fill: '#8884d8' }}
                        />
                      ) : null
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              {/* Section C */}
              <Card className="p-2">
                <h3 className="text-sm mb-2">
                  C. MAPE by Category / Function (Drilldown View)
                </h3>
                <Table>
                  <TableHeader>
                    <TableRow className="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200">
                      <TableHead>Function / Segment</TableHead>
                      <TableHead>Current MAPE</TableHead>
                      <TableHead>12 Mths Avg.</TableHead>
                      <TableHead>Notes</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataC.map((row) => (
                      <TableRow key={row.function}>
                        <TableCell>{row.function}</TableCell>
                        <TableCell>{row.current}</TableCell>
                        <TableCell>{row.avg}</TableCell>
                        <TableCell>{row.notes}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Card>

              {/* Section D */}
              <Card className="p-2 mb-2">
                <h3 className="text-sm mb-2">D. Root Cause</h3>
                <Table>
                  <TableHeader>
                    <TableRow className="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200">
                      <TableHead>Month</TableHead>
                      <TableHead>Spike Cause Identified</TableHead>
                      <TableHead>Action Taken</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataD.map((row) => (
                      <TableRow key={row.month}>
                        <TableCell>{row.month}</TableCell>
                        <TableCell>{row.cause}</TableCell>
                        <TableCell>{row.action}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Card>
            </div>
          </div>
        )}
        </>
      )}
      </CardContent>
    </Card>
    
  );
};
