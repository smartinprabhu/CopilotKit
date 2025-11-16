import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ComposedChart, Area, AreaChart, Legend, ReferenceLine, Cell } from 'recharts';
import { Calendar, Users, TrendingUp, TrendingDown, Clock, ArrowRight, Target, Zap, RefreshCw, Settings } from "lucide-react";
import { AVAILABLE_MODELS } from "../../tacticalPlan/models/shared/constants";
import { ALL_BUSINESS_UNITS, BUSINESS_UNIT_CONFIG } from "../../tacticalPlan/planComponents/capacity-insights/types";
import type { ModelType } from "../../tacticalPlan/models/shared/interfaces";

// SummaryCard component matching CapacityLook format
type SummaryCardProps = {
  title: string;
  color: string;
  value: number;
  trendData?: { value: number }[];
};

const SummaryCard = ({ title, value, color, trendData = [] }: SummaryCardProps) => (
  <Card className="bg-background text-black dark:text-white p-4 w-full shadow-lg">
    <CardContent className="flex flex-col gap-2">
      <div>
        <span className="text-sm text-gray-400">{title}</span>
        <div className="text-2xl font-semibold">{value}</div>
      </div>
      {/* Tiny Line Chart (Sparkline) */}
      <div className="h-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trendData}>
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={false}
            />
            <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </CardContent>
  </Card>
);

interface TacticalLandingPageProps {
  onStartPlanning: () => void;
  planData?: any;
  onDataChange?: (data: any) => void;
  selectedModel?: ModelType;
  onModelChange?: (model: ModelType) => void;
}

const TacticalLandingPage: React.FC<TacticalLandingPageProps> = ({
  onStartPlanning,
  planData,
  onDataChange,
  selectedModel = 'volume-backlog',
  onModelChange
}) => {
  const [selectedWeeks, setSelectedWeeks] = useState("12");
  const [selectedBU, setSelectedBU] = useState<string>("POS"); // Default to POS
  const [selectedLOBs, setSelectedLOBs] = useState<string[]>(["Chat"]); // Default to Chat, multi-select


  // Get actual business units (remove "All" option)
  const businessUnits = ALL_BUSINESS_UNITS;

  // Get LOBs based on selected BU
  const getAvailableLOBs = () => {
    if (selectedBU in BUSINESS_UNIT_CONFIG) {
      return BUSINESS_UNIT_CONFIG[selectedBU as keyof typeof BUSINESS_UNIT_CONFIG].lonsOfBusiness;
    }
    return [];
  };

  const availableLOBs = getAvailableLOBs();

  // Helper function to get current fiscal week and generate next weeks from current date
  const getCurrentFiscalWeekAndGenerateNext = (numWeeks: number) => {
    const today = new Date();
    const currentYear = today.getFullYear();

    // Function to determine if year is leap
    const isLeapYear = (year: number) => (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;

    // Function to get fiscal year start date
    const getFiscalYearStart = (year: number) => {
      const isLeap = isLeapYear(year);
      let fiscalYearStart: Date;

      if (isLeap) {
        const feb1st = new Date(Date.UTC(year, 1, 1)); // February 1st
        let dayOfWeek = feb1st.getUTCDay();
        dayOfWeek = (dayOfWeek === 0) ? 6 : dayOfWeek - 1; // Monday = 0, Sunday = 6
        fiscalYearStart = new Date(Date.UTC(year, 1, 1 - dayOfWeek));
      } else {
        const jan22nd = new Date(Date.UTC(year, 0, 22)); // January 22nd
        let dayOfWeek = jan22nd.getUTCDay();
        dayOfWeek = (dayOfWeek === 0) ? 6 : dayOfWeek - 1;
        fiscalYearStart = new Date(Date.UTC(year, 0, 22 - dayOfWeek));
      }

      return fiscalYearStart;
    };

    // Find current fiscal week
    let currentFiscalYear = currentYear;
    let fiscalYearStart = getFiscalYearStart(currentFiscalYear);

    // If today is before this year's fiscal start, use previous year
    if (today < fiscalYearStart) {
      currentFiscalYear = currentYear - 1;
      fiscalYearStart = getFiscalYearStart(currentFiscalYear);
    }

    // Calculate current week number
    const daysDiff = Math.floor((today.getTime() - fiscalYearStart.getTime()) / (1000 * 60 * 60 * 24));
    const currentFiscalWeekNumber = Math.floor(daysDiff / 7) + 1;

    // Generate next weeks starting from current week
    const weeks = [];
    const startOfCurrentWeek = new Date(fiscalYearStart);
    startOfCurrentWeek.setUTCDate(fiscalYearStart.getUTCDate() + (currentFiscalWeekNumber - 1) * 7);

    for (let i = 0; i < numWeeks; i++) {
      const weekStartDate = new Date(startOfCurrentWeek);
      weekStartDate.setUTCDate(startOfCurrentWeek.getUTCDate() + i * 7);

      const weekEndDate = new Date(weekStartDate);
      weekEndDate.setUTCDate(weekStartDate.getUTCDate() + 6);

      // Format dates as "03 Nov - 09 Nov" for hover only
      const formatDateForHover = (date: Date): string => {
        const day = date.getUTCDate().toString().padStart(2, '0');
        const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
        return `${day} ${month}`;
      };

      const actualFiscalWeek = currentFiscalWeekNumber + i;
      const hoverDateRange = `${formatDateForHover(weekStartDate)} - ${formatDateForHover(weekEndDate)}`;

      weeks.push({
        fiscalWeek: actualFiscalWeek,
        weekStartDate,
        weekEndDate,
        hoverDateRange: hoverDateRange,
        year: weekStartDate.getUTCFullYear()
      });
    }

    return weeks;
  };

  // Generate data based on actual tactical planning structure
  const generateWeeklyCapacityData = () => {
    const weeks = [];
    const numWeeks = parseInt(selectedWeeks);
    const fiscalWeeks = getCurrentFiscalWeekAndGenerateNext(numWeeks);

    for (let i = 0; i < numWeeks && i < fiscalWeeks.length; i++) {
      const fiscalWeek = fiscalWeeks[i];

      // Show actual data for most weeks to have good comparison with required
      // This gives us actual data through most of the planning period
      const isWeekInPast = i < Math.min(numWeeks - 2, Math.floor(numWeeks * 0.8)); // Show actual for 80% of weeks, leave last 20% as future

      // Simulate data based on model type and filters
      const baseRequired = selectedModel === 'fix-fte' ? 80 : 120;
      const modelMultiplier = selectedModel === 'billable-hours' ? 0.8 : 1.0;

      const required = Math.round((baseRequired + Math.sin((i + 1) * 0.5) * 20) * modelMultiplier);
      const planned = Math.round(required * (0.95 + Math.random() * 0.1));
      const actual = isWeekInPast ? Math.round(planned * (0.98 + Math.random() * 0.04)) : null; // Only past weeks have actual

      weeks.push({
        week: `WK${fiscalWeek.fiscalWeek}`, // Simple week number for X-axis
        weekLabel: `WK${fiscalWeek.fiscalWeek}`, // Clean X-axis label
        dateRange: fiscalWeek.hoverDateRange, // "03 Nov - 09 Nov" format for hover
        required,
        planned,
        actual,
        overUnder: (actual || planned) - required,
        isPast: isWeekInPast,
        isFuture: !isWeekInPast
      });
    }
    return weeks;
  };

  const generateAttritionData = () => {
    const data = [];

    // Get last 8 weeks (historical) - going backwards from current week
    const historicalWeeks = getCurrentFiscalWeekAndGenerateNext(1); // Get current week info
    const currentWeekStart = historicalWeeks[0].weekStartDate;

    // Generate last 8 weeks (going backwards)
    for (let i = 7; i >= 0; i--) {
      const weekStartDate = new Date(currentWeekStart);
      weekStartDate.setUTCDate(currentWeekStart.getUTCDate() - (i * 7));

      const weekEndDate = new Date(weekStartDate);
      weekEndDate.setUTCDate(weekStartDate.getUTCDate() + 6);

      // Calculate fiscal week number for this historical week
      const daysDiff = Math.floor((weekStartDate.getTime() - historicalWeeks[0].weekStartDate.getTime()) / (1000 * 60 * 60 * 24));
      const fiscalWeekNumber = historicalWeeks[0].fiscalWeek + Math.floor(daysDiff / 7);

      const formatDateForHover = (date: Date): string => {
        const day = date.getUTCDate().toString().padStart(2, '0');
        const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
        return `${day} ${month}`;
      };

      const formatDatePart = (date: Date): string =>
        `${(date.getUTCMonth() + 1).toString().padStart(2, '0')}/${date.getUTCDate().toString().padStart(2, '0')}`;

      const baseRate = selectedModel === 'fix-hc' ? 6 : 8;

      data.push({
        period: `WK${fiscalWeekNumber}`,
        periodLabel: `WK${fiscalWeekNumber}(${formatDatePart(weekStartDate)}-${formatDatePart(weekEndDate)})`,
        dateRange: `${formatDateForHover(weekStartDate)} - ${formatDateForHover(weekEndDate)}`,
        type: 'Historical',
        attritionRate: Number((baseRate + Math.random() * 4).toFixed(2)),
        count: Math.round(15 + Math.random() * 10),
        isPast: true
      });
    }

    // Coming 4 weeks (planned) - starting from current week
    const futureWeeks = getCurrentFiscalWeekAndGenerateNext(4);
    for (let i = 0; i < 4 && i < futureWeeks.length; i++) {
      const fiscalWeek = futureWeeks[i];
      const baseRate = selectedModel === 'fix-hc' ? 5 : 6;

      data.push({
        period: `WK${fiscalWeek.fiscalWeek}`,
        periodLabel: `WK${fiscalWeek.fiscalWeek}(${fiscalWeek.dateRange})`,
        dateRange: fiscalWeek.hoverDateRange,
        type: 'Planned',
        attritionRate: Number((baseRate + Math.random() * 3).toFixed(2)),
        count: Math.round(10 + Math.random() * 8),
        isPast: false
      });
    }
    return data;
  };

  const generateAHTData = () => {
    const data = [];
    const baseAHT = selectedLOBs.some(lob => lob.includes('Chat')) ? 8 : 12;

    // Get last 8 weeks (historical) - going backwards from current week
    const historicalWeeks = getCurrentFiscalWeekAndGenerateNext(1); // Get current week info
    const currentWeekStart = historicalWeeks[0].weekStartDate;

    // Generate last 8 weeks (going backwards)
    for (let i = 7; i >= 0; i--) {
      const weekStartDate = new Date(currentWeekStart);
      weekStartDate.setUTCDate(currentWeekStart.getUTCDate() - (i * 7));

      const weekEndDate = new Date(weekStartDate);
      weekEndDate.setUTCDate(weekStartDate.getUTCDate() + 6);

      // Calculate fiscal week number for this historical week
      const daysDiff = Math.floor((weekStartDate.getTime() - historicalWeeks[0].weekStartDate.getTime()) / (1000 * 60 * 60 * 24));
      const fiscalWeekNumber = historicalWeeks[0].fiscalWeek + Math.floor(daysDiff / 7);

      const formatDateForHover = (date: Date): string => {
        const day = date.getUTCDate().toString().padStart(2, '0');
        const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
        return `${day} ${month}`;
      };

      const formatDatePart = (date: Date): string =>
        `${(date.getUTCMonth() + 1).toString().padStart(2, '0')}/${date.getUTCDate().toString().padStart(2, '0')}`;

      data.push({
        period: `WK${fiscalWeekNumber}`,
        periodLabel: `WK${fiscalWeekNumber}(${formatDatePart(weekStartDate)}-${formatDatePart(weekEndDate)})`,
        dateRange: `${formatDateForHover(weekStartDate)} - ${formatDateForHover(weekEndDate)}`,
        type: 'Actual',
        aht: Number((baseAHT + Math.random() * 2).toFixed(2)),
        target: baseAHT,
        isPast: true
      });
    }

    // Coming 4 weeks (planned) - starting from current week
    const futureWeeks = getCurrentFiscalWeekAndGenerateNext(4);
    for (let i = 0; i < 4 && i < futureWeeks.length; i++) {
      const fiscalWeek = futureWeeks[i];

      data.push({
        period: `WK${fiscalWeek.fiscalWeek}`,
        periodLabel: `WK${fiscalWeek.fiscalWeek}(${fiscalWeek.dateRange})`,
        dateRange: fiscalWeek.hoverDateRange,
        type: 'Planned',
        aht: Number((baseAHT - 0.5 + Math.random() * 1).toFixed(2)),
        target: baseAHT,
        isPast: false
      });
    }
    return data;
  };

  const generateNewHireData = () => {
    const data = [];
    const numWeeks = parseInt(selectedWeeks);
    const fiscalWeeks = getCurrentFiscalWeekAndGenerateNext(numWeeks);

    for (let i = 0; i < numWeeks && i < fiscalWeeks.length; i++) {
      const fiscalWeek = fiscalWeeks[i];
      const hasClass = Math.random() > 0.4;
      const baseSize = selectedModel === 'billable-hours' ? 5 : 12;

      data.push({
        week: `WK${fiscalWeek.fiscalWeek}`,
        weekLabel: `WK${fiscalWeek.fiscalWeek}(${fiscalWeek.dateRange})`,
        dateRange: fiscalWeek.hoverDateRange, // Use "02 Jan - 08 Jan" format
        newHires: hasClass ? Math.round(baseSize + Math.random() * 10) : 0,
        classSize: hasClass ? Math.round(15 + Math.random() * 10) : 0,
        startDate: fiscalWeek.weekStartDate.toISOString().split('T')[0] // YYYY-MM-DD format
      });
    }
    return data;
  };

  // Generate data based on current selections
  const weeklyCapacityData = generateWeeklyCapacityData();
  const attritionData = generateAttritionData();
  const ahtData = generateAHTData();
  const newHireData = generateNewHireData();

  // Calculate KPIs
  const calculateKPIs = () => {
    const totalRequired = weeklyCapacityData.reduce((sum, week) => sum + week.required, 0);
    const totalPlanned = weeklyCapacityData.reduce((sum, week) => sum + week.planned, 0);
    const actualWeeks = weeklyCapacityData.filter(w => w.actual !== null);
    const totalActual = actualWeeks.reduce((sum, week) => sum + (week.actual || 0), 0);
    const avgAttrition = attritionData.filter(d => d.isPast).reduce((sum, d) => sum + d.attritionRate, 0) / 8;
    const avgAHT = ahtData.filter(d => d.isPast).reduce((sum, d) => sum + d.aht, 0) / 8;
    const weeksOverStaffed = weeklyCapacityData.filter(w => w.overUnder > 0).length;
    const weeksUnderStaffed = weeklyCapacityData.filter(w => w.overUnder < 0).length;

    return {
      avgRequiredHC: Math.round(totalRequired / weeklyCapacityData.length),
      avgPlannedHC: Math.round(totalPlanned / weeklyCapacityData.length),
      avgActualHC: actualWeeks.length > 0 ? Math.round(totalActual / actualWeeks.length) : 0,
      totalVariance: totalPlanned - totalRequired,
      avgAttritionRate: avgAttrition,
      avgAHT: avgAHT,
      weeksOverStaffed,
      weeksUnderStaffed
    };
  };

  const kpis = calculateKPIs();

  const kpiData = [
    { title: "Avg Required HC", value: kpis.avgRequiredHC, target: "120", status: "neutral", icon: Users },
    { title: "Avg Actual HC", value: kpis.avgActualHC, target: "115", status: kpis.avgActualHC > 115 ? "above" : "below", icon: Target },
    { title: "Attrition Rate", value: `${kpis.avgAttritionRate.toFixed(1)}%`, target: "8%", status: kpis.avgAttritionRate > 8 ? "above" : "below", icon: TrendingUp },
    { title: "Average AHT", value: `${kpis.avgAHT.toFixed(1)}m`, target: "12m", status: kpis.avgAHT < 12 ? "below" : "above", icon: Clock },
  ];

  return (
    <div className="min-h-screen bg-background p-3">


      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <Zap className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-semibold text-foreground">Tactical Capacity Plan Insights</h1>
        </div>
      </div>

      {/* Filter Controls */}
      <Card className="mb-4">
        <CardHeader className="pb-4">
          <CardTitle className="text-md">Configuration Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-6 gap-6 items-center">
            <div className="">
              <Label className="text-xs text-foreground">Model</Label>
              <Select value={selectedModel} onValueChange={onModelChange}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {AVAILABLE_MODELS.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="">
              <Label className="text-xs text-foreground">Time Period</Label>
              <Select value={selectedWeeks} onValueChange={setSelectedWeeks}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="4">4 Weeks</SelectItem>
                  <SelectItem value="8">8 Weeks</SelectItem>
                  <SelectItem value="12">12 Weeks</SelectItem>
                  <SelectItem value="16">16 Weeks</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="">
              <Label className="text-xs text-foreground">Business Unit</Label>
              <Select value={selectedBU} onValueChange={(value) => {
                setSelectedBU(value);
                setSelectedLOBs(["Chat"]); // Reset LOBs when BU changes
              }}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {businessUnits.map(bu => (
                    <SelectItem key={bu} value={bu}>{bu}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="">
              <Label className="text-xs text-foreground">LOB</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start">
                    {selectedLOBs.length === 0
                      ? "Select LOBs..."
                      : selectedLOBs.length === 1
                        ? selectedLOBs[0]
                        : `${selectedLOBs.length} LOBs selected`
                    }
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-2">
                  <div className="space-y-2">
                    {availableLOBs.map(lob => (
                      <div key={lob} className="flex items-center space-x-2">
                        <Checkbox
                          id={lob}
                          checked={selectedLOBs.includes(lob)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedLOBs([...selectedLOBs, lob]);
                            } else {
                              setSelectedLOBs(selectedLOBs.filter(l => l !== lob));
                            }
                          }}
                        />
                        <label htmlFor={lob} className="text-sm font-medium">
                          {lob}
                        </label>
                      </div>
                    ))}
                  </div>
                </PopoverContent>
              </Popover>
            </div>

            <div className="mt-3">
              <Badge variant="outline" className="border-primary text-primary">
                {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.complexity} Complexity
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>


      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        {kpiData.map((kpi, index) => {
          const IconComponent = kpi.icon;
          return (
            <Card key={index} className="executive-card shadow-sm">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">{kpi.title}</p>
                    <p className="text-3xl font-bold">{kpi.value}</p>
                    <p className="text-xs text-muted-foreground">Target: {kpi.target}</p>
                  </div>
                  <div className="flex flex-col items-end">
                    <IconComponent className="h-8 w-8 mb-2" />
                    {kpi.status && (
                      <div className={`flex items-center gap-1 ${kpi.status === 'above' ? 'text-green-400' : 'text-red-400'}`}>
                        {kpi.status === 'above' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="space-y-6">
        {/* Main Capacity Planning Chart - Matching CapacityLook Format */}
        <div className="executive-card shadow-sm p-3">
          <h2 className="text-md font-semibold text-foreground flex items-center justify-between">
            <span>Headcount Planning - Next {selectedWeeks} Weeks ({selectedBU} - {selectedLOBs.join(', ')})</span>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                {kpis.weeksOverStaffed} weeks over-staffed
              </Badge>
              <Badge variant="outline">
                {kpis.weeksUnderStaffed} weeks under-staffed
              </Badge>
              <Badge variant="outline" className="border-primary text-primary">
                {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}
              </Badge>
            </div>
          </h2>

          <div className="flex flex-col lg:flex-row gap-4 items-start">
            {/* Summary Cards - 20% width on large screens */}
            <div className="w-full lg:w-1/5 space-y-4 mt-2">
              <SummaryCard
                title="Required HC"
                color="#f97316"
                value={kpis.avgRequiredHC}
                trendData={weeklyCapacityData.map((d) => ({ value: d.required }))}
              />
              <SummaryCard
                title="Actual HC"
                color="#22c55e"
                value={kpis.avgActualHC}
                trendData={weeklyCapacityData.filter(d => d.actual !== null).map((d) => ({ value: d.actual || 0 }))}
              />
              <SummaryCard
                title="Over/Under HC"
                color="#999"
                value={kpis.totalVariance}
                trendData={weeklyCapacityData.map((d) => ({ value: d.overUnder }))}
              />
            </div>

            {/* Chart - 80% width on large screens */}
            <div className="w-full lg:w-4/5 space-y-4">
              <h2 className="text-sm font-semibold text-foreground pl-10">
                Required Vs Actual Trend
              </h2>
              <div className="w-full h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={weeklyCapacityData}
                    margin={{ top: 20, right: 20, bottom: 20, left: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="weekLabel" stroke="#ccc" />
                    <YAxis yAxisId="left" stroke="#ccc" />
                    <YAxis yAxisId="right" orientation="right" stroke="#ccc" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#333', border: 'none' }}
                      labelFormatter={(label, payload) => {
                        const data = payload?.[0]?.payload;
                        return data?.dateRange ? `${label} (${data.dateRange})` : label;
                      }}
                    />
                    <Legend />
                    <ReferenceLine
                      y={0}
                      yAxisId="right"
                      stroke="red"
                      strokeDasharray="3 3"
                      strokeWidth={1}
                    />
                    <Line yAxisId="left" type="monotone" dataKey="required" stroke="#f97316" strokeWidth={2} name="Required" />
                    {weeklyCapacityData.some(d => d.actual) && (
                      <Line yAxisId="left" type="monotone" dataKey="actual" stroke="#22c55e" strokeWidth={2} name="Actual" />
                    )}
                    <Bar yAxisId="right" dataKey="overUnder" barSize={20} fill="#999" name="O/U">
                      {weeklyCapacityData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.overUnder >= 0 ? '#008001CA' : '#ff0200CA'}
                        />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>


        {/* New Hire Classes Chart */}
        <div className="executive-card shadow-sm p-3">
          <h2 className="text-md font-semibold text-foreground">
            New Hire Classes - Coming 12 Weeks ({selectedBU} - {selectedLOBs.join(', ')})
          </h2>
          <div className="w-full h-80 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={newHireData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="week" stroke="#ccc" />
                <YAxis stroke="#ccc" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#333', border: 'none' }}
                  labelFormatter={(label, payload) => {
                    const data = payload?.[0]?.payload;
                    return data?.dateRange ? `${label} (${data.dateRange})` : label;
                  }}
                />
                <Legend />
                <Bar dataKey="newHires" fill="#22c55e" name="New Hires" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>


        {/* Attrition and AHT Charts Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Attrition Rate Chart */}
          <div className="executive-card shadow-sm p-3">
            <h2 className="text-md font-semibold text-foreground">
              Attrition Rate - Last 8 Weeks vs Coming 4 Weeks ({selectedBU})
            </h2>
            <div className="w-full h-80 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={attritionData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                  <defs>
                    <linearGradient id="attritionGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="period" stroke="#ccc" />
                  <YAxis stroke="#ccc" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#333', border: 'none' }}
                    formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Attrition Rate']}
                    labelFormatter={(label, payload) => {
                      const data = payload?.[0]?.payload;
                      return data?.dateRange ? `${label} (${data.dateRange})` : label;
                    }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="attritionRate"
                    stroke="#f97316"
                    fillOpacity={1}
                    fill="url(#attritionGradient)"
                    name="Attrition Rate %"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AHT Chart */}
          <div className="executive-card shadow-sm p-3">
            <h2 className="text-md font-semibold text-foreground">
              Average Handle Time - Last 8 Weeks vs Coming 4 Weeks ({selectedLOBs.join(', ')})
            </h2>
            <div className="w-full h-80 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ahtData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="period" stroke="#ccc" />
                  <YAxis stroke="#ccc" domain={['dataMin - 0.5', 'dataMax + 0.5']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#333', border: 'none' }}
                    formatter={(value, name) => [`${Number(value).toFixed(2)} min`, name]}
                    labelFormatter={(label, payload) => {
                      const data = payload?.[0]?.payload;
                      return data?.dateRange ? `${label} (${data.dateRange})` : label;
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="aht" stroke="#3b82f6" strokeWidth={3} name="Actual AHT" />
                  <Line type="monotone" dataKey="target" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Target AHT" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>


        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-chart-blue">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Calendar className="h-6 w-6" />
                <h3 className="font-semibold text-foreground">Flexible Scheduling</h3>
              </div>
              <p className="text-foreground text-sm">
                Plan workforce schedules with drag-and-drop interface and real-time capacity adjustments.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-chart-blue">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Users className="h-6 w-6" />
                <h3 className="font-semibold text-foreground">Skill-Based Planning</h3>
              </div>
              <p className="text-foreground text-sm">
                Match agent skills with demand requirements for optimal service delivery.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-chart-blue">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <TrendingUp className="h-6 w-6" />
                <h3 className="font-semibold text-foreground">Real-time Analytics</h3>
              </div>
              <p className="text-foreground text-sm">
                Monitor performance metrics and adjust plans based on live operational data.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Action Button */}
        <div className="text-center pt-6">
          <Button
            onClick={onStartPlanning}
            size="lg"
            className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 text-lg font-medium"
          >
            Start Planning
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default TacticalLandingPage;