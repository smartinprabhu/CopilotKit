import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ComposedChart, Area, AreaChart, Legend, ReferenceLine, Cell } from 'recharts';
import { Calendar, Users, TrendingUp, TrendingDown, Clock, ArrowRight, Target, RefreshCw } from "lucide-react";
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

interface StrategicLandingPageProps {
  onStartPlanning: (planningType?: "annual" | "monthly") => void;
  planData?: any;
  onDataChange?: (data: any) => void;
  selectedModel?: ModelType;
  onModelChange?: (model: ModelType) => void;
}

const StrategicLandingPage: React.FC<StrategicLandingPageProps> = ({
  onStartPlanning,
  planData,
  onDataChange,
  selectedModel = 'volume-backlog',
  onModelChange
}) => {
  const [selectedMonths, setSelectedMonths] = useState("12");
  const [selectedBU, setSelectedBU] = useState<string>("POS"); // Default to POS
  const [selectedLOBs, setSelectedLOBs] = useState<string[]>(["Chat"]); // Default to Chat, multi-select
  const [planningType, setPlanningType] = useState<"annual" | "monthly">("annual"); // New toggle

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

  // Generate capacity data based on planning type
  const generateCapacityData = () => {
    const periods = [];
    const numPeriods = parseInt(selectedMonths);
    const currentYear = 2025; // Start from 2025 fiscal year

    if (planningType === 'annual') {
      // Annual planning shows monthly data
      const startMonth = 1; // January

      for (let i = 0; i < numPeriods; i++) {
        const monthDate = new Date(currentYear, startMonth - 1 + i, 1);
        const monthEndDate = new Date(currentYear, startMonth + i, 0);

        // Show actual data for first 8-10 months to have good comparison with required
        // This gives us actual data through most of the planning period
        const isMonthInPast = i < Math.min(10, numPeriods - 2); // Show actual for most periods, leave last 2 as future

        const monthLabel = `${monthDate.toLocaleDateString('en-US', { month: 'short' })} ${monthDate.getFullYear()}`;
        const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} (${monthDate.getFullYear()})`;

        // Higher numbers for monthly aggregation
        const baseRequired = selectedModel === 'fix-fte' ? 320 : 480;
        const modelMultiplier = selectedModel === 'billable-hours' ? 0.8 : 1.0;

        const required = Math.round((baseRequired + Math.sin((i + 1) * 0.3) * 80) * modelMultiplier);
        const planned = Math.round(required * (0.95 + Math.random() * 0.1));
        const actual = isMonthInPast ? Math.round(planned * (0.98 + Math.random() * 0.04)) : null;

        periods.push({
          month: `M${i + 1}`,
          monthLabel: monthLabel,
          dateRange: fullDateRange,
          required,
          planned,
          actual,
          overUnder: (actual || planned) - required,
          isPast: isMonthInPast,
          isFuture: !isMonthInPast
        });
      }
    } else {
      // Monthly planning shows weekly data (like tactical plan)
      const startWeek = 1;

      for (let i = 0; i < numPeriods; i++) {
        const weekStartDate = new Date(currentYear, 0, 1 + (i * 7)); // Start from Jan 1st, add weeks
        const weekEndDate = new Date(weekStartDate);
        weekEndDate.setDate(weekStartDate.getDate() + 6);

        // Show actual data for most weeks to have good comparison with required
        // This gives us actual data through most of the planning period
        const isWeekInPast = i < Math.min(numPeriods - 3, numPeriods * 0.75); // Show actual for 75% of periods, leave last 25% as future

        const weekLabel = `Wk ${i + 1}`;
        const fullDateRange = `${weekStartDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${weekEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} (${weekStartDate.getFullYear()})`;

        // Lower numbers for weekly data
        const baseRequired = selectedModel === 'fix-fte' ? 80 : 120;
        const modelMultiplier = selectedModel === 'billable-hours' ? 0.8 : 1.0;

        const required = Math.round((baseRequired + Math.sin((i + 1) * 0.5) * 20) * modelMultiplier);
        const planned = Math.round(required * (0.95 + Math.random() * 0.1));
        const actual = isWeekInPast ? Math.round(planned * (0.98 + Math.random() * 0.04)) : null;

        periods.push({
          month: `W${i + 1}`,
          monthLabel: weekLabel,
          dateRange: fullDateRange,
          required,
          planned,
          actual,
          overUnder: (actual || planned) - required,
          isPast: isWeekInPast,
          isFuture: !isWeekInPast
        });
      }
    }
    return periods;
  };

  const generateAttritionData = () => {
    const data = [];
    const today = new Date();

    // Last 2 months (historical)
    for (let i = -1; i <= 0; i++) {
      const monthDate = new Date(today.getFullYear(), today.getMonth() + i, 1);
      const monthEndDate = new Date(today.getFullYear(), today.getMonth() + i + 1, 0);

      const dateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short' })} ${monthDate.getFullYear()}`;
      const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

      const baseRate = selectedModel === 'fix-hc' ? 6 : 8;
      data.push({
        period: `M${2 + i}`,
        periodLabel: dateRange,
        dateRange: fullDateRange,
        type: 'Historical',
        attritionRate: Number((baseRate + Math.random() * 4).toFixed(2)),
        count: Math.round(60 + Math.random() * 40), // Higher numbers for monthly
        isPast: true
      });
    }
    // Coming 2 months (planned)
    for (let i = 1; i <= 2; i++) {
      const monthDate = new Date(today.getFullYear(), today.getMonth() + i, 1);
      const monthEndDate = new Date(today.getFullYear(), today.getMonth() + i + 1, 0);

      const dateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short' })} ${monthDate.getFullYear()}`;
      const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

      const baseRate = selectedModel === 'fix-hc' ? 5 : 6;
      data.push({
        period: `M${2 + i}`,
        periodLabel: dateRange,
        dateRange: fullDateRange,
        type: 'Planned',
        attritionRate: Number((baseRate + Math.random() * 3).toFixed(2)),
        count: Math.round(40 + Math.random() * 32), // Lower planned attrition
        isPast: false
      });
    }
    return data;
  };

  const generateAHTData = () => {
    const data = [];
    const baseAHT = selectedLOBs.some(lob => lob.includes('Chat')) ? 8 : 12;
    const today = new Date();

    // Last 2 months (actual)
    for (let i = -1; i <= 0; i++) {
      const monthDate = new Date(today.getFullYear(), today.getMonth() + i, 1);
      const monthEndDate = new Date(today.getFullYear(), today.getMonth() + i + 1, 0);

      const dateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short' })} ${monthDate.getFullYear()}`;
      const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

      data.push({
        period: `M${2 + i}`,
        periodLabel: dateRange,
        dateRange: fullDateRange,
        type: 'Actual',
        aht: Number((baseAHT + Math.random() * 2).toFixed(2)),
        target: baseAHT,
        isPast: true
      });
    }
    // Coming 2 months (planned)
    for (let i = 1; i <= 2; i++) {
      const monthDate = new Date(today.getFullYear(), today.getMonth() + i, 1);
      const monthEndDate = new Date(today.getFullYear(), today.getMonth() + i + 1, 0);

      const dateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short' })} ${monthDate.getFullYear()}`;
      const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

      data.push({
        period: `M${2 + i}`,
        periodLabel: dateRange,
        dateRange: fullDateRange,
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
    const today = new Date();
    const numMonths = parseInt(selectedMonths);

    for (let i = 1; i <= numMonths; i++) {
      const monthDate = new Date(today.getFullYear(), today.getMonth() + (i - 1), 1);
      const monthEndDate = new Date(today.getFullYear(), today.getMonth() + i, 0);

      const dateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}`;
      const fullDateRange = `${monthDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${monthEndDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

      const hasClass = Math.random() > 0.3; // 70% chance of having a class for monthly
      const baseSize = selectedModel === 'billable-hours' ? 20 : 48; // Higher numbers for monthly

      data.push({
        month: `M${i}`,
        monthLabel: dateRange,
        dateRange: fullDateRange,
        newHires: hasClass ? Math.round(baseSize + Math.random() * 40) : 0,
        classSize: hasClass ? Math.round(60 + Math.random() * 40) : 0,
        startDate: monthDate.toISOString().split('T')[0] // YYYY-MM-DD format
      });
    }
    return data;
  };

  // Generate data based on current selections
  const capacityData = generateCapacityData();
  const attritionData = generateAttritionData();
  const ahtData = generateAHTData();
  const newHireData = generateNewHireData();

  // Calculate KPIs
  const calculateKPIs = () => {
    const totalRequired = capacityData.reduce((sum, period) => sum + period.required, 0);
    const totalPlanned = capacityData.reduce((sum, period) => sum + period.planned, 0);
    const actualPeriods = capacityData.filter(p => p.actual !== null);
    const totalActual = actualPeriods.reduce((sum, period) => sum + (period.actual || 0), 0);
    const avgAttrition = attritionData.filter(d => d.isPast).reduce((sum, d) => sum + d.attritionRate, 0) / 2;
    const avgAHT = ahtData.filter(d => d.isPast).reduce((sum, d) => sum + d.aht, 0) / 2;
    const periodsOverStaffed = capacityData.filter(p => p.overUnder > 0).length;
    const periodsUnderStaffed = capacityData.filter(p => p.overUnder < 0).length;

    return {
      avgRequiredHC: Math.round(totalRequired / capacityData.length),
      avgPlannedHC: Math.round(totalPlanned / capacityData.length),
      avgActualHC: actualPeriods.length > 0 ? Math.round(totalActual / actualPeriods.length) : 0,
      totalVariance: totalPlanned - totalRequired,
      avgAttritionRate: avgAttrition,
      avgAHT: avgAHT,
      periodsOverStaffed,
      periodsUnderStaffed
    };
  };

  const kpis = calculateKPIs();

  const kpiData = [
    { title: "Avg Required HC", value: kpis.avgRequiredHC, target: "480", status: "neutral", icon: Users },
    { title: "Avg Actual HC", value: kpis.avgActualHC, target: "470", status: kpis.avgActualHC > 470 ? "above" : "below", icon: Target },
    { title: "Attrition Rate", value: `${kpis.avgAttritionRate.toFixed(1)}%`, target: "8%", status: kpis.avgAttritionRate > 8 ? "above" : "below", icon: TrendingUp },
    { title: "Average AHT", value: `${kpis.avgAHT.toFixed(1)}m`, target: "12m", status: kpis.avgAHT < 12 ? "below" : "above", icon: Clock },
  ];

  return (
    <div className="min-h-screen bg-background p-3">


      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <Calendar className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-semibold text-foreground">Strategic Capacity Plan Insights</h1>
        </div>
      </div>

      {/* Filter Controls */}

      <Card className="mb-4">
        <CardHeader className="pb-4">
          <CardTitle className="text-md">Configuration Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-6 gap-6 items-center">
            <div>
              <Label htmlFor="lob" className="text-xs">Planning Type</Label>
              <Select value={planningType} onValueChange={(value: "annual" | "monthly") => setPlanningType(value)}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="annual">Annual Plan</SelectItem>
                  <SelectItem value="monthly">Monthly Plan</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="">
              <Label className="text-xs">Model</Label>
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
              <Label className="text-xs">Time Period</Label>
              <Select value={selectedMonths} onValueChange={setSelectedMonths}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="3">3 Months</SelectItem>
                  <SelectItem value="6">6 Months</SelectItem>
                  <SelectItem value="12">12 Months</SelectItem>
                  <SelectItem value="18">18 Months</SelectItem>
                  <SelectItem value="24">24 Months</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">Business Unit</Label>
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
            <div>
              <Label className="text-xs">LOB</Label>
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
            <span>{planningType === 'annual' ? 'Annual' : 'Monthly'} Headcount Planning - Next {selectedMonths} {planningType === 'annual' ? 'Months' : 'Weeks'} ({selectedBU} - {selectedLOBs.join(', ')})</span>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                {kpis.periodsOverStaffed} {planningType === 'annual' ? 'months' : 'weeks'} over-staffed
              </Badge>
              <Badge variant="outline">
                {kpis.periodsUnderStaffed} {planningType === 'annual' ? 'months' : 'weeks'} under-staffed
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
                trendData={capacityData.map((d) => ({ value: d.required }))}
              />
              <SummaryCard
                title="Actual HC"
                color="#22c55e"
                value={kpis.avgActualHC}
                trendData={capacityData.filter(d => d.actual !== null).map((d) => ({ value: d.actual || 0 }))}
              />
              <SummaryCard
                title="Over/Under HC"
                color="#999"
                value={kpis.totalVariance}
                trendData={capacityData.map((d) => ({ value: d.overUnder }))}
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
                    data={capacityData}
                    margin={{ top: 20, right: 20, bottom: 20, left: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="monthLabel" stroke="#ccc" />
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
                    {capacityData.some(d => d.actual) && (
                      <Line yAxisId="left" type="monotone" dataKey="actual" stroke="#22c55e" strokeWidth={2} name="Actual" />
                    )}
                    <Bar yAxisId="right" dataKey="overUnder" barSize={20} fill="#999" name="O/U">
                      {capacityData.map((entry, index) => (
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
            New Hire Classes - Coming {selectedMonths} Months ({selectedBU} - {selectedLOBs.join(', ')})
          </h2>
          <div className="w-full h-80 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={newHireData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="monthLabel" stroke="#ccc" />
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
              Attrition Rate - Last 2 Months vs Coming 2 Months ({selectedBU})
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
                  <XAxis dataKey="periodLabel" stroke="#ccc" />
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
              Average Handle Time - Last 2 Months vs Coming 2 Months ({selectedLOBs.join(', ')})
            </h2>
            <div className="w-full h-80 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ahtData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="periodLabel" stroke="#ccc" />
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
                <Target className="h-6 w-6" />
                <h3 className="font-semibold text-foreground text-lg">Long-term Planning</h3>
              </div>
              <p className="text-foreground text-sm">
                Strategic workforce planning with monthly granularity for comprehensive business alignment.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-chart-blue">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <TrendingUp className="h-6 w-6" />
                <h3 className="font-semibold text-foreground text-lg">Trend Analysis</h3>
              </div>
              <p className="text-foreground text-sm">
                Analyze long-term trends and patterns for strategic decision making and resource allocation.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-chart-blue">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Calendar className="h-6 w-6" />
                <h3 className="font-semibold text-foreground text-lg">Annual Planning</h3>
              </div>
              <p className="text-foreground text-am">
                Comprehensive annual and multi-year capacity planning with scenario modeling capabilities.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Action Button */}
        <div className="text-center pt-6">
          <Button
            onClick={() => onStartPlanning(planningType)}
            size="lg"
            className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 text-lg font-medium"
          >
            Start {planningType === 'annual' ? 'Annual' : 'Monthly'} Planning
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default StrategicLandingPage;