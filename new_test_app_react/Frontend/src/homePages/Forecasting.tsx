import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingUp, AlertCircle, Target, Calendar, BarChart3, Zap } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { LineChart, Line, ReferenceLine, Cell, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

const getRandomInt = (min, max) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

// Utility to generate random float between min and max
const getRandomFloat = (min, max) => {
  return Math.random() * (max - min) + min;
};

const generateForecastData = () => {
  const forecastData = [];
  const baseDate = new Date();

  for (let i = 0; i < 30; i++) {
    const date = new Date(baseDate);
    date.setDate(baseDate.getDate() + i);

    const formattedDate = date.toISOString().split("T")[0];
    const baseDemand = 1200 + getRandomInt(-200, 200);

    forecastData.push({
      date: formattedDate,
      predicted_demand: baseDemand,
      actual_demand: i < 7 ? baseDemand + getRandomInt(-50, 50) : null,
      staffing_level: Math.max(Math.floor(baseDemand * 0.8), 800),
      sla_impact: (() => {
        const targetSLA = 85; // target line
        const maxDeviationPercent = 15; // max ± deviation allowed
        if (i >= 7 || !baseDemand) return null;

        const actual = baseDemand + getRandomInt(-50, 50);
        const deviation = (actual - baseDemand) / baseDemand; // % deviation

        // Apply deviation but clamp impact
        let sla = targetSLA * (1 - deviation);

        // Clamp around target ±maxDeviationPercent
        sla = Math.max(targetSLA - maxDeviationPercent, Math.min(targetSLA + maxDeviationPercent, sla));

        return parseFloat(sla.toFixed(1));
      })(),
      confidence: parseFloat((getRandomFloat(0.85, 0.98) * 100).toFixed(1)),
    });
  }

  return forecastData;
};

const Forecasting = ({ defaultDecimalPrecisions, children }) => {
  const [forecastData, setForecastData] = useState(generateForecastData());
  const [loading, setLoading] = useState(false);
  const [scenarioMultiplier, setScenarioMultiplier] = useState([1.0]);
  const [scenarioResults, setScenarioResults] = useState(null);

  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` : `${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(defaultDecimalPrecisions)}K` : `${(value / 1000).toFixed(2)}K`;
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  const formatTooltipNumber = (value: number) => {
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  /* useEffect(() => {
     fetchForecastData();
   }, []);
 
   const fetchForecastData = async () => {
     try {
       setLoading(true);
       const response = await axios.get(`${API}/forecasting/demand`);
       setForecastData(response.data);
     } catch (error) {
       console.error('Error fetching forecast data:', error);
     } finally {
       setLoading(false);
     }
   }; */

  const runScenarioAnalysis = () => {
    const multiplier = scenarioMultiplier[0];
    const baseRevenue = 5000000; // $5M monthly revenue
    const baseCost = 2000000; // $2M monthly cost

    const results = {
      demandChange: ((multiplier - 1) * 100).toFixed(1),
      staffingNeed: Math.round(multiplier * 100 - 100),
      slaImpact: Math.max(70, 95 - (multiplier - 1) * 50),
      revenueAtRisk: multiplier > 1.1 ? (multiplier - 1) * baseRevenue * 0.3 : 0,
      additionalCost: multiplier > 1 ? (multiplier - 1) * baseCost * 0.8 : 0
    };

    setScenarioResults(results);
  };

  const shortTermData = forecastData.slice(0, 7);
  const longTermData = forecastData.slice(0, 30);

  const varianceData = forecastData.slice(0, 14).map(item => ({
    ...item,
    variance: item.actual_demand ?
      ((item.actual_demand - item.predicted_demand) / item.predicted_demand * 100) : 0,
    staffingGap: item.predicted_demand - item.staffing_level
  }));

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <h1 className="text-xl font-bold text-foreground">Forecasting Insights</h1>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="executive-card">
              <CardContent className="p-6">
                <div className="animate-pulse space-y-4">
                  <div className="h-6 bg-muted rounded w-1/2"></div>
                  <div className="h-64 bg-muted rounded"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3 animate-fadeIn mb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground">Forecasting Insights</h1>
          <p className="text-muted-foreground mt-1">
            Strategic workforce demand analysis and scenario planning
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {children}
          <Badge variant="outline" className="text-blue-400 border-blue-400">
            <TrendingUp className="w-4 h-4 mr-2" />
            Forecast Accuracy: 92.3%
          </Badge>
        </div>
      </div>

      {/* Key Forecast Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <Calendar className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">7-Day Outlook</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">1,285</p>
              <p className="text-sm text-muted-foreground">Avg Daily Demand</p>
              <Badge variant="outline" className="text-green-400 border-green-400">
                +8% vs Last Week
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400">
                <Target className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">SLA Risk</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">15%</p>
              <p className="text-sm text-muted-foreground">Breach Probability</p>
              <Badge variant="outline" className="text-amber-400 border-amber-400">
                Medium Risk
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-red-500/20 text-red-400">
                <AlertCircle className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Revenue at Risk</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">$1.2M</p>
              <p className="text-sm text-muted-foreground">Due to Understaffing</p>
              <Badge variant="outline" className="text-red-400 border-red-400">
                High Impact
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <BarChart3 className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Confidence Level</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">94.2%</p>
              <p className="text-sm text-muted-foreground">Forecast Reliability</p>
              <Badge variant="outline" className="text-green-400 border-green-400">
                Very High
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="short-term" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="short-term">Short-term (7 days)</TabsTrigger>
          <TabsTrigger value="long-term">Long-term (30 days)</TabsTrigger>
          <TabsTrigger value="variance">Variance Analysis</TabsTrigger>
          <TabsTrigger value="scenario">Scenario Testing</TabsTrigger>
        </TabsList>

        <TabsContent value="short-term" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Demand vs Staffing - Next 7 Days</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <ComposedChart data={shortTermData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={formatNumber} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '8px'
                      }}
                      formatter={formatTooltipNumber}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="predicted_demand"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      stroke="#3b82f6"
                      strokeWidth={2}
                      name="Predicted Demand"
                    />
                    <Bar
                      dataKey="staffing_level"
                      fill="#22c55e"
                      fillOpacity={0.8}
                      name="Staffing Level"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">SLA Impact Forecast</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={shortTermData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={formatNumber} domain={[0, 100]} />
                    {/* Target line at 85% */}
                    <ReferenceLine
                      y={85}               // target value
                      stroke="#22c55e"
                      strokeWidth={2}     // color for the target line
                      label={{
                        position: "insideTop", // moves label above the line
                        value: "Target SLA 85%",
                        fill: "#22c55e",
                        fontSize: 12,
                        offset: -15,              // optional spacing from the line
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '8px'
                      }}
                      formatter={formatTooltipNumber}
                    />
                    <Area
                      type="monotone"
                      dataKey="sla_impact"
                      stroke="#f59e0b"
                      fill="url(#slaGradient)"
                      strokeWidth={3}
                      name="SLA Impact Score (%)"
                    />
                    <defs>
                      <linearGradient id="slaGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="long-term" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">30-Day Strategic Forecast</CardTitle>
              <p className="text-sm text-muted-foreground">
                Long-term workforce planning and budget implications
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={longTermData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis tickFormatter={formatNumber} yAxisId="left" />
                  <YAxis tickFormatter={formatNumber} yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(0,0,0,0.8)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '8px'
                    }}
                    formatter={formatTooltipNumber}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="predicted_demand"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    name="Predicted Demand"
                    yAxisId="left"
                  />
                  <Line
                    type="monotone"
                    dataKey="staffing_level"
                    stroke="#22c55e"
                    strokeWidth={3}
                    dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
                    name="Current Staffing"
                    yAxisId="left"
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Confidence Level (%}"
                    yAxisId="right"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="variance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Forecast vs Actual Variance</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={varianceData.filter(d => d.actual_demand)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={formatNumber} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '8px'
                      }}
                      formatter={formatTooltipNumber}
                    />
                    <Bar
                      dataKey="variance"
                      fill="#f59e0b"
                      name="Variance %"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Staffing Gap Heatmap</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={varianceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={formatNumber} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <p className="font-semibold">{label}</p>
                            {payload.map((entry) => (
                              <div key={entry.name} className="flex items-center space-x-2">
                                <span>{entry.name}: {formatTooltipNumber(Number(entry.value))}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Bar
                      dataKey="staffingGap"
                      name="Staffing Gap"
                    >
                      {varianceData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.staffingGap < 0 ? "#ef4444" : "#22c55e"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="scenario" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                <Zap className="w-5 h-5 text-primary" />
                <span>Scenario Testing: What-If Analysis</span>
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Test different demand scenarios and their impact on operations
              </p>
            </CardHeader>
            <CardContent className="space-y-6 pb-4 pl-4 pr-4">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-foreground">
                    Demand Change: {((scenarioMultiplier[0] - 1) * 100).toFixed(1)}%
                  </label>
                  <div className="mt-2">
                    <Slider
                      value={scenarioMultiplier}
                      onValueChange={setScenarioMultiplier}
                      max={1.5}
                      min={0.7}
                      step={0.05}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>-30%</span>
                      <span>Baseline</span>
                      <span>+50%</span>
                    </div>
                  </div>
                </div>

                <Button onClick={runScenarioAnalysis} className="w-full">
                  Run Scenario Analysis
                </Button>
              </div>

              {scenarioResults && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-4 border-t border-border">
                  <div className="p-4 rounded-lg bg-muted/20">
                    <h4 className="font-medium text-foreground">Staffing Impact</h4>
                    <p className="text-2xl font-bold text-blue-400">
                      {scenarioResults.staffingNeed > 0 ? '+' : ''}{scenarioResults.staffingNeed}%
                    </p>
                    <p className="text-sm text-muted-foreground">Additional Staff Needed</p>
                  </div>

                  <div className="p-4 rounded-lg bg-muted/20">
                    <h4 className="font-medium text-foreground">SLA Impact</h4>
                    <p className="text-2xl font-bold text-amber-400">
                      {scenarioResults.slaImpact.toFixed(1)}%
                    </p>
                    <p className="text-sm text-muted-foreground">Expected SLA Level</p>
                  </div>

                  <div className="p-4 rounded-lg bg-muted/20">
                    <h4 className="font-medium text-foreground">Revenue at Risk</h4>
                    <p className="text-2xl font-bold text-red-400">
                      ${(scenarioResults.revenueAtRisk / 1000000).toFixed(1)}M
                    </p>
                    <p className="text-sm text-muted-foreground">Potential Revenue Loss</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Forecasting;