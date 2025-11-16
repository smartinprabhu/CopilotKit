import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Users, DollarSign, TrendingUp, AlertTriangle, PieChart, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Line, Area, AreaChart } from 'recharts';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

const Planning = ({ defaultDecimalPrecisions, children }) => {
  const [planningData, setPlanningData] = useState([
    { "period": "Q1 2025", "capacity": 950, "demand": 1200, "gap": -250, "cost_impact": 125000, "headcount_needed": 25 },
    { "period": "Q2 2025", "capacity": 1100, "demand": 1350, "gap": -250, "cost_impact": 150000, "headcount_needed": 30 },
    { "period": "Q3 2025", "capacity": 1250, "demand": 1400, "gap": -150, "cost_impact": 89000, "headcount_needed": 18 },
    { "period": "Q4 2025", "capacity": 1300, "demand": 1500, "gap": -200, "cost_impact": 110000, "headcount_needed": 22 }
  ]);
  const [loading, setLoading] = useState(false);

  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` : `${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(defaultDecimalPrecisions)}K` : `${(value / 1000).toFixed(2)}K`;
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  const formatTooltipNumber = (value: number) => {
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  useEffect(() => {
    const unitCost = 500;

    const forecast = [
      { period: "Q1 2025", capacity: 950, demand: 1200 },
      { period: "Q2 2025", capacity: 1100, demand: 1350 },
      { period: "Q3 2025", capacity: 1250, demand: 1400 },
      { period: "Q4 2025", capacity: 1300, demand: 1500 },
    ];

    const updatedForecast = forecast.map(item => {
      const gap = item.capacity - item.demand; // negative if demand > capacity
      const cost_impact = Math.abs(item.capacity) * unitCost;

      return {
        ...item,
        gap,
        cost_impact,
        headcount_needed: Math.ceil(Math.abs(gap) / 10), // example: 10 units per headcount
      };
    });

    setPlanningData(updatedForecast);

  }, []);

  /* const fetchPlanningData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/planning/capacity`);
      setPlanningData(response.data);
    } catch (error) {
      console.error('Error fetching planning data:', error);
    } finally {
      setLoading(false);
    }
  }; */

  // Generate additional mock data for comprehensive planning
  const skillsData = [
    { skill: 'Customer Service', current: 85, needed: 120, gap: -35, cost: 175000 },
    { skill: 'Technical Support', current: 45, needed: 60, gap: -15, cost: 125000 },
    { skill: 'Sales Support', current: 30, needed: 35, gap: -5, cost: 50000 },
    { skill: 'Bilingual Support', current: 25, needed: 40, gap: -15, cost: 120000 }
  ];

  const diversityData = [
    { category: 'Gender Balance', current: 65, target: 50, status: 'needs_improvement' },
    { category: 'Age Diversity', current: 78, target: 80, status: 'on_track' },
    { category: 'Experience Mix', current: 82, target: 75, status: 'exceeds' },
    { category: 'Cultural Diversity', current: 45, target: 60, status: 'needs_improvement' }
  ];

  const costAnalysis = [
    { category: 'Direct Hiring', cost: 450000, percentage: 45, impact: 'high' },
    { category: 'Overtime', cost: 280000, percentage: 28, impact: 'medium' },
    { category: 'Outsourcing', cost: 180000, percentage: 18, impact: 'low' },
    { category: 'Training', cost: 90000, percentage: 9, impact: 'high' }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'exceeds':
        return 'text-green-400 bg-green-500/20 border-green-400';
      case 'on_track':
        return 'text-blue-400 bg-blue-500/20 border-blue-400';
      case 'needs_improvement':
        return 'text-amber-400 bg-amber-500/20 border-amber-400';
      default:
        return 'text-gray-400 bg-gray-500/20 border-gray-400';
    }
  };

  const getImpactColor = (impact) => {
    switch (impact) {
      case 'high':
        return '#ef4444';
      case 'medium':
        return '#f59e0b';
      case 'low':
        return '#22c55e';
      default:
        return '#6b7280';
    }
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <h1 className="text-xl font-bold text-foreground">Planning Tools</h1>
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

  const totalGap = planningData.reduce((sum, item) => sum + Math.abs(item.gap), 0);
  const totalCost = planningData.reduce((sum, item) => sum + item.cost_impact, 0);
  const avgUtilization = 76.5; // Mock data

  return (
    <div className="p-3 space-y-3 animate-fadeIn mb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground">Planning Tools</h1>
          <p className="text-muted-foreground mt-1">
            Strategic workforce capacity planning and optimization
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {children}
          <Badge variant="outline" className="text-purple-400 border-purple-400">
            <BarChart3 className="w-4 h-4 mr-2" />
            Q1 2025 Planning Cycle
          </Badge>
        </div>
      </div>

      {/* Key Planning Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-red-500/20 text-red-400">
                <Users className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Capacity Gap</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">{totalGap}</p>
              <p className="text-sm text-muted-foreground">FTE Shortage</p>
              <Badge variant="outline" className="text-red-400 border-red-400">
                Critical Gap
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400">
                <DollarSign className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Budget Impact</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">
                ${(totalCost / 1000000).toFixed(1)}M
              </p>
              <p className="text-sm text-muted-foreground">Additional Investment</p>
              <Badge variant="outline" className="text-amber-400 border-amber-400">
                18% Over Budget
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <TrendingUp className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Utilization Target</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">{avgUtilization}%</p>
              <p className="text-sm text-muted-foreground">Current Average</p>
              <Badge variant="outline" className="text-blue-400 border-blue-400">
                Below Target (80%)
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <PieChart className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Efficiency Score</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">8.2</p>
              <p className="text-sm text-muted-foreground">Out of 10</p>
              <Badge variant="outline" className="text-green-400 border-green-400">
                Good Performance
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="capacity" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="capacity">Capacity Analysis</TabsTrigger>
          <TabsTrigger value="skills">Skill Gap Analysis</TabsTrigger>
          <TabsTrigger value="diversity">Diversity & Inclusion</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="capacity" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Quarterly Capacity vs Demand</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <ComposedChart data={planningData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" />
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
                    <Bar dataKey="capacity" fill="#22c55e" name="Current Capacity" />
                    <Bar dataKey="demand" fill="#3b82f6" name="Demand" />
                    <Area dataKey="gap" fill="#ef4444" stroke="#ef4444" name="Gap" />
                  </ComposedChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Headcount Planning by Quarter</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={planningData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" />
                    <YAxis tickFormatter={formatNumber} />
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
                      dataKey="capacity"
                      stroke="#8b5cf6"
                      fill="url(#headcountGradient)"
                      strokeWidth={3}
                      name="Additional Headcount Needed"
                    />
                    <defs>
                      <linearGradient id="headcountGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Cost Impact Analysis</CardTitle>
              <p className="text-sm text-muted-foreground">
                Financial implications of capacity planning decisions
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={planningData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis tickFormatter={formatNumber} />
                  <Tooltip contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px'
                  }} formatter={(value) => [`$${(value / 1000).toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2)}K`, 'Cost Impact']} />
                  <Bar dataKey="cost_impact" fill="#f59e0b" name="Cost Impact ($)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="skills" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Skill Gap Analysis by Category</CardTitle>
              <p className="text-sm text-muted-foreground">
                Identify critical skill shortages and investment priorities
              </p>
            </CardHeader>
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="space-y-6">
                {skillsData.map((skill, index) => (
                  <div key={index} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-foreground">{skill.skill}</h4>
                        <p className="text-sm text-muted-foreground">
                          Current: {skill.current} | Needed: {skill.needed} | Gap: {skill.gap}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-foreground">${(skill.cost / 1000).toFixed(0)}K</p>
                        <p className="text-sm text-muted-foreground">Investment</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Current Capacity</span>
                        <span>{((skill.current / skill.needed) * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={(skill.current / skill.needed) * 100} className="w-full" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="diversity" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Diversity & Inclusion Scorecard</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-6">
                  {diversityData.map((item, index) => (
                    <div key={index} className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-foreground">{item.category}</h4>
                        <Badge variant="outline" className={getStatusColor(item.status)}>
                          {item.current}%
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Target: {item.target}%</span>
                          <span>Current: {item.current}%</span>
                        </div>
                        <Progress value={(item.current / item.target) * 100} className="w-full" />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Workforce Composition Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={diversityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis tickFormatter={formatNumber} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '8px'
                      }}
                      formatter={formatTooltipNumber}
                    />
                    <Bar dataKey="current" fill="#3b82f6" name="Current %" />
                    <Bar dataKey="target" fill="#22c55e" name="Target %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="costs" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Cost Breakdown by Strategy</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  {costAnalysis.map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-muted/20">
                      <div className="flex items-center space-x-3">
                        <div
                          className="w-4 h-4 rounded"
                          style={{ backgroundColor: getImpactColor(item.impact) }}
                        />
                        <div>
                          <h4 className="font-medium text-foreground">{item.category}</h4>
                          <p className="text-sm text-muted-foreground">{item.impact} impact</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-foreground">${(item.cost / 1000).toFixed(0)}K</p>
                        <p className="text-sm text-muted-foreground">{item.percentage}%</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">ROI Analysis</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Return on investment by staffing strategy
                </p>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={costAnalysis}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis tickFormatter={formatNumber} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <p className="font-semibold">{label}</p>
                            {payload.map((entry) => (
                              <div key={entry.name} className="flex items-center space-x-2">
                                <span>{entry.name}: {`$${(Number(entry.value) / 1000).toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2)}K`}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Bar
                      dataKey="cost"
                      name="Investment Cost ($)"
                    >
                      {costAnalysis.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getImpactColor(entry.impact)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Planning;