import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Calendar, Users, Clock, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Cell, PieChart, Pie } from 'recharts';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

// Utility to generate random integer between min and max
const getRandomInt = (min, max) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

// Utility to generate random float between min and max
const getRandomFloat = (min, max) => {
  return Math.random() * (max - min) + min;
};

const generateScheduleData = () => {
  const scheduleData = [];
  const baseDate = new Date();
  const shifts = ["Morning", "Afternoon", "Evening", "Night"];

  for (let i = 0; i < 14; i++) { // 2 weeks of schedule data
    const date = new Date(baseDate);
    date.setDate(baseDate.getDate() + i);

    const formattedDate = date.toISOString().split("T")[0];

    shifts.forEach((shift) => {
      const scheduled = getRandomInt(80, 120);
      const actual = scheduled + getRandomInt(-15, 5);

      scheduleData.push({
        date: formattedDate,
        shift: shift,
        scheduled_agents: scheduled,
        actual_agents: actual,
        adherence: Math.min(100, Math.max(70, (actual / scheduled) * 100)),
        coverage: Math.min(100, Math.max(60, getRandomFloat(75, 95))),
      });
    });
  }

  return scheduleData;
};

const Scheduling = ({ defaultDecimalPrecisions, children }) => {
  const [scheduleData, setScheduleData] = useState(generateScheduleData());
  const [loading, setLoading] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState('today');


  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` : `${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(defaultDecimalPrecisions)}K` : `${(value / 1000).toFixed(2)}K`;
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  const formatTooltipNumber = (value: number) => {
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };


  const shiftsHours = {
    Morning: [6, 12],
    Afternoon: [12, 18],
    Evening: [18, 24],
    Night: [0, 6],
  };

  const generateHourlyData = (scheduleData) => {
    const hourlyData = [];

    // Create a map for each date-hour
    scheduleData.forEach(({ date, shift, adherence, coverage }) => {
      const [startHour, endHour] = shiftsHours[shift];
      for (let h = startHour; h < endHour; h++) {
        const hourLabel = `${h.toString().padStart(2, "0")}:00`;
        hourlyData.push({
          date,
          hour: hourLabel,
          adherence,
          coverage,
        });
      }
    });

    // Optionally, aggregate multiple days by hour (average adherence/coverage)
    const aggregated = [];
    const hourMap = {};

    hourlyData.forEach(({ hour, adherence, coverage }) => {
      if (!hourMap[hour]) {
        hourMap[hour] = { totalAdherence: 0, totalCoverage: 0, count: 0 };
      }
      hourMap[hour].totalAdherence += adherence;
      hourMap[hour].totalCoverage += coverage;
      hourMap[hour].count += 1;
    });

    for (const hour in hourMap) {
      const { totalAdherence, totalCoverage, count } = hourMap[hour];
      aggregated.push({
        hour,
        adherence: parseFloat((totalAdherence / count).toFixed(1)),
        coverage: parseFloat((totalCoverage / count).toFixed(1)),
      });
    }

    // Sort by hour
    return aggregated.sort((a, b) => parseInt(a.hour) - parseInt(b.hour));
  };

  const hourlyData = generateHourlyData(scheduleData);


  /* useEffect(() => {
    fetchScheduleData();
  }, []);

  const fetchScheduleData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/scheduling/coverage`);
      setScheduleData(response.data);
    } catch (error) {
      console.error('Error fetching schedule data:', error);
    } finally {
      setLoading(false);
    }
  }; */

  const processDataByPeriod = (data, period) => {
    const now = new Date();
    const today = now.toISOString().split('T')[0];

    switch (period) {
      case 'today':
        return data.filter(item => item.date === today);
      case 'week':
        const weekStart = new Date(now);
        weekStart.setDate(now.getDate() - now.getDay());
        return data.filter(item => {
          const itemDate = new Date(item.date);
          return itemDate >= weekStart && itemDate <= now;
        });
      case 'month':
        return data.filter(item => {
          const itemDate = new Date(item.date);
          return itemDate.getMonth() === now.getMonth() && itemDate.getFullYear() === now.getFullYear();
        });
      default:
        return data;
    }
  };

  const filteredData = processDataByPeriod(scheduleData, selectedPeriod);

  // Calculate key metrics
  const avgAdherence = filteredData.length > 0
    ? filteredData.reduce((sum, item) => sum + item.adherence, 0) / filteredData.length
    : 0;

  const avgCoverage = filteredData.length > 0
    ? filteredData.reduce((sum, item) => sum + item.coverage, 0) / filteredData.length
    : 0;

  const totalScheduled = filteredData.reduce((sum, item) => sum + item.scheduled_agents, 0);
  const totalActual = filteredData.reduce((sum, item) => sum + item.actual_agents, 0);

  // Generate productivity and wellness data
  const productivityData = [
    { team: 'Team Alpha', aht: 245, sla: 89, utilization: 82, wellness: 8.2 },
    { team: 'Team Beta', aht: 268, sla: 85, utilization: 78, wellness: 7.8 },
    { team: 'Team Gamma', aht: 221, sla: 92, utilization: 85, wellness: 8.5 },
    { team: 'Team Delta', aht: 299, sla: 78, utilization: 71, wellness: 6.9 }
  ];

  /* const shiftCoverage = [
    { shift: 'Morning', coverage: 95, agents: 45, status: 'good' },
    { shift: 'Afternoon', coverage: 88, agents: 52, status: 'warning' },
    { shift: 'Evening', coverage: 76, agents: 38, status: 'critical' },
    { shift: 'Night', coverage: 82, agents: 25, status: 'warning' }
  ]; */


  // Aggregate totals by shift
  const shiftCoverage = filteredData.reduce((acc, curr) => {
    const existing = acc.find((item) => item.shift === curr.shift);
    if (existing) {
      existing.agents += curr.actual_agents;
      existing.scheduled += curr.scheduled_agents;
      existing.totalAdherence += curr.adherence;
      existing.count += 1;
    } else {
      acc.push({
        shift: curr.shift,
        agents: curr.actual_agents,
        scheduled: curr.scheduled_agents,
        totalAdherence: curr.adherence,
        count: 1,
      });
    }
    return acc;
  }, []);

  // Compute average adherence per shift
  const shiftCoverageResult = shiftCoverage.map((item) => ({
    shift: item.shift,
    agents: item.agents,
    scheduled: item.scheduled,
    adherence: parseFloat((item.totalAdherence / item.count).toFixed(1)), // average
  }));


  const getStatusColor = (status) => {
    switch (status) {
      case 'good':
        return '#22c55e';
      case 'warning':
        return '#f59e0b';
      case 'critical':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const shiftColors = {
    Morning: '#22c55e',   // green
    Afternoon: '#f59e0b', // amber
    Evening: '#3b82f6',   // blue
    Night: '#8b5cf6',     // purple
  };

  const wellnessColors = ['#22c55e', '#f59e0b', '#ef4444'];

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <h1 className="text-xl font-bold text-foreground">Scheduling & Workforce Control</h1>
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
          <h1 className="text-xl font-bold text-foreground">Scheduling & Workforce Control</h1>
          <p className="text-muted-foreground mt-1">
            Real-time workforce optimization and schedule management
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {children}
          { /* <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">Today</SelectItem>
              <SelectItem value="week">This Week</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
            </SelectContent>
          </Select> */ }
          <Badge variant="outline" className="text-green-400 border-green-400">
            <Activity className="w-4 h-4 mr-2" />
            Real-time Data
          </Badge>
        </div>
      </div>

      {/* Key Scheduling Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <Users className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Coverage Rate</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">{avgCoverage.toFixed(1)}%</p>
              <p className="text-sm text-muted-foreground">Average Shift Coverage</p>
              <Badge variant="outline" className={
                avgCoverage >= 90 ? "text-green-400 border-green-400" :
                  avgCoverage >= 80 ? "text-amber-400 border-amber-400" :
                    "text-red-400 border-red-400"
              }>
                {avgCoverage >= 90 ? 'Excellent' : avgCoverage >= 80 ? 'Good' : 'Needs Attention'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <CheckCircle className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Schedule Adherence</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">{avgAdherence.toFixed(1)}%</p>
              <p className="text-sm text-muted-foreground">Agent Adherence Rate</p>
              <Badge variant="outline" className="text-green-400 border-green-400">
                Above Target (85%)
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Staffing Gap</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">{totalScheduled - totalActual}</p>
              <p className="text-sm text-muted-foreground">Agents Short</p>
              <Badge variant="outline" className="text-amber-400 border-amber-400">
                Monitor Closely
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="executive-card">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3 mb-3">
              <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                <Clock className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground">Overtime Hours</h3>
            </div>
            <div className="space-y-2">
              <p className="text-2xl font-bold text-foreground">1,247</p>
              <p className="text-sm text-muted-foreground">This Week</p>
              <Badge variant="outline" className="text-purple-400 border-purple-400">
                +23% vs Last Week
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="coverage" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="coverage">Coverage View</TabsTrigger>
          <TabsTrigger value="productivity">Team Productivity</TabsTrigger>
          <TabsTrigger value="wellness">Employee Wellness</TabsTrigger>
          <TabsTrigger value="optimization">Schedule Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="coverage" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Schedule Adherence by Shift</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={filteredData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="shift" />
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
                    <Bar dataKey="adherence" fill="#22c55e" name="Adherence %" />
                    <Bar dataKey="coverage" fill="#3b82f6" name="Coverage %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Shift Coverage Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <PieChart>
                    <Pie
                      data={shiftCoverageResult}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="adherence"
                      nameKey="shift"
                    >
                      {shiftCoverageResult.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={shiftColors[entry.shift]} />
                      ))}
                    </Pie>
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                            <p className="font-semibold">{label}</p>
                            {payload.map((entry) => (
                              <div key={entry.name} className="flex items-center space-x-2">
                                <span
                                  className="w-3 h-3 rounded-full"
                                  style={{ backgroundColor: shiftColors[entry.payload.shift] }}
                                />
                                <span>{entry.name}: {formatTooltipNumber(Number(entry.value))}</span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Scheduled vs Actual Staffing Trend</CardTitle>
              <p className="text-sm text-muted-foreground">
                Real-time staffing levels across all shifts
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={hourlyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" interval={0} />
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
                  <Line
                    type="monotone"
                    dataKey="adherence"
                    stroke="#22c55e"
                    strokeWidth={3}
                    dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
                    name="Scheduled"
                  />
                  <Line
                    type="monotone"
                    dataKey="coverage"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    name="Actual"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="productivity" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Team Performance Leaderboard</CardTitle>
              <p className="text-sm text-muted-foreground">
                Balanced scorecards showing productivity and quality metrics
              </p>
            </CardHeader>
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="space-y-4">
                {productivityData.sort((a, b) => b.sla - a.sla).map((team, index) => (
                  <div key={team.team} className="flex items-center justify-between p-4 rounded-lg bg-muted/20">
                    <div className="flex items-center space-x-4">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${index === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                        index === 1 ? 'bg-gray-500/20 text-gray-400' :
                          index === 2 ? 'bg-amber-600/20 text-amber-600' :
                            'bg-muted text-muted-foreground'
                        }`}>
                        {index + 1}
                      </div>
                      <div>
                        <h4 className="font-semibold text-foreground">{team.team}</h4>
                        <p className="text-sm text-muted-foreground">
                          AHT: {team.aht}s | Utilization: {team.utilization}%
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="font-bold text-foreground">{team.sla}%</p>
                        <p className="text-sm text-muted-foreground">SLA</p>
                      </div>
                      <Badge variant="outline" className={
                        team.sla >= 90 ? "text-green-400 border-green-400" :
                          team.sla >= 80 ? "text-amber-400 border-amber-400" :
                            "text-red-400 border-red-400"
                      }>
                        {team.sla >= 90 ? 'Excellent' : team.sla >= 80 ? 'Good' : 'Needs Improvement'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Team Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={productivityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="team" />
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
                  <Bar dataKey="sla" fill="#22c55e" name="SLA %" />
                  <Bar dataKey="utilization" fill="#3b82f6" name="Utilization %" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="wellness" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Employee Wellness Index</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Burnout risk indicators and engagement scores
                </p>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  {productivityData.map((team, index) => (
                    <div key={team.team} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-foreground">{team.team}</h4>
                        <div className="flex items-center space-x-2">
                          <span className="text-2xl font-bold text-foreground">{team.wellness}</span>
                          <span className="text-sm text-muted-foreground">/10</span>
                        </div>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all duration-300"
                          style={{
                            width: `${(team.wellness / 10) * 100}%`,
                            backgroundColor: team.wellness >= 8 ? '#22c55e' : team.wellness >= 6 ? '#f59e0b' : '#ef4444'
                          }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {team.wellness >= 8 ? 'Excellent wellness' :
                          team.wellness >= 6 ? 'Monitor for burnout' :
                            'High burnout risk'}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Wellness vs Performance Correlation</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={productivityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="team" />
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
                    <Line
                      type="monotone"
                      dataKey="wellness"
                      stroke="#8b5cf6"
                      strokeWidth={3}
                      dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 6 }}
                      name="Wellness Score"
                    />
                    <Line
                      type="monotone"
                      dataKey="sla"
                      stroke="#22c55e"
                      strokeWidth={3}
                      dot={{ fill: '#22c55e', strokeWidth: 2, r: 6 }}
                      name="SLA Performance"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Schedule Optimization Controls</CardTitle>
              <p className="text-sm text-muted-foreground">
                What-if scenarios for schedule adjustments
              </p>
            </CardHeader>
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-semibold text-foreground">Quick Actions</h4>
                  <div className="space-y-2">
                    <Button className="w-full justify-start" variant="outline">
                      <Users className="w-4 h-4 mr-2" />
                      Add 5 agents to Evening shift
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Clock className="w-4 h-4 mr-2" />
                      Extend Morning shift by 1 hour
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Activity className="w-4 h-4 mr-2" />
                      Enable cross-training alerts
                    </Button>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-foreground">Impact Prediction</h4>
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-green-500/20">
                      <p className="text-sm font-medium text-green-400">SLA Improvement</p>
                      <p className="text-lg font-bold text-foreground">+8.5%</p>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-500/20">
                      <p className="text-sm font-medium text-amber-400">Cost Impact</p>
                      <p className="text-lg font-bold text-foreground">+$12K/week</p>
                    </div>
                    <div className="p-3 rounded-lg bg-blue-500/20">
                      <p className="text-sm font-medium text-blue-400">Coverage Increase</p>
                      <p className="text-lg font-bold text-foreground">+15%</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Scheduling;