import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Users,
  Target,
  Star,
  Activity,
  Smartphone
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

const Dashboard = ({ defaultDecimalPrecisions, children }) => {
  const [kpis, setKpis] = useState([
    { "name": "SLA Compliance", "value": 89.5, "target": 95.0, "unit": "%", "trend": "down", "status": "warning" },
    { "name": "Average Handle Time", "value": 285, "target": 240, "unit": "seconds", "trend": "up", "status": "critical" },
    { "name": "Forecast Accuracy", "value": 92.3, "target": 90.0, "unit": "%", "trend": "up", "status": "good" },
    { "name": "Schedule Adherence", "value": 87.2, "target": 85.0, "unit": "%", "trend": "up", "status": "good" },
    { "name": "Utilization", "value": 78.9, "target": 80.0, "unit": "%", "trend": "stable", "status": "warning" },
    { "name": "Service Level", "value": 82.1, "target": 85.0, "unit": "%", "trend": "down", "status": "warning" },
    { "name": "Customer Satisfaction", "value": 4.2, "target": 4.5, "unit": "stars", "trend": "stable", "status": "warning" },
    { "name": "Employee Engagement", "value": 7.8, "target": 8.0, "unit": "score", "trend": "up", "status": "good" },
    { "name": "Attrition Risk", "value": 12.5, "target": 10.0, "unit": "%", "trend": "up", "status": "critical" },
    { "name": "Digital Adoption", "value": 65.3, "target": 70.0, "unit": "%", "trend": "up", "status": "warning" }
  ]);
  const [alerts, setAlerts] = useState([
    { "type": "sla_risk", "severity": "high", "title": "SLA Breach Risk", "message": "Service level dropping below 95% for premium customers" },
    { "type": "staffing_shortage", "severity": "critical", "title": "Critical Staffing Shortage", "message": "25% understaffed for evening shift" },
    { "type": "abnormal_aht", "severity": "medium", "title": "AHT Spike Detected", "message": "Average handle time increased by 15%" }
  ]);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState([]);

  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` : `${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(defaultDecimalPrecisions)}K` : `${(value / 1000).toFixed(2)}K`;
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  const formatTooltipNumber = (value: number) => {
    return value ? value.toFixed(defaultDecimalPrecisions > 0 ? defaultDecimalPrecisions : 2) : "0.00";
  };

  useEffect(() => {
    generateChartData();
  }, []);

  /*  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [kpiResponse, alertResponse] = await Promise.all([
        axios.get(`${API}/dashboard/kpis`),
        axios.get(`${API}/dashboard/alerts`)
      ]);
      
      setKpis(kpiResponse.data);
      setAlerts(alertResponse.data);
      
      // Generate chart data
      generateChartData();
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }; */

  const generateChartData = () => {
    const data = [];
    const now = new Date();

    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        sla: 82 + Math.random() * 15,
        aht: 240 + Math.random() * 60,
        utilization: 75 + Math.random() * 20,
        satisfaction: (4.0 + Math.random() * 1.0) * 20
      });
    }

    setChartData(data);
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <Minus className="w-4 h-4 text-blue-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'good':
        return 'text-green-400 bg-green-500/20';
      case 'warning':
        return 'text-amber-400 bg-amber-500/20';
      case 'critical':
        return 'text-red-400 bg-red-500/20';
      default:
        return 'text-blue-400 bg-blue-500/20';
    }
  };

  const getKpiIcon = (name) => {
    switch (name.toLowerCase()) {
      case 'sla compliance':
        return <Target className="w-5 h-5" />;
      case 'average handle time':
        return <Clock className="w-5 h-5" />;
      case 'utilization':
        return <Users className="w-5 h-5" />;
      case 'customer satisfaction':
        return <Star className="w-5 h-5" />;
      case 'employee engagement':
        return <Activity className="w-5 h-5" />;
      case 'digital adoption':
        return <Smartphone className="w-5 h-5" />;
      default:
        return <Target className="w-5 h-5" />;
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'high':
        return <AlertTriangle className="w-5 h-5 text-orange-400" />;
      case 'medium':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'low':
        return <CheckCircle className="w-5 h-5 text-blue-400" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-gray-400" />;
    }
  };

  const pieData = [
    { name: 'On Track', value: 65, color: '#22c55e' },
    { name: 'At Risk', value: 25, color: '#f59e0b' },
    { name: 'Critical', value: 10, color: '#ef4444' }
  ];

  if (loading) {
    return (
      <div className="p-3 space-y-3">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-foreground">Executive Dashboard</h1>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {[...Array(10)].map((_, i) => (
            <Card key={i} className="executive-card">
              <CardContent className="p-6">
                <div className="animate-pulse space-y-3">
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-8 bg-muted rounded w-1/2"></div>
                  <div className="h-3 bg-muted rounded w-full"></div>
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
          <h1 className="text-xl font-bold text-foreground">Executive Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            CXO-grade workforce management control tower
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {children}
          <Badge variant="outline" className="text-green-400 border-green-400">
            Live Data
          </Badge>
          <span className="text-sm text-muted-foreground">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <Alert key={alert.id} className={`alert-banner ${alert.severity}`}>
              <div className="flex items-center space-x-3">
                {getSeverityIcon(alert.severity)}
                <div className="flex-1">
                  <AlertDescription className="font-medium">
                    {alert.title}
                  </AlertDescription>
                  <p className="text-sm text-muted-foreground mt-1">
                    {alert.message}
                  </p>
                </div>
                <Badge variant="outline" className={getStatusColor(alert.severity)}>
                  {alert.severity.toUpperCase()}
                </Badge>
              </div>
            </Alert>
          ))}
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
        {kpis.map((kpi) => (
          <Card key={kpi.id} className="executive-card animate-slideInUp">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className={`p-2 rounded-lg ${getStatusColor(kpi.status)}`}>
                  {getKpiIcon(kpi.name)}
                </div>
                {getTrendIcon(kpi.trend)}
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-sm text-muted-foreground">
                  {kpi.name}
                </h3>
                <div className="flex items-baseline space-x-2">
                  <span className="text-2xl font-bold text-foreground">
                    {kpi.value}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {kpi.unit}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-muted-foreground">
                    Target: {kpi.target}{kpi.unit}
                  </span>
                  <Badge
                    variant="outline"
                    className={`text-xs ${getStatusColor(kpi.status)}`}
                  >
                    {kpi.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* SLA Trend */}
        <Card className="executive-card">
          <CardHeader className="p-6">
            <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
              <Target className="w-5 h-5 text-primary" />
              <span>SLA Performance Trend</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
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
                <Area
                  type="monotone"
                  dataKey="sla"
                  stroke="#22c55e"
                  fill="url(#slaGradient)"
                  strokeWidth={2}
                />
                <defs>
                  <linearGradient id="slaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Performance Distribution */}
        <Card className="executive-card">
          <CardHeader className="p-6">
            <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
              <Activity className="w-5 h-5 text-primary" />
              <span>Manpower Performance Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
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

        {/* Multi-metric Trend */}
        <Card className="executive-card lg:col-span-2">
          <CardHeader className="p-6">
            <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
              <TrendingUp className="w-5 h-5 text-primary" />
              <span>Key Metrics Overview</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                {/* Left Y axis for SLA & Utilization */}
                <YAxis tickFormatter={formatNumber} yAxisId="left" domain={[0, 100]} />

                {/* Right Y axis for CSAT % */}
                <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
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
                  dataKey="sla"
                  stroke="#22c55e"
                  strokeWidth={3}
                  dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
                  name="SLA %"
                  yAxisId="left"
                />
                <Line
                  type="monotone"
                  dataKey="utilization"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                  name="Utilization %"
                  yAxisId="left"
                />
                <Line
                  type="monotone"
                  dataKey="satisfaction"
                  stroke="#f59e0b"
                  strokeWidth={3}
                  dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
                  name="CSAT %"
                  yAxisId="right"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;