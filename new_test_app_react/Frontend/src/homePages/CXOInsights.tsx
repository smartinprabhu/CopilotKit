import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Crown, DollarSign, Heart, Cog, Cpu, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

const CXOInsights = ({ children }) => {
  const [insights, setInsights] = useState([
    { "role": "CEO", "metric_name": "Revenue Protection", "value": 2.3, "insight": "SLA breaches risk $2.3M in revenue", "action_required": true },
    { "role": "CFO", "metric_name": "Cost Variance", "value": 18.5, "insight": "18.5% over budget due to overtime", "action_required": true },
    { "role": "CHRO", "metric_name": "Attrition Risk", "value": 12.5, "insight": "High attrition in customer service", "action_required": true },
    { "role": "COO", "metric_name": "Efficiency Gap", "value": 15.2, "insight": "15.2% efficiency gap in operations", "action_required": true },
    { "role": "CIO", "metric_name": "Digital Adoption", "value": 65.3, "insight": "Below target digital adoption rate", "action_required": false }
  ]);
  const [loading, setLoading] = useState(false);

  /* useEffect(() => {
     fetchCXOInsights();
   }, []);
 
   const fetchCXOInsights = async () => {
     try {
       setLoading(true);
       const response = await axios.get(`${API}/insights/cxo`);
       setInsights(response.data);
     } catch (error) {
       console.error('Error fetching CXO insights:', error);
     } finally {
       setLoading(false);
     }
   }; */

  // Mock data for comprehensive CXO dashboards
  const ceoData = {
    revenueProtection: 2.3,
    customerImpact: 4.2,
    slaCompliance: 89.5,
    brandRisk: 'Medium',
    strategicAlignment: 78,
    growthImpact: 15.2
  };

  const cfoData = {
    costVariance: 18.5,
    roi: -8.2,
    budgetUtilization: 112,
    costPerContact: 12.50,
    overtimeCosts: 280000,
    efficiency: 76.3
  };

  const chroData = {
    attritionRisk: 12.5,
    engagement: 7.8,
    diversityScore: 65,
    skillsGap: 23,
    wellnessIndex: 7.2,
    talentPipeline: 58
  };

  const cooData = {
    operationalEfficiency: 84.7,
    processBottlenecks: 3,
    resourceUtilization: 78.9,
    qualityScore: 4.2,
    capacityGap: 15,
    automationIndex: 42
  };

  const cioData = {
    digitalAdoption: 65.3,
    systemUptime: 99.2,
    automationROI: 125,
    techDebt: 'High',
    supportLoad: 890,
    innovationIndex: 3.8
  };

  const radarData = [
    { subject: 'Revenue', A: 89, fullMark: 100 },
    { subject: 'Cost', A: 75, fullMark: 100 },
    { subject: 'People', A: 68, fullMark: 100 },
    { subject: 'Operations', A: 82, fullMark: 100 },
    { subject: 'Technology', A: 71, fullMark: 100 },
    { subject: 'Strategy', A: 77, fullMark: 100 }
  ];

  const getCXOIcon = (role) => {
    switch (role) {
      case 'CEO':
        return <Crown className="w-5 h-5" />;
      case 'CFO':
        return <DollarSign className="w-5 h-5" />;
      case 'CHRO':
        return <Heart className="w-5 h-5" />;
      case 'COO':
        return <Cog className="w-5 h-5" />;
      case 'CIO':
        return <Cpu className="w-5 h-5" />;
      default:
        return <TrendingUp className="w-5 h-5" />;
    }
  };

  const getStatusIcon = (actionRequired) => {
    return actionRequired ?
      <AlertCircle className="w-4 h-4 text-red-400" /> :
      <CheckCircle className="w-4 h-4 text-green-400" />;
  };

  if (loading) {
    return (
      <div className="p-3 space-y-3">
        <h1 className="text-xl font-bold text-foreground">CXO Insights</h1>
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
          <h1 className="text-xl font-bold text-foreground">CXO Insights</h1>
          <p className="text-muted-foreground mt-1">
            Executive-grade insights for strategic decision making
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {children}
          <Badge variant="outline" className="text-purple-400 border-purple-400">
            <Crown className="w-4 h-4 mr-2" />
            Executive View
          </Badge>
        </div>
      </div>

      {/* High-Level Insights Summary */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {insights.map((insight) => (
          <Card key={insight.id} className="executive-card">
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="flex items-center justify-between mb-2">
                <div className={`p-2 rounded-lg ${insight.role === 'CEO' ? 'bg-purple-500/20 text-purple-400' :
                    insight.role === 'CFO' ? 'bg-green-500/20 text-green-400' :
                      insight.role === 'CHRO' ? 'bg-pink-500/20 text-pink-400' :
                        insight.role === 'COO' ? 'bg-blue-500/20 text-blue-400' :
                          'bg-orange-500/20 text-orange-400'
                  }`}>
                  {getCXOIcon(insight.role)}
                </div>
                {getStatusIcon(insight.action_required)}
              </div>
              <h3 className="font-semibold text-sm text-foreground mb-1">{insight.role}</h3>
              <p className="text-xl font-bold text-foreground mb-1">{insight.value}</p>
              <p className="text-xs text-muted-foreground">{insight.metric_name}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Strategic Alignment Radar */}
      <Card className="executive-card">
        <CardHeader className="p-6">
          <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
            <TrendingUp className="w-5 h-5 text-primary" />
            <span>Strategic Alignment Dashboard</span>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Overall organizational performance across key dimensions
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar
                name="Performance"
                dataKey="A"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Tabs defaultValue="ceo" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="ceo">CEO</TabsTrigger>
          <TabsTrigger value="cfo">CFO</TabsTrigger>
          <TabsTrigger value="chro">CHRO</TabsTrigger>
          <TabsTrigger value="coo">COO</TabsTrigger>
          <TabsTrigger value="cio">CIO</TabsTrigger>
        </TabsList>

        <TabsContent value="ceo" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Crown className="w-5 h-5 text-purple-400" />
                  <span>Revenue Protection</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-red-400">${ceoData.revenueProtection}M</p>
                    <p className="text-sm text-muted-foreground">At Risk Revenue</p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>SLA Impact</span>
                      <span>{ceoData.slaCompliance}%</span>
                    </div>
                    <Progress value={ceoData.slaCompliance} className="w-full" />
                  </div>
                  <Badge variant="outline" className="text-red-400 border-red-400 w-full justify-center">
                    High Priority Action Required
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Customer Experience Impact</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-foreground">{ceoData.customerImpact}</p>
                    <p className="text-sm text-muted-foreground">CSAT Score</p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Brand Risk</span>
                      <Badge variant="outline" className="text-amber-400 border-amber-400">
                        {ceoData.brandRisk}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Growth Impact</span>
                      <span className="text-sm font-semibold text-foreground">{ceoData.growthImpact}%</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Strategic Alignment</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-foreground">{ceoData.strategicAlignment}%</p>
                    <p className="text-sm text-muted-foreground">Alignment Score</p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Target Achievement</span>
                      <span>{ceoData.strategicAlignment}%</span>
                    </div>
                    <Progress value={ceoData.strategicAlignment} className="w-full" />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Workforce strategy aligned with business objectives
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="cfo" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <DollarSign className="w-5 h-5 text-green-400" />
                  <span>Cost Variance Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 rounded-lg bg-red-500/20">
                      <p className="text-2xl font-bold text-red-400">{cfoData.costVariance}%</p>
                      <p className="text-sm text-muted-foreground">Over Budget</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-amber-500/20">
                      <p className="text-2xl font-bold text-amber-400">${cfoData.costPerContact}</p>
                      <p className="text-sm text-muted-foreground">Cost per Contact</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Budget Utilization</span>
                      <span>{cfoData.budgetUtilization}%</span>
                    </div>
                    <Progress value={Math.min(cfoData.budgetUtilization, 100)} className="w-full" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">ROI & Efficiency Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={[
                    { month: 'Jan', roi: -5, efficiency: 72 },
                    { month: 'Feb', roi: -3, efficiency: 75 },
                    { month: 'Mar', roi: -8, efficiency: 76 },
                    { month: 'Apr', roi: -6, efficiency: 78 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip contentStyle={{
                      backgroundColor: 'rgba(0,0,0,0.8)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '8px'
                    }} />
                    <Area type="monotone" dataKey="efficiency" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
                    <Area type="monotone" dataKey="roi" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Overtime Cost Impact</CardTitle>
            </CardHeader>
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 rounded-lg bg-muted/20">
                  <p className="text-2xl font-bold text-foreground">${(cfoData.overtimeCosts / 1000).toFixed(0)}K</p>
                  <p className="text-sm text-muted-foreground">Monthly Overtime</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-muted/20">
                  <p className="text-2xl font-bold text-foreground">{cfoData.roi}%</p>
                  <p className="text-sm text-muted-foreground">Workforce ROI</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-muted/20">
                  <p className="text-2xl font-bold text-foreground">{cfoData.efficiency}%</p>
                  <p className="text-sm text-muted-foreground">Efficiency Score</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chro" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Heart className="w-5 h-5 text-pink-400" />
                  <span>People Analytics</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 rounded-lg bg-red-500/20">
                      <p className="text-2xl font-bold text-red-400">{chroData.attritionRisk}%</p>
                      <p className="text-sm text-muted-foreground">Attrition Risk</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-green-500/20">
                      <p className="text-2xl font-bold text-green-400">{chroData.engagement}</p>
                      <p className="text-sm text-muted-foreground">Engagement Score</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Diversity Score</span>
                        <span>{chroData.diversityScore}%</span>
                      </div>
                      <Progress value={chroData.diversityScore} className="w-full" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Wellness Index</span>
                        <span>{chroData.wellnessIndex}/10</span>
                      </div>
                      <Progress value={(chroData.wellnessIndex / 10) * 100} className="w-full" />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Skills & Development</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={[
                    { category: 'Technical', current: 75, target: 85 },
                    { category: 'Soft Skills', current: 68, target: 80 },
                    { category: 'Leadership', current: 45, target: 70 },
                    { category: 'Digital', current: 62, target: 75 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip contentStyle={{
                      backgroundColor: 'rgba(0,0,0,0.8)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '8px'
                    }} />
                    <Bar dataKey="current" fill="#3b82f6" name="Current" />
                    <Bar dataKey="target" fill="#22c55e" name="Target" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="coo" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Cog className="w-5 h-5 text-blue-400" />
                  <span>Operational Excellence</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 rounded-lg bg-blue-500/20">
                      <p className="text-2xl font-bold text-blue-400">{cooData.operationalEfficiency}%</p>
                      <p className="text-sm text-muted-foreground">Efficiency</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-amber-500/20">
                      <p className="text-2xl font-bold text-amber-400">{cooData.processBottlenecks}</p>
                      <p className="text-sm text-muted-foreground">Bottlenecks</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Resource Utilization</span>
                      <span>{cooData.resourceUtilization}%</span>
                    </div>
                    <Progress value={cooData.resourceUtilization} className="w-full" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Quality & Capacity</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={[
                    { week: 'W1', quality: 4.1, capacity: 88 },
                    { week: 'W2', quality: 4.2, capacity: 85 },
                    { week: 'W3', quality: 4.0, capacity: 82 },
                    { week: 'W4', quality: 4.3, capacity: 90 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" />
                    <YAxis />
                    <Tooltip contentStyle={{
                      backgroundColor: 'rgba(0,0,0,0.8)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '8px'
                    }} />
                    <Line type="monotone" dataKey="quality" stroke="#22c55e" strokeWidth={3} />
                    <Line type="monotone" dataKey="capacity" stroke="#3b82f6" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="cio" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Cpu className="w-5 h-5 text-orange-400" />
                  <span>Digital Transformation</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 rounded-lg bg-orange-500/20">
                      <p className="text-2xl font-bold text-orange-400">{cioData.digitalAdoption}%</p>
                      <p className="text-sm text-muted-foreground">Digital Adoption</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-green-500/20">
                      <p className="text-2xl font-bold text-green-400">{cioData.systemUptime}%</p>
                      <p className="text-sm text-muted-foreground">System Uptime</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Automation ROI</span>
                      <span>{cioData.automationROI}%</span>
                    </div>
                    <Progress value={Math.min(cioData.automationROI, 100)} className="w-full" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="text-lg text-foreground">Technology Performance</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 rounded-lg bg-muted/20">
                      <p className="text-2xl font-bold text-foreground">{cioData.supportLoad}</p>
                      <p className="text-sm text-muted-foreground">Support Tickets</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-muted/20">
                      <p className="text-2xl font-bold text-foreground">{cioData.innovationIndex}</p>
                      <p className="text-sm text-muted-foreground">Innovation Index</p>
                    </div>
                  </div>
                  <Badge variant="outline" className="text-amber-400 border-amber-400 w-full justify-center">
                    Tech Debt: {cioData.techDebt}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CXOInsights;