import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MessageSquare, Send, Bot, Trash2, User, Filter, Download, Search, Zap } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';

/* const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`; */

const Interactivity = ({ userData, businessData }) => {
  const [messages, setMessages] = useState(localStorage.getItem("ai_global_messages") ? JSON.parse(localStorage.getItem("ai_global_messages") || "[]") : []);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const [filters, setFilters] = useState({
    timeRange: 'today',
    businessUnit: 'all',
    region: 'all',
    channel: 'all'
  });

  const WEBAPPAPIURL = `${AppConfig.API_URL}/`;

  const [searchQuery, setSearchQuery] = useState('');
  const [chatHistory, setChatHistory] = useState(localStorage.getItem("ai_global_messages") ? JSON.parse(localStorage.getItem("ai_global_messages") || "[]") : []);

  /* useEffect(() => {
    fetchChatHistory();
  }, []);

  const fetchChatHistory = async () => {
    try {
      const response = await axios.get(`${API}/chat/history/${sessionId}`);
      setChatHistory(response.data);
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  }; */

  useEffect(() => {
    if (messages && messages.length) {
      localStorage.setItem("ai_global_messages", JSON.stringify(messages));
    }
  }, [JSON.stringify(messages)]);

  const clearChat = () => {
    localStorage.setItem("ai_global_messages", JSON.stringify([]));
    setMessages([]);
  }

  const chatEndRef = React.useRef<HTMLDivElement>(null);

  function convertMarkdownWithChartsToHtml(text: string) {
    let html = text
      .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
      .replace(/\n/g, "<br/>");

    const quickChartRegex = /(https?:\/\/quickchart\.io\/chart\?c=[^\s]+)/g;

    html = html.replace(quickChartRegex, (match) => {
      try {
        let url = decodeURIComponent(match);
        const cParamMatch = url.match(/c=({.*})/);

        if (cParamMatch) {
          const chartJson = cParamMatch[1];

          // ✅ Parse JSON string to object, then stringify properly
          const chartObj = JSON.parse(chartJson);
          const chartUrl = `https://quickchart.io/chart?c=${encodeURIComponent(JSON.stringify(chartObj))}`;

          return `<br/><div style="width:100%; height:400px;"><iframe style="width:150%; height:210%; transform:scale(0.7); transform-origin:top left;" src="${chartUrl}" frameBorder="0"></iframe></div><br/>`;
        }

        // fallback if no c param found
        return `<br/><iframe src="${match}" width="100%" height="200" frameBorder="0"></iframe><br/>`;
      } catch (err) {
        console.error("Error processing QuickChart URL:", err);
        return `<br/><iframe src="${match}" width="100%" height="200" frameBorder="0"></iframe><br/>`;
      }
    });

    html = html.replace("![Chart](", "")
    return html;
  }



  // Scroll to the latest message when messages update
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

  // Focus when component first mounts
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  // Focus again whenever message is cleared
  useEffect(() => {
    if (textareaRef.current && newMessage === "" && !loading) {
      textareaRef.current.focus();
    }
  }, [newMessage, loading]);


  const sendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: newMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {

      const params = new URLSearchParams({
        user: JSON.stringify({ id: userData?.sub, name: userData?.name }),
        company: AuthService.getCompanyId().toString(),
        business_unit: JSON.stringify({ id: businessData?.id, name: businessData?.display_name }),
        token: AuthService.getAccessToken(),
        prompt: newMessage,
      });


      const response = await axios.get(`${WEBAPPAPIURL}webhook/88c0f99b-d43b-4e26-85e0-6cd3e39fe88d?${params.toString()}`);

      const result = response.data?.length ? response.data[0] : false;

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: result?.output || "Sorry unable to process your request",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
      setNewMessage('');
      toast.success('Response generated successfully');
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to get AI response');

      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'I apologize, but I\'m experiencing technical difficulties. Please try again later.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickQuestions = [
    "Show me the top 3 MAPE values for all LOBs.",
    "Compare Prophet vs SARIMAX and show me the MAPE% for Case Type 4 for last year.",
    "What's the Least MAPE for chat lob ?",
    "Give me the short-term outlook for July 2025 week for all Case type 1.",
    "Forecast weekend vs weekday volumes separately.",
    "What's the forecast for Case type 4",
    "Set Sarimax model as default for Case Type 2",
    "Forecast between July 2024 to June 2025 for chat and phone lob."
  ];

  const drillDownLevels = [
    { level: 'Organization', value: 'Global Workforce Management' },
    { level: 'Region', value: 'North America' },
    { level: 'Business Unit', value: 'Customer Service' },
    { level: 'Team', value: 'Premium Support' },
    { level: 'Individual', value: 'Agent Performance' }
  ];

  const handleQuickQuestion = (question) => {
    setNewMessage(question);
    // setTimeout(() => sendMessage(), 100);
  };

  const exportData = () => {
    toast.success('Data export initiated');
    // Implement export functionality
  };

  const applyFilters = () => {
    toast.success('Filters applied successfully');
    // Implement filter logic
  };

  return (
    <div className="p-3 space-y-3 animate-fadeIn mb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground">Executive Interactivity</h1>
          <p className="text-muted-foreground mt-1">
            AI-powered insights and drill-down analytics
          </p>
        </div>
        <Badge variant="outline" className="text-blue-400 border-blue-400">
          <Zap className="w-4 h-4 mr-2" />
          AI Assistant Active
        </Badge>
      </div>

      <Tabs defaultValue="chat" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="chat">AI Assistant</TabsTrigger>
          <TabsTrigger value="drilldown">Drill-down Analysis</TabsTrigger>
          <TabsTrigger value="filters">Advanced Filters</TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[70vh]">
            {/* Chat Interface */}
            <div className="lg:col-span-2">
              <Card className="chat-card overflow-y-auto h-[70vh] flex flex-col">
                {/* Sticky Header */}
                <CardHeader className="z-[1010] p-3 sticky top-0 bg-background border-b">
                  <div className="flex items-start justify-between w-full">
                    {/* Left content */}
                    <div>
                      <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                        <Bot className="w-5 h-5 text-primary" />
                        <span>Executive AI Assistant</span>
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        Ask questions about workforce metrics, trends, and strategic insights
                      </p>
                    </div>

                    {/* Clear history icon */}
                    <button
                      type="button"
                      className="flex items-center space-x-1 text-sm text-muted-foreground hover:text-red-500"
                      aria-label="Clear History"
                      onClick={() => clearChat()}
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>Clear</span>
                    </button>
                  </div>
                </CardHeader>

                {/* Scrollable messages */}
                <CardContent className="flex-1 overflow-y-auto p-3">
                  <div className="space-y-4">
                    {messages.length === 0 && (
                      <div className="text-center text-muted-foreground py-8">
                        <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Welcome! I'm your executive AI assistant.</p>
                        <p className="text-sm mt-2">
                          Ask me anything about your workforce metrics and performance.
                        </p>
                      </div>
                    )}

                    {messages.map((message) => {
                      const isBotChart = message.type === "bot" && (
                        /\[chart\]\s*({[\s\S]*?})/.test(message.content) ||
                        /https?:\/\/quickchart\.io\/chart\?c=/.test(message.content)
                      );

                      return (
                        <div
                          key={message.id}
                          className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
                        >
                          <div
                            className={`rounded-lg p-4 ${message.type === "user"
                              ? "bg-primary text-primary-foreground ml-4 max-w-[80%]"
                              : `bg-muted mr-4 ${isBotChart ? "w-full p-0" : "max-w-[80%]"}` // ✅ full width bubble for chart
                              }`}
                          >
                            <div className={isBotChart ? 'flex items-start space-x-2' : ''}>
                              {message.type === "bot" && <Bot className="w-4 h-4 mt-1 text-primary" />}
                              {message.type === "user" && <User className="w-4 h-4 mt-1" />}
                              <div className="flex-1">
                                {message.type === "user" && (
                                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                                )}
                                {message.type === "bot" && (
                                  <div
                                    className={`text-sm whitespace-pre-wrap ${isBotChart ? "w-full" : ""}`}
                                    dangerouslySetInnerHTML={{
                                      __html: convertMarkdownWithChartsToHtml(message.content),
                                    }}
                                  />
                                )}
                                <p className="text-xs opacity-70 mt-2">
                                  {new Date(message?.timestamp).toLocaleTimeString([], {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                  })}
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}



                    <div ref={chatEndRef} />

                    {loading && (
                      <div className="flex justify-start">
                        <div className="bg-muted rounded-lg p-4 mr-4">
                          <div className="flex items-center space-x-2">
                            <Bot className="w-4 h-4 text-primary" />
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                              <div
                                className="w-2 h-2 bg-primary rounded-full animate-pulse"
                                style={{ animationDelay: "0.2s" }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-primary rounded-full animate-pulse"
                                style={{ animationDelay: "0.4s" }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>

                {/* Sticky Input */}
                <div className="sticky bottom-0 left-0 w-full bg-background border-t p-4 flex space-x-2">
                  <Textarea
                    value={newMessage}
                    ref={textareaRef}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about SLAs, staffing, costs, or any workforce metric..."
                    className="flex-1 min-h-[60px] max-h-[120px]"
                  />
                  <Button
                    onClick={sendMessage}
                    disabled={!newMessage.trim() || loading}
                    className="self-end"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </Card>
            </div>

            {/* Quick Questions */}
            <div className="space-y-6 overflow-y-auto h-[70vh] pr-2">
              <Card className="executive-card">
                <CardHeader className="p-6">
                  <CardTitle className="text-lg text-foreground">Quick Questions</CardTitle>
                </CardHeader>
                <CardContent className="pb-4 pl-4 pr-4">
                  <div className="space-y-2">
                    {quickQuestions.map((question, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        className="w-full text-left justify-start h-auto p-3 text-sm"
                        onClick={() => handleQuickQuestion(question)}
                      >
                        <MessageSquare className="w-4 h-4 mr-2 flex-shrink-0" />
                        <span className="text-wrap">{question}</span>
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="executive-card">
                <CardHeader className="p-6">
                  <CardTitle className="text-lg text-foreground">Recent Insights</CardTitle>
                </CardHeader>
                <CardContent className="pb-4 pl-4 pr-4">
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-amber-500/20">
                      <p className="text-sm font-medium text-amber-400">SLA Alert</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Service level dropping below 80% for premium customers
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-red-500/20">
                      <p className="text-sm font-medium text-red-400">Staffing Shortage</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        25% understaffed for evening shift today
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-green-500/20">
                      <p className="text-sm font-medium text-green-400">Performance Win</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Team Alpha exceeded SLA targets by 8%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="drilldown" className="space-y-6">
          <Card className="executive-card">
            <CardHeader className="p-6">
              <CardTitle className="text-lg text-foreground">Hierarchical Drill-down Navigation</CardTitle>
              <p className="text-sm text-muted-foreground">
                Navigate from organization-level metrics down to individual performance
              </p>
            </CardHeader>
            <CardContent className="pb-4 pl-4 pr-4">
              <div className="space-y-4">
                {drillDownLevels.map((level, index) => (
                  <div key={level.level}>
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/20 hover:bg-muted/30 transition-colors cursor-pointer">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${index === 0 ? 'bg-purple-500/20 text-purple-400' :
                          index === 1 ? 'bg-blue-500/20 text-blue-400' :
                            index === 2 ? 'bg-green-500/20 text-green-400' :
                              index === 3 ? 'bg-amber-500/20 text-amber-400' :
                                'bg-red-500/20 text-red-400'
                          }`}>
                          {index + 1}
                        </div>
                        <div>
                          <h4 className="font-semibold text-foreground">{level.level}</h4>
                          <p className="text-sm text-muted-foreground">{level.value}</p>
                        </div>
                      </div>
                      <Badge variant="outline">
                        View Details
                      </Badge>
                    </div>
                    {index < drillDownLevels.length - 1 && (
                      <div className="flex justify-center my-2">
                        <div className="w-px h-8 bg-border"></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <Separator className="my-6" />

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg bg-muted/20">
                  <h4 className="font-semibold text-foreground mb-2">Current View</h4>
                  <p className="text-sm text-muted-foreground">Premium Support Team</p>
                  <p className="text-lg font-bold text-foreground mt-2">45 Agents</p>
                </div>
                <div className="p-4 rounded-lg bg-muted/20">
                  <h4 className="font-semibold text-foreground mb-2">Performance</h4>
                  <p className="text-sm text-muted-foreground">SLA Compliance</p>
                  <p className="text-lg font-bold text-green-400 mt-2">94.2%</p>
                </div>
                <div className="p-4 rounded-lg bg-muted/20">
                  <h4 className="font-semibold text-foreground mb-2">Utilization</h4>
                  <p className="text-sm text-muted-foreground">Current Level</p>
                  <p className="text-lg font-bold text-blue-400 mt-2">87.5%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="filters" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Filter className="w-5 h-5 text-primary" />
                  <span>Advanced Filters</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4 space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-foreground mb-2 block">Time Range</label>
                    <Select value={filters.timeRange} onValueChange={(value) => setFilters(prev => ({ ...prev, timeRange: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="today">Today</SelectItem>
                        <SelectItem value="week">This Week</SelectItem>
                        <SelectItem value="month">This Month</SelectItem>
                        <SelectItem value="quarter">This Quarter</SelectItem>
                        <SelectItem value="year">This Year</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground mb-2 block">Business Unit</label>
                    <Select value={filters.businessUnit} onValueChange={(value) => setFilters(prev => ({ ...prev, businessUnit: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Units</SelectItem>
                        <SelectItem value="customer_service">Customer Service</SelectItem>
                        <SelectItem value="technical_support">Technical Support</SelectItem>
                        <SelectItem value="sales_support">Sales Support</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground mb-2 block">Region</label>
                    <Select value={filters.region} onValueChange={(value) => setFilters(prev => ({ ...prev, region: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Regions</SelectItem>
                        <SelectItem value="north_america">North America</SelectItem>
                        <SelectItem value="europe">Europe</SelectItem>
                        <SelectItem value="asia_pacific">Asia Pacific</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground mb-2 block">Channel</label>
                    <Select value={filters.channel} onValueChange={(value) => setFilters(prev => ({ ...prev, channel: value }))}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Channels</SelectItem>
                        <SelectItem value="phone">Phone</SelectItem>
                        <SelectItem value="email">Email</SelectItem>
                        <SelectItem value="chat">Chat</SelectItem>
                        <SelectItem value="social">Social</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-foreground mb-2 block">Search Metrics</label>
                  <div className="flex space-x-2">
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search for specific metrics or KPIs..."
                    />
                    <Button size="icon" variant="outline">
                      <Search className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <Button onClick={applyFilters} className="flex-1">
                    Apply Filters
                  </Button>
                  <Button variant="outline" onClick={() => setFilters({ timeRange: 'today', businessUnit: 'all', region: 'all', channel: 'all' })}>
                    Reset
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="executive-card">
              <CardHeader className="p-6">
                <CardTitle className="flex items-center space-x-2 text-lg text-foreground">
                  <Download className="w-5 h-5 text-primary" />
                  <span>Export & Reporting</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4 pl-4 pr-4 space-y-4">
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start" onClick={exportData}>
                    <Download className="w-4 h-4 mr-2" />
                    Export Current View (Excel)
                  </Button>
                  <Button variant="outline" className="w-full justify-start" onClick={exportData}>
                    <Download className="w-4 h-4 mr-2" />
                    Generate Executive Report (PDF)
                  </Button>
                  <Button variant="outline" className="w-full justify-start" onClick={exportData}>
                    <Download className="w-4 h-4 mr-2" />
                    Schedule Automated Report
                  </Button>
                </div>

                <Separator />

                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground">Real-time Alerts</h4>
                  <div className="space-y-2">
                    <div className="p-2 rounded bg-red-500/20 text-red-400 text-sm">
                      🚨 SLA breach risk detected
                    </div>
                    <div className="p-2 rounded bg-amber-500/20 text-amber-400 text-sm">
                      ⚠️ Staffing shortage in 2 hours
                    </div>
                    <div className="p-2 rounded bg-green-500/20 text-green-400 text-sm">
                      ✅ Performance target exceeded
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Interactivity;