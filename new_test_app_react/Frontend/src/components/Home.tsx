import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Zap, CalendarCheck, LineChart, MonitorSmartphone, BarChart2, Settings, LayoutDashboard,
  Bot,
  LucideComputer,
} from "lucide-react";
import { Chip, Tooltip } from "@mui/material"
import { useNavigate } from "react-router-dom";

import { Skeleton } from "@/components/ui/skeleton";

import { UnitSelection } from './UnitSelection';
import LoadingSkeleton from "./LoadingSkeleton";
import ContactRates from './ContactRates';
import StaticAHTTrendChart from './StaticAHTTrendChart'
import StaticAttritionLineChart from './StaticAttritionLineChart'
import StaticAttritionPieChart from './StaticAttritionPieChart'
import NpsDashboard from "./NPSDashboard";
import CapacityLook from './CapacityLook'
import ExecutiveDashboard from '../homePages/Dashboard';
import InterActivity from '../homePages/Interactivity';
import CXOInsights from "@/homePages/CXOInsights";
import Forecasting from "@/homePages/Forecasting";
import Scheduling from "@/homePages/Scheduling";
import Planning from "@/homePages/Planning";

const Home = ({ options, businessData, userData, configLoading, defaultConfig, parametersList, defaultDecimalPrecisions, overallOldData, children, overallData, overallLoading, setActiveMainTab, setBusinessData }) => {


  const isManager = defaultConfig && defaultConfig.length > 0 ? defaultConfig.find(item => item.name === 'is_manager')?.value === 'true' || false : false;

  const [activeTab, setActiveTab] = useState<"Executive" | "Forecasting" | "PlanningHome" | "SchedulingHome" | "CXO" | "InterActivity" | "Apps" | "Contact Rates" | "Capacity Outlook" | "AHT Trend" | "Attrition" | "NPS-CSAT">("");

  React.useEffect(() => {
    if (!configLoading) {
      if (isManager) {
        setActiveTab('Executive')
      } else {
        setActiveTab('NPS-CSAT')
      }
    }
  }, [defaultConfig, configLoading]);

  const features = [
    {
      title: "Forecasting",
      icon: <LineChart className="h-7 w-7 text-blue-600" />,
      description: "Predict trends, stay ahead, act with confidence.",
    },
    {
      title: "Planning",
      icon: <CalendarCheck className="h-7 w-7 text-green-600" />,
      description: "Strategize resources, align goals, execute flawlessly.",
    },
    {
      title: "Scheduling",
      icon: <Zap className="h-7 w-7 text-yellow-500" />,
      description: "Optimize shifts, balance workload, maximize efficiency.",
    },
    {
      title: "Realtime Ops Support",
      icon: <MonitorSmartphone className="h-7 w-7 text-rose-600" />,
      description: "Monitor live, resolve fast, keep operations smooth.",
    },
    {
      title: "Dashboards & Analytics",
      icon: <LayoutDashboard className="h-7 w-7 text-indigo-600" />,
      description: "Visualize performance, identify patterns, and unlock insights.",
    },
    {
      title: "Agent Gauri (Chat bot)",
      icon: <Bot className="h-7 w-7 text-cyan-600" />,
      description: "AI-powered assistant to guide agents and answer questions.",
    },
    {
      title: "AI Agent",
      icon: <LucideComputer className="h-7 w-7 text-teal-600" />,
      description: "Manage and monitor operations on the go.",
    },
    {
      title: "CXO Insights",
      icon: <BarChart2 className="h-7 w-7 text-purple-600" />,
      description: "Executive dashboards for data-driven support decisions.",
    },
    {
      title: "Settings",
      icon: <Settings className="h-7 w-7 text-gray-600" />,
      description: "Customize and control your system settings.",
    },
  ];


  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const navigate = useNavigate(); // from react-router-dom

  const handleCardClick = (feature) => {
    if (feature.title === "CXO Insights") {
      setActiveTab("Contact Rates");
    } else if (feature.title === "AI Agent") {
      navigate("/new-agent");
    } else if (feature.title === "Settings") {
      navigate("/company");
    } else {
      setSelectedFeature(feature);
      setIsModalOpen(true);
    }
  };


  if (configLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <Card key={index} className="relative shadow-md w-full h-48 p-4">
            <Skeleton className="h-4 w-3/4 mb-2" />
            <Skeleton className="h-8 w-1/2 mb-4" />
            <Skeleton className="h-4 w-1/2 mb-4" />
            <Skeleton className="h-4 w-full" />
          </Card>
        ))}
      </div>
    );
  }


  return (
    <Card className="bg-white dark:bg-background">
      <CardContent className="pt-2 mt-[-10px] mb-10">
        <div className="mb-6">
          {/* <div className="flex flex-col space-y-4 sm:flex-row sm:justify-between sm:space-y-0 mb-6">
            <div className="flex items-center gap-2">
              <h3 className="text-xl font-semibold">Home Dashboard</h3>
            </div>
          </div> */}

          {!isManager && (
            <div className="flex border-b">
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "NPS-CSAT" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("NPS-CSAT")}
              >
                NPS-CSAT
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "Contact Rates" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("Contact Rates")}
              >
                Contact Rates
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "Capacity Outlook" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("Capacity Outlook")}
              >
                Capacity Outlook
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "AHT Trend" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("AHT Trend")}
              >
                AHT Trend
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "Attrition" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("Attrition")}
              >
                Attrition
              </button>
            </div>
          )}
          {isManager && (
            <div className="flex border-b">
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "Executive" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("Executive")}
              >
                Executive Dashboard
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "Forecasting" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("Forecasting")}
              >
                Forecasting
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "PlanningHome" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("PlanningHome")}
              >
                Planning
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "SchedulingHome" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("SchedulingHome")}
              >
                Scheduling
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "CXO" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("CXO")}
              >
                CXO Insights
              </button>
              <button
                className={`py-2 px-4 text-sm font-medium ${activeTab === "InterActivity" ? "border-b-2 border-primary text-primary" : ""}`}
                onClick={() => setActiveTab("InterActivity")}
              >
                Interactivity
              </button>
            </div>
          )}
        </div>

        <div className={isManager ? 'mt-0' : 'mt-4'}>
          {activeTab === "Apps" &&
            <div className="w-full flex">
              <div className="w-full sm:w-1/2">
                <h2 className="text-xl mb-2">Welcome to AForce360.ai!</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {features.map((feature) => (
                    <div
                      key={feature.title}
                      onClick={() => handleCardClick(feature)}
                      className="flex cursor-pointer flex-col items-center justify-center text-center p-6 bg-card bg-accent border shadow-sm rounded-lg"
                    >
                      <div className="mb-2">{feature.icon}</div>
                      <h3 className="text-lg font-semibold mb-1 flex items-center">
                        {feature.title}
                        {feature.title === 'Scheduling' && (
                          <Tooltip title="Roadmap" arrow>
                            <Chip label="R" size="small" className="colour ml-2" sx={{ color: 'var(--primary-foreground)' }} />
                          </Tooltip>
                        )}
                        {feature.title.includes('Realtime Ops Support') && (
                          <Tooltip title="Roadmap" arrow>
                            <Chip label="R" size="small" className="colour ml-2" sx={{ color: 'var(--primary-foreground)' }} />
                          </Tooltip>
                        )}
                        {feature.title.includes('Dashboards & Analytics') && (
                          <Tooltip title="Roadmap" arrow>
                            <Chip label="R" size="small" className="colour ml-2" sx={{ color: 'var(--primary-foreground)' }} />
                          </Tooltip>
                        )}
                        {feature.title.includes('Agent Gauri (Chat bot)') && (
                          <Tooltip title="Roadmap" arrow>
                            <Chip label="R" size="small" className="colour ml-2" sx={{ color: 'var(--primary-foreground)' }} />
                          </Tooltip>
                        )}
                        {feature.title.includes('Mobile App') && (
                          <Tooltip title="Roadmap" arrow>
                            <Chip label="R" size="small" className="colour ml-2" sx={{ color: 'var(--primary-foreground)' }} />
                          </Tooltip>
                        )}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-300">{feature.description}</p>
                    </div>
                  ))}
                  <UnitSelection
                    open={isModalOpen}
                    feature={selectedFeature}
                    options={options}
                    setBusinessData={setBusinessData}
                    setActiveTab={setActiveMainTab}
                    onClose={() => setIsModalOpen(false)}
                  />
                </div>
              </div>
            </div>
          }
          {activeTab === "Contact Rates" && (
            overallLoading ? (
              <div className="space-y-4">
                <LoadingSkeleton />
              </div>
            ) : (
              <ContactRates defaultDecimalPrecisions={defaultDecimalPrecisions} parametersList={parametersList} children={children} options={options} overallOldData={overallOldData} overallData={overallData} overallLoading={overallLoading} />
            )
          )}
          {activeTab === "Capacity Outlook" &&
            <div>
              <CapacityLook hideSummary={false} options={options && options.length > 0 ? options.map(trace => trace.display_name) : []} />
            </div>
          }
          {activeTab === "AHT Trend" && (
            <div className="w-full flex gap-4">
              <div className="w-full sm:w-1/2 h-auto p-4">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Average Handling Time Line Graph</h3>
                <StaticAHTTrendChart />
              </div>
              <div className="w-full sm:w-1/2 h-auto">
                <div className="p-6 space-y-6 executive-card shadow-sm">
                  {/* Intro */}
                  <section className="space-y-2">
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">AHT Trend</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      <strong>AHT Trend:</strong> AHT trend over the past (X months/quarters) shows a (rise/drop/stable pattern),
                      indicating a shift in operational efficiency and potentially in customer interaction complexity.
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      Along with it, we will showcase key AHT drivers and business impact.
                    </p>
                  </section>

                  {/* AHT Description */}
                  <section className="space-y-4">
                    <h4 className="text-base font-semibold">AHT Trend Description</h4>

                    {/* A. Trend Insight */}
                    <div className="space-y-1">
                      <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">A. Trend Insight</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        <strong>Direction:</strong> AHT has (increased/decreased) by X% over the last (timeframe).
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        <strong>Why it matters:</strong> This reflects (improved agent efficiency / increased complexity of queries / system-related delays).
                      </p>
                    </div>

                    {/* B. Key Drivers */}
                    <div className="space-y-1">
                      <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">B. Key Drivers (Root Causes)</h5>
                      <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                        <li>Upward trend is primarily driven by: product/process changes, system issues, new campaign handling, knowledge gaps, etc.</li>
                        <li>Downward trend is due to: training programs, self-service adoption, tool enhancements, etc.</li>
                      </ul>
                    </div>

                    {/* C. Business Impact */}
                    <div className="space-y-1">
                      <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">C. Business Impact</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        This change in AHT has a direct impact on operational cost and customer satisfaction.
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        A 10-second reduction in AHT translates into $X savings per month, assuming current call volume.
                      </p>
                    </div>

                    {/* D. Strategic Actions */}
                    <div className="space-y-1">
                      <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">D. Strategic Actions</h5>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        We are implementing initiatives like automation, better knowledge management, and agent coaching to optimize AHT without compromising quality.
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Monitoring will continue to ensure changes are sustainable.
                      </p>
                    </div>

                    {/* E. Visual Representation */}
                    <div className="space-y-1">
                      <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">E. Visual Representation</h5>
                      <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                        <li>Include a clear line chart showing monthly/quarterly AHT trend.</li>
                        <li>Add annotations for notable events (e.g., system upgrades, new product launches).</li>
                      </ul>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          )}
          {activeTab === "Attrition" && (
            <div className="w-full flex gap-4">
              <div className="w-full sm:w-1/2 h-auto p-4">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Employee Attrition Rate</h3>
                <StaticAttritionLineChart />
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Percentage</h3>
                <StaticAttritionPieChart />
              </div>
              <div className="w-full sm:w-1/2 h-auto">
                <div className="p-6 space-y-6 executive-card shadow-sm">
                  {/* Header */}
                  <section className="space-y-2">
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Attrition Overview</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      Data shows attrition trends (monthly/quarterly/yearly), broken down BU-wise and LOB-wise.
                      Further segmentation includes role and tenure buckets for deeper insight.
                    </p>
                  </section>

                  {/* A. What is Included */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">A. What is Included</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>
                        <strong>Overall Attrition Rate:</strong> For the reporting period (e.g., monthly, quarterly, yearly).
                      </li>
                      <li>
                        <strong>Comparison with Previous Period:</strong> Highlights increase or decrease in attrition rate vs prior period.
                      </li>
                      <li>
                        <strong>High-Level Insight:</strong> Indicates trend direction (rise/drop/stable) and its strategic implications on workforce planning, talent retention, and cost.
                      </li>
                    </ul>
                  </section>

                  {/* B. Trend Visualization */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">B. Trend Visualization</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>
                        <strong>Line Chart:</strong> Shows overall attrition rate over the last 6–12 months to capture trends.
                      </li>
                      <li>
                        <strong>Bar Chart:</strong> Voluntary vs Involuntary attrition for the selected timeframe.
                      </li>
                      <li>
                        <strong>Geo Map (if applicable):</strong> Displays attrition by location or site for multi-region operations.
                      </li>
                      <li>
                        <strong>Heat Map:</strong> Visualizes attrition across departments or function levels to detect high-risk areas.
                      </li>
                    </ul>
                  </section>

                  {/* C. Attrition Breakdown Insight */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">C. Attrition Breakdown Insight</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>
                        <strong>Role/Team Level:</strong> Bar chart or table showing % attrition grouped by job roles or teams.
                      </li>
                      <li>
                        <strong>Tenure Buckets:</strong> Pie chart breaking down attrition by tenure bands
                        (e.g., 0–6 months, 6–12 months, 1–2 years, etc.) to highlight early exits or long-term churn patterns.
                      </li>
                    </ul>
                  </section>
                </div>
              </div>
            </div>
          )}

          {activeTab === "NPS-CSAT" && (
            <div className="w-full flex gap-2">
              <div className="w-full sm:w-1/2 h-auto">
                <div className="p-6 space-y-6 shadow-sm executive-card">
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">NPS–CSAT Dashboard</h3>
                  <NpsDashboard />
                </div>
              </div>
              <div className="w-full sm:w-1/2 h-auto">
                <div className="p-6 space-y-6 executive-card shadow-sm">
                  {/* Header */}
                  <section className="space-y-2">
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">NPS–CSAT Overview</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      The current NPS and CSAT scores are tracked weekly/monthly by LOB, BU, and overall, and compared against business targets.
                    </p>
                  </section>

                  {/* A. NPS Trend */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">A. NPS Trend</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>Line graph showing the trend over the last 6–12 months or quarters.</li>
                      <li>Key events annotated (e.g., product launches, service changes) for context.</li>
                    </ul>
                  </section>

                  {/* B. NPS Breakdown */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">B. NPS Breakdown</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>
                        <strong>Promoters / Passives / Detractors:</strong> Display percentage and count.
                      </li>
                      <li>
                        <strong>Trend Comparison:</strong> Show current period vs previous period.
                      </li>
                      <li>
                        <strong>Segmented NPS:</strong> Breakdown by channel, product, region, or customer segment.
                      </li>
                    </ul>
                  </section>

                  {/* C. CSAT Trend */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">C. CSAT Trend</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>Line or bar chart showing CSAT trends by week or month.</li>
                      <li>Compare actual CSAT with internal targets or industry benchmarks.</li>
                    </ul>
                  </section>

                  {/* D. CSAT Breakdown */}
                  <section className="space-y-2">
                    <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">D. CSAT Breakdown</h5>
                    <ul className="list-disc pl-6 text-sm text-gray-600 dark:text-gray-300 space-y-1">
                      <li>
                        <strong>By Channel:</strong> Voice, chat, cases, self-service.
                      </li>
                      <li>
                        <strong>By Function:</strong> CSAT broken down by team, function, or product line.
                      </li>
                      <li>
                        <strong>Top Drop Drivers:</strong> Factors such as long wait times, agent knowledge gaps, and resolution delays.
                      </li>
                    </ul>
                  </section>
                </div>
              </div>
            </div>
          )}

          {activeTab === "Executive" &&
            <div>
              <ExecutiveDashboard defaultDecimalPrecisions={defaultDecimalPrecisions} children={children} />
            </div>
          }

          {activeTab === "InterActivity" &&
            <div>
              <InterActivity defaultDecimalPrecisions={defaultDecimalPrecisions} businessData={businessData} userData={userData} />
            </div>
          }

          {activeTab === "CXO" &&
            <div>
              <CXOInsights defaultDecimalPrecisions={defaultDecimalPrecisions} children={children} />
            </div>
          }

          {activeTab === "Forecasting" &&
            <div>
              <Forecasting defaultDecimalPrecisions={defaultDecimalPrecisions} children={children} />
            </div>
          }

          {activeTab === "PlanningHome" &&
            <div>
              <Planning defaultDecimalPrecisions={defaultDecimalPrecisions} children={children} />
            </div>
          }

          {activeTab === "SchedulingHome" &&
            <div>
              <Scheduling defaultDecimalPrecisions={defaultDecimalPrecisions} children={children} />
            </div>
          }


        </div>
      </CardContent>
    </Card>
  );
};

export default Home;
