import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Zap, CalendarCheck, LineChart, MonitorSmartphone, BarChart2, Settings } from "lucide-react";
import { useNavigate } from "react-router-dom";

import StrategicLandingPage from "./landingPage/StrategicLandingPage";
import PlanningTab from "../tacticalPlan/NewPage";

import type { ModelType } from "../tacticalPlan/models/shared/interfaces"


const PlanSelection = ({ navigateSimulator, businessId }) => {
  const [selectedModel, setSelectedModel] = useState<ModelType>('volume-backlog');

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
  } else if (feature.title === "Settings") {
    navigate("/company");
  } else {
    setSelectedFeature(feature);
    setIsModalOpen(true);
  }
};


  const [activeTab, setActiveTab] = useState<"Insights" | "Annual" | "Monthly">("Insights");

  return (
    <Card>
      <CardContent className="pt-2 mt-[-10px] mb-10">
        <div className="mb-6">
    
          <div className="flex border-b">
              <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "Insights" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("Insights")}
            >
              Insights
            </button>
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "Annual" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("Annual")}
            >
              Annual Plan
            </button>
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "Monthly" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("Monthly")}
            >
              Monthly Actual
            </button>
          </div>
        </div>

        <div className="mt-4">
          {activeTab === "Insights" && (
            <StrategicLandingPage 
              onStartPlanning={() => setActiveTab("Annual")} 
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          )}
          {activeTab === "Annual" && 
            <PlanningTab monthly navigateSimulator={navigateSimulator} businessId={businessId} selectedModel={selectedModel} />
          }
          {activeTab === "Monthly" && (
            <PlanningTab monthly navigateSimulator={navigateSimulator} businessId={businessId} selectedModel={selectedModel} />
          )}
         
        </div>
      </CardContent>
    </Card>
  );
};

export default PlanSelection;
