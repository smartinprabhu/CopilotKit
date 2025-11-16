import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import PlanningTab from "../../tacticalPlan/NewPage";

import TacticalLandingPage from "./TacticalLandingPage";
import type { ModelType } from "../../tacticalPlan/models/shared/interfaces";

const TacticalPlanSelection = ({ navigateSimulator, businessId }) => {
  const [activeTab, setActiveTab] = useState<"Overview" | "Plan">("Overview");
  const [planData, setPlanData] = useState(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('volume-backlog');

  // Handle data changes from plan page
  const handlePlanDataChange = (newData: any) => {
    setPlanData(newData);
    // This will trigger re-render of overview with updated data
  };

  // Handle model changes
  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);
    // This will update both overview and plan pages
  };

  return (
    <Card>
      <CardContent className="pt-2 mt-[-10px] mb-10">
        <div className="mb-6">
          <div className="flex border-b">
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "Overview" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("Overview")}
            >
             Insights
            </button>
            <button
              className={`py-2 px-4 text-sm font-medium ${activeTab === "Plan" ? "border-b-2 border-primary text-primary" : ""}`}
              onClick={() => setActiveTab("Plan")}
            >
              Detailed Planning
            </button>
          </div>
        </div>

        <div className="mt-4">
          {activeTab === "Overview" && (
            <TacticalLandingPage 
              onStartPlanning={() => setActiveTab("Plan")} 
              planData={planData}
              onDataChange={handlePlanDataChange}
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
            />
          )}
          {activeTab === "Plan" && (
            <PlanningTab 
              whatIfDetails={{}} 
              monthly={false} 
              navigateSimulator={navigateSimulator} 
              businessId={businessId}
              selectedModel={selectedModel}
              onDataChange={handlePlanDataChange}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TacticalPlanSelection;