import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, Table2, TrendingUp } from "lucide-react";
import { useCoAgentStateRender } from "@copilotkit/react-core";
import { useCopilotChatSuggestions } from "@copilotkit/react-ui";
import { useBusinessDataContext } from "./contexts/BusinessDataContext";
import { useBusinessCopilotActions } from "./hooks/useCopilotActions";
import { useBusinessCopilotReadables } from "./hooks/useCopilotReadables";
import { businessSuggestions } from "./lib/prompts";

export function BusinessDashboard() {
  const [viewMode, setViewMode] = useState<"overview" | "forecast" | "analysis">("overview");
  const { businessData, setBusinessData, forecastData, setForecastData } = useBusinessDataContext();
  const [loading, setLoading] = useState(true);

  // Set up Copilot actions and readables
  useBusinessCopilotActions();
  useBusinessCopilotReadables();

  // Dynamic suggestions based on context
  useCopilotChatSuggestions({
    instructions: businessSuggestions,
    minSuggestions: 2,
    maxSuggestions: 4,
  });

  // Agent state rendering for visualizations
  // Placeholder for DynamicVisualization component
  const DynamicVisualization = ({ state, status }: { state: any, status: any }) => {
    return (
      <div>
        {/* This is a placeholder for the dynamic visualization component. */}
        {/* It will be implemented in a future step. */}
        <pre>{JSON.stringify(state, null, 2)}</pre>
      </div>
    );
  };

  useCoAgentStateRender({
    name: "business_insights_agent",
    render: (props) => {
      return <DynamicVisualization state={props.state} status={props.status} />;
    },
  });

  useEffect(() => {
    loadInitialData();
  }, []);

  async function loadInitialData() {
    // Load business data
    setLoading(false);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight">
          Business Insights Dashboard
        </h1>
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === "overview" ? "default" : "outline"}
            size="sm"
            onClick={() => setViewMode("overview")}
          >
            <Table2 className="mr-2 h-4 w-4" />
            Overview
          </Button>
          <Button
            variant={viewMode === "forecast" ? "default" : "outline"}
            size="sm"
            onClick={() => setViewMode("forecast")}
          >
            <TrendingUp className="mr-2 h-4 w-4" />
            Forecast
          </Button>
          <Button
            variant={viewMode === "analysis" ? "default" : "outline"}
            size="sm"
            onClick={() => setViewMode("analysis")}
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            Analysis
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            {viewMode === "overview" && "Business Units Overview"}
            {viewMode === "forecast" && "Forecast Analysis"}
            {viewMode === "analysis" && "AI-Powered Insights"}
          </CardTitle>
          <CardDescription>
            {viewMode === "overview" && "Monitor key metrics across all business units"}
            {viewMode === "forecast" && "View predictive analytics and trends"}
            {viewMode === "analysis" && "Get AI-generated insights and recommendations"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {viewMode === 'overview' && (
            <div>
              <h3 className="text-lg font-semibold">Business Unit Data</h3>
              <pre>{JSON.stringify(businessData, null, 2)}</pre>
            </div>
          )}
          {viewMode === 'forecast' && (
            <div>
              <h3 className="text-lg font-semibold">Forecast Data</h3>
              <pre>{JSON.stringify(forecastData, null, 2)}</pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
