import { BusinessDashboard } from "@/copilot/BusinessDashboard";
import { ChatInterface } from "@/copilot/ChatInterface";

export function AIInsightsPage() {
  return (
    <div className="flex h-screen">
      <div className="flex-1 p-6">
        <BusinessDashboard />
      </div>
      <ChatInterface />
    </div>
  );
}
