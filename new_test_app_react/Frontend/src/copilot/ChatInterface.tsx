import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { useBusinessDataContext } from "./contexts/BusinessDataContext";
import { businessInstructions } from "./lib/prompts";

export function ChatInterface() {
  const { businessData, forecastData } = useBusinessDataContext();

  const instructions = businessInstructions
    .replace("{businessData}", JSON.stringify(businessData))
    .replace("{forecastData}", JSON.stringify(forecastData));

  return (
    <div className="flex h-full w-80 flex-col border-l bg-background">
      <div className="flex items-center justify-between border-b px-4 py-4">
        <h2 className="font-semibold">Business Insights AI</h2>
      </div>
      <CopilotChat
        className="flex-1 min-h-0 py-4"
        instructions={instructions}
      />
    </div>
  );
}
