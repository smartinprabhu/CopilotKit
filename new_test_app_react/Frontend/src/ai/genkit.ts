
// Mock implementation of the genkit ai interface
export const ai = {
  prompt: async (prompt: string) => {
    console.log("AI prompt called with:", prompt);
    
    // Return a mock response object
    return {
      text: async () => {
        return JSON.stringify({
          suggestedGroupings: [
            ["US Phone", "EU Phone"],
            ["APAC Chat", "Global Email"]
          ],
          reasoning: "Mock reasoning for AI response"
        });
      }
    };
  }
};
