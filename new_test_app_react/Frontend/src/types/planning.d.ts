
// Type definitions to fix PlanningTab.tsx type errors

// Define possible LOB (Line of Business) values
export type LOBType = 
  | "Customer Returns"
  | "US Chat"
  | "US Phone"
  | "Core Support"
  | "Inventory Management"
  | "Dispute Management"
  | "IBE Management"
  | "FC Liaison"
  | "Flex Team"
  | "Help Desk"
  | "Onboarding"
  | "Customer Success"
  | "KYC"
  | "Tech Support"
  | "Product Support"
  | "Walmart Cash"
  | "Walmart Import";

// This makes the type definitions available globally in the project
declare global {
  interface Window {
    LOBTypes: LOBType[];
  }
  
  // Add a global namespace for LOB types
  namespace NodeJS {
    interface Global {
      LOBTypes: LOBType[];
    }
  }
}

// Create a module augmentation for the PlanningTab component
declare module "*.tsx" {
  interface PlanningTabProps {
    lobOptions?: LOBType[];
  }
  
  // Export functions that might be causing type errors
  export function handleLOBSelection(lob: LOBType): void;
  export function filterDataByLOB(lob: LOBType): any;
  export function getLOBMetrics(lob: LOBType): any;
}

// Make sure the declarations are merged properly
export {};
