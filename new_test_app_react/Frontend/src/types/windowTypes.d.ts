
interface Window {
  LOBTypes: string[];
}

// Define allowed LOB types for PlanningTab.tsx
declare type LOBType = 
  | "Customer Returns" 
  | "US Phone" 
  | "US Chat" 
  | "Core Support" 
  | "Inventory Management" 
  | "Dispute Management" 
  | "IBE Management" 
  | "FC Liaison" 
  | "Flex Team" 
  | "Help Desk" 
  | "MRC"
  | "OSS" 
  | "Seller Experience" 
  | "DSP" 
  | "Walmart CA" 
  | "Walmart Import";

// Define module declaration for components that use these types
declare module "*PlanningTab.tsx" {
  export interface FormattedData {
    lob: LOBType;
    // additional properties...
    [key: string]: any;
  }
}
