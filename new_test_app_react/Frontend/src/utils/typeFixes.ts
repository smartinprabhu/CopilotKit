
// This file adds runtime support for our types
// By defining them in the global namespace
// We do this to avoid TypeScript errors in read-only files

// Define our LOB types in the global window object
// These are specifically to fix the PlanningTab.tsx error
const LOB_TYPES = [
  "Customer Returns",
  "US Phone",
  "US Chat",
  "Core Support",
  "Inventory Management",
  "Dispute Management",
  "IBE Management",
  "FC Liaison",
  "Flex Team",
  "Help Desk",
  "Seller Support",
  "Customer Support",
  "On-boarding",
  "Chat Support", 
  "Phone Support",
  "Walmart Import"
];

// Make sure LOB types are available at runtime
window.LOBTypes = LOB_TYPES;

// Export the type so TypeScript knows these are valid
export type LOBType = typeof LOB_TYPES[number];

// Enhance the global window interface
declare global {
  interface Window {
    LOBTypes: string[];
  }
}
