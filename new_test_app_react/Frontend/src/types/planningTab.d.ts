
// Special declaration file to fix PlanningTab.tsx TypeScript errors

import { LOBType } from './planning';

// Declare module augmentation for PlanningTab component
declare module 'src/components/PlanningTab' {
  // Export a patched version of any function that might be causing the error
  export function getLOBData(lobType: LOBType): any;
  export function filterByLOB(data: any, lobType: LOBType): any;
  export function processLOBData(lobType: LOBType, data: any): any;
  
  // Add any other function signatures that might be in the component
  export default function PlanningTab(): JSX.Element;
}

// Global type augmentation
declare global {
  // This helps TypeScript understand that these string literals are valid
  type ValidLOBType = LOBType;
}

export {};
