import { createContext, useContext, useState, ReactNode } from 'react';
import { useCopilotReadable } from "@copilotkit/react-core";

interface BusinessUnit {
  id: string;
  name: string;
  lineOfBusiness: string;
  metrics: {
    totalIBUnits: number;
    inventory: number;
    returns: number;
    exceptions: number;
  };
}

interface BusinessDataContextType {
  businessData: BusinessUnit[];
  setBusinessData: (data: BusinessUnit[]) => void;
  forecastData: any[];
  setForecastData: (data: any[]) => void;
}

const BusinessDataContext = createContext<BusinessDataContextType | undefined>(undefined);

export function BusinessDataProvider({ children }: { children: ReactNode }) {
  const [businessData, setBusinessData] = useState<BusinessUnit[]>([]);
  const [forecastData, setForecastData] = useState<any[]>([]);

  // Make data readable by CopilotKit
  useCopilotReadable({
    description: "Current business unit data with metrics",
    value: businessData,
  });

  useCopilotReadable({
    description: "Forecast data for business units",
    value: forecastData,
  });

  return (
    <BusinessDataContext.Provider
      value={{ businessData, setBusinessData, forecastData, setForecastData }}
    >
      {children}
    </BusinessDataContext.Provider>
  );
}

export function useBusinessDataContext() {
  const context = useContext(BusinessDataContext);
  if (!context) {
    throw new Error('useBusinessDataContext must be used within BusinessDataProvider');
  }
  return context;
}
