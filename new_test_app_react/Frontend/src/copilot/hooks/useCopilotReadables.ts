import { useCopilotReadable } from "@copilotkit/react-core";
import { useBusinessDataContext } from "../contexts/BusinessDataContext";

export function useBusinessCopilotReadables() {
  const { businessData, forecastData } = useBusinessDataContext();

  // Make business data readable
  useCopilotReadable({
    description: `Current business unit data including:
    - Unit IDs and names
    - Lines of business
    - Current metrics (inventory, returns, exceptions)
    - Performance indicators`,
    value: businessData,
  });

  // Make forecast data readable
  useCopilotReadable({
    description: `Forecast data including:
    - Predicted values for future weeks
    - Confidence intervals (upper/lower bounds)
    - Model used for predictions
    - Accuracy metrics`,
    value: forecastData,
  });

  // Make model information readable
  useCopilotReadable({
    description: `Available forecasting models:
    - Prophet: Best for seasonal patterns
    - ARIMA: Good for trends
    - LightGBM: Fast and accurate
    - XGBoost: Handles complex patterns
    - SARIMAX: Seasonal with external factors
    - Ensemble: Combined predictions
    - CatBoost: Handles categorical data`,
    value: {
      models: ["Prophet", "ARIMA", "LightGBM", "XGBoost", "SARIMAX", "Ensemble", "CatBoost"],
      defaultModel: "Prophet",
      defaultWeeks: 12,
    },
  });
}
