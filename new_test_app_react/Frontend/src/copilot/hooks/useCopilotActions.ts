import { useCopilotAction } from "@copilotkit/react-core";
import { useBusinessDataContext } from "../contexts/BusinessDataContext";

export function useBusinessCopilotActions() {
  const { setBusinessData, setForecastData } = useBusinessDataContext();

  // Fetch business units
  useCopilotAction({
    name: "fetchBusinessUnits",
    description: "Fetch all business unit data with current metrics",
    parameters: [],
    handler: async () => {
      const response = await fetch('/api/business-units');
      const data = await response.json();
      setBusinessData(data);
      return data;
    },
  });

  // Generate forecast
  useCopilotAction({
    name: "generateForecast",
    description: "Generate forecast for business metrics",
    parameters: [
      {
        name: "weeks",
        type: "number",
        description: "Number of weeks to forecast (1-52)",
        required: true,
      },
      {
        name: "modelType",
        type: "string",
        description: "Model: Prophet, ARIMA, LightGBM, XGBoost, SARIMAX, Ensemble, CatBoost",
        required: true,
      },
      {
        name: "variables",
        type: "array",
        description: "Variables to forecast",
        required: false,
      },
    ],
    handler: async ({ weeks, modelType, variables }) => {
      const formData = new FormData();
      formData.append('weeks', weeks.toString());
      formData.append('model_type', modelType);

      const response = await fetch('http://localhost:5011/forecast', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setForecastData(data.future_df);
      return {
        forecast: data.future_df,
        metrics: data.metrics,
        historical: data.dz_df,
      };
    },
  });

  // Analyze forecasts with AI
  useCopilotAction({
    name: "analyzeForecastsWithAI",
    description: "Get AI-powered insights on forecast data",
    parameters: [
      {
        name: "weeks",
        type: "number",
        required: true,
      },
      {
        name: "modelType",
        type: "string",
        required: true,
      },
    ],
    handler: async ({ weeks, modelType }) => {
      const formData = new FormData();
      formData.append('weeks', weeks.toString());
      formData.append('model_type', modelType);

      const response = await fetch('http://localhost:5011/analyze_forecasts', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      return data.insights;
    },
  });

  // Compare models
  useCopilotAction({
    name: "compareModels",
    description: "Compare accuracy of different forecasting models",
    parameters: [
      {
        name: "weeks",
        type: "number",
        required: true,
      },
      {
        name: "models",
        type: "array",
        description: "Array of model names to compare",
        required: true,
      },
    ],
    handler: async ({ weeks, models }) => {
      const results = [];

      for (const model of models) {
        const formData = new FormData();
        formData.append('weeks', weeks.toString());
        formData.append('model_type', model);

        const response = await fetch('http://localhost:5011/forecast', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        results.push({
          model,
          metrics: data.metrics,
        });
      }

      return results;
    },
  });
}
