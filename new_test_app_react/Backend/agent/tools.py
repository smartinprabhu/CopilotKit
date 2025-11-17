GENERATE_BUSINESS_INSIGHTS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_business_insights",
        "description": """Generate comprehensive business insights including:
        - Performance analysis across business units
        - Trend identification
        - Risk assessment
        - Recommendations

        Returns structured insights with metrics and visualizations.""",
        "parameters": {
            "type": "object",
            "properties": {
                "insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Insight category: performance, risk, opportunity, trend"
                            },
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"]
                            },
                            "metrics": {
                                "type": "object",
                                "description": "Relevant metrics supporting the insight"
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["category", "title", "description", "severity"]
                    }
                }
            },
            "required": ["insights"]
        }
    }
}

GENERATE_FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_forecast",
        "description": """Generate forecast using specified ML model.

        Available models:
        - Prophet: Time series with seasonality
        - ARIMA: Auto-regressive integrated moving average
        - LightGBM: Gradient boosting
        - XGBoost: Extreme gradient boosting
        - SARIMAX: Seasonal ARIMA with exogenous variables
        - Ensemble: Combined model predictions
        - CatBoost: Categorical boosting

        Returns forecast data with confidence intervals.""",
        "parameters": {
            "type": "object",
            "properties": {
                "weeks": {
                    "type": "number",
                    "description": "Number of weeks to forecast (1-52)"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["Prophet", "ARIMA", "LightGBM", "XGBoost", "SARIMAX", "Ensemble", "CatBoost"]
                },
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Total IB Units", "inventory", "returns", "exceptions", "wfs_china"]
                    }
                }
            },
            "required": ["weeks", "model_type"]
        }
    }
}

GENERATE_VISUALIZATION_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_visualization",
        "description": """Generate dynamic chart configuration for business data.

        Supported chart types:
        - line: Time series trends
        - bar: Comparisons
        - pie: Distributions
        - scatter: Correlations
        - area: Cumulative trends

        Returns Recharts-compatible configuration.""",
        "parameters": {
            "type": "object",
            "properties": {
                "chartType": {
                    "type": "string",
                    "enum": ["line", "bar", "pie", "scatter", "area"]
                },
                "data": {
                    "type": "array",
                    "description": "Data points for visualization"
                },
                "xAxis": {"type": "string"},
                "yAxis": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["chartType", "data", "title"]
        }
    }
}
