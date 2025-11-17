"""
Business domain knowledge base for the AI agent
"""

BUSINESS_UNITS_KNOWLEDGE = """
Business Units Overview:
- Each business unit represents a distinct operational division
- Units are categorized by Line of Business (LOB)
- Performance is measured across multiple KPIs

Lines of Business:
1. Retail Operations
2. E-commerce Fulfillment
3. B2B Distribution
4. International Markets
5. Direct-to-Consumer

Key Performance Indicators:
- Total IB Units: Inventory buffer to handle demand fluctuations
- Inventory Turnover: How quickly stock is sold and replaced
- Return Rate: Percentage of products returned by customers
- Exception Rate: Supply chain issues requiring intervention
- Fulfillment Speed: Time from order to delivery
"""

FORECASTING_MODELS_KNOWLEDGE = """
Forecasting Models Explained:

1. Prophet (Facebook's Time Series Model)
   - Best for: Data with strong seasonal patterns
   - Strengths: Handles missing data, outliers, holidays
   - Use when: You have daily/weekly data with clear seasonality
   - Accuracy: High for seasonal business patterns

2. ARIMA (AutoRegressive Integrated Moving Average)
   - Best for: Stationary time series data
   - Strengths: Classical statistical approach, interpretable
   - Use when: Data shows clear trends without complex seasonality
   - Accuracy: Good for short-term forecasts

3. LightGBM (Light Gradient Boosting Machine)
   - Best for: Complex patterns with multiple features
   - Strengths: Fast training, handles large datasets
   - Use when: You have additional features beyond time
   - Accuracy: Very high with proper feature engineering

4. XGBoost (Extreme Gradient Boosting)
   - Best for: Non-linear relationships
   - Strengths: Robust to outliers, feature importance
   - Use when: Data has complex interactions
   - Accuracy: Excellent for structured data

5. SARIMAX (Seasonal ARIMA with Exogenous Variables)
   - Best for: Seasonal data with external factors
   - Strengths: Incorporates external variables
   - Use when: External factors influence your metrics
   - Accuracy: High for seasonal patterns with regressors

6. Ensemble (Combined Models)
   - Best for: Maximum accuracy
   - Strengths: Reduces individual model weaknesses
   - Use when: Accuracy is critical
   - Accuracy: Typically highest overall

7. CatBoost (Categorical Boosting)
   - Best for: Data with categorical features
   - Strengths: Handles categories natively
   - Use when: Business units have categorical attributes
   - Accuracy: Excellent for mixed data types

Model Selection Guidelines:
- Start with Prophet for initial analysis
- Use Ensemble for production forecasts
- Compare multiple models for validation
- Consider computational cost vs accuracy trade-offs
"""

METRICS_INTERPRETATION = """
Metric Interpretation Guide:

Total IB Units (Inventory Buffer):
- Optimal Range: 2-4 weeks of demand
- Too High: Excess holding costs, obsolescence risk
- Too Low: Stockout risk, lost sales
- Action Triggers:
  * >20% above target: Review procurement
  * <20% below target: Expedite orders

Inventory Management:
- Turnover Ratio: Sales / Average Inventory
- Industry Benchmark: 8-12 times per year
- Low Turnover: Slow-moving stock, capital tied up
- High Turnover: Efficient operations, potential stockout risk

Customer Returns:
- Acceptable Rate: <5% for most categories
- High Returns: Quality issues, wrong descriptions
- Seasonal Patterns: Higher during holidays
- Cost Impact: 15-30% of product value

Inbound Exceptions:
- Target: <2% of total shipments
- Common Causes: Delays, damage, quantity mismatches
- Impact: Disrupts planning, increases costs
- Prevention: Supplier quality programs

WFS China Operations:
- Special Considerations: Longer lead times, customs
- Peak Seasons: Chinese New Year, Singles Day
- Risk Factors: Geopolitical, regulatory changes
"""

BUSINESS_INSIGHTS_TEMPLATES = """
Insight Generation Templates:

Performance Insight:
- Category: performance
- Title: "[Unit] shows [trend] in [metric]"
- Description: Detailed analysis with numbers
- Severity: Based on deviation from target
- Recommendations: Specific actions

Risk Insight:
- Category: risk
- Title: "Potential [risk type] in [area]"
- Description: Risk factors and probability
- Severity: Impact assessment
- Recommendations: Mitigation strategies

Opportunity Insight:
- Category: opportunity
- Title: "[Opportunity] identified in [area]"
- Description: Potential benefits
- Severity: Priority level
- Recommendations: Implementation steps

Trend Insight:
- Category: trend
- Title: "[Metric] trending [direction]"
- Description: Pattern analysis
- Severity: Significance level
- Recommendations: Strategic adjustments
"""
