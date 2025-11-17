from typing import List, Dict, Any
from copilotkit import CopilotKitState

class BusinessInsightsState(CopilotKitState):
    """
    State for the business insights agent.
    Inherits from CopilotKitState for CopilotKit integration.
    """
    business_units: List[Dict[str, Any]] = []
    forecast_data: List[Dict[str, Any]] = []
    analysis_results: Dict[str, Any] = {}
    visualizations: List[Dict[str, Any]] = []
    selected_model: str = "Prophet"
    forecast_weeks: int = 12
