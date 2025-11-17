"""
Integration layer between LangGraph agent and existing Flask API
"""
import httpx
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class FlaskAPIClient:
    """Client for interacting with existing Flask forecasting API"""

    def __init__(self, base_url: str = "http://localhost:5011"):
        self.base_url = base_url

    async def generate_forecast(
        self,
        weeks: int,
        model_type: str,
        use_regressors: bool = False,
        file_data: bytes = None
    ) -> Dict[str, Any]:
        """
        Call the existing /forecast endpoint
        """
        try:
            form_data = {
                'weeks': str(weeks),
                'model_type': model_type,
                'use_regressors': 'true' if use_regressors else 'false'
            }

            files = {}
            if file_data:
                files['file'] = ('data.xlsx', file_data)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/forecast",
                    data=form_data,
                    files=files if files else None,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Error calling forecast API: {e}")
            return {"error": str(e)}

    async def analyze_forecasts(
        self,
        weeks: int,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Call the existing /analyze_forecasts endpoint
        """
        try:
            form_data = {
                'weeks': str(weeks),
                'model_type': model_type
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/analyze_forecasts",
                    data=form_data,
                    timeout=120
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Error calling analyze API: {e}")
            return {"error": str(e)}

# Global client instance
flask_client = FlaskAPIClient()
