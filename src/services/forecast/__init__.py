"""
Forecast Service - AutoGluon-based forecasting
"""
from src.services.forecast.forecast_service import AutoGluonForecastService, get_forecast_service

# Re-export for convenience
__all__ = [
    "AutoGluonForecastService",
    "get_forecast_service",
]
