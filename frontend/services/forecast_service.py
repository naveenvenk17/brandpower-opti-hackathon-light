"""
Frontend Forecast Service - Simple Interface to Backend

Provides:
1. get_baseline_forecast() - Load baseline forecast
2. simulate_with_upload() - Simulate with user marketing data
"""


import pandas as pd
import logging
from typing import Optional

from production_scripts.services.forecast_simulate_service import ForecastSimulateService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrandPowerForecastService:
    """
    Frontend service for brand power forecasting

    Methods:
    - get_baseline_forecast: Get baseline predictions
    - simulate_with_upload: Simulate with new marketing spends and get uplift
    """

    def __init__(
        self,
        model_path: str = "production_scripts/models/brand_power_forecaster.pkl",
        forecast_features_path: str = "data/forecast_features.csv",
        baseline_forecast_path: str = "data/baseline_forecast.csv"
    ):
        """
        Initialize service

        Args:
            model_path: Path to trained model
            forecast_features_path: Path to forecast features (108 rows)
            baseline_forecast_path: Path to baseline forecast (pre-computed)
        """
        # Initialize backend service
        self.service = ForecastSimulateService(
            model_path=model_path,
            forecast_features_path=forecast_features_path
        )

        # Load baseline forecast
        self.baseline_forecast_path = baseline_forecast_path
        try:
            self.baseline_forecast = pd.read_csv(baseline_forecast_path)
            logger.info(f"✓ Baseline forecast loaded: {len(self.baseline_forecast)} rows")
        except FileNotFoundError:
            logger.warning(f"Baseline forecast not found, generating...")
            self.baseline_forecast = self.service.forecast_baseline(save_path=baseline_forecast_path)

    def get_baseline_forecast(self, country: Optional[str] = None) -> pd.DataFrame:
        """
        Get baseline forecast

        Args:
            country: Optional country filter (e.g., "brazil", "us", "colombia")

        Returns:
            DataFrame with columns: country, brand, year, quarter, predicted_power
        """
        baseline = self.baseline_forecast.copy()

        if country:
            baseline = baseline[baseline['country'].str.lower() == country.lower()]

        return baseline

    def simulate_with_upload(
        self,
        uploaded_data: pd.DataFrame,
        country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Simulate brand power with uploaded marketing data

        Args:
            uploaded_data: DataFrame with WEEKLY marketing data
                          Columns: country, brand, year, quarter, month, week_of_month,
                                  digital_spend, tv_spend, traditional_spend,
                                  sponsorship_spend, other_spend, power (ignored)
            country: Optional country filter

        Returns:
            DataFrame with columns: country, brand, year, quarter,
                                   baseline_power, simulated_power, uplift, uplift_pct
        """
        logger.info("Simulating with uploaded data...")

        # Filter by country if specified
        if country:
            uploaded_data = uploaded_data[uploaded_data['country'].str.lower() == country.lower()].copy()

        # Call backend service
        result = self.service.simulate(
            uploaded_template=uploaded_data,
            baseline_forecast=self.baseline_forecast
        )

        logger.info(f"✓ Simulation complete: {len(result)} rows")
        logger.info(f"   Average uplift: {result['uplift'].mean():.4f} ({result['uplift_pct'].mean():.2f}%)")

        return result
