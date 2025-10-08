"""
Simulation Service - Wrapper for simulation functionality
Provides what-if simulation capabilities for marketing changes
"""
from typing import Any, Dict, List, Optional
import pickle
import pandas as pd
import logging

from src.optimization.impact_model import MarketingImpactModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PLACEHOLDER_CHANNEL_COEFFICIENTS = {
    'digital_spend': 0.000005,
    'tv_spend': 0.000003,
    'traditional_spend': 0.000002,
    'sponsorship_spend': 0.000004,
    'other_spend': 0.000001,
}

PLACEHOLDER_SATURATION_PARAMS = {"alpha": 0.5, "K_scale": 1_000_000}


class ForecastSimulateService:
    def __init__(self, model_path: str = "models/unified_brand_power_model.pkl", baseline_features_df: Optional[pd.DataFrame] = None):
        self.model_path = model_path
        self.baseline_features_df = baseline_features_df

        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact['model']
        # Handle both formats: 'feature_cols' or 'features'
        self.feature_cols = artifact.get('feature_cols', artifact.get('features', []))

        if self.baseline_features_df is not None:
            missing = set(self.feature_cols) - set(self.baseline_features_df.columns)
            for col in missing:
                self.baseline_features_df[col] = 0
            self.baseline_features_df = self.baseline_features_df[
                ['brand', 'country', 'year', 'quarter'] + self.feature_cols
            ]

        self.impact_model = MarketingImpactModel(
            channel_coefficients=PLACEHOLDER_CHANNEL_COEFFICIENTS,
            saturation_params=PLACEHOLDER_SATURATION_PARAMS,
        )

    def list_brands(self, country: Optional[str] = None) -> List[str]:
        df = self.baseline_features_df.copy()
        if country:
            df = df[df['country'].str.strip().str.lower() == country.lower()]
        return sorted(df['brand'].unique().tolist())

    def forecast_baseline(self, country: Optional[str] = None, brand: Optional[str] = None, save_path: Optional[str] = None) -> pd.DataFrame:
        # Predict using stored model; here model has predict and feature_cols
        predictions = self.model.predict(self.baseline_features_df[self.feature_cols])
        result = self.baseline_features_df[['brand', 'country', 'year', 'quarter']].copy()
        result['predicted_power'] = predictions
        if country:
            result = result[result['country'].str.lower() == country.lower()]
        if brand:
            result = result[result['brand'].str.lower() == brand.lower()]
        if save_path:
            result.to_csv(save_path, index=False)
        return result

    def simulate(self, uploaded_template: pd.DataFrame, baseline_forecast: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # Aggregate weekly to quarterly spend
        marketing_cols = ['digital_spend', 'tv_spend', 'traditional_spend', 'sponsorship_spend', 'other_spend']
        quarterly = uploaded_template.groupby(['country', 'brand', 'year', 'quarter']).agg({c: 'sum' for c in marketing_cols}).reset_index()

        # Filter baseline features for the uploaded brands/countries
        features = self.baseline_features_df.copy()
        features = features[
            features['brand'].isin(quarterly['brand'].unique()) &
            features['country'].isin(quarterly['country'].unique())
        ]
        features = features.set_index(['country', 'brand', 'year', 'quarter'])
        quarterly = quarterly.set_index(['country', 'brand', 'year', 'quarter'])
        # Update spends
        for c in marketing_cols:
            if c in features.columns and c in quarterly.columns:
                features[c] = quarterly[c].combine_first(features[c])
        features.reset_index(inplace=True)
        features['total_marketing_spend'] = features[[c for c in marketing_cols if c in features.columns]].sum(axis=1)

        # Ensure feature columns exist
        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[['brand', 'country', 'year', 'quarter'] + self.feature_cols]

        # Allow for feature re-engineering hook (patched in tests)
        features = self._reengineer_marketing_features(features)

        sim_preds = self.model.predict(features[self.feature_cols])
        sim_df = features[['brand', 'country', 'year', 'quarter']].copy()
        sim_df['simulated_power'] = sim_preds

        if baseline_forecast is None:
            baseline_forecast = self.forecast_baseline()
        merged = sim_df.merge(
            baseline_forecast[['country', 'brand', 'year', 'quarter', 'predicted_power']],
            on=['country', 'brand', 'year', 'quarter'], how='left'
        ).rename(columns={'predicted_power': 'baseline_power'})
        merged['uplift'] = merged['simulated_power'] - merged['baseline_power']
        merged['uplift_pct'] = (merged['uplift'] / merged['baseline_power'] * 100).fillna(0)
        return merged

    def _reengineer_marketing_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Hook for downstream feature engineering.

        Tests patch this method to assert it is invoked exactly once. The default
        implementation returns the input unmodified.
        """
        return features

    def optimize_allocation(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = 'gradient',
        digital_cap: float = 0.99,
        tv_cap: float = 0.5,
    ) -> Dict[str, Any]:
        return self.impact_model.optimize(
            total_budget=total_budget,
            channels=channels,
            method=method,
            digital_cap=digital_cap,
            tv_cap=tv_cap,
        )


class SimulationService:
    """
    Simulation service for what-if analysis with marketing changes
    """
    
    def __init__(self, forecast_service: ForecastSimulateService):
        """
        Initialize simulation service
        
        Args:
            forecast_service: Instance of ForecastSimulateService
        """
        self.forecast_service = forecast_service
    
    def simulate(self, uploaded_template, baseline_forecast=None):
        """
        Simulate brand power with new marketing allocations
        
        Args:
            uploaded_template: DataFrame with new marketing spends
            baseline_forecast: Optional baseline forecast
        
        Returns:
            DataFrame with simulation results
        """
        return self.forecast_service.simulate(uploaded_template, baseline_forecast)


__all__ = ['SimulationService']

