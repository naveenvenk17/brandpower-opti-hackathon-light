"""
Brand Power Forecast and Simulate Service

Provides two main functions:
1. forecast() - Generate baseline forecast from forecast_features.csv
2. simulate() - Update marketing spends, re-engineer features, predict uplift
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
import pickle

from production_scripts.models.marketing.impact_model import MarketingImpactModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for channel coefficients (example values)
# In a real scenario, these would be derived from a trained model
PLACEHOLDER_CHANNEL_COEFFICIENTS = {
    'digital_spend': 0.000005,
    'tv_spend': 0.000003,
    'traditional_spend': 0.000002,
    'sponsorship_spend': 0.000004,
    'other_spend': 0.000001
}

# Placeholder for saturation parameters (example values)
# In a real scenario, these would be derived from data analysis
PLACEHOLDER_SATURATION_PARAMS = {
    'alpha': 0.5,  # Diminishing returns parameter
    'K_scale': 1_000_000 # Scale factor for K (half-saturation point)
}


class ForecastSimulateService:
    """
    Service for forecasting and simulating brand power with marketing changes
    """

    def __init__(
        self,
        model_path: str = "production_scripts/models/brand_power_forecaster.pkl",
        forecast_features_path: str = "data/forecast_features.csv"
    ):
        """
        Initialize service

        Args:
            model_path: Path to trained LightGBM model (.pkl file)
            forecast_features_path: Path to forecast features CSV
        """
        self.model_path = model_path
        self.forecast_features_path = forecast_features_path

        # Load LightGBM model
        logger.info(f"Loading LightGBM model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_artifact = pickle.load(f)
        self.model = model_artifact['model']
        self.feature_cols = model_artifact['feature_cols']
        logger.info(f"✓ LightGBM model loaded with {len(self.feature_cols)} features.")

        # Load forecast features (baseline)
        logger.info(f"Loading forecast features from {forecast_features_path}...")
        self.baseline_features_df = pd.read_csv(forecast_features_path)
        logger.info(f"✓ Loaded {len(self.baseline_features_df)} forecast rows")

        # Ensure all feature columns are present in the baseline features dataframe
        missing_features = set(self.feature_cols) - set(self.baseline_features_df.columns)
        if missing_features:
            logger.warning(f"Missing features in forecast_features.csv: {missing_features}. Filling with 0.")
            for col in missing_features:
                self.baseline_features_df[col] = 0

        # Reorder columns to match training data
        self.baseline_features_df = self.baseline_features_df[
            ['brand', 'country', 'year', 'quarter'] + self.feature_cols
        ]

        # Initialize MarketingImpactModel for optimization
        self.impact_model = MarketingImpactModel(
            channel_coefficients=PLACEHOLDER_CHANNEL_COEFFICIENTS,
            saturation_params=PLACEHOLDER_SATURATION_PARAMS
        )
        logger.info("✓ MarketingImpactModel initialized for optimization.")

    def optimize_allocation(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = 'gradient',
        digital_cap: float = 0.99,
        tv_cap: float = 0.5
    ) -> Dict[str, Any]:
        """
        Optimize marketing allocation using the MarketingImpactModel.

        Args:
            total_budget: Total marketing budget.
            channels: List of channels to optimize.
            method: Optimization method ('gradient' or 'evolutionary').
            digital_cap: Maximum fraction of budget for digital.
            tv_cap: Maximum fraction of budget for TV.

        Returns:
            Dictionary containing optimal allocation, expected lift, ROI, etc.
        """
        logger.info(f"Starting optimization for total budget: {total_budget}")
        optimization_results = self.impact_model.optimize(
            total_budget=total_budget,
            channels=channels,
            method=method,
            digital_cap=digital_cap,
            tv_cap=tv_cap
        )
        logger.info("✓ Optimization complete.")
        return optimization_results

    def _quarter_to_month(self, quarter_str):
        q_map = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
        return q_map[quarter_str]

    def forecast_baseline(self, country: Optional[str] = None, brand: Optional[str] = None, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate baseline forecast for all rows in forecast_features.csv

        Args:
            country: Optional country to filter the forecast.
            brand: Optional brand to filter the forecast.
            save_path: Optional path to save the forecast.

        Returns:
            DataFrame with columns: brand, country, year, quarter, predicted_power
        """
        logger.info("Generating baseline forecast...")

        # Make predictions using the LightGBM model
        predictions = self.model.predict(self.baseline_features_df[self.feature_cols])

        result = self.baseline_features_df[['brand', 'country', 'year', 'quarter']].copy()
        result['predicted_power'] = predictions

        if country:
            result = result[result['country'] == country]
        if brand:
            result = result[result['brand'] == brand]

        logger.info(f"✓ Generated {len(result)} baseline forecasts")
        logger.info(f"   Power range: [{result['predicted_power'].min():.4f}, {result['predicted_power'].max():.4f}]")

        # Save if path provided
        if save_path:
            result.to_csv(save_path, index=False)
            logger.info(f"✓ Saved baseline forecast to {save_path}")

        return result

    def simulate(
        self,
        uploaded_template: pd.DataFrame,
        baseline_forecast: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Simulate brand power with new marketing allocations

        Args:
            uploaded_template: Weekly data with new marketing spends
                              Columns: country, brand, year, quarter, month, week_of_month,
                                      digital_spend, tv_spend, traditional_spend,
                                      sponsorship_spend, other_spend, power (ignored)
            baseline_forecast: Optional baseline forecast to calculate uplift against
                              If None, will generate baseline first

        Returns:
            DataFrame with columns: country, brand, year, quarter,
                                   baseline_power, simulated_power, uplift, uplift_pct
        """
        logger.info("Starting simulation...")
        logger.info(f"   Input: {len(uploaded_template)} weekly rows")

        # Step 1: Aggregate weekly -> quarterly
        logger.info("Step 1: Aggregating weekly spends to quarterly...")

        marketing_cols = ['digital_spend', 'tv_spend', 'traditional_spend',
                         'sponsorship_spend', 'other_spend']

        quarterly_data = uploaded_template.groupby(['country', 'brand', 'year', 'quarter']).agg({
            col: 'sum' for col in marketing_cols
        }).reset_index()

        logger.info(f"   Aggregated to {len(quarterly_data)} quarterly rows")

        # Step 2: Load baseline forecast features (already loaded in __init__)
        logger.info("Step 2: Preparing features for simulation...")
        simulated_features_df = self.baseline_features_df.copy()

        # Step 3: Update marketing columns
        logger.info("Step 3: Updating marketing columns with new values...")

        # Create merge key
        merge_keys = ['country', 'brand', 'year', 'quarter']

        # Set index for efficient update
        simulated_features_df = simulated_features_df.set_index(merge_keys)
        quarterly_data = quarterly_data.set_index(merge_keys)

        # Update the spend columns
        simulated_features_df.update(quarterly_data[marketing_cols])
        simulated_features_df.reset_index(inplace=True)

        # Update total_marketing_spend
        simulated_features_df['total_marketing_spend'] = simulated_features_df[marketing_cols].sum(axis=1)

        logger.info("   ✓ Marketing columns updated")

        # Step 4: Re-engineer marketing-dependent features
        logger.info("Step 4: Re-engineering marketing-dependent features...")
        simulated_features_df = self._reengineer_marketing_features(simulated_features_df)
        logger.info("   ✓ Marketing features re-engineered")

        # Step 5: Predict with updated features using LightGBM
        logger.info("Step 5: Predicting power with updated features using LightGBM...")

        # Ensure all required feature columns are present in simulated_features_df
        missing_features = set(self.feature_cols) - set(simulated_features_df.columns)
        if missing_features:
            logger.warning(f"   Missing features in simulated data: {missing_features}. Filling with 0.")
            for col in missing_features:
                simulated_features_df[col] = 0

        # Reorder columns to match training data
        simulated_features_df = simulated_features_df[['brand', 'country', 'year', 'quarter'] + self.feature_cols]

        # Make predictions
        simulated_predictions = self.model.predict(simulated_features_df[self.feature_cols])

        simulated_result = simulated_features_df[['brand', 'country', 'year', 'quarter']].copy()
        simulated_result['simulated_power'] = simulated_predictions

        logger.info(f"   ✓ Predicted {len(simulated_result)} values")
        logger.info(f"   Power range: [{simulated_result['simulated_power'].min():.4f}, {simulated_result['simulated_power'].max():.4f}]")

        # Step 6: Get baseline forecast if not provided
        if baseline_forecast is None:
            logger.info("Step 6: Generating baseline forecast for comparison...")
            baseline_forecast = self.forecast_baseline()

        # Step 7: Calculate uplift
        logger.info("Step 7: Calculating uplift...")

        result = simulated_result.merge(
            baseline_forecast[['country', 'brand', 'year', 'quarter', 'predicted_power']],
            on=['country', 'brand', 'year', 'quarter'],
            how='left'
        )
        result = result.rename(columns={'predicted_power': 'baseline_power'})

        # Calculate uplift
        result['uplift'] = result['simulated_power'] - result['baseline_power']
        result['uplift_pct'] = (result['uplift'] / result['baseline_power'] * 100).fillna(0)

        logger.info("   ✓ Uplift calculated")
        logger.info(f"   Average uplift: {result['uplift'].mean():.4f} ({result['uplift_pct'].mean():.2f}%)")

        logger.info("✓ Simulation complete!")

        return result

    def _reengineer_marketing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-engineer marketing-dependent features after updating marketing spends

        Re-calculates:
        - Adstock features (for each channel)
        - Saturation features
        - Synergy features
        - Share of voice
        - Marketing interactions

        Args:
            df: DataFrame with updated marketing columns

        Returns:
            DataFrame with re-engineered marketing features
        """
        # Sort by country, brand, year, quarter for time-series features
        df = df.sort_values(['country', 'brand', 'year', 'quarter']).reset_index(drop=True)

        marketing_cols = ['digital_spend', 'tv_spend', 'traditional_spend',
                         'sponsorship_spend', 'other_spend']

        # 1. Adstock features (decay rate = 0.3)
        decay_rate = 0.3

        for col in marketing_cols:
            adstock_col = f"{col}_adstock"
            if adstock_col in self.feature_cols:
                # Calculate adstock per brand
                df[adstock_col] = 0.0
                for (country, brand), group in df.groupby(['country', 'brand']):
                    indices = group.index
                    values = group[col].values
                    adstock = np.zeros(len(values))

                    for i in range(len(values)):
                        adstock[i] = values[i]
                        if i > 0:
                            adstock[i] += decay_rate * adstock[i-1]

                    df.loc[indices, adstock_col] = adstock

        # 2. Saturation features (diminishing returns)
        alpha = 0.5  # Saturation parameter

        for col in marketing_cols:
            saturated_col = f"{col}_saturated"
            if saturated_col in self.feature_cols:
                # Apply saturation transformation: x^alpha
                df[saturated_col] = np.power(df[col] + 1, alpha) - 1

        # 3. Synergy features
        synergy_pairs = [
            ('digital_spend', 'tv_spend', 'digital_tv_synergy'),
            ('digital_spend', 'traditional_spend', 'digital_traditional_synergy'),
            ('tv_spend', 'sponsorship_spend', 'tv_sponsorship_synergy')
        ]

        for col1, col2, synergy_col in synergy_pairs:
            if synergy_col in self.feature_cols:
                # Synergy = sqrt(col1 * col2)
                df[synergy_col] = np.sqrt(df[col1] * df[col2])

        # 4. Share of voice (SOV) - marketing spend relative to competitors
        for col in ['total_marketing_spend'] + marketing_cols:
            sov_col = f"{col}_sov"
            if sov_col in self.feature_cols:
                # Calculate per country-quarter
                df[sov_col] = 0.0
                for (country, year, quarter), group in df.groupby(['country', 'year', 'quarter']):
                    total = group[col].sum()
                    if total > 0:
                        df.loc[group.index, sov_col] = group[col] / total

        # 5. Marketing velocity (change from previous quarter)
        for col in marketing_cols:
            velocity_col = f"{col}_velocity"
            if velocity_col in self.feature_cols:
                df[velocity_col] = 0.0
                for (country, brand), group in df.groupby(['country', 'brand']):
                    indices = group.index
                    values = group[col].values
                    velocity = np.diff(values, prepend=values[0])
                    df.loc[indices, velocity_col] = velocity

        # 6. Marketing ROI estimate (power / marketing spend)
        roi_col = 'historical_marketing_roi'
        if roi_col in self.feature_cols:
            df[roi_col] = 0.0
            # For forecast data, we don't have actual power, so use a default or skip
            # This feature is best calculated from historical data

        # 7. Marketing intensity ratios
        intensity_ratios = [
            ('digital_spend', 'digital_intensity'),
            ('tv_spend', 'tv_intensity'),
            ('traditional_spend', 'traditional_intensity'),
            ('sponsorship_spend', 'sponsorship_intensity'),
            ('other_spend', 'other_intensity')
        ]

        for spend_col, intensity_col in intensity_ratios:
            if intensity_col in self.feature_cols:
                total = df['total_marketing_spend']
                df[intensity_col] = np.where(total > 0, df[spend_col] / total, 0)

        # 8. Competitor spend (sum of other brands in same country-quarter)
        competitor_col = 'competitor_total_spend'
        if competitor_col in self.feature_cols:
            df[competitor_col] = 0.0
            for (country, year, quarter), group in df.groupby(['country', 'year', 'quarter']):
                for idx in group.index:
                    brand = df.loc[idx, 'brand']
                    other_brands = group[group['brand'] != brand]
                    df.at[idx, competitor_col] = other_brands['total_marketing_spend'].sum()

        return df


def forecast_baseline(
    model_path: str = "production_scripts/models/brand_power_forecaster.pkl",
    forecast_features_path: str = "data/forecast_features.csv",
    save_path: str = "data/baseline_forecast.csv"
) -> pd.DataFrame:
    """
    Standalone function to generate baseline forecast

    Returns:
        DataFrame with baseline predictions
    """
    service = ForecastSimulateService(
        model_path=model_path,
        forecast_features_path=forecast_features_path
    )

    return service.forecast_baseline(save_path=save_path)


def simulate_with_template(
    uploaded_template_path: str,
    model_path: str = "production_scripts/models/brand_power_forecaster.pkl",
    forecast_features_path: str = "data/forecast_features.csv",
    baseline_forecast_path: str = "data/baseline_forecast.csv",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Standalone function to simulate with uploaded template

    Args:
        uploaded_template_path: Path to uploaded CSV with new marketing spends
        model_path: Path to trained model
        forecast_features_path: Path to forecast features
        baseline_forecast_path: Path to baseline forecast
        output_path: Optional path to save simulation results

    Returns:
        DataFrame with simulation results and uplift
    """
    service = ForecastSimulateService(
        model_path=model_path,
        forecast_features_path=forecast_features_path
    )

    # Load uploaded template
    uploaded_template = pd.read_csv(uploaded_template_path)

    # Load baseline forecast
    baseline_forecast = pd.read_csv(baseline_forecast_path)

    # Simulate
    result = service.simulate(uploaded_template, baseline_forecast)

    # Save if path provided
    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"✓ Saved simulation results to {output_path}")

    return result


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE: Generate Baseline Forecast")
    print("="*80 + "\n")

    baseline = forecast_baseline(
        save_path="data/baseline_forecast.csv"
    )

    print("\nBaseline Forecast:")
    print(baseline.head(10))
    print(f"\nTotal forecasts: {len(baseline)}")
    print(f"Power range: [{baseline['predicted_power'].min():.4f}, {baseline['predicted_power'].max():.4f}]")

    print("\n" + "="*80)
    print("EXAMPLE: Simulate with Template")
    print("="*80 + "\n")

    # Use Brazil template as example
    simulation = simulate_with_template(
        uploaded_template_path="frontend/data/upload_template_brazil.csv",
        output_path="data/simulation_result.csv"
    )

    print("\nSimulation Results:")
    print(simulation.head(10))
    print(f"\nAverage uplift: {simulation['uplift'].mean():.4f} ({simulation['uplift_pct'].mean():.2f}%)")
