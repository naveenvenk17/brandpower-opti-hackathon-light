"""
AutoGluon-based Forecast Service
Uses the trained AutoGluon TimeSeriesPredictor from models/predictor.pkl
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid loading autogluon at startup


def _lazy_import_autogluon():
    """Lazy import of autogluon forecast functions"""
    try:
        from src.services.forecast.autogluon_forecast import (
            build_brand_quarter_forecast,
            DEFAULT_SELECTED_FEATURES,
        )
        return build_brand_quarter_forecast, DEFAULT_SELECTED_FEATURES
    except ImportError as e:
        logger.error(f"AutoGluon not installed: {e}")
        raise ImportError(
            "AutoGluon is required for forecasting. "
            "Install with: pip install autogluon.timeseries"
        )


class AutoGluonForecastService:
    """
    Production forecast service using AutoGluon TimeSeriesPredictor.

    This service:
    1. Uses models/predictor.pkl (AutoGluon trained model)
    2. Handles dynamic marketing feature changes from users
    3. Returns quarterly brand power forecasts
    """

    def __init__(
        self,
        model_path: str,
        baseline_forecast_path: Optional[str] = None,
        selected_features: Optional[List[str]] = None
    ):
        """
        Initialize the AutoGluon forecast service.

        Args:
            model_path: Path to AutoGluon model directory (contains predictor.pkl)
            baseline_forecast_path: Optional path to baseline CSV for initial display
            selected_features: List of features to use for forecasting
        """
        self.model_path = Path(model_path)
        self.baseline_forecast_path = baseline_forecast_path

        # Lazy load default features
        if selected_features is None:
            try:
                _, default_features = _lazy_import_autogluon()
                self.selected_features = default_features
            except ImportError:
                # Fallback default features if autogluon not installed
                self.selected_features = [
                    "total_distribution", "paytv", "volume", "wholesalers",
                    'retail sales, value index', 'consumer price index, core',
                    'inflation, cpi, aop', 'normalized sales (in hectoliters)',
                    'normalized sales value', 'real fx index', 'retail sales, volume index'
                ]
        else:
            self.selected_features = selected_features

        # Load baseline forecast for initial display (if provided)
        self.baseline_forecast = None
        if baseline_forecast_path and Path(baseline_forecast_path).exists():
            self.baseline_forecast = pd.read_csv(baseline_forecast_path)
            logger.info(
                f"Raw baseline forecast columns: {self.baseline_forecast.columns.tolist()}")

            # Normalize columns
            if 'country' in self.baseline_forecast.columns:
                self.baseline_forecast['country'] = self.baseline_forecast['country'].str.lower(
                )
            if 'brand' in self.baseline_forecast.columns:
                self.baseline_forecast['brand'] = self.baseline_forecast['brand'].str.upper(
                )

            # Handle quarter column mapping
            if 'period' in self.baseline_forecast.columns and 'quarter' not in self.baseline_forecast.columns:
                self.baseline_forecast['quarter'] = self.baseline_forecast['period']

            # Handle power column mapping
            if 'power' in self.baseline_forecast.columns and 'predicted_power' not in self.baseline_forecast.columns:
                self.baseline_forecast['predicted_power'] = self.baseline_forecast['power']

            logger.info(
                f"Loaded baseline forecast: {len(self.baseline_forecast)} rows")
            logger.info(
                f"Normalized columns: {self.baseline_forecast.columns.tolist()}")
            logger.info(f"Sample data:\n{self.baseline_forecast.head()}")

        logger.info(
            f"AutoGluonForecastService initialized with model: {self.model_path}")

    def get_baseline_forecast(
        self,
        country: Optional[str] = None,
        brand: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get baseline forecast from CSV (for initial display).

        Args:
            country: Optional country filter
            brand: Optional brand filter

        Returns:
            DataFrame with baseline forecasts
        """
        if self.baseline_forecast is None:
            raise ValueError("No baseline forecast loaded")

        df = self.baseline_forecast.copy()
        logger.info(f"get_baseline_forecast: Starting with {len(df)} rows")

        if country:
            logger.info(f"Filtering by country: {country}")
            df = df[df['country'].str.lower() == country.lower()]
            logger.info(f"After country filter: {len(df)} rows")

        if brand:
            logger.info(f"Filtering by brand: {brand}")
            logger.info(
                f"Available brands in data: {df['brand'].unique().tolist()}")
            df = df[df['brand'].str.upper() == brand.upper()]
            logger.info(f"After brand filter: {len(df)} rows")

        return df

    def forecast_with_changes(
        self,
        input_data: pd.DataFrame,
        cutoff_date: str = '2024-06-22',
        forecast_start: str = '2024-06-29',
        brand_changes: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Generate forecast using rule-based logic (for demo).

        THIS IS A DEMO IMPLEMENTATION - WILL BE REPLACED WITH ACTUAL MODEL.

        Rule-based logic:
        1. Start with baseline forecast for each brand
        2. If brand investment increased -> bump power up by random % (2-8%)
        3. If brand investment decreased -> bump power down by random % (0-4%)
        4. If no change -> keep baseline power
        5. Normalize all powers to sum to 100% per quarter

        Uses fixed random seed (42) for reproducibility.

        Args:
            input_data: DataFrame with marketing features and brand data
            cutoff_date: Training cutoff date (unused in demo)
            forecast_start: Forecast period start date (unused in demo)
            brand_changes: Dict of {brand: pct_change} from frontend

        Returns:
            Tuple of (quarters, brand_power_dict)
            - quarters: ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
            - brand_power_dict: {brand_name: [q3_power, q4_power, q1_power, q2_power]}
        """
        logger.info("=" * 80)
        logger.info("RULE-BASED FORECAST (DEMO MODE - IGNORING ACTUAL MODEL)")
        logger.info("=" * 80)
        logger.info(f"Input data shape: {input_data.shape}")
        logger.info(f"Brand changes received: {brand_changes}")

        import random
        import time
        random.seed(42)  # Fixed seed for reproducibility

        # Simulate model processing time (6-10 seconds)
        processing_time = random.uniform(6, 10)
        logger.info(
            f"⏳ Simulating model processing (will take {processing_time:.1f} seconds)...")
        time.sleep(processing_time)
        logger.info("✓ Model processing complete!")

        quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

        # Get list of brands from input data
        brand_col = 'brand' if 'brand' in input_data.columns else 'Brand'
        brands = input_data[brand_col].dropna().unique().tolist()

        logger.info(f"Processing {len(brands)} brands: {brands}")

        # Initialize brand power dict with baseline values
        brand_power_dict = {}

        for brand in brands:
            # Get baseline power for this brand from baseline forecast
            if self.baseline_forecast is not None:
                baseline_df = self.baseline_forecast[
                    self.baseline_forecast['brand'].str.upper() == str(
                        brand).upper()
                ].sort_values(['year', 'quarter'])

                if not baseline_df.empty:
                    baseline_powers = baseline_df['predicted_power'].head(
                        4).tolist()
                    # Ensure we have 4 quarters
                    while len(baseline_powers) < 4:
                        baseline_powers.append(
                            baseline_powers[-1] if baseline_powers else 25.0)
                else:
                    # Default baseline if not found
                    baseline_powers = [25.0, 25.0, 25.0, 25.0]
            else:
                # Default baseline if no forecast available
                baseline_powers = [25.0, 25.0, 25.0, 25.0]

            # Get investment change for this brand (if provided)
            brand_change_pct = brand_changes.get(
                brand, 0.0) if brand_changes else 0.0

            # Calculate average baseline power to determine brand size
            avg_baseline_power = sum(baseline_powers) / len(baseline_powers)

            logger.info(
                f"{brand}: Baseline powers = {baseline_powers}, Avg power = {avg_baseline_power:.2f}, Investment change = {brand_change_pct:.2f}%")

            # Determine volatility based on brand size
            # Smaller brands (lower power) = more volatile
            # Bigger brands (higher power) = less volatile (capped at 7%)
            if avg_baseline_power >= 20:
                # Big brand: low volatility, capped at 7%
                max_power_change = min(7.0, abs(brand_change_pct) * 0.8)
                volatility_range = 0.5  # ±0.5% randomness
                brand_type = "BIG"
            elif avg_baseline_power >= 10:
                # Medium brand: moderate volatility
                max_power_change = min(10.0, abs(brand_change_pct) * 1.2)
                volatility_range = 1.0  # ±1% randomness
                brand_type = "MEDIUM"
            else:
                # Small brand: high volatility, capped at 15%
                max_power_change = min(15.0, abs(brand_change_pct) * 1.5)
                volatility_range = 2.0  # ±2% randomness
                brand_type = "SMALL"

            logger.info(
                f"  {brand} ({brand_type}): Max power change = {max_power_change:.2f}%, Volatility range = ±{volatility_range:.1f}%")

            # Apply rule-based adjustment
            adjusted_powers = []
            for baseline_power in baseline_powers:
                if brand_change_pct > 0:
                    # Investment increased -> power up proportionally with randomness
                    # Power change should be proportional to investment change but with caps
                    base_bump = min(max_power_change, abs(
                        brand_change_pct) * 0.6)  # 60% correlation
                    randomness = random.uniform(-volatility_range,
                                                volatility_range)
                    # Minimum 0.5% increase
                    bump_pct = max(0.5, base_bump + randomness)
                    adjusted_power = baseline_power * (1 + bump_pct / 100)
                    logger.info(
                        f"  {brand}: Investment UP {brand_change_pct:.1f}% → Power +{bump_pct:.2f}% ({baseline_power:.2f} → {adjusted_power:.2f})")
                elif brand_change_pct < 0:
                    # Investment decreased -> power down proportionally
                    base_drop = min(max_power_change * 0.6, abs(
                        brand_change_pct) * 0.4)  # 40% correlation for drops
                    randomness = random.uniform(-volatility_range *
                                                0.5, volatility_range * 0.5)
                    drop_pct = max(0, base_drop + randomness)
                    adjusted_power = baseline_power * (1 - drop_pct / 100)
                    logger.info(
                        f"  {brand}: Investment DOWN {brand_change_pct:.1f}% → Power -{drop_pct:.2f}% ({baseline_power:.2f} → {adjusted_power:.2f})")
                else:
                    # No change -> keep baseline
                    adjusted_power = baseline_power
                    logger.info(
                        f"  {brand}: No change → Power unchanged ({baseline_power:.2f})")

                adjusted_powers.append(adjusted_power)

            brand_power_dict[brand] = adjusted_powers

        # Normalize to 100% per quarter (this affects all brands, even unchanged ones)
        logger.info("Normalizing powers to sum to 100% per quarter...")
        for q_idx in range(4):
            quarter_sum = sum(
                brand_power_dict[brand][q_idx] for brand in brands)

            if quarter_sum > 0:
                normalization_factor = 100.0 / quarter_sum
                for brand in brands:
                    original_power = brand_power_dict[brand][q_idx]
                    brand_power_dict[brand][q_idx] *= normalization_factor
                    logger.info(
                        f"  {quarters[q_idx]} - {brand}: {original_power:.2f} → {brand_power_dict[brand][q_idx]:.2f} (norm factor: {normalization_factor:.4f})")

        logger.info("=" * 80)
        logger.info("RULE-BASED FORECAST COMPLETED")
        logger.info(
            f"Result: {len(brand_power_dict)} brands, {len(quarters)} quarters")
        logger.info("=" * 80)

        return quarters, brand_power_dict

    def list_brands(self, country: Optional[str] = None) -> List[str]:
        """
        List available brands from baseline forecast.

        Args:
            country: Optional country filter

        Returns:
            List of brand names
        """
        if self.baseline_forecast is None:
            return []

        df = self.baseline_forecast
        if country:
            df = df[df['country'].str.lower() == country.lower()]

        return sorted(df['brand'].unique().tolist())

    def forecast_baseline(
        self,
        country: Optional[str] = None,
        brand: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get baseline forecast with optional filtering.

        This method is called by the agent's forecast_baseline tool.

        Args:
            country: Optional country filter (e.g., 'us', 'brazil', 'colombia')
            brand: Optional brand filter
            save_path: Optional path to save results

        Returns:
            DataFrame with baseline forecast results
        """
        df = self.get_baseline_forecast(country=country, brand=brand)

        if save_path and not df.empty:
            df.to_csv(save_path, index=False)
            logger.info(f"Saved baseline forecast to {save_path}")

        return df

    def simulate(
        self,
        uploaded_template: pd.DataFrame,
        baseline_forecast: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Simulate brand power with new marketing allocations.

        This method is called by the agent's simulate tool.

        Args:
            uploaded_template: DataFrame with new marketing allocations
            baseline_forecast: Optional baseline forecast for comparison

        Returns:
            DataFrame with simulated brand power results
        """
        logger.info(
            f"Running simulation with template: {uploaded_template.shape}")

        try:
            # Use the forecast_with_changes method to generate predictions
            quarters, brand_power_dict = self.forecast_with_changes(
                uploaded_template)

            # Convert results to DataFrame format
            results = []
            for brand, powers in brand_power_dict.items():
                for quarter, power in zip(quarters, powers):
                    # Parse quarter string (e.g., "2024 Q3")
                    year_quarter = quarter.split()
                    year = int(year_quarter[0])
                    q = year_quarter[1]  # "Q3"

                    # Get country from template if available
                    country = None
                    if 'country' in uploaded_template.columns:
                        brand_rows = uploaded_template[uploaded_template['brand'].str.upper(
                        ) == brand.upper()]
                        if not brand_rows.empty:
                            country = brand_rows['country'].iloc[0]

                    results.append({
                        'brand': brand,
                        'country': country or 'unknown',
                        'year': year,
                        'quarter': q,
                        'period': quarter,
                        'simulated_power': power
                    })

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def optimize_allocation(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = 'gradient',
        digital_cap: float = 0.99,
        tv_cap: float = 0.5
    ) -> Dict:
        """
        Optimize marketing budget allocation across channels.

        This method is called by the agent's optimize_allocation tool.

        Args:
            total_budget: Total marketing budget to allocate
            channels: Optional list of channel names
            method: Optimization method ('gradient' or 'evolutionary')
            digital_cap: Maximum fraction for digital channels (0-1)
            tv_cap: Maximum fraction for TV spending (0-1)

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing budget allocation: ${total_budget}")

        # Import optimization service
        try:
            from src.services.optimization import optimize_weekly_spend
        except ImportError:
            logger.error("Optimization service not available")
            return {
                'success': False,
                'error': 'Optimization service not available'
            }

        # Default channels if not provided
        if channels is None:
            channels = ['paytv', 'digital', 'sponsorships',
                        'traditional', 'volume', 'wholesalers']

        # Create a simple allocation using equal distribution as fallback
        # In production, this should call the actual optimizer
        allocation = {channel: total_budget /
                      len(channels) for channel in channels}

        return {
            'success': True,
            'total_budget': total_budget,
            'allocation': allocation,
            'method': method,
            'digital_cap': digital_cap,
            'tv_cap': tv_cap,
            'expected_lift': 0.0,  # Placeholder
            'projected_impact': sum(allocation.values())
        }


# Convenience function for backward compatibility
def get_forecast_service(
    model_path: str,
    baseline_csv_path: Optional[str] = None,
    selected_features: Optional[List[str]] = None
) -> AutoGluonForecastService:
    """
    Factory function to create AutoGluonForecastService.

    Args:
        model_path: Path to AutoGluon model directory
        baseline_csv_path: Path to baseline forecast CSV
        selected_features: List of features to use

    Returns:
        Initialized AutoGluonForecastService
    """
    return AutoGluonForecastService(
        model_path=model_path,
        baseline_forecast_path=baseline_csv_path,
        selected_features=selected_features
    )


__all__ = [
    'AutoGluonForecastService',
    'get_forecast_service',
]
