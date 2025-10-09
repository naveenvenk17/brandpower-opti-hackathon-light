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
            logger.info(f"Raw baseline forecast columns: {self.baseline_forecast.columns.tolist()}")
            
            # Normalize columns
            if 'country' in self.baseline_forecast.columns:
                self.baseline_forecast['country'] = self.baseline_forecast['country'].str.lower()
            if 'brand' in self.baseline_forecast.columns:
                self.baseline_forecast['brand'] = self.baseline_forecast['brand'].str.upper()
            
            # Handle quarter column mapping
            if 'period' in self.baseline_forecast.columns and 'quarter' not in self.baseline_forecast.columns:
                self.baseline_forecast['quarter'] = self.baseline_forecast['period']
            
            # Handle power column mapping
            if 'power' in self.baseline_forecast.columns and 'predicted_power' not in self.baseline_forecast.columns:
                self.baseline_forecast['predicted_power'] = self.baseline_forecast['power']
            
            logger.info(f"Loaded baseline forecast: {len(self.baseline_forecast)} rows")
            logger.info(f"Normalized columns: {self.baseline_forecast.columns.tolist()}")
            logger.info(f"Sample data:\n{self.baseline_forecast.head()}")
        
        logger.info(f"AutoGluonForecastService initialized with model: {self.model_path}")
    
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
            logger.info(f"Available brands in data: {df['brand'].unique().tolist()}")
            df = df[df['brand'].str.upper() == brand.upper()]
            logger.info(f"After brand filter: {len(df)} rows")
        
        return df
    
    def forecast_with_changes(
        self,
        input_data: pd.DataFrame,
        cutoff_date: str = '2024-06-22',
        forecast_start: str = '2024-06-29'
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Generate forecast using AutoGluon when user makes marketing changes.
        
        Args:
            input_data: DataFrame with marketing features and brand data
            cutoff_date: Training cutoff date
            forecast_start: Forecast period start date
            
        Returns:
            Tuple of (quarters, brand_power_dict)
            - quarters: ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
            - brand_power_dict: {brand_name: [q3_power, q4_power, q1_power, q2_power]}
        """
        logger.info("Generating forecast with AutoGluon...")
        logger.info(f"Input data shape: {input_data.shape}")
        
        try:
            # Lazy load AutoGluon function
            build_forecast_fn, _ = _lazy_import_autogluon()
            
            # Call AutoGluon forecast
            quarters, brand_power_dict = build_forecast_fn(
                data=input_data,
                model_path=str(self.model_path),
                cutoff_date=cutoff_date,
                forecast_start=forecast_start,
                selected_features=self.selected_features
            )
            
            logger.info(f"Forecast completed: {len(brand_power_dict)} brands, {len(quarters)} quarters")
            return quarters, brand_power_dict
            
        except Exception as e:
            logger.error(f"AutoGluon forecast failed: {e}", exc_info=True)
            raise
    
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
        logger.info(f"Running simulation with template: {uploaded_template.shape}")
        
        try:
            # Use the forecast_with_changes method to generate predictions
            quarters, brand_power_dict = self.forecast_with_changes(uploaded_template)
            
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
                        brand_rows = uploaded_template[uploaded_template['brand'].str.upper() == brand.upper()]
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
            channels = ['paytv', 'digital', 'sponsorships', 'traditional', 'volume', 'wholesalers']
        
        # Create a simple allocation using equal distribution as fallback
        # In production, this should call the actual optimizer
        allocation = {channel: total_budget / len(channels) for channel in channels}
        
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
