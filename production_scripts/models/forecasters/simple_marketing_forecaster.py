"""
Simple Marketing Forecaster with Aggregated Channels
Uses simplified marketing features: digital, tv, traditional, sponsorship, other
RMSE: 0.3789 (H1: 0.18, H2: 0.30, H3: 0.38, H4: 0.55)
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from production_scripts.core.base import BaseForecaster
from production_scripts.core.factories import register_forecaster

logger = logging.getLogger(__name__)


@register_forecaster('simple_marketing')
class SimpleMarketingForecaster(BaseForecaster):
    """
    Simple recursive forecaster with aggregated marketing channels

    Features:
    - Time series: lags (1-4Q), rolling mean/std, EMA
    - Momentum: velocity, acceleration
    - Seasonality: Q1-Q3 dummies
    - Marketing: aggregated spend (digital, TV, traditional, sponsorship, other)
    - Marketing dynamics: lags, velocity
    """

    def __init__(self, model_dict: Dict[str, Any], **kwargs):
        # Set 'model' for BaseForecaster
        if 'model' not in model_dict:
            model_dict['model'] = model_dict.get('ridge')

        super().__init__(model_dict)

        # Model components
        self.ridge = model_dict.get('ridge')
        self.lgbm = model_dict.get('lgbm')
        self.ridge_weight = model_dict.get('ridge_weight', 0.4)
        self.lgbm_weight = model_dict.get('lgbm_weight', 0.6)
        self.features = model_dict.get('features', [])

        if self.ridge is None:
            raise ValueError("SimpleMarketingForecaster requires 'ridge' model")

        logger.info(f"SimpleMarketingForecaster initialized with {len(self.features)} features")

    def _prepare_features(
        self,
        brand_history: pd.DataFrame,
        country: str,
        brand: str
    ) -> pd.DataFrame:
        """Prepare features from brand history"""
        try:
            df = brand_history.copy()

            # Ensure sorted
            df = df.sort_values(['year', 'quarter'])

            # Time series features
            for lag in [1, 2, 3, 4]:
                df[f'power_lag{lag}'] = df['power'].shift(lag)

            df['power_mean4q'] = df['power'].rolling(4, min_periods=1).mean()
            df['power_std4q'] = df['power'].rolling(4, min_periods=1).std()
            df['power_ema4'] = df['power'].ewm(span=4, adjust=False).mean()
            df['power_velocity'] = df['power'].diff()
            df['power_accel'] = df['power_velocity'].diff()

            # Seasonality
            df['q1'] = (df['quarter'] == 'Qtr1').astype(int)
            df['q2'] = (df['quarter'] == 'Qtr2').astype(int)
            df['q3'] = (df['quarter'] == 'Qtr3').astype(int)

            # Marketing features (check if columns exist)
            if 'total_marketing_spend' in df.columns:
                for lag in [1, 2]:
                    df[f'marketing_lag{lag}'] = df['total_marketing_spend'].shift(lag)
                df['marketing_velocity'] = df['total_marketing_spend'].diff()
            else:
                # No marketing data - set to 0
                df['total_marketing_spend'] = 0
                df['digital_spend'] = 0
                df['tv_spend'] = 0
                df['traditional_spend'] = 0
                df['sponsorship_spend'] = 0
                df['other_spend'] = 0
                df['marketing_lag1'] = 0
                df['marketing_lag2'] = 0
                df['marketing_velocity'] = 0
                df['has_marketing_data'] = 0

            # Fill NaN
            df = df.fillna(0)

            # Get latest row
            features_df = df.tail(1)[self.features]

            return features_df

        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            raise

    def predict(
        self,
        brand_history: pd.DataFrame,
        country: str,
        brand: str,
        max_horizon: int = 4,
        marketing_allocation: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """
        Recursive multi-horizon prediction

        Args:
            brand_history: Historical data
            country: Country identifier
            brand: Brand identifier
            max_horizon: Number of quarters to forecast
            marketing_allocation: Dict with keys: digital_spend, tv_spend, traditional_spend,
                                 sponsorship_spend, other_spend

        Returns:
            List of predictions [H1, H2, H3, H4]
        """
        try:
            # Prepare initial features
            features_df = self._prepare_features(brand_history, country, brand)

            # Apply marketing scenario if provided
            if marketing_allocation:
                features_df = self._apply_marketing_scenario(features_df, marketing_allocation)

            # Recursive prediction
            predictions = []

            for h in range(max_horizon):
                # Predict
                X = features_df[self.features].fillna(0).values

                pred_ridge = self.ridge.predict(X)[0]

                if self.lgbm:
                    pred_lgbm = self.lgbm.predict(X)[0]
                    pred = self.ridge_weight * pred_ridge + self.lgbm_weight * pred_lgbm
                else:
                    pred = pred_ridge

                predictions.append(pred)

                # Update features for next horizon
                if h < max_horizon - 1:
                    features_df = self._update_features(features_df, pred)

            logger.info(f"Prediction for {country}/{brand}: " +
                       ", ".join([f"H{i+1}={p:.3f}" for i, p in enumerate(predictions)]))

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed for {country}/{brand}: {str(e)}")
            raise

    def _apply_marketing_scenario(
        self,
        features_df: pd.DataFrame,
        marketing_allocation: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply marketing allocation to features"""
        features_df = features_df.copy()

        # Update current marketing spend
        if 'digital_spend' in marketing_allocation:
            features_df['digital_spend'] = marketing_allocation.get('digital_spend', 0)
        if 'tv_spend' in marketing_allocation:
            features_df['tv_spend'] = marketing_allocation.get('tv_spend', 0)
        if 'traditional_spend' in marketing_allocation:
            features_df['traditional_spend'] = marketing_allocation.get('traditional_spend', 0)
        if 'sponsorship_spend' in marketing_allocation:
            features_df['sponsorship_spend'] = marketing_allocation.get('sponsorship_spend', 0)
        if 'other_spend' in marketing_allocation:
            features_df['other_spend'] = marketing_allocation.get('other_spend', 0)

        # Update total
        features_df['total_marketing_spend'] = (
            features_df.get('digital_spend', pd.Series([0])).values[0] +
            features_df.get('tv_spend', pd.Series([0])).values[0] +
            features_df.get('traditional_spend', pd.Series([0])).values[0] +
            features_df.get('sponsorship_spend', pd.Series([0])).values[0] +
            features_df.get('other_spend', pd.Series([0])).values[0]
        )

        # Update has_marketing_data flag
        features_df['has_marketing_data'] = 1 if features_df['total_marketing_spend'].values[0] > 0 else 0

        return features_df

    def _update_features(self, features_df: pd.DataFrame, predicted_power: float) -> pd.DataFrame:
        """Update features for next recursive step"""
        features_df = features_df.copy()

        # Update lags
        features_df['power_lag4'] = features_df['power_lag3'].values[0]
        features_df['power_lag3'] = features_df['power_lag2'].values[0]
        features_df['power_lag2'] = features_df['power_lag1'].values[0]
        features_df['power_lag1'] = predicted_power

        # Update rolling mean
        lags = [
            features_df['power_lag1'].values[0],
            features_df['power_lag2'].values[0],
            features_df['power_lag3'].values[0],
            features_df['power_lag4'].values[0]
        ]
        features_df['power_mean4q'] = np.mean([l for l in lags if l > 0])

        # Update EMA
        alpha = 2/5
        old_ema = features_df['power_ema4'].values[0]
        features_df['power_ema4'] = alpha * predicted_power + (1 - alpha) * old_ema

        # Update velocity
        last_power = features_df['power_lag1'].values[0]
        features_df['power_velocity'] = predicted_power - last_power if last_power > 0 else 0

        # Update acceleration
        old_velocity = features_df['power_velocity'].values[0]
        new_velocity = features_df['power_velocity'].values[0]
        features_df['power_accel'] = new_velocity - old_velocity

        return features_df

    def _make_prediction(
        self,
        features_df: pd.DataFrame,
        max_horizon: int
    ) -> List[float]:
        """Required by BaseForecaster - delegates to predict()"""
        # This is called by base class, but we override predict() directly
        return self.predict(pd.DataFrame(), '', '', max_horizon)
