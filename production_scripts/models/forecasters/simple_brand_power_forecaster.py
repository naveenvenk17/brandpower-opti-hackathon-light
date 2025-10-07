"""
Simple Brand Power Forecaster - Streamlined Model

A straightforward forecaster using LightGBM for brand power prediction.
Designed to work with pre-engineered features and make direct predictions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available, using GradientBoostingRegressor")

logger = logging.getLogger(__name__)


class SimpleBrandPowerForecaster:
    """
    Simple forecaster for brand power using engineered features.

    This forecaster expects:
    - Pre-engineered features with all necessary lags, trends, and marketing variables
    - Direct prediction (no recursive forecasting)
    - Works with the 146 selected features from brand_power_engineered_with_selection.csv
    """

    def __init__(self, model=None, features=None, scaler=None):
        """
        Initialize the forecaster

        Args:
            model: Trained model (LGBMRegressor or similar)
            features: List of feature names in correct order
            scaler: Optional StandardScaler for feature normalization
        """
        self.model = model
        self.features = features or []
        self.scaler = scaler

        if self.model is None:
            logger.warning("No model provided - forecaster needs to be trained")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              params: Optional[Dict[str, Any]] = None):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training target (power values)
            params: Optional model parameters
        """
        # Store feature names
        self.features = list(X_train.columns)

        # Default parameters
        if params is None:
            if LGBM_AVAILABLE:
                params = {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbose': -1
                }
            else:
                params = {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }

        # Initialize and train model
        if LGBM_AVAILABLE:
            self.model = LGBMRegressor(**params)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**params)

        logger.info(f"Training model with {len(self.features)} features...")
        self.model.fit(X_train, y_train)
        logger.info("Model training complete")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure features are in correct order
        if not all(f in X.columns for f in self.features):
            missing = [f for f in self.features if f not in X.columns]
            raise ValueError(f"Missing features: {missing[:10]}...")

        X_ordered = X[self.features]

        # Handle any missing values
        X_ordered = X_ordered.fillna(0)

        # Make predictions
        predictions = self.model.predict(X_ordered)

        return predictions

    def predict_single(self, features_dict: Dict[str, float]) -> float:
        """
        Predict for a single observation

        Args:
            features_dict: Dictionary of feature_name -> value

        Returns:
            Single prediction value
        """
        # Convert dict to DataFrame
        X = pd.DataFrame([features_dict])

        # Make prediction
        pred = self.predict(X)[0]

        return pred

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            })

            return importance_df.sort_values('importance', ascending=False).head(top_n)
        else:
            return pd.DataFrame()

    def save(self, filepath: str):
        """Save the model"""
        import pickle

        model_dict = {
            'model': self.model,
            'features': self.features,
            'scaler': self.scaler,
            'lgbm_available': LGBM_AVAILABLE
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load a saved model"""
        import pickle

        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)

        forecaster = cls(
            model=model_dict['model'],
            features=model_dict['features'],
            scaler=model_dict.get('scaler')
        )

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Features: {len(forecaster.features)}")

        return forecaster


def train_simple_model(df: pd.DataFrame, target_col: str = 'power',
                       test_size: float = 0.2) -> SimpleBrandPowerForecaster:
    """
    Convenience function to train a simple model

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction for test set

    Returns:
        Trained forecaster
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split (maintaining time order if data is sorted)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    forecaster = SimpleBrandPowerForecaster()
    forecaster.train(X_train, y_train)

    # Evaluate
    train_pred = forecaster.predict(X_train)
    test_pred = forecaster.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    logger.info(f"\nModel Performance:")
    logger.info(f"  Train RMSE: {train_rmse:.4f}")
    logger.info(f"  Test RMSE:  {test_rmse:.4f}")
    logger.info(f"  Test MAE:   {test_mae:.4f}")
    logger.info(f"  Test R²:    {test_r2:.4f}")

    return forecaster
