"""
Simple Brand Power Forecaster Training Script

Trains a LightGBM model on engineered features for brand power forecasting.
Designed for production use with forecast() and simulate() functions.

Usage:
    python -m production_scripts.scripts.train_simple_forecaster
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_simple_forecaster(
    data_path: str = "data/brand_power_engineered_with_selection.csv",
    model_output_path: str = "production_scripts/models/brand_power_forecaster.pkl",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a simple LightGBM forecaster for brand power

    Args:
        data_path: Path to training data CSV
        model_output_path: Path to save trained model
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    """

    logger.info("="*80)
    logger.info("TRAINING SIMPLE BRAND POWER FORECASTER")
    logger.info("="*80)

    # Load data
    logger.info(f"\n1. Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Separate features and target
    target = 'power'
    id_cols = ['brand', 'country', 'year', 'quarter']

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data!")

    feature_cols = [col for col in df.columns if col not in id_cols + [target]]

    X = df[feature_cols]
    y = df[target]
    ids = df[id_cols]

    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Target: {target}")
    logger.info(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")

    # Train-test split
    logger.info(f"\n2. Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, shuffle=True
    )

    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")

    # Train LightGBM model
    logger.info("\n3. Training LightGBM model...")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': random_state
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # Evaluate on train and test
    logger.info("\n4. Model Evaluation:")

    for name, X_eval, y_eval in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_pred = model.predict(X_eval, num_iteration=model.best_iteration)

        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)
        mape = np.mean(np.abs((y_eval - y_pred) / y_eval)) * 100

        logger.info(f"\n   {name} Set:")
        logger.info(f"   - RMSE: {rmse:.4f}")
        logger.info(f"   - MAE:  {mae:.4f}")
        logger.info(f"   - R²:   {r2:.4f}")
        logger.info(f"   - MAPE: {mape:.2f}%")

    # Feature importance
    logger.info("\n5. Top 20 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"   {row['feature']:50s} {row['importance']:10.0f}")

    # Save model and metadata
    logger.info(f"\n6. Saving model to {model_output_path}...")

    model_artifact = {
        'model': model,
        'feature_cols': feature_cols,
        'target': target,
        'train_rmse': np.sqrt(mean_squared_error(y_train, model.predict(X_train, num_iteration=model.best_iteration))),
        'test_rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test, num_iteration=model.best_iteration))),
        'feature_importance': feature_importance.to_dict('records'),
        'params': params,
        'best_iteration': model.best_iteration
    }

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model_artifact, f)

    logger.info(f"   Model saved successfully!")
    logger.info(f"   Best iteration: {model.best_iteration}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80 + "\n")

    return model_artifact


if __name__ == "__main__":
    # Train from project root
    train_simple_forecaster(
        data_path="data/brand_power_engineered_with_selection.csv",
        model_output_path="production_scripts/models/brand_power_forecaster.pkl"
    )
