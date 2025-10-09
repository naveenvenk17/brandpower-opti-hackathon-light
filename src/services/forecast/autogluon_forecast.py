"""
AutoGluon Time Series Forecasting Module
Migrated from hackathon_website_lightweight/frontend/train_weekly_2.py and scaled_forecast.py

This module provides forecasting capabilities using AutoGluon TimeSeriesPredictor.
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
from typing import Optional, List, Tuple

filterwarnings("ignore")

# Path to pre-fitted scaler (saved from training)
SCALER_PATH = Path(__file__).parent.parent.parent.parent / "models" / "scaler.pkl"


# Default selected features for Colombia model
DEFAULT_SELECTED_FEATURES = [
    "total_distribution", "paytv", "volume", "wholesalers",
    'retail sales, value index', 'consumer price index, core',
    'inflation, cpi, aop', 'normalized sales (in hectoliters)',
    'normalized sales value', 'real fx index', 'retail sales, volume index'
]


def load_scaler(scaler_path: Optional[str] = None) -> Optional[StandardScaler]:
    """
    Load pre-fitted scaler from pickle file.
    
    Args:
        scaler_path: Path to scaler.pkl file. If None, uses default SCALER_PATH.
        
    Returns:
        Loaded StandardScaler or None if file doesn't exist
    """
    if scaler_path is None:
        scaler_path = SCALER_PATH
    
    scaler_path = Path(scaler_path)
    
    if not scaler_path.exists():
        print(f"⚠️  Warning: Scaler not found at {scaler_path}")
        print("   Falling back to fit_transform (not recommended for production)")
        return None
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            print(f"⚠️  Warning: Loaded scaler is not fitted")
            return None
        
        print(f"✓ Loaded pre-fitted scaler from {scaler_path}")
        print(f"  Features: {len(scaler.feature_names_in_)} columns")
        return scaler
        
    except Exception as e:
        print(f"⚠️  Error loading scaler: {e}")
        return None


def create_features(df: pd.DataFrame, target_col: str = 'power') -> pd.DataFrame:
    """
    Create lag and rolling window features for time series forecasting.

    Args:
        df: Input DataFrame with time series data
        target_col: Name of the target column

    Returns:
        DataFrame with engineered features
    """
    print(f"Creating features... Shape: {df.shape}")
    try:
        df_sorted = df.sort_values(['item_id', 'timestamp'])
    except KeyError:
        print("Warning: item_id column not found, sorting by timestamp only")
        df_sorted = df.sort_values(['timestamp'])

    # Lag features
    for lag in range(1, 13):
        try:
            df_sorted[f'{target_col}_lag_{lag}'] = (
                df_sorted.groupby('item_id')[target_col].shift(lag)
            )
        except KeyError:
            print(f"Warning: Cannot create lag feature for lag {lag}")
            df_sorted[f'{target_col}_lag_{lag}'] = None

    # Rolling window features
    for window in [4, 8, 12, 16]:
        try:
            shifted = df_sorted.groupby('item_id')[target_col].shift(1)
            df_sorted[f'{target_col}_rolling_mean_{window}'] = (
                shifted.groupby(df_sorted['item_id']).transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            df_sorted[f'{target_col}_rolling_median_{window}'] = (
                shifted.groupby(df_sorted['item_id']).transform(
                    lambda x: x.rolling(window=window, min_periods=1).median())
            )
            df_sorted[f'{target_col}_rolling_std_{window}'] = (
                shifted.groupby(df_sorted['item_id']).transform(
                    lambda x: x.rolling(window=window, min_periods=1).std())
            )
        except KeyError:
            print(f"Warning: Cannot create rolling features for window {window}")

    # Expanding features
    try:
        shifted = df_sorted.groupby('item_id')[target_col].shift(1)
        df_sorted[f'{target_col}_expanding_mean'] = (
            shifted.groupby(df_sorted['item_id']).transform(
                lambda x: x.expanding().mean())
        )
        df_sorted[f'{target_col}_expanding_median'] = (
            shifted.groupby(df_sorted['item_id']).transform(
                lambda x: x.expanding().median())
        )
        df_sorted[f'{target_col}_expanding_std'] = (
            shifted.groupby(df_sorted['item_id']).transform(
                lambda x: x.expanding().std())
        )
    except KeyError:
        print("Warning: Cannot create expanding features")

    # Time components
    if 'year' not in df_sorted.columns:
        df_sorted['year'] = df_sorted['timestamp'].dt.year
    if 'month' not in df_sorted.columns:
        df_sorted['month'] = df_sorted['timestamp'].dt.month
    if 'week_of_month' not in df_sorted.columns:
        df_sorted['week_of_month'] = ((df_sorted['timestamp'].dt.day - 1) // 7) + 1

    print(f"Shape after feature engineering: {df_sorted.shape}")
    return df_sorted


def preprocess_data_for_inference(
    df: pd.DataFrame,
    selected_features: List[str],
    cutoff_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess data for AutoGluon inference.

    Args:
        df: Input DataFrame
        selected_features: List of feature column names
        cutoff_date: Optional cutoff date for training period

    Returns:
        Preprocessed DataFrame ready for forecasting
    """
    print("Starting data preprocessing for inference...")
    df_prep = df.copy()

    # Rename columns if needed
    column_mapping = {
        'sales_hectoliters': 'normalized sales (in hectoliters)',
        'sales_value': 'normalized sales value'
    }
    df_prep = df_prep.rename(columns=column_mapping)

    # Scale power column
    if 'power' in df_prep.columns:
        df_prep['power'] = df_prep['power'] * 1000

    # Drop unnecessary columns
    drop_cols = ['meaning', 'difference', 'salience', 'premium']
    existing_drop_cols = [col for col in drop_cols if col in df_prep.columns]
    if existing_drop_cols:
        df_prep = df_prep.drop(columns=existing_drop_cols)

    # Normalize numerical columns using pre-fitted scaler
    # BEST PRACTICE: Load scaler fitted on training data and use transform() only
    print("Loading pre-fitted scaler for normalization...")
    scaler = load_scaler()
    
    if scaler is not None:
        # Use pre-fitted scaler (PRODUCTION MODE)
        scaler_features = scaler.feature_names_in_.tolist()
        
        # Find common features between scaler and current data
        available_features = [f for f in scaler_features if f in df_prep.columns]
        missing_features = [f for f in scaler_features if f not in df_prep.columns]
        
        if missing_features:
            print(f"⚠️  Warning: {len(missing_features)} features missing from data")
            print(f"   Missing: {missing_features[:5]}..." if len(missing_features) > 5 else f"   Missing: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                df_prep[feat] = 0.0
        
        # Apply pre-fitted scaler transform (NOT fit_transform!)
        try:
            df_prep[scaler_features] = scaler.transform(df_prep[scaler_features])
            print(f"✓ Applied pre-fitted scaler to {len(scaler_features)} features")
            print("  This ensures consistent scaling with training data!")
        except Exception as e:
            print(f"⚠️  Error applying scaler: {e}")
            print("   Falling back to fit_transform")
            numerical_columns = df_prep.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_normalize = [
                col for col in numerical_columns if col not in ['power', 'year', 'month', 'week_of_month']
            ]
            if columns_to_normalize:
                fallback_scaler = StandardScaler()
                df_prep[columns_to_normalize] = fallback_scaler.fit_transform(df_prep[columns_to_normalize])
                print(f"   Normalized {len(columns_to_normalize)} columns with fallback scaler")
    else:
        # Fallback: fit_transform on inference data (NOT RECOMMENDED)
        print("⚠️  Using fallback scaling (fit_transform on inference data)")
        numerical_columns = df_prep.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_normalize = [
            col for col in numerical_columns if col not in ['power', 'year', 'month', 'week_of_month']
        ]
        
        if columns_to_normalize:
            fallback_scaler = StandardScaler()
            df_prep[columns_to_normalize] = fallback_scaler.fit_transform(df_prep[columns_to_normalize])
            print(f"   Normalized {len(columns_to_normalize)} numerical columns")

    # Create channel groups (matching source logic)
    channel_groups = {
        "Digital": ["digitaldisplayandsearch", "digitalvideo", "meta", "tiktok", "twitter", "youtube"],
        "Influencer": ["influencer"],
        "TV": ["opentv", "paytv"],
        "OOH_Audio": ["radio", "streamingaudio", "ooh"],
        "Events_Sponsorship": ["brand events", "sponsorship", "others"]
    }

    for group_name, columns in channel_groups.items():
        available_cols = [col for col in columns if col in df_prep.columns]
        if available_cols:
            df_prep[group_name] = df_prep[available_cols].sum(axis=1)

    # Ensure time columns are numeric
    for col in ['year', 'month', 'week_of_month']:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').astype('Int64')

    # Create timestamp
    df_prep['year'] = pd.to_numeric(df_prep['year'], errors='coerce').astype(int)
    df_prep['month'] = pd.to_numeric(df_prep['month'], errors='coerce').astype(int)
    df_prep['week_of_month'] = pd.to_numeric(df_prep['week_of_month'], errors='coerce').astype(int)

    df_prep['month_start'] = pd.to_datetime({
        'year': df_prep['year'],
        'month': df_prep['month'],
        'day': 1
    })
    df_prep['timestamp'] = df_prep['month_start'] + \
        pd.to_timedelta((df_prep['week_of_month'] - 1) * 7, unit='days')
    df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp']).dt.tz_localize(None)
    df_prep.drop('month_start', axis=1, inplace=True)

    # Create item_id
    if 'country' not in df_prep.columns:
        df_prep['country'] = 'COLOMBIA'

    try:
        df_prep['item_id'] = df_prep['country'] + "_" + df_prep['brand']
    except KeyError:
        if 'brand' in df_prep.columns:
            df_prep['item_id'] = df_prep['country'] + '_' + df_prep['brand'].astype(str)
        else:
            df_prep['item_id'] = 'unknown_brand_' + df_prep.index.astype(str)

    # Select columns
    keep_columns = ['item_id', 'timestamp', 'power', 'brand'] + selected_features
    available_columns = [col for col in keep_columns if col in df_prep.columns]
    missing_columns = [col for col in keep_columns if col not in df_prep.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")

    df_final = df_prep[available_columns].copy()

    # Apply cutoff if specified
    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        df_final = df_final[df_final['timestamp'] <= cutoff_date].copy()
        print(f"Applied cutoff date: {cutoff_date}")

    print(f"Final preprocessed data shape: {df_final.shape}")
    return df_final


def generate_forecast_with_model(
    predictor: TimeSeriesPredictor,
    preprocessed_data: pd.DataFrame,
    prediction_length: int = 53
) -> pd.DataFrame:
    """
    Generate forecast using a trained AutoGluon model.

    Args:
        predictor: Trained TimeSeriesPredictor
        preprocessed_data: Preprocessed input data
        prediction_length: Number of steps to forecast

    Returns:
        DataFrame with predictions
    """
    print("Generating forecast with loaded model...")

    # Create features
    data_with_features = create_features(preprocessed_data, target_col='power')

    # Ensure timestamp is proper datetime
    if not pd.api.types.is_datetime64_any_dtype(data_with_features['timestamp']):
        data_with_features['timestamp'] = pd.to_datetime(
            data_with_features['timestamp'], errors='coerce', infer_datetime_format=True)
        n_invalid = data_with_features['timestamp'].isna().sum()
        if n_invalid > 0:
            print(f"Warning: {n_invalid} invalid timestamps found and dropped.")
            data_with_features.dropna(subset=['timestamp'], inplace=True)

    # Convert to TimeSeriesDataFrame
    print("Converting to TimeSeriesDataFrame format...")
    ts_data = TimeSeriesDataFrame.from_data_frame(
        data_with_features,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    print(f"Input TimeSeriesDataFrame shape: {ts_data.shape}")

    # Generate predictions
    print("Generating predictions from model...")
    # PRIORITY: Use models that actually utilize marketing features (covariates)
    # DirectTabular and RecursiveTabular use XGBoost/LightGBM which leverage features heavily
    # DeepAR, PatchTST, ChronosFineTuned can also use covariates
    try:
        # Try DirectTabular first - best for using marketing features
        predictions = predictor.predict(data=ts_data, model='DirectTabular')
        print("✓ Using DirectTabular model (XGBoost-based, uses marketing features)")
    except Exception as e:
        print(f"DirectTabular failed: {e}, trying RecursiveTabular...")
        try:
            # RecursiveTabular is also feature-aware
            predictions = predictor.predict(data=ts_data, model='RecursiveTabular')
            print("✓ Using RecursiveTabular model (uses marketing features)")
        except Exception as e2:
            print(f"RecursiveTabular failed: {e2}, trying DeepAR...")
            try:
                # DeepAR can use covariates
                predictions = predictor.predict(data=ts_data, model='DeepAR')
                print("✓ Using DeepAR model (can use marketing features)")
            except Exception as e3:
                print(f"DeepAR failed: {e3}, falling back to WeightedEnsemble...")
                # WeightedEnsemble combines all models including feature-aware ones
                predictions = predictor.predict(data=ts_data)
                print("⚠️  Using WeightedEnsemble (best available, includes feature-aware models)")

    print("Processing prediction results...")
    predictions_df = predictions.reset_index()

    # Add time components
    predictions_df['year'] = predictions_df['timestamp'].dt.year
    predictions_df['month'] = predictions_df['timestamp'].dt.month
    predictions_df['week_of_month'] = ((predictions_df['timestamp'].dt.day - 1) // 7 + 1)

    # Filter to first 4 weeks of each month
    predictions_df = predictions_df[predictions_df['week_of_month'] <= 4]

    print(f"Predictions after filtering to 4 weeks per month: {predictions_df.shape}")
    return predictions_df


def format_submission(
    predictions_df: pd.DataFrame,
    forecast_start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Format predictions into submission format with normalization.

    Args:
        predictions_df: Raw predictions from model
        forecast_start_date: Optional start date for forecast period

    Returns:
        Formatted submission DataFrame
    """
    print("Formatting submission...")
    submission = predictions_df.copy()

    # Filter to forecast period if specified
    if forecast_start_date:
        forecast_start_date = pd.to_datetime(forecast_start_date)
        submission = submission[submission['timestamp'] >= forecast_start_date]
        print(f"Filtered to forecast period starting: {forecast_start_date}")

    # Extract country and brand from item_id
    if 'item_id' in submission.columns:
        submission['country'] = submission['item_id'].str.split('_').str[0]
        submission['brand'] = submission['item_id'].str.split('_').str[1]
    else:
        print("Warning: item_id column not found")
        submission['country'] = 'unknown'
        submission['brand'] = 'unknown'

    # Rename prediction column
    submission.rename(columns={'mean': 'predicted_power'}, inplace=True)

    # Normalize to sum to 100,000 per group
    group_cols = ['year', 'month', 'week_of_month', 'country']
    group_sums = submission.groupby(group_cols)['predicted_power'].transform('sum')

    submission['normalized_predicted_power'] = 0.0
    mask_nonzero = group_sums > 0
    submission.loc[mask_nonzero, 'normalized_predicted_power'] = (
        submission.loc[mask_nonzero, 'predicted_power'] *
        100000 / group_sums.loc[mask_nonzero]
    )

    # Final predicted power (divide by 1000)
    submission['Predicted Power'] = submission['normalized_predicted_power'] / 1000

    # Select final columns
    final_submission = submission[[
        'country', 'brand', 'year', 'month', 'week_of_month', 'Predicted Power']].copy()

    print(f"Final submission shape: {final_submission.shape}")
    return final_submission


def month_to_quarter(month: int) -> str:
    """Convert month number to quarter string."""
    if month in [1, 2, 3]:
        return 'Qtr1'
    elif month in [4, 5, 6]:
        return 'Qtr2'
    elif month in [7, 8, 9]:
        return 'Qtr3'
    elif month in [10, 11, 12]:
        return 'Qtr4'
    else:
        return None


def complete_post_processing(predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete post-processing: normalize weekly to 100, then calculate quarterly averages.

    Args:
        predictions_df: Predictions DataFrame with 'Predicted Power' column

    Returns:
        Tuple of (normalized_weekly, quarterly_data)
    """
    print("="*60)
    print("POST-PROCESSING: Normalize Weekly + Calculate Quarterly Averages")
    print("="*60)

    df_norm = predictions_df.copy()

    # Normalize weekly power to sum to 100 per week
    print("Normalizing weekly power to sum to 100 per week...")
    group_cols = ['year', 'month', 'week_of_month', 'country']
    group_sums = df_norm.groupby(group_cols)['Predicted Power'].transform('sum')

    df_norm['Normalized Weekly Power'] = 0.0
    mask_nonzero = group_sums > 0
    df_norm.loc[mask_nonzero, 'Normalized Weekly Power'] = (
        df_norm.loc[mask_nonzero, 'Predicted Power'] *
        100 / group_sums.loc[mask_nonzero]
    )

    # Calculate quarterly averages
    print("Calculating quarterly averages...")
    df_norm['Quarter'] = df_norm['month'].apply(month_to_quarter)

    quarterly_data = df_norm.groupby(
        ['year', 'Quarter', 'country', 'brand']
    )['Normalized Weekly Power'].mean().reset_index()

    quarterly_data.rename(
        columns={'Normalized Weekly Power': 'Quarterly Avg Power'}, inplace=True)

    print(f"Quarterly data shape: {quarterly_data.shape}")
    print("="*60)
    print("POST-PROCESSING COMPLETED")
    print("="*60)

    return df_norm, quarterly_data


def build_brand_quarter_forecast(
    data: pd.DataFrame,
    model_path: str,
    cutoff_date: str = '2024-06-22',
    forecast_start: str = '2024-06-29',
    selected_features: Optional[List[str]] = None,
) -> Tuple[List[str], dict]:
    """
    Run AutoGluon forecast and return per-brand values for 4 quarters.

    Args:
        data: Input DataFrame with marketing and brand data
        model_path: Path to trained AutoGluon model
        cutoff_date: Training cutoff date
        forecast_start: Forecast period start date
        selected_features: List of features to use

    Returns:
        Tuple of (forecast_quarters, brand_to_values_dict)
        where brand_to_values_dict maps brand -> list of 4 quarterly power values
    """
    print("\n" + "="*80)
    print("AUTOGLUON BRAND QUARTER FORECAST")
    print("="*80)

    if selected_features is None:
        selected_features = DEFAULT_SELECTED_FEATURES

    # Load model
    predictor = TimeSeriesPredictor.load(model_path, require_version_match=False)
    print(f"Model loaded from: {model_path}")

    # Preprocess
    preprocessed = preprocess_data_for_inference(data, selected_features, cutoff_date)
    if preprocessed is None or len(preprocessed) == 0:
        # Retry without cutoff
        preprocessed = preprocess_data_for_inference(data, selected_features, cutoff_date=None)
        if preprocessed is None or len(preprocessed) == 0:
            raise ValueError("Preprocessing produced no rows")

    # Generate forecast
    predictions_df = generate_forecast_with_model(predictor, preprocessed)

    # Format submission
    final_submission = format_submission(predictions_df, forecast_start)

    # Post-process: normalize weekly and calculate quarterly averages
    _, quarterly_data = complete_post_processing(final_submission)

    # Map quarters to expected format
    forecast_quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
    quarter_mapping = {'Qtr3': 'Q3', 'Qtr4': 'Q4', 'Qtr1': 'Q1', 'Qtr2': 'Q2'}

    # Build brand-to-values dictionary
    simulated_data = {}
    brands = quarterly_data['brand'].unique()

    for brand in brands:
        brand_df = quarterly_data[quarterly_data['brand'] == brand]
        values = []
        for quarter in forecast_quarters:
            year, q = quarter.split(' ')
            year = int(year)
            qtr_str = f'Qtr{q[1]}'  # Convert 'Q3' to 'Qtr3'

            matching = brand_df[
                (brand_df['year'] == year) & (brand_df['Quarter'] == qtr_str)
            ]
            if not matching.empty:
                values.append(float(matching['Quarterly Avg Power'].iloc[0]))
            else:
                values.append(0.0)

        simulated_data[brand] = values

    print("="*80)
    print("FORECAST COMPLETED")
    print("="*80)

    return forecast_quarters, simulated_data


__all__ = [
    'create_features',
    'load_scaler',
    'preprocess_data_for_inference',
    'generate_forecast_with_model',
    'format_submission',
    'complete_post_processing',
    'build_brand_quarter_forecast',
    'DEFAULT_SELECTED_FEATURES',
    'SCALER_PATH',
]
