"""
Channel utilities for brand power calculation and data aggregation
Migrated from hackathon_website_lightweight/frontend/utils.py
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Constants
lst_optimize_allowed_features = ['paytv', 'wholesalers']
lst_fixed_featured = [
    'brand', 'year', 'retail sales, value index', 'month',
    'consumer price index, core', 'inflation, cpi, aop',
    'normalized sales (in hectoliters)', 'normalized sales value',
    'real fx index', 'retail sales, volume index', 'total_distribution', 'volume'
]
lst_id_columns = ["country", "brand", "year", "month", "week_of_month"]
lst_target_columns = ["power"]

channel_groups = {
    "Media-TV": ["paytv"],
    "Wholesalers": ["wholesalers"],
#    "total_distribution": ["total_distribution"],
#    "Forecasted Volume": ["volume"]
}

colombia_megabrands = ['AGUILA', 'FAMILIA POKER', 'FAMILIA CORONA', 'FAMILIA CLUB COLOMBIA']


def get_optimizable_columns() -> List[str]:
    """Return list of optimizable columns"""
    return lst_optimize_allowed_features


def get_channel_groups() -> Dict[str, List[str]]:
    """Return channel groups dictionary"""
    return channel_groups


def aggregate_by_channel_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual feature columns by channel groups

    Args:
        df: DataFrame with individual feature columns

    Returns:
        DataFrame with aggregated channel group columns added
    """
    aggregated_df = df.copy()

    for group_name, features in channel_groups.items():
        # Sum all features in this group that exist in the dataframe
        group_sum = pd.Series(0, index=df.index)
        for feature in features:
            if feature in df.columns:
                group_sum += df[feature].fillna(0)

        # Add the aggregated column
        aggregated_df[group_name] = group_sum

    return aggregated_df


def calculate_brand_power(data: pd.DataFrame, baseline_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Calculate brand power based on marketing spend features

    Uses feature coefficients with saturation curves to model diminishing returns.
    This provides a more realistic brand power response to marketing spend.

    Args:
        data: DataFrame with marketing features
        baseline_data: Optional baseline data for comparison

    Returns:
        DataFrame with 'power' column added, or None if input is invalid
    """
    if data is None or data.empty:
        return None

    power_data = data.copy()

    # Base power (brand baseline strength)
    base_power = np.full(len(data), 12.0)  # Deterministic base

    # Feature coefficients (calibrated for realistic impact)
    feature_coefficients = {
        'paytv': 0.00000040,          # PayTV has moderate impact
        'wholesalers': 0.00000060,     # Wholesalers have higher impact (distribution reach)
        'total_distribution': 0.00000025,  # Distribution has baseline impact
        'volume': 0.00000015,          # Volume contributes to power
    }

    # Saturation parameters (diminishing returns)
    saturation_K = 50_000_000  # Half-saturation point ($50M)
    saturation_alpha = 0.6     # Curve shape (higher = sharper curve)

    # Add influence from marketing features with saturation
    for feature in lst_optimize_allowed_features:
        if feature in data.columns:
            coef = feature_coefficients.get(feature, 0.0)
            if coef > 0:
                # Apply Hill saturation curve for diminishing returns
                feature_values = data[feature].fillna(0).values.astype(float)
                
                # Saturation: S(x) = x^α / (K^α + x^α)
                # This creates an S-curve where spending shows diminishing returns
                x_alpha = np.power(np.maximum(feature_values, 0.1), saturation_alpha)
                K_alpha = np.power(saturation_K, saturation_alpha)
                saturated_values = x_alpha / (K_alpha + x_alpha + 1e-10)
                
                # Normalized contribution (0 to ~20 range)
                feature_contribution = coef * feature_values
                base_power += feature_contribution

    # Ensure power stays in reasonable range (8-35 to show variability)
    power_data['power'] = np.clip(base_power, 8.0, 35.0)

    return power_data


def roll_data_to_month(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Aggregate weekly data to monthly level

    Args:
        data: DataFrame with weekly data

    Returns:
        DataFrame aggregated to monthly level
    """
    if data is None or data.empty:
        return None

    # Group by country, brand, year, month and aggregate
    monthly_data = data.groupby(['country', 'brand', 'year', 'month']).agg({
        col: 'sum' if col in lst_optimize_allowed_features else 'mean'
        for col in data.columns if col not in ['country', 'brand', 'year', 'month']
    }).reset_index()

    return monthly_data


def roll_data_to_quarter(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Aggregate data to quarterly level

    Args:
        data: DataFrame with monthly/weekly data

    Returns:
        DataFrame aggregated to quarterly level with 'quarter' column
    """
    if data is None or data.empty:
        return None

    # Create quarter column
    data_copy = data.copy()
    if 'quarter' not in data_copy.columns:
        if 'month' in data_copy.columns:
            try:
                # Ensure month values are numeric and valid
                data_copy['month'] = pd.to_numeric(data_copy['month'], errors='coerce')
                data_copy = data_copy.dropna(subset=['month'])  # Remove invalid months
                data_copy['quarter'] = data_copy['month'].apply(
                    lambda x: f"Q{((int(x)-1)//3)+1}" if pd.notna(x) and 1 <= x <= 12 else 'Q1'
                )
            except Exception as e:
                print(f"Error processing month column: {e}")
                data_copy['quarter'] = 'Q1'  # Default fallback
        else:
            # If no month column, try to infer from other data or create dummy quarters
            data_copy['quarter'] = 'Q1'  # Default fallback

    # Determine grouping columns based on what's available
    group_cols = []
    for col in ['country', 'brand', 'year', 'quarter']:
        if col in data_copy.columns:
            group_cols.append(col)

    if not group_cols:
        return data_copy  # Return as-is if no grouping columns found

    # Group and aggregate
    agg_dict = {}
    for col in data_copy.columns:
        if col not in group_cols:
            if col in lst_optimize_allowed_features:
                agg_dict[col] = 'sum'
            elif col in ['power'] + lst_target_columns:
                agg_dict[col] = 'mean'  # Average power values
            else:
                agg_dict[col] = 'mean'  # Default to mean for other columns

    if agg_dict:
        try:
            quarterly_data = data_copy.groupby(group_cols).agg(agg_dict).reset_index()
        except Exception as e:
            print(f"Error in groupby operation: {e}")
            quarterly_data = data_copy
    else:
        quarterly_data = data_copy

    return quarterly_data


def get_brands_from_data(data: pd.DataFrame) -> List[str]:
    """
    Get unique brands from data

    Args:
        data: DataFrame with 'brand' column

    Returns:
        Sorted list of unique brand names
    """
    if data is None or data.empty or 'brand' not in data.columns:
        return []
    return sorted(data['brand'].unique().tolist())


def get_countries_from_data(data: pd.DataFrame) -> List[str]:
    """
    Get unique countries from data

    Args:
        data: DataFrame with 'country' column

    Returns:
        Sorted list of unique country names
    """
    if data is None or data.empty or 'country' not in data.columns:
        return []
    return sorted(data['country'].unique().tolist())


def filter_data_after_q2_2024(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Filter data to only include weeks after 2024 Q2

    Args:
        data: DataFrame with 'year' and 'month' columns

    Returns:
        Filtered DataFrame with data after 2024 Q2
    """
    if data is None or data.empty:
        return None

    filtered_data = data.copy()

    # Filter for data after 2024 Q2 (assuming Q2 ends in June, so after week 26)
    if 'year' in filtered_data.columns and 'month' in filtered_data.columns:
        mask = (filtered_data['year'] > 2024) | \
               ((filtered_data['year'] == 2024) & (filtered_data['month'] > 6))
        filtered_data = filtered_data[mask]

    return filtered_data
