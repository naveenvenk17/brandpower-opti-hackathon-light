import pandas as pd
import numpy as np
import os


# Updated to match the simplified marketing spend columns from backend
lst_optimize_allowed_features = ["digital_spend", "tv_spend", "traditional_spend", "sponsorship_spend", "other_spend"]

# Keep original features list for backward compatibility (if needed)
lst_optimize_allowed_features_legacy = ["brand events", "brand promotion", "digitaldisplayandsearch", "digitalvideo", "influencer",
                                 "meta", "ooh", "opentv", "others", "paytv", "radio", "sponsorship", "streamingaudio", "tiktok", "twitter", "youtube",]

lst_fixed_featured = ["avg_prcp", "consumer price index, core", "consumer_price_inflation", "cpi_adjusted_personal_income", "discount", "discounted_price_ratio", "domestic demand % of gdp", "gdp per capita, lcu", "inflation, cpi, aop", "personal disposable income, lcu",
                      "population, growth", "population, total", "population, working age", "Private consumption including NPISHs, real, LCU", "real fx index", "real_disposable_income", "retail sales, value index", "retail sales, volume index", "unemployment rate", "week_of_month"]
lst_id_columns = ["country", "brand", "year", "month", "week_of_month", "quarter"]
lst_target_columns = ["power"]


def load_data(file_path):
    """Load CSV data from file path"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        return None


def roll_data_to_month(data):
    """Aggregate weekly data to monthly level"""
    if data is None or data.empty:
        return None

    # Group by country, brand, year, month and aggregate
    monthly_data = data.groupby(['country', 'brand', 'year', 'month']).agg({
        col: 'sum' if col in lst_optimize_allowed_features else 'mean'
        for col in data.columns if col not in ['country', 'brand', 'year', 'month']
    }).reset_index()

    return monthly_data


def roll_data_to_quarter(data):
    """Aggregate data to quarterly level"""
    if data is None or data.empty:
        return None

    # Create quarter column
    data_copy = data.copy()
    if 'quarter' not in data_copy.columns:
        if 'month' in data_copy.columns:
            try:
                # Ensure month values are numeric and valid
                data_copy['month'] = pd.to_numeric(
                    data_copy['month'], errors='coerce')
                data_copy = data_copy.dropna(
                    subset=['month'])  # Remove invalid months
                data_copy['quarter'] = data_copy['month'].apply(
                    lambda x: f"Q{((int(x)-1)//3)+1}" if pd.notna(x) and 1 <= x <= 12 else 'Q1')
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
            quarterly_data = data_copy.groupby(
                group_cols).agg(agg_dict).reset_index()
        except Exception as e:
            print(f"Error in groupby operation: {e}")
            quarterly_data = data_copy
    else:
        quarterly_data = data_copy

    return quarterly_data


def forecast_power(data, filters=None):
    """Generate forecasted power values (placeholder with random data)"""
    if data is None or data.empty:
        return None

    # Create a copy of the data
    forecast_data = data.copy()

    # Apply filters if provided
    if filters:
        for col, value in filters.items():
            if col in forecast_data.columns:
                forecast_data[col] = value

    # Generate random power values as placeholder
    np.random.seed(42)  # For reproducible results
    base_power = np.random.uniform(5, 20, len(forecast_data))

    # Add some variation based on optimizable features
    for feature in lst_optimize_allowed_features:
        if feature in forecast_data.columns:
            base_power += forecast_data[feature] * \
                np.random.uniform(0.001, 0.01)

    forecast_data['power'] = base_power
    return forecast_data


def optimize_for_given_power(data, target_power):
    """Optimize features for given power target (placeholder)"""
    if data is None or data.empty:
        return None

    optimized_data = data.copy()

    # Generate random optimized values
    np.random.seed(123)
    for feature in lst_optimize_allowed_features:
        if feature in optimized_data.columns:
            optimized_data[feature] = np.random.uniform(
                0, 1000, len(optimized_data))

    return optimized_data


def plot_data(data):
    """Plot data (placeholder - will be implemented in main app)"""
    pass


def optimize_forecast(data, country=None, brand=None):
    """
    Placeholder function for forecast optimization
    This will be replaced with actual optimization logic later
    """
    import random

    if data is None or data.empty:
        return None

    # Generate random optimized forecast
    optimized_data = data.copy()

    # Add some random optimization adjustments
    for col in lst_optimize_allowed_features:
        if col in optimized_data.columns:
            # Apply random optimization factor
            optimization_factor = random.uniform(0.9, 1.15)  # ±15% adjustment
            optimized_data[col] = optimized_data[col] * optimization_factor

    return optimized_data


def calculate_brand_power(data, baseline_data=None):
    """
    Placeholder function for brand power calculation
    Returns simulated power values based on input data
    """
    import random

    if data is None or data.empty:
        return None

    # Generate random power values
    np.random.seed(42)  # For reproducible results

    power_data = data.copy()

    # Calculate power based on marketing spend (placeholder logic)
    base_power = np.random.uniform(8, 25, len(data))

    # Add influence from marketing features
    for feature in lst_optimize_allowed_features:
        if feature in data.columns:
            # Each feature contributes to power
            feature_contribution = data[feature] * \
                np.random.uniform(0.001, 0.005)
            base_power += feature_contribution

    power_data['power'] = base_power

    return power_data


def generate_quarterly_forecast(brands, quarters, country=None):
    """
    Generate quarterly forecast data for given brands and quarters
    Placeholder function returning random but realistic values
    """
    import random

    forecast_data = {}

    # Country-specific base power ranges
    country_ranges = {
        'Brazil': (12, 28),
        'Colombia': (10, 24),
        'US': (15, 32),
        'default': (10, 25)
    }

    power_range = country_ranges.get(country, country_ranges['default'])

    for brand in brands:
        brand_values = []

        # Generate base power for this brand
        base_power = random.uniform(*power_range)

        # Add quarterly trend (slight growth over time)
        for i, quarter in enumerate(quarters):
            # Add trend and some randomness
            trend_factor = 1 + (i * 0.02)  # 2% growth per quarter
            random_factor = random.uniform(0.85, 1.15)  # ±15% variation

            quarterly_power = base_power * trend_factor * random_factor
            brand_values.append(max(0, quarterly_power))

        forecast_data[brand] = brand_values

    return forecast_data


def simulate_marketing_impact(baseline_data, adjustments=None):
    """
    Simulate the impact of marketing adjustments on brand power
    Placeholder function with random variations
    """
    import random

    if baseline_data is None:
        return None

    simulated_data = {}

    for brand, values in baseline_data.items():
        simulated_values = []

        for value in values:
            # Apply random marketing impact
            if adjustments:
                # Use adjustments to influence the simulation
                impact_factor = random.uniform(
                    0.95, 1.08)  # Marketing can improve by up to 8%
            else:
                # No adjustments, just natural variation
                impact_factor = random.uniform(
                    0.92, 1.05)  # ±5% natural variation

            simulated_value = value * impact_factor
            simulated_values.append(max(0, simulated_value))

        simulated_data[brand] = simulated_values

    return simulated_data


def get_optimizable_columns():
    """Return list of optimizable columns"""
    return lst_optimize_allowed_features


def get_brands_from_data(data):
    """Get unique brands from data"""
    if data is None or data.empty or 'brand' not in data.columns:
        return []
    return sorted(data['brand'].unique().tolist())


def get_countries_from_data(data):
    """Get unique countries from data"""
    if data is None or data.empty or 'country' not in data.columns:
        return []
    return sorted(data['country'].unique().tolist())


def filter_data_after_q2_2024(data):
    """Filter data to only include weeks after 2024 Q2"""
    if data is None or data.empty:
        return None

    filtered_data = data.copy()

    # Filter for data after 2024 Q2 (assuming Q2 ends in June, so after week 26)
    if 'year' in filtered_data.columns and 'month' in filtered_data.columns:
        mask = (filtered_data['year'] > 2024) | \
               ((filtered_data['year'] == 2024) & (filtered_data['month'] > 6))
        filtered_data = filtered_data[mask]

    return filtered_data
