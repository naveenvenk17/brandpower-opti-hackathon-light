import os
import pandas as pd

from autogluon.timeseries import TimeSeriesPredictor
from frontend.train_weekly_2 import (
    preprocess_data_for_inference,
    generate_forecast_with_model,
    format_submission,
    complete_post_processing,
)


DEFAULT_SELECTED_FEATURES = [
    "Digital", "Influencer", "TV", "OOH_Audio", "Events_Sponsorship",
    'retail sales, value index',
    'Private consumption including NPISHs, real, LCU',
    'consumer_price_inflation',
    'consumer price index, core',
    'unemployment rate',
    'gdp per capita, lcu',
    'population, growth',
    'cpi_adjusted_personal_income',
    'inflation, cpi, aop'
]


def _map_qtr_to_q(qtr: str) -> str:
    """Map 'Qtr1' -> 'Q1' style labels."""
    if isinstance(qtr, str) and qtr.startswith('Qtr'):
        return 'Q' + qtr.replace('Qtr', '')
    return qtr


def build_brand_quarter_forecast(
    data: pd.DataFrame,
    model_path: str = r'd:\Projects\hackathon_2025\autogluon_power_model',
    cutoff_date: str = '2024-06-22',
    forecast_start: str = '2024-06-29',
    selected_features: list[str] | None = None,
) -> tuple[list[str], dict[str, list[float]]]:
    """Run scaled AutoGluon forecast and return per-brand values for 4 quarters.

    Returns (forecast_quarters, brand_to_values) where forecast_quarters is
    ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2'] and brand_to_values maps brand
    -> list of 4 floats. Raises exceptions if the model or outputs are invalid.
    """
    print("\n" + "="*80)
    print("CHECKPOINT 1: Starting build_brand_quarter_forecast")
    print("="*80)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print("CHECKPOINT 2: Model path exists, loading predictor...")
    predictor = TimeSeriesPredictor.load(model_path)
    print(f"CHECKPOINT 3: Predictor loaded successfully")

    print("CHECKPOINT 4: Setting features and normalizing columns...")
    if selected_features is None:
        selected_features = DEFAULT_SELECTED_FEATURES

    # Normalize expected id columns casing to match training pipeline expectations
    data_in = data.copy()
    rename_map = {}
    for src, dst in [('Brand', 'brand'), ('Country', 'country'), ('Year', 'year'), ('Month', 'month'), ('Week', 'week_of_month')]:
        if src in data_in.columns and dst not in data_in.columns:
            rename_map[src] = dst
    if rename_map:
        data_in = data_in.rename(columns=rename_map)

    print("\nCHECKPOINT 5: INPUT SUMMARY")
    print(f"Incoming data shape: {data.shape}")
    print(f"Incoming data columns: {list(data.columns)[:20]}...")
    print(f"Renamed (if any): {rename_map}")

    print("CHECKPOINT 6: Starting preprocessing...")
    preprocessed = preprocess_data_for_inference(
        data_in, selected_features, cutoff_date
    )
    print(
        f"CHECKPOINT 7: Preprocessed shape (with cutoff): {len(preprocessed) if preprocessed is not None else 'None'}")

    if preprocessed is None or len(preprocessed) == 0:
        # Retry without cutoff once, to avoid empty data
        print("CHECKPOINT 8: Empty after cutoff; retrying without cutoff...")
        preprocessed = preprocess_data_for_inference(
            data_in, selected_features, cutoff_date=None
        )
        print(
            f"CHECKPOINT 9: Preprocessed shape (no cutoff): {len(preprocessed) if preprocessed is not None else 'None'}")
        if preprocessed is None or len(preprocessed) == 0:
            raise ValueError(
                "Preprocessing produced no rows; ensure required identifiers/time coverage exist.")

    print("CHECKPOINT 10: Generating forecast with model...")
    predictions_df = generate_forecast_with_model(predictor, preprocessed)
    print(
        f"CHECKPOINT 11: Predictions shape after model: {len(predictions_df)}")

    print("CHECKPOINT 12: Formatting submission...")
    final_submission = format_submission(predictions_df, forecast_start)
    print(f"CHECKPOINT 13: Final submission shape: {len(final_submission)}")

    # Ensure weekly normalization to 100 and compute quarterly averages
    print("CHECKPOINT 14: Starting post-processing (normalization + quarterly aggregation)...")
    normalized_weekly, quarterly_df = complete_post_processing(
        final_submission)
    print(f"CHECKPOINT 15: Post-processing complete")
    print(
        f"  Weekly normalized shape: {len(normalized_weekly)} | Quarterly rows: {len(quarterly_df)}")
    print(f"  Quarterly df columns: {list(quarterly_df.columns)}")
    if not quarterly_df.empty:
        print(f"  Quarterly df sample (first 3):\n{quarterly_df.head(3)}")

    # Map quarter names and build quarter label as 'YYYY QX'
    print("CHECKPOINT 16: Mapping quarter names...")
    quarterly_df = quarterly_df.copy()
    quarterly_df['Quarter'] = quarterly_df['Quarter'].apply(_map_qtr_to_q)
    quarterly_df['quarter_label'] = quarterly_df['year'].astype(
        str) + ' ' + quarterly_df['Quarter'].astype(str)
    print(
        f"  Unique quarter labels: {sorted(quarterly_df['quarter_label'].unique().tolist())}")

    # Use fixed forecast quarters as required by UI
    forecast_quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
    print(f"CHECKPOINT 17: Expected forecast quarters: {forecast_quarters}")

    # Brands list from data (case-insensitive consistency)
    if 'brand' not in data_in.columns:
        raise ValueError(
            "Input data must contain a 'brand' column for forecasting")
    brands_all = sorted(pd.Series(data_in['brand']).dropna().unique().tolist())
    print(f"CHECKPOINT 18: Extracted {len(brands_all)} brands from input data")

    # Validate coverage and construct output mapping
    print(
        f"CHECKPOINT 19: Building brand-quarter mapping for {len(brands_all)} brands x {len(forecast_quarters)} quarters...")
    brand_to_values: dict[str, list[float]] = {}
    brands_processed = 0
    for brand in brands_all:
        vals: list[float] = []
        for quarter in forecast_quarters:
            slice_q = quarterly_df[
                (quarterly_df['brand'] == brand) &
                (quarterly_df['quarter_label'] == quarter)
            ]
            if slice_q.empty:
                print(
                    f"  ERROR: Missing forecast for brand '{brand}' in quarter '{quarter}'")
                print(
                    f"    Available brands in quarterly_df: {quarterly_df['brand'].unique().tolist()[:5]}...")
                print(
                    f"    Available quarters in quarterly_df: {quarterly_df['quarter_label'].unique().tolist()}")
                raise ValueError(
                    f"Missing forecast for brand '{brand}' in quarter '{quarter}'"
                )
            # Single value per brand-quarter (Quarterly Avg Power)
            vals.append(float(slice_q['Quarterly Avg Power'].mean()))
        brand_to_values[brand] = vals
        brands_processed += 1
        if brands_processed % 5 == 0 or brands_processed == len(brands_all):
            print(f"  Processed {brands_processed}/{len(brands_all)} brands")

    print(f"CHECKPOINT 20: All brands mapped successfully")

    # Validate that each quarter sums to 100 across brands
    print(f"CHECKPOINT 21: Validating quarterly sums (should be ~100 per quarter)...")
    for quarter in forecast_quarters:
        sum_q = 0.0
        for brand in brands_all:
            sum_q += brand_to_values[brand][forecast_quarters.index(quarter)]
        print(f"  Quarter {quarter} sum across brands: {sum_q:.4f}")
        # Relaxed validation: warn if not exactly 100, but don't fail
        if abs(sum_q - 100.0) > 0.5:
            print(
                f"  WARNING: Quarter {quarter} sum is {sum_q:.4f}, expected ~100.0")

    print(f"CHECKPOINT 22: Returning forecast_quarters and brand_to_values")
    print(f"  Total brands in output: {len(brand_to_values)}")
    print(
        f"  Sample brand: {list(brand_to_values.keys())[0]} -> {brand_to_values[list(brand_to_values.keys())[0]]}")
    print("="*80)

    return forecast_quarters, brand_to_values
