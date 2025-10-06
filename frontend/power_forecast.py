import pandas as pd
import os
from datetime import datetime

channel_groups = {
    "Digital": ["digitaldisplayandsearch", "digitalvideo", "meta", "tiktok", "twitter", "youtube"],
    "Influencer": ["influencer"],
    "TV": ["opentv", "paytv"],
    "OOH_Audio": ["radio", "streamingaudio", "ooh"],
    "Events_Sponsorship": ["brand events", "sponsorship", "others"]
}


def complete_inference_pipeline(data_path, model_path='./autogluon_power_model', selected_features=None):
    """
    Complete inference pipeline: load data, preprocess, load model, generate forecast, format submission.
    """
    print("="*60)
    print("COMPLETE INFERENCE PIPELINE")
    print("="*60)

    # Default features for country 1
    if selected_features is None:
        selected_features = [
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

    # Load data
    print(f"Loading data from: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.pkl'):
        df = pd.read_pickle(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .pkl")

    print(f"Data loaded with shape: {df.shape}")

    # Load model
    predictor = load_trained_model(model_path)
    if predictor is None:
        return None

    # Preprocess data (use cutoff for training period)
    cutoff_date = '2024-06-22'  # 2024-06 week 4
    preprocessed_data = preprocess_data_for_inference(
        df, selected_features, cutoff_date)

    # Generate forecast
    predictions = generate_forecast_with_model(predictor, preprocessed_data)

    # Format submission
    forecast_start = '2024-06-29'  # First week after cutoff
    final_submission = format_submission(predictions, forecast_start)

    print("="*60)
    print("INFERENCE PIPELINE COMPLETED")
    print("="*60)

    return final_submission, predictor


def month_to_quarter(month):
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


def normalize_weekly_power_to_100(df):
    """
    Normalize weekly power so each (year, month, week_of_month, country) group sums to 100.
    """
    print("Normalizing weekly power to sum to 100 per week...")
    df_norm = df.copy()

    # Group by time period and country for normalization
    group_cols = ['year', 'month', 'week_of_month', 'country']
    group_sums = df_norm.groupby(
        group_cols)['Predicted Power'].transform('sum')

    # Normalize to sum to 100 per group
    df_norm['Normalized Weekly Power'] = 0.0
    mask_nonzero = group_sums > 0
    df_norm.loc[mask_nonzero, 'Normalized Weekly Power'] = (
        df_norm.loc[mask_nonzero, 'Predicted Power'] *
        100 / group_sums.loc[mask_nonzero]
    )

    # Verify normalization
    normalized_group_sums = df_norm.groupby(
        group_cols)['Normalized Weekly Power'].sum()
    print(
        f"Number of groups with sum = 100: {(normalized_group_sums.round(2) == 100.0).sum()}")
    print(
        f"Number of groups with sum = 0: {(normalized_group_sums == 0).sum()}")
    print(f"Total number of groups: {len(normalized_group_sums)}")

    return df_norm


def calculate_quarterly_averages(df):
    """
    Calculate quarterly averages from weekly normalized data.
    """
    print("Calculating quarterly averages...")
    df_quarterly = df.copy()

    # Add quarter column
    df_quarterly['Quarter'] = df_quarterly['month'].apply(month_to_quarter)

    # Calculate quarterly averages
    quarterly_avg_power = df_quarterly.groupby(
        ['year', 'Quarter', 'country', 'brand']
    )['Normalized Weekly Power'].mean().reset_index()

    quarterly_avg_power.rename(
        columns={'Normalized Weekly Power': 'Quarterly Avg Power'}, inplace=True)

    print(f"Quarterly data shape: {quarterly_avg_power.shape}")
    print(f"Unique quarters: {quarterly_avg_power['Quarter'].unique()}")

    # Show quarterly sums by country
    quarterly_country_sums = quarterly_avg_power.groupby(
        ['year', 'Quarter', 'country'])['Quarterly Avg Power'].sum()
    print("\nSum of quarterly average power by country and quarter:")
    print(quarterly_country_sums.head(10))

    return quarterly_avg_power


def complete_post_processing(predictions_df):
    """
    Complete post-processing pipeline: normalize weekly to 100, then calculate quarterly averages.
    """
    print("="*60)
    print("COMPLETE POST-PROCESSING PIPELINE")
    print("="*60)

    # Step 1: Normalize weekly power to 100
    normalized_weekly = normalize_weekly_power_to_100(predictions_df)

    # Step 2: Calculate quarterly averages
    quarterly_data = calculate_quarterly_averages(normalized_weekly)

    print("="*60)
    print("POST-PROCESSING COMPLETED")
    print("="*60)

    return normalized_weekly, quarterly_data


def load_model_and_predict_with_postprocessing(data_path, model_path=r'd:\Projects\hackathon_2025\autogluon_power_model', save_results=True):
    """
    Load the trained model, generate predictions, and apply post-processing.

    Args:
        data_path (str): Path to the CSV data file
        model_path (str): Path to the trained AutoGluon model directory
        save_results (bool): Whether to save results to CSV file

    Returns:
        dict: Dictionary containing 'predictions', 'normalized_weekly', and 'quarterly' DataFrames
    """

    print("Loading model and generating predictions with post-processing...")
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")

    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return None

    try:
        # Step 1: Run the complete inference pipeline
        print("\nStep 1: Generating base predictions...")
        final_submission, predictor = complete_inference_pipeline(
            data_path=data_path,
            model_path=model_path
        )

        # Step 2: Apply post-processing
        print("\nStep 2: Applying post-processing...")
        normalized_weekly, quarterly_data = complete_post_processing(
            final_submission)

        results = {
            'predictions': final_submission,
            'normalized_weekly': normalized_weekly,
            'quarterly': quarterly_data
        }

        if save_results:
            # Save results with timestamp
            current_time = datetime.now()
            date_str = current_time.strftime("%m%d")
            time_str = current_time.strftime("%H%M")

            os.makedirs("submission", exist_ok=True)

            # Save original predictions
            output_path = f"submission/predictions_{date_str}_{time_str}.csv"
            final_submission.to_csv(output_path, index=False)
            print(f"Original predictions saved to: {output_path}")

            # Save normalized weekly data
            weekly_path = f"submission/normalized_weekly_{date_str}_{time_str}.csv"
            normalized_weekly.to_csv(weekly_path, index=False)
            print(f"Normalized weekly data saved to: {weekly_path}")

            # Save quarterly data
            quarterly_path = f"submission/quarterly_averages_{date_str}_{time_str}.csv"
            quarterly_data.to_csv(quarterly_path, index=False)
            print(f"Quarterly averages saved to: {quarterly_path}")

        # Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Original predictions: {len(final_submission)} rows")
        print(f"Normalized weekly: {len(normalized_weekly)} rows")
        print(f"Quarterly averages: {len(quarterly_data)} rows")
        print(f"Countries: {final_submission['country'].nunique()}")
        print(f"Brands: {final_submission['brand'].nunique()}")
        print(
            f"Date range: {final_submission['year'].min()}-{final_submission['month'].min()} to {final_submission['year'].max()}-{final_submission['month'].max()}")
        print(
            f"Quarters covered: {sorted(quarterly_data['Quarter'].unique())}")

        return results

    except Exception as e:
        print(f"ERROR: Failed to generate predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_post_processing_results(results):
    """Display detailed results from post-processing."""
    if results is None:
        print("No results to display.")
        return

    print("\n" + "="*60)
    print("POST-PROCESSING RESULTS DETAILS")
    print("="*60)

    # Original predictions
    print("\n1. ORIGINAL PREDICTIONS (first 5):")
    print(results['predictions'][['country', 'brand', 'year',
          'month', 'week_of_month', 'Predicted Power']].head())

    # Normalized weekly
    print("\n2. NORMALIZED WEEKLY (first 5):")
    print(results['normalized_weekly'][['country', 'brand', 'year', 'month',
          'week_of_month', 'Predicted Power', 'Normalized Weekly Power']].head())

    # Quarterly averages
    print("\n3. QUARTERLY AVERAGES (first 10):")
    print(results['quarterly'].head(10))

    # Verification: Check that weekly sums equal 100
    print("\n4. VERIFICATION - Weekly sums (should be 100):")
    weekly_sums = results['normalized_weekly'].groupby(
        ['year', 'month', 'week_of_month', 'country'])['Normalized Weekly Power'].sum()
    print(f"Min weekly sum: {weekly_sums.min():.2f}")
    print(f"Max weekly sum: {weekly_sums.max():.2f}")
    print(f"Mean weekly sum: {weekly_sums.mean():.2f}")
    print(
        f"Groups with sum = 100: {(weekly_sums.round(2) == 100.0).sum()}/{len(weekly_sums)}")

    # Quarterly summary by country
    print("\n5. QUARTERLY SUMMARY BY COUNTRY:")
    quarterly_country_totals = results['quarterly'].groupby(
        ['year', 'Quarter', 'country'])['Quarterly Avg Power'].sum()
    print(quarterly_country_totals)


# --- Lightweight forecast helper that uses model scaling if available ---
def forecast_with_model_and_scaling(data: pd.DataFrame, model_path: str = r'd:\Projects\hackathon_2025\autogluon_power_model'):
    """Try model forecast with x1000 scaling via train_weekly_2 if available; else None."""
    try:
        from autogluon.timeseries import TimeSeriesPredictor  # type: ignore
        try:
            from train_weekly_2 import (
                preprocess_data_for_inference,
                generate_forecast_with_model,
                format_submission,
            )  # type: ignore
        except Exception:
            return None

        selected_features = [
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
        cutoff_date = '2024-06-22'
        forecast_start = '2024-06-29'

        predictor = TimeSeriesPredictor.load(model_path)
        pre_df = preprocess_data_for_inference(
            data, selected_features, cutoff_date)
        preds = generate_forecast_with_model(predictor, pre_df)
        final_submission = format_submission(preds, forecast_start)
        return final_submission  # Contains 'Predicted Power'
    except Exception:
        return None


def forecast_power_scaled_or_placeholder(data: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """Forecast power using model scaling if available; else fallback placeholder.

    Returns a DataFrame with id columns and a 'power' column.
    """
    model_path = model_path or r'd:\Projects\hackathon_2025\autogluon_power_model'

    model_result = forecast_with_model_and_scaling(data, model_path)
    if model_result is not None and isinstance(model_result, pd.DataFrame) and 'Predicted Power' in model_result.columns:
        df_out = model_result.copy()
        return df_out.rename(columns={'Predicted Power': 'power'})

    # Fallback: placeholder power calculation
    return calculate_brand_power(data)


# Example usage and testing
if __name__ == "__main__":
    # Replace with your actual data path
    data_file = r'D:\Projects\hackathon_website_lightweight\data\weekly_colombia.csv'

    print("Running enhanced prediction with post-processing...")

    # Generate predictions with post-processing
    results = load_model_and_predict_with_postprocessing(
        data_file, save_results=True)

    if results is not None:
        # Show detailed results
        show_post_processing_results(results)

        print("\n" + "="*60)
        print("SUCCESS: All processing completed!")
        print("Check the 'submission' folder for saved CSV files.")
        print("="*60)
    else:
        print("Failed to generate predictions.")
