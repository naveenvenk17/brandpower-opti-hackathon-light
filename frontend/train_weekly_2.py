# %%
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor
import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
import tempfile

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

filterwarnings("ignore")

# display 1000 rows and 1000 columns
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


def read_pkl(file):
    return pd.read_pickle(file)

# %%


def select_best_features(df, target, k=10, time_limit=60*30):
    train_data = df.copy()
    train_data = train_data.fillna(0)
    with tempfile.TemporaryDirectory() as temp_dir:
        predictor = TabularPredictor(
            label=target,
            path=temp_dir,
            verbosity=0
        ).fit(
            train_data,
            time_limit=time_limit,
            presets='medium_quality_faster_train'
        )
        feature_importance = predictor.feature_importance(train_data)
    feature_scores = []
    for feature, importance in zip(feature_importance.index, feature_importance.values):
        if isinstance(importance, (np.ndarray, list)):
            importance_value = float(importance[0]) if len(
                importance) > 0 else 0.0
        else:
            importance_value = float(importance)
        feature_scores.append((feature, importance_value))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    feature_scores_df = pd.DataFrame(feature_scores, columns=[
        'feature', 'importance_score'])
    feature_scores_df['rank'] = range(1, len(feature_scores_df) + 1)
    top_k_features = [feature for feature, _ in feature_scores[:k]]
    return top_k_features, feature_scores_df

# %%


def create_features(df, target_col='power'):
    print(f"Shape of the dataframe: {df.shape}")
    try:
        df_sorted = df.sort_values(['item_id', 'timestamp'])
    except KeyError:
        print("Warning: item_id column not found, sorting by timestamp only")
        df_sorted = df.sort_values(['timestamp'])
    for lag in range(1, 13):
        try:
            df_sorted[f'{target_col}_lag_{lag}'] = (
                df_sorted.groupby('item_id')[target_col].shift(lag)
            )
        except KeyError:
            print(
                f"Warning: Cannot create lag feature for lag {lag}, item_id not found")
            df_sorted[f'{target_col}_lag_{lag}'] = None
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
            print(
                f"Warning: Cannot create rolling features for window {window}, item_id not found")
            df_sorted[f'{target_col}_rolling_mean_{window}'] = None
            df_sorted[f'{target_col}_rolling_median_{window}'] = None
            df_sorted[f'{target_col}_rolling_std_{window}'] = None
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
        print("Warning: Cannot create expanding features, item_id not found")
        df_sorted[f'{target_col}_expanding_mean'] = None
        df_sorted[f'{target_col}_expanding_median'] = None
        df_sorted[f'{target_col}_expanding_std'] = None
    if 'year' in df_sorted.columns:
        df_sorted['year'] = pd.to_numeric(
            df_sorted['year'], errors='coerce').astype('Int64')
    else:
        print("Warning: 'year' column not found, extracting from timestamp")
        df_sorted['year'] = df_sorted['timestamp'].dt.year
    if 'month' in df_sorted.columns:
        df_sorted['month'] = pd.to_numeric(
            df_sorted['month'], errors='coerce').astype('Int64')
    else:
        print("Warning: 'month' column not found, extracting from timestamp")
        df_sorted['month'] = df_sorted['timestamp'].dt.month
    if 'week_of_month' in df_sorted.columns:
        df_sorted['week_of_month'] = pd.to_numeric(
            df_sorted['week_of_month'], errors='coerce').astype('Int64')
    else:
        print("Warning: 'week_of_month' column not found, creating from timestamp")
        df_sorted['week_of_month'] = (
            (df_sorted['timestamp'].dt.day - 1) // 7) + 1
    print(f"Shape after feature engineering: {df_sorted.shape}")
    return df_sorted

# %%


def ensure_regular_frequency(df, freq='W'):
    print(f"Ensuring regular {freq} frequency...")
    item_ids = df['item_id'].unique()
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
    complete_data = []
    for item_id in item_ids:
        for date in full_date_range:
            complete_data.append({'item_id': item_id, 'timestamp': date})
    complete_df = pd.DataFrame(complete_data)
    regular_df = complete_df.merge(df, on=['item_id', 'timestamp'], how='left')
    regular_df = regular_df.sort_values(['item_id', 'timestamp'])

    def fill_group(group):
        return group.ffill().bfill()
    regular_df = regular_df.groupby(
        'item_id', group_keys=False).apply(fill_group)
    print(f"Original shape: {df.shape}, Regular shape: {regular_df.shape}")
    return regular_df


def prepare_data_for_forecasting(df_normalized, selected_features):
    print("Preparing data for forecasting...")
    df_prep = df_normalized.copy()
    print(f"Available columns in df_prep: {df_prep.columns.tolist()[:10]}...")
    print(f"Total columns: {len(df_prep.columns)}")
    print(f"Has 'country' column: {'country' in df_prep.columns}")
    print(f"Has 'brand' column: {'brand' in df_prep.columns}")
    print(f"Has 'month' column: {'month' in df_prep.columns}")
    print(f"Has 'week_of_month' column: {'week_of_month' in df_prep.columns}")
    df_prep['year'] = pd.to_numeric(
        df_prep['year'], errors='coerce').astype(int)
    df_prep['month'] = pd.to_numeric(
        df_prep['month'], errors='coerce').astype(int)
    df_prep['week_of_month'] = pd.to_numeric(
        df_prep['week_of_month'], errors='coerce').astype(int)
    df_prep['month_start'] = pd.to_datetime({
        'year': df_prep['year'],
        'month': df_prep['month'],
        'day': 1
    })
    df_prep['timestamp'] = df_prep['month_start'] + \
        pd.to_timedelta((df_prep['week_of_month'] - 1) * 7, unit='days')
    df_prep['timestamp'] = pd.to_datetime(
        df_prep['timestamp']).dt.tz_localize(None)
    df_prep.drop('month_start', axis=1, inplace=True)
    try:
        df_prep['item_id'] = df_prep['country'] + "_" + df_prep['brand']
        print("Successfully created item_id column")
    except KeyError as e:
        print(f"Error creating item_id: Missing column {e}")
        print(f"Available columns: {df_prep.columns.tolist()}")
        if 'brand' in df_prep.columns:
            df_prep['item_id'] = 'unknown_' + df_prep['brand'].astype(str)
            print("Created fallback item_id using brand only")
        else:
            df_prep['item_id'] = 'unknown_brand_' + df_prep.index.astype(str)
            print("Created fallback item_id using index")
    keep_columns = ['item_id', 'timestamp', 'power'] + selected_features
    available_columns = [col for col in keep_columns if col in df_prep.columns]
    missing_columns = [
        col for col in keep_columns if col not in df_prep.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    df_final = df_prep[available_columns].copy()

    # --- SPLIT TRAIN/TEST BASED ON CUTOFF: train till 2024 month 6 week 4, forecast 52 weeks ---
    # Find the cutoff timestamp for 2024-06 week 4
    cutoff_year = 2024
    cutoff_month = 6
    cutoff_week_of_month = 4
    cutoff_date = pd.to_datetime({'year': [cutoff_year], 'month': [cutoff_month], 'day': [1]})[
        0] + pd.to_timedelta((cutoff_week_of_month - 1) * 7, unit='days')
    print(f"Train/test cutoff timestamp: {cutoff_date}")

    # All rows with timestamp <= cutoff_date are train, after are test
    train_data = df_final[df_final['timestamp'] <= cutoff_date].copy()
    test_data = df_final[df_final['timestamp'] > cutoff_date].copy()

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(
        f"Unique country-brand combinations: {train_data['item_id'].nunique()}")
    print(
        f"Date range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")

    return train_data, test_data

# %%


def evaluate_forecast(submission_df):
    print("Evaluating forecast performance at the week level...")
    print(f"Submission shape: {submission_df.shape}")
    if 'item_id' in submission_df.columns:
        print(
            f"Unique country-brand combinations: {submission_df['item_id'].nunique()}")
    else:
        print("No item_id column found in submission")
        print(f"Available columns: {submission_df.columns.tolist()}")
    if 'year' in submission_df.columns and 'month' in submission_df.columns and 'week_of_month' in submission_df.columns:
        print(
            f"Submission covers months: {sorted(submission_df['month'].unique())}")
        print(
            f"Submission covers years: {sorted(submission_df['year'].unique())}")
        print(
            f"Submission covers weeks of month: {sorted(submission_df['week_of_month'].unique())}")
    else:
        print("Warning: 'year', 'month', and/or 'week_of_month' columns not found in submission")
        print(f"Available columns: {submission_df.columns.tolist()}")
    print("Evaluation metrics would be calculated here with actual targets")


def submission_template(forecast_df, cycle_start_date):
    submission = forecast_df.copy()
    submission = submission[submission['timestamp'] >= cycle_start_date]
    if 'item_id' in submission.columns:
        submission['country'] = submission['item_id'].str.split('_').str[0]
        submission['brand'] = submission['item_id'].str.split('_').str[1]
    else:
        print("Warning: item_id column not found in forecast_df")
        print(f"Available columns: {submission.columns.tolist()}")
        submission['country'] = 'unknown'
        submission['brand'] = 'unknown'
    submission['year'] = submission['timestamp'].dt.year.astype(int)
    submission['month'] = submission['timestamp'].dt.month.astype(int)
    submission['week_of_month'] = (
        (submission['timestamp'].dt.day - 1) // 7 + 1).astype(int)
    submission.rename(
        columns={'y_pred_forecast': 'predicted_power'}, inplace=True)
    return submission[['country', 'brand', 'year', 'month', 'week_of_month', 'predicted_power']]

# %%


def autogluon_timeseries_pipeline(df_normalized, selected_features, time_limit=60*30):
    print("="*60)
    print("AUTOGLUON TIMESERIES FORECASTING PIPELINE (WEEKLY LEVEL)")
    print("="*60)

    # Step 1: Prepare data
    train_data, test_data = prepare_data_for_forecasting(
        df_normalized, selected_features)

    # Step 2: Create features
    train_data_feat = create_features(train_data, target_col='power')
    test_data_feat = create_features(test_data, target_col='power')

    for df in [train_data_feat, test_data_feat]:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(
                df['timestamp'], errors='coerce', infer_datetime_format=True)
            n_invalid = df['timestamp'].isna().sum()
            if n_invalid > 0:
                print(
                    f"Warning: {n_invalid} invalid timestamps found and dropped.")
                df.dropna(subset=['timestamp'], inplace=True)
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df.sort_values(['item_id', 'timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    print("\nEnsuring regular weekly frequency...")
    train_data_feat = ensure_regular_frequency(train_data_feat, freq='W')
    test_data_feat = ensure_regular_frequency(test_data_feat, freq='W')

    print("\nConverting to TimeSeriesDataFrame format...")

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_data_feat,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_data_feat,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    print(f"Training TimeSeriesDataFrame shape: {train_ts.shape}")
    print(f"Test TimeSeriesDataFrame shape: {test_ts.shape}")

    print("\nInitializing AutoGluon TimeSeriesPredictor...")

    prediction_length = 53

    predictor = TimeSeriesPredictor(
        target='power',
        prediction_length=prediction_length,
        path='./autogluon_power_model',
        eval_metric='RMSE',
        verbosity=2,
        freq='W',
    )

    print("\nTraining the model...")
    print(f"Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")

    fit_kwargs = {
        'train_data': train_ts,
        'presets': 'best_quality',
        'time_limit': time_limit,
        'random_seed': 42
    }

    predictor.fit(**fit_kwargs)

    print("\nGenerating predictions...")

    # For forecasting, we want to predict the next 52 weeks after the last train date
    # So, we need to pass only the test covariates for the forecast period
    # We'll use test_ts as known_covariates, but only for the forecast period

    predictions = predictor.predict(
        data=train_ts,
        known_covariates=test_ts
    )

    print("\nProcessing prediction results...")

    predictions_df = predictions.reset_index()
    test_df = test_ts.reset_index()

    if 'item_id' in predictions_df.columns and 'item_id' in test_df.columns:
        forecast_results = predictions_df[['timestamp', 'item_id', 'mean']].merge(
            test_df[['timestamp', 'item_id', 'power']],
            on=['timestamp', 'item_id'],
            how='left'
        )
    else:
        print("Warning: item_id column missing from predictions or test data")
        print(f"Predictions columns: {predictions_df.columns.tolist()}")
        print(f"Test columns: {test_df.columns.tolist()}")
        forecast_results = predictions_df[['timestamp', 'mean']].merge(
            test_df[['timestamp', 'power']],
            on=['timestamp'],
            how='left'
        )
        forecast_results['item_id'] = 'unknown_item'

    forecast_results.rename(columns={
        'power': 'target',
        'mean': 'y_pred_forecast'
    }, inplace=True)

    print("\nFiltering to first 4 weeks of each month...")

    forecast_results['year'] = forecast_results['timestamp'].dt.year
    forecast_results['month'] = forecast_results['timestamp'].dt.month
    forecast_results['week_of_month'] = (
        (forecast_results['timestamp'].dt.day - 1) // 7 + 1)

    forecast_results = forecast_results[forecast_results['week_of_month'] <= 4]
    print(f"After filtering to 4 weeks per month: {forecast_results.shape}")

    print("\nCreating submission format...")

    # The forecast start date is the first week after the cutoff (2024-06 week 4)
    forecast_start = pd.to_datetime({'year': [2024], 'month': [6], 'day': [1]})[
        0] + pd.to_timedelta((4 - 1) * 7, unit='days') + pd.DateOffset(weeks=1)

    final_submission = submission_template(forecast_results, forecast_start)

    evaluate_forecast(final_submission)

    from datetime import datetime
    current_time = datetime.now()
    date_str = current_time.strftime("%m%d")
    time_str = current_time.strftime("%H%M")

    # output_path = f"submission/power_forecast_weekly_{date_str}_{time_str}.csv"
    # final_submission.to_csv(output_path, index=False)
    # print(f"\nResults saved to: {output_path}")

    print("="*60)
    print("FORECASTING PIPELINE COMPLETED")
    print("="*60)

    return final_submission, predictor

# %% [markdown]
# ## Data Prep - Add files here

# %%
# country 1


df1 = pd.read_csv(
    r'D:\Projects\hackathon_website_lightweight\data\weekly_colombia.csv')
drop_cols = ['meaning', 'difference', 'salience', 'premium']
df1['power'] = df1['power'] * 1000
df1 = df1.drop(columns=drop_cols)

df_norm1 = df1.copy()

numerical_columns = df_norm1.select_dtypes(
    include=[np.number]).columns.tolist()
columns_to_normalize = [
    col for col in numerical_columns if col not in ['power', 'year', 'month', 'week_of_month']]

scaler = StandardScaler()
df_norm1[columns_to_normalize] = scaler.fit_transform(
    df_norm1[columns_to_normalize])

print(df_norm1.shape)

# %%
channel_groups = {
    "Digital": ["digitaldisplayandsearch", "digitalvideo", "meta", "tiktok", "twitter", "youtube"],
    "Influencer": ["influencer"],
    "TV": ["opentv", "paytv"],
    "OOH_Audio": ["radio", "streamingaudio", "ooh"],
    "Events_Sponsorship": ["brand events", "sponsorship", "others"]
}

for group_name, columns in channel_groups.items():
    df_norm1[group_name] = df_norm1[columns].sum(axis=1)

# %%
df_norm1 = df_norm1[['country', 'brand', 'year', 'month', 'week_of_month', 'power', "Digital", "Influencer", "TV", "OOH_Audio", "Events_Sponsorship", "avg_prcp", "consumer price index, core", "consumer_price_inflation", "cpi_adjusted_personal_income", "discount", "discounted_price_ratio", "domestic demand % of gdp", "gdp per capita, lcu", "inflation, cpi, aop", "personal disposable income, lcu",
                     "population, growth", "population, total", "population, working age", "Private consumption including NPISHs, real, LCU", "real fx index", "real_disposable_income", "retail sales, value index", "retail sales, volume index", "unemployment rate"]]

# %%
# END

# %%
# selected_features_country1, feature_scores_df_country1 = select_best_features(
#     df_norm1, 'power', k=15, time_limit=60*2)

# %%
# selected_features_country1

# %%
# store selected features
selected_features_country1 = ["Digital", "Influencer", "TV", "OOH_Audio", "Events_Sponsorship", 'retail sales, value index',
                              'Private consumption including NPISHs, real, LCU',
                              'consumer_price_inflation',
                              'consumer price index, core',
                              'unemployment rate',
                              'gdp per capita, lcu',
                              'population, growth',
                              'cpi_adjusted_personal_income',
                              'inflation, cpi, aop']
selected_features_country2 = ['brand', 'year', 'total_prcp', 'month', 'NBA', 'Nascar', 'num_dry_days',
                              'Football', 'avg_prcp', 'Rodeo', 'avg_weekend_prcp']
selected_features_country3 = ['brand', 'consumer price index, core', 'normalized sales (in hectoliters)', 'normalized sales value', 'inflation, cpi, aop', 'real fx index',
                              'retail sales, volume index', 'year']


# %%
print(df_norm1.loc[df_norm1['power'].notnull(),
      ['year', 'month', 'week_of_month']])

# %%

# %%
training_time_limit = 60 * 5

# Run the pipeline
# print("starting country 1")
# final_submission_country1, trained_predictor_country1 = autogluon_timeseries_pipeline(
#     df_normalized=df_norm1,
#     selected_features=selected_features_country1,
#     time_limit=training_time_limit
# )

# %%
# Save the trained predictor
# Saves to the path specified during initialization
# trained_predictor_country1.save()

# Load the trained predictor (example usage)
# trained_predictor_country1_test = TimeSeriesPredictor.load(
#     trained_predictor_country1.path)

# %%


def forecast_with_trained_predictor(predictor, data, known_covariates=None):
    """
    Forecast using a trained predictor. Data must have an 'item_id' column.
    """
    if 'item_id' not in data.columns:
        raise ValueError(
            "Input data must contain an 'item_id' column for forecasting.")
    ts_data = TimeSeriesDataFrame.from_data_frame(data)
    return predictor.predict(ts_data, known_covariates=known_covariates)


# Example usage:
# Ensure df_norm1 has an 'item_id' column before calling this function
# forecast_df = forecast_with_trained_predictor(
#     trained_predictor_country1_test, df_norm1)


# %%
# final_submission_country1

# %%
# Filter final_submission_country1 for rows where year > 2024 or (year == 2024 and month >= 7)
# final_submission_country1 = final_submission_country1[
#     (final_submission_country1['year'] > 2024) |
#     ((final_submission_country1['year'] == 2024)
#      & (final_submission_country1['month'] >= 7))
# ]


# %%
# final_submission_country1.shape

# %%
# print(final_submission_country1.head(1))
# print(final_submission_country2.head(1))
# print(final_submission_country3.head(1))

# %%
# final_submission_country1[['year', 'month', 'week_of_month']].drop_duplicates()

# %%
# print(df_norm1['brand'].nunique())
# print(final_submission_country1['brand'].nunique())

# %%
# final_submission = pd.concat(
#     [final_submission_country1, final_submission_country2, final_submission_country3])

# %%
# final_submission = final_submission_country1.copy()


# %%
# final_submission.shape


# %%
# final_submission[['year', 'month']].drop_duplicates()

# %%
# month_power_sum = final_submission.groupby(['year', 'month', 'week_of_month', 'country'])[
#     'predicted_power'].sum()
# print("Sum of predicted power in each month:")
# print(month_power_sum.tail())

# %%
# group_cols = ['year', 'month', 'week_of_month', 'country']
# group_sums = final_submission.groupby(
#     group_cols)['predicted_power'].transform('sum')

# %%
# final_submission['normalized_predicted_power'] = 0.0
# mask_nonzero = group_sums > 0
# final_submission.loc[mask_nonzero, 'normalized_predicted_power'] = (
#     final_submission.loc[mask_nonzero, 'predicted_power'] *
#     100000 / group_sums.loc[mask_nonzero]
# )

# %%
# normalized_group_sums = final_submission.groupby(
#     group_cols)['normalized_predicted_power'].sum()
# print("Sum of normalized predicted power in each (year, month, week_of_month, country) group (should be 100,000):")
# print(normalized_group_sums.head(10))

# %%
# print("\nPreview of normalized data:")
# print(final_submission[['country', 'brand', 'year', 'month',
#       'week_of_month', 'predicted_power', 'normalized_predicted_power']].head())

# %%
# print(
#     f"\nNumber of groups with sum = 100,000: {(normalized_group_sums.round(2) == 100000.0).sum()}")
# print(f"Number of groups with sum = 0: {(normalized_group_sums == 0).sum()}")
# print(f"Total number of groups: {len(normalized_group_sums)}")

# %%
# final_submission['Predicted Power'] = final_submission['normalized_predicted_power'] / 1000

# %%
# predicted_power_sum = final_submission.groupby(['year', 'month', 'week_of_month'])[
#     'Predicted Power'].sum().reset_index()
# print("Sum of predicted power for each (year, month, week_of_month):")
# print(predicted_power_sum)

# %%


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


# final_submission['Quarter'] = final_submission['month'].apply(month_to_quarter)

# %%
# quarterly_avg_power = final_submission.groupby(
#     ['year', 'Quarter', 'country', 'brand']
# )['Predicted Power'].mean().reset_index()

# print("Average predicted power per quarter:")
# print(quarterly_avg_power.head())

# %%
# quarterly_avg_power.shape

# %%


# %%
# quarterly_avg_power.head(2)

# %%
# month_power_sum = quarterly_avg_power.groupby(['year', 'Quarter', 'country'])[
#     'Predicted Power'].sum()
# print("Sum of predicted power in each quarter:")
# print(month_power_sum)

# %%
# quarterly_avg_power.shape

# %%
# quarterly_avg_power['brand'].value_counts()

# %%


def load_trained_model(model_path='./autogluon_power_model'):
    """Load the trained AutoGluon TimeSeriesPredictor model."""
    try:
        predictor = TimeSeriesPredictor.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return predictor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_data_for_inference(df, selected_features, cutoff_date=None):
    """
    Preprocess data for inference using the same steps as training.
    Returns preprocessed data ready for forecasting.
    """
    print("Starting data preprocessing for inference...")

    # Make a copy to avoid modifying original data
    df_prep = df.copy()

    # Apply power scaling if needed
    if 'power' in df_prep.columns:
        df_prep['power'] = df_prep['power'] * 1000

    # Drop unnecessary columns
    drop_cols = ['meaning', 'difference', 'salience', 'premium']
    existing_drop_cols = [col for col in drop_cols if col in df_prep.columns]
    if existing_drop_cols:
        df_prep = df_prep.drop(columns=existing_drop_cols)
        print(f"Dropped columns: {existing_drop_cols}")

    # Normalize numerical columns (excluding power, year, month, week_of_month)
    numerical_columns = df_prep.select_dtypes(
        include=[np.number]).columns.tolist()
    columns_to_normalize = [
        col for col in numerical_columns if col not in ['power', 'year', 'month', 'week_of_month']
    ]

    if columns_to_normalize:
        scaler = StandardScaler()
        df_prep[columns_to_normalize] = scaler.fit_transform(
            df_prep[columns_to_normalize])
        print(f"Normalized {len(columns_to_normalize)} numerical columns")

    # Create channel groups (if applicable)
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
            print(f"Created {group_name} from {len(available_cols)} columns")

    # Select final columns based on the pattern from training
    final_columns = ['country', 'brand', 'year', 'month', 'week_of_month', 'power',
                     "Digital", "Influencer", "TV", "OOH_Audio", "Events_Sponsorship",
                     "avg_prcp", "consumer price index, core", "consumer_price_inflation",
                     "cpi_adjusted_personal_income", "discount", "discounted_price_ratio",
                     "domestic demand % of gdp", "gdp per capita, lcu", "inflation, cpi, aop",
                     "personal disposable income, lcu", "population, growth", "population, total",
                     "population, working age", "Private consumption including NPISHs, real, LCU",
                     "real fx index", "real_disposable_income", "retail sales, value index",
                     "retail sales, volume index", "unemployment rate"]

    available_final_cols = [
        col for col in final_columns if col in df_prep.columns]
    df_prep = df_prep[available_final_cols]

    # Ensure required time columns exist
    if 'year' in df_prep.columns:
        df_prep['year'] = pd.to_numeric(
            df_prep['year'], errors='coerce').astype('Int64')
    if 'month' in df_prep.columns:
        df_prep['month'] = pd.to_numeric(
            df_prep['month'], errors='coerce').astype('Int64')
    if 'week_of_month' in df_prep.columns:
        df_prep['week_of_month'] = pd.to_numeric(
            df_prep['week_of_month'], errors='coerce').astype('Int64')

    # Create timestamp
    df_prep['year'] = pd.to_numeric(
        df_prep['year'], errors='coerce').astype(int)
    df_prep['month'] = pd.to_numeric(
        df_prep['month'], errors='coerce').astype(int)
    df_prep['week_of_month'] = pd.to_numeric(
        df_prep['week_of_month'], errors='coerce').astype(int)

    df_prep['month_start'] = pd.to_datetime({
        'year': df_prep['year'],
        'month': df_prep['month'],
        'day': 1
    })
    df_prep['timestamp'] = df_prep['month_start'] + \
        pd.to_timedelta((df_prep['week_of_month'] - 1) * 7, unit='days')
    df_prep['timestamp'] = pd.to_datetime(
        df_prep['timestamp']).dt.tz_localize(None)
    df_prep.drop('month_start', axis=1, inplace=True)

    # Create item_id
    try:
        df_prep['item_id'] = df_prep['country'] + "_" + df_prep['brand']
        print("Successfully created item_id column")
    except KeyError as e:
        print(f"Error creating item_id: Missing column {e}")
        if 'brand' in df_prep.columns:
            df_prep['item_id'] = 'unknown_' + df_prep['brand'].astype(str)
            print("Created fallback item_id using brand only")
        else:
            df_prep['item_id'] = 'unknown_brand_' + df_prep.index.astype(str)
            print("Created fallback item_id using index")

    # Select relevant columns
    keep_columns = ['item_id', 'timestamp', 'power'] + selected_features
    available_columns = [col for col in keep_columns if col in df_prep.columns]
    missing_columns = [
        col for col in keep_columns if col not in df_prep.columns]

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


def generate_forecast_with_model(predictor, preprocessed_data, prediction_length=53):
    """
    Generate forecast using the loaded model and preprocessed data.
    """
    print("Generating forecast with loaded model...")

    # Create features for the data
    data_with_features = create_features(preprocessed_data, target_col='power')

    # Ensure regular frequency
    data_regular = ensure_regular_frequency(data_with_features, freq='W')

    # Convert to TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(
        data_regular,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    print(f"Input TimeSeriesDataFrame shape: {ts_data.shape}")

    # Generate predictions
    predictions = predictor.predict(data=ts_data)

    print("Processing prediction results...")
    predictions_df = predictions.reset_index()

    # Add time components for filtering
    predictions_df['year'] = predictions_df['timestamp'].dt.year
    predictions_df['month'] = predictions_df['timestamp'].dt.month
    predictions_df['week_of_month'] = (
        (predictions_df['timestamp'].dt.day - 1) // 7 + 1)

    # Filter to first 4 weeks of each month
    predictions_df = predictions_df[predictions_df['week_of_month'] <= 4]

    print(
        f"Predictions after filtering to 4 weeks per month: {predictions_df.shape}")

    return predictions_df


def format_submission(predictions_df, forecast_start_date=None):
    """
    Format predictions into submission format with normalization.
    """
    print("Formatting submission...")

    # Create submission dataframe
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

    # Group by time period and country for normalization
    group_cols = ['year', 'month', 'week_of_month', 'country']
    group_sums = submission.groupby(
        group_cols)['predicted_power'].transform('sum')

    # Normalize to sum to 100,000 per group
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


# %%
# Example usage:
# Load the trained model and generate predictions
# final_submission, predictor = complete_inference_pipeline(
#     data_path=r'D:\Projects\hackathon_website_lightweight\data\weekly_colombia.csv',
#     model_path='./autogluon_power_model'
# )

# %%
