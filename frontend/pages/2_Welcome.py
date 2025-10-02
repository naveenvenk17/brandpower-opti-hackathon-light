#!/usr/bin/env python3
"""
BrandCompass.ai - Welcome Page
Brand analysis and simulation interface
"""

from utils import get_optimizable_columns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import sys
import os
import io
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="Welcome - BrandCompass.ai",
    page_icon="🎯",
    layout="wide"
)

# Initialize experiment session state
if 'saved_experiments' not in st.session_state:
    st.session_state.saved_experiments = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .increase {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    .decrease {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    .unchanged {
        background-color: #e2e3e5 !important;
        color: #383d41 !important;
    }
</style>
""", unsafe_allow_html=True)


def get_optimizable_features():
    """Get list of optimizable marketing features"""
    try:
        return get_optimizable_columns()
    except:
        # Fallback list if utils import fails
        return [
            'brand_events', 'brand_promotion', 'digitaldisplayandsearch',
            'digitalvideo', 'influencer', 'meta', 'ooh', 'opentv',
            'others', 'paytv', 'radio', 'sponsorship', 'streamingaudio',
            'tiktok', 'twitter', 'youtube'
        ]


def convert_to_percentage(df, group_cols=['Year', 'Month', 'Week', 'week_of_month']):
    """Convert values to percentages within each week across all brands"""
    if df is None or df.empty:
        return df

    df_pct = df.copy()
    optimizable_features = get_optimizable_features()

    # Get columns that exist in the dataframe - handle both cases
    available_features = [
        col for col in optimizable_features if col in df_pct.columns]

    # Handle flexible group column names - use only time-based columns for grouping
    # This ensures we group by week across all brands, not just within selected brand
    flexible_group_cols = []
    for col in group_cols:
        if col in df_pct.columns:
            flexible_group_cols.append(col)
        elif col.lower() in df_pct.columns:
            flexible_group_cols.append(col.lower())
        elif col.upper() in df_pct.columns:
            flexible_group_cols.append(col.upper())

    if not available_features or not flexible_group_cols:
        return df_pct

    # Calculate percentages within each week across all brands
    for feature in available_features:
        if feature in df_pct.columns:
            try:
                # Group by time period (week) and calculate percentage across all brands
                # This will show what percentage each brand contributes to the total in that week
                df_pct[feature] = df_pct.groupby(flexible_group_cols)[feature].transform(
                    lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
                )
            except Exception as e:
                # If groupby fails, just return original values
                pass

    return df_pct


def load_baseline_forecast():
    """Load baseline forecast data from CSV file"""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to frontend directory, then into data
    baseline_path = os.path.join(os.path.dirname(
        current_dir), "data", "baseline_forecast.csv")

    # Alternative paths to try
    alternative_paths = [
        # Current working directory
        os.path.join("data", "baseline_forecast.csv"),
        os.path.join("..", "data", "baseline_forecast.csv"),  # One level up
        os.path.join(current_dir, "..", "data",
                     "baseline_forecast.csv"),  # Relative to script
        "frontend/data/baseline_forecast.csv",  # From project root
    ]

    # Try the main path first
    try:
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
            return baseline_df
    except Exception as e:
        pass

    # Try alternative paths
    for alt_path in alternative_paths:
        try:
            if os.path.exists(alt_path):
                baseline_df = pd.read_csv(alt_path)
                return baseline_df
        except Exception as e:
            continue

    st.error(f"❌ Could not find baseline_forecast.csv in any of these locations:")
    st.error(f"   - {baseline_path}")
    for alt_path in alternative_paths:
        st.error(f"   - {alt_path}")
    return None


def get_baseline_data_for_brands(baseline_df, brands):
    """Extract baseline data for specific brands and quarters"""
    if baseline_df is None or baseline_df.empty:
        return None

    baseline_data = {}
    quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

    # The CSV format is: year,period,period_type,country,brand,power
    # where period is like 'Q3', 'Q4', 'Q1', 'Q2'

    for brand in brands:
        brand_values = []

        for quarter in quarters:
            # Parse quarter format
            if quarter == '2024 Q3':
                csv_year, csv_period = 2024, 'Q3'
            elif quarter == '2024 Q4':
                csv_year, csv_period = 2024, 'Q4'
            elif quarter == '2025 Q1':
                csv_year, csv_period = 2025, 'Q1'
            elif quarter == '2025 Q2':
                csv_year, csv_period = 2025, 'Q2'
            else:
                csv_year, csv_period = 2024, 'Q3'  # Default

            # Look for matching brand and quarter in baseline data
            # Try exact brand match first (case-insensitive)
            matching_rows = baseline_df[
                (baseline_df['year'] == csv_year) &
                (baseline_df['period'] == csv_period) &
                (baseline_df['brand'].str.lower() == brand.lower())
            ]

            if not matching_rows.empty:
                power_value = matching_rows['power'].iloc[0]
                brand_values.append(power_value)
            else:
                # If no exact match, try partial match
                partial_match = baseline_df[
                    (baseline_df['year'] == csv_year) &
                    (baseline_df['period'] == csv_period) &
                    (baseline_df['brand'].str.contains(
                        brand, case=False, na=False))
                ]

                if not partial_match.empty:
                    power_value = partial_match['power'].iloc[0]
                    brand_values.append(power_value)
                else:
                    # Use average for that quarter if no brand match
                    quarter_avg = baseline_df[
                        (baseline_df['year'] == csv_year) &
                        (baseline_df['period'] == csv_period)
                    ]['power'].mean()

                    if pd.notna(quarter_avg):
                        brand_values.append(quarter_avg)
                    else:
                        brand_values.append(15.0)  # Default fallback

        baseline_data[brand] = brand_values

    return baseline_data


def simulate_outcomes_from_changes(baseline_data, user_changes=None):
    """Simulate outcomes based on user changes to marketing spend"""
    if user_changes is None:
        # No changes made, return baseline data (0% change)
        return baseline_data.copy()

    simulated_data = {}

    for brand, baseline_values in baseline_data.items():
        simulated_values = []

        for value in baseline_values:
            # Apply user changes impact (placeholder logic)
            # In real implementation, this would use ML models
            impact_factor = 1.0  # Start with no change

            # Simple placeholder: sum of percentage changes in marketing spend
            if user_changes:
                total_change = 0
                change_count = 0

                for channel, change_pct in user_changes.items():
                    if change_pct != 0:
                        total_change += change_pct
                        change_count += 1

                if change_count > 0:
                    avg_change = total_change / change_count
                    # Convert marketing spend change to power impact (simplified)
                    # 1% spend change = 0.1% power change
                    impact_factor = 1 + (avg_change * 0.001)

            simulated_value = value * impact_factor
            simulated_values.append(max(0, simulated_value))

        simulated_data[brand] = simulated_values

    return simulated_data


def save_experiment(name, baseline_data, simulated_data, original_data, column_changes=None):
    """Save current experiment state"""
    experiment = {
        'name': name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_data': baseline_data.copy() if baseline_data is not None else None,
        'simulated_data': simulated_data.copy() if simulated_data is not None else None,
        'original_data': original_data.copy() if original_data is not None and not original_data.empty else None,
        'column_changes': column_changes.copy() if column_changes is not None else None
    }
    return experiment


def create_excel_export(experiments):
    """Create Excel file with one tab per experiment with weekly decomposed data"""
    output = io.BytesIO()

    print(f"DEBUG: Starting Excel export for {len(experiments)} experiments")

    # Create Excel file with experiment data sheets
    sheets_created = 0

    # Collect data for all experiments first
    experiment_data_list = []

    for experiment in experiments:
        # Prepare data for this experiment
        experiment_name = experiment['name']
        baseline_data = experiment.get('baseline_data', {})
        simulated_data = experiment.get('simulated_data', {})
        original_data = experiment.get('original_data')

        print(f"DEBUG: Processing experiment '{experiment_name}'")
        print(f"DEBUG: original_data is None: {original_data is None}")
        print(
            f"DEBUG: original_data.empty: {original_data.empty if original_data is not None else 'N/A'}")
        print(
            f"DEBUG: baseline_data keys: {list(baseline_data.keys()) if baseline_data else 'None'}")
        print(
            f"DEBUG: simulated_data keys: {list(simulated_data.keys()) if simulated_data else 'None'}")

        # Create detailed data if original data exists
        if original_data is not None and not original_data.empty:
            try:
                print(
                    f"DEBUG: Processing data for experiment '{experiment_name}'")
                print(f"DEBUG: original_data shape: {original_data.shape}")
                print(
                    f"DEBUG: original_data columns: {list(original_data.columns)}")

                # Create mapping of power values by brand and quarter
                brand_baseline_map = {}
                brand_simulated_map = {}
                quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

                if baseline_data:
                    for brand in baseline_data:
                        brand_baseline_map[brand] = dict(
                            zip(quarters, baseline_data[brand]))
                    print(
                        f"DEBUG: Created baseline map for brands: {list(brand_baseline_map.keys())}")

                if simulated_data:
                    for brand in simulated_data:
                        brand_simulated_map[brand] = dict(
                            zip(quarters, simulated_data[brand]))
                    print(
                        f"DEBUG: Created simulated map for brands: {list(brand_simulated_map.keys())}")

                # Get column names
                brand_col = 'Brand' if 'Brand' in original_data.columns else 'brand'
                month_col = 'Month' if 'Month' in original_data.columns else 'month'

                print(
                    f"DEBUG: Using brand_col='{brand_col}', month_col='{month_col}'")

                if brand_col in original_data.columns and month_col in original_data.columns:
                    # Create a copy of the original data to add power columns
                    export_data = original_data.copy()

                    print(f"DEBUG: Adding power columns to export_data")
                    # Add baseline_power and simulated_power columns
                    export_data['baseline_power'] = None
                    export_data['simulated_power'] = None

                    # Fill power values based on brand and month (mapped to quarter)
                    power_assignments = 0
                    for idx, row in export_data.iterrows():
                        brand = row[brand_col]
                        month = row[month_col]
                        # Get year for quarter mapping
                        year = row.get('year', row.get('Year', 2024))

                        # Convert month to quarter label
                        if isinstance(month, (int, float)):
                            month_num = int(month)
                            year_num = int(year) if isinstance(
                                year, (int, float)) else 2024

                            if month_num >= 1 and month_num <= 3:
                                quarter_label = f"{year_num} Q1"
                            elif month_num >= 4 and month_num <= 6:
                                quarter_label = f"{year_num} Q2"
                            elif month_num >= 7 and month_num <= 9:
                                quarter_label = f"{year_num} Q3"
                            elif month_num >= 10 and month_num <= 12:
                                quarter_label = f"{year_num} Q4"
                            else:
                                quarter_label = f"{year_num} Q1"  # fallback
                        else:
                            # If month is not numeric, try to infer quarter from year
                            year_num = int(year) if isinstance(
                                year, (int, float)) else 2024
                            quarter_label = f"{year_num} Q1"  # fallback

                        # Set power values for this row
                        baseline_assigned = False
                        simulated_assigned = False

                        if brand in brand_baseline_map and quarter_label in brand_baseline_map[brand]:
                            export_data.at[idx,
                                           'baseline_power'] = brand_baseline_map[brand][quarter_label]
                            baseline_assigned = True

                        if brand in brand_simulated_map and quarter_label in brand_simulated_map[brand]:
                            export_data.at[idx,
                                           'simulated_power'] = brand_simulated_map[brand][quarter_label]
                            simulated_assigned = True

                        if baseline_assigned or simulated_assigned:
                            power_assignments += 1

                    print(
                        f"DEBUG: Assigned power values to {power_assignments} rows")

                    # Filter to 2024-7 to 2025-6 range
                    year_col = 'Year' if 'Year' in export_data.columns else 'year'
                    month_col = 'Month' if 'Month' in export_data.columns else 'month'

                    if year_col in export_data.columns and month_col in export_data.columns:
                        filtered_data = export_data[
                            (export_data[year_col] > 2024) |
                            ((export_data[year_col] == 2024)
                             & (export_data[month_col] >= 7))
                        ].copy()
                    else:
                        filtered_data = export_data.copy()

                    # Sort by brand, year, month, week
                    sort_cols = []
                    if brand_col in filtered_data.columns:
                        sort_cols.append(brand_col)
                    if year_col in filtered_data.columns:
                        sort_cols.append(year_col)
                    if month_col in filtered_data.columns:
                        sort_cols.append(month_col)
                    if 'week_of_month' in filtered_data.columns:
                        sort_cols.append('week_of_month')

                    if sort_cols:
                        filtered_data = filtered_data.sort_values(sort_cols)

                    # Drop unwanted columns
                    columns_to_drop = ['meaning', 'difference',
                                       'salience', 'power', 'premium']
                    filtered_data = filtered_data.drop(
                        columns=[col for col in columns_to_drop if col in filtered_data.columns])

                    # Store the processed data
                    experiment_data_list.append({
                        'name': experiment_name,
                        'data': filtered_data
                    })

                    print(
                        f"DEBUG: Successfully processed data for '{experiment_name}'")
                else:
                    print(
                        f"DEBUG: Required columns not found for '{experiment_name}' (brand_col='{brand_col}', month_col='{month_col}')")
            except Exception as e:
                print(
                    f"DEBUG: Failed to export detailed data for '{experiment_name}': {str(e)}")
        else:
            print(f"DEBUG: No valid data for experiment '{experiment_name}'")

    # Now create Excel file with the processed data
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if experiment_data_list:
            # Create sheets for experiments with data
            for experiment_data in experiment_data_list:
                experiment_data['data'].to_excel(
                    writer, sheet_name=experiment_data['name'][:31], index=False)
                print(f"DEBUG: Created sheet for '{experiment_data['name']}'")
        else:
            # Create an empty sheet if no data
            empty_df = pd.DataFrame()
            empty_df.to_excel(writer, sheet_name='No_Data', index=False)
            print("DEBUG: Created empty sheet 'No_Data'")

    output.seek(0)
    return output


def create_comparison_chart(baseline_data, simulated_data, quarters, selected_brand=None):
    """Create comparison chart for baseline vs simulated"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Baseline vs Simulated Power Forecast"]
    )

    brands_to_show = [selected_brand] if selected_brand else list(
        baseline_data.keys())

    colors = px.colors.qualitative.Set1

    for i, brand in enumerate(brands_to_show):
        if brand in baseline_data and brand in simulated_data:
            # Baseline line
            fig.add_trace(
                go.Scatter(
                    x=quarters,
                    y=baseline_data[brand],
                    mode='lines+markers',
                    name=f'{brand} - Baseline',
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8)
                )
            )

            # Simulated line
            fig.add_trace(
                go.Scatter(
                    x=quarters,
                    y=simulated_data[brand],
                    mode='lines+markers',
                    name=f'{brand} - Simulated',
                    line=dict(color=colors[i %
                              len(colors)], width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                )
            )

    fig.update_layout(
        title="Power Forecast Comparison",
        xaxis_title="Quarter",
        yaxis_title="Power",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def main():
    """Main welcome page"""

    # Create sidebar for experiments
    with st.sidebar:
        st.markdown("### 🧪 Saved Experiments")

        # Display saved experiments
        if st.session_state.saved_experiments:
            for i, exp in enumerate(st.session_state.saved_experiments):
                with st.expander(f"{exp['name']} ({exp['timestamp']})"):
                    st.write(f"**Changes:**")
                    if exp.get('column_changes'):
                        for col, data in exp['column_changes'].items():
                            st.write(
                                f"• {col.replace('_', ' ').title()}: {data['change_pct']:+.1f}%")
                    else:
                        st.write("No changes recorded")

                    # Load and Delete experiment buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📂 Load", key=f"load_{i}"):
                            # Load experiment data into current working state
                            experiment = st.session_state.saved_experiments[i]
                            st.session_state.baseline_data = experiment.get(
                                'baseline_data')
                            st.session_state.simulated_data = experiment.get(
                                'simulated_data')
                            st.session_state.original_data = experiment.get(
                                'original_data')
                            st.session_state.column_changes = experiment.get(
                                'column_changes')
                            st.session_state.calculation_done = True  # Mark as calculated
                            st.success(
                                f"✅ Loaded experiment '{experiment['name']}'")
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            experiment_name = st.session_state.saved_experiments[i]['name']
                            st.session_state.saved_experiments.pop(i)
                            st.success(
                                f"🗑️ Deleted experiment '{experiment_name}'")
                            st.rerun()

            # Export and Clear All buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export All to Excel", type="primary"):
                    if st.session_state.saved_experiments:
                        excel_data = create_excel_export(
                            st.session_state.saved_experiments)
                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_data,
                            file_name=f"brandcompass_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            with col2:
                if st.button("🗑️ Clear All Experiments", type="secondary"):
                    experiment_count = len(st.session_state.saved_experiments)
                    st.session_state.saved_experiments = []
                    st.success(
                        f"🗑️ Cleared all {experiment_count} experiments")
        else:
            st.info("No experiments saved yet. Save your first experiment below!")

    # Check if user came from home page
    if 'uploaded_data' not in st.session_state or st.session_state.uploaded_data is None:
        st.error(
            "⚠️ No data found. Please go back to the home page and upload your data.")
        if st.button("🏠 Go to Home"):
            st.switch_page("BrandCompass.py")
        return

    # Initialize session state for this page
    if 'percentage_view' not in st.session_state:
        st.session_state.percentage_view = False
    if 'calculation_done' not in st.session_state:
        st.session_state.calculation_done = False
    if 'baseline_data' not in st.session_state:
        st.session_state.baseline_data = None
    if 'simulated_data' not in st.session_state:
        st.session_state.simulated_data = None
    if 'baseline_df' not in st.session_state:
        st.session_state.baseline_df = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'column_changes' not in st.session_state:
        st.session_state.column_changes = None

    # Header
    st.markdown('<h1 class="main-header">🎯 Welcome to Analysis</h1>',
                unsafe_allow_html=True)

    # Get uploaded data and filter from July 2024 onwards
    df = st.session_state.uploaded_data

    # Filter data to show only from July 2024 onwards
    year_col = 'Year' if 'Year' in df.columns else 'year'
    month_col = 'Month' if 'Month' in df.columns else 'month'
    brand_col = 'Brand' if 'Brand' in df.columns else 'brand'

    if year_col in df.columns and month_col in df.columns:
        df_filtered = df[
            (df[year_col] > 2024) |
            ((df[year_col] == 2024) & (df[month_col] >= 7))
        ].copy()
    else:
        df_filtered = df.copy()

    # Store original data for comparison
    if st.session_state.original_data is None:
        st.session_state.original_data = df_filtered.copy()

    # Load baseline forecast data
    if st.session_state.baseline_df is None:
        st.session_state.baseline_df = load_baseline_forecast()

    # Get all brands from user data
    brands = sorted(df_filtered[brand_col].unique().tolist(
    )) if brand_col in df_filtered.columns else ['Brand1', 'Brand2']

    # Load baseline data for these brands immediately
    if st.session_state.baseline_data is None and st.session_state.baseline_df is not None:
        st.session_state.baseline_data = get_baseline_data_for_brands(
            st.session_state.baseline_df, brands)
        # Initialize simulated data as copy of baseline (0% change initially)
        st.session_state.simulated_data = st.session_state.baseline_data.copy(
        ) if st.session_state.baseline_data else None
        st.session_state.calculation_done = True  # Show results immediately

    # Section: Inputs
    st.markdown('<div class="section-header">🎛️ Analysis Parameters</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Brand multi-select
        st.markdown("**📊 Select Brands:**")
        selected_brands = st.multiselect(
            "Choose brands for analysis:",
            brands,
            default=[brands[0]] if brands else [],  # Default to first brand
            help="Select one or more brands for analysis"
        )

    with col2:
        # Year multi-select
        years = sorted(df_filtered[year_col].unique().tolist()
                       ) if year_col in df_filtered.columns else [2024, 2025]
        st.markdown("**📅 Select Years:**")
        selected_years = st.multiselect(
            "Choose years for analysis:",
            years,
            default=[2024] if 2024 in years else [years[0]] if years else [],
            help="Select one or more years for analysis"
        )

    with col3:
        # Month multi-select
        months = sorted(df_filtered[month_col].unique().tolist()
                        ) if month_col in df_filtered.columns else list(range(7, 13))
        st.markdown("**📆 Select Months:**")
        selected_months = st.multiselect(
            "Choose months for analysis:",
            months,
            default=[7] if 7 in months else [months[0]] if months else [],
            help="Select one or more months for analysis"
        )

    # Section: Data Display
    st.markdown('<div class="section-header">📊 Uploaded Data with Optimizable Features</div>',
                unsafe_allow_html=True)

    # Toggle for percentage view
    col1, col2 = st.columns([1, 4])
    with col1:
        percentage_toggle = st.toggle(
            "📊 Switch to % View",
            value=st.session_state.percentage_view,
            help="Toggle between raw values and percentage distribution"
        )

        if percentage_toggle != st.session_state.percentage_view:
            st.session_state.percentage_view = percentage_toggle
            st.rerun()

    # Create base dataframe for percentage calculation (all data, no filters)
    # This ensures percentages are calculated against the full dataset
    base_df_for_percentage = df_filtered.copy()  # Use all July 2024+ data

    # Filter data based on multi-select selections for display only
    filtered_df = df_filtered.copy()

    # Apply multi-select filters for display only
    if selected_brands and brand_col in df_filtered.columns:
        filtered_df = filtered_df[filtered_df[brand_col].isin(selected_brands)]

    if selected_years and year_col in df_filtered.columns:
        filtered_df = filtered_df[filtered_df[year_col].isin(selected_years)]

    if selected_months and month_col in df_filtered.columns:
        filtered_df = filtered_df[filtered_df[month_col].isin(selected_months)]

    # Show selection summary
    if selected_brands or selected_years or selected_months:
        summary_parts = []
        if selected_brands:
            summary_parts.append(f"Brands: {', '.join(selected_brands)}")
        if selected_years:
            summary_parts.append(
                f"Years: {', '.join(map(str, selected_years))}")
        if selected_months:
            summary_parts.append(
                f"Months: {', '.join(map(str, selected_months))}")

    # Get optimizable features
    optimizable_features = get_optimizable_features()

    # Build display columns using flexible column names
    base_columns = []
    if brand_col in filtered_df.columns:
        base_columns.append(brand_col)
    if year_col in filtered_df.columns:
        base_columns.append(year_col)
    if month_col in filtered_df.columns:
        base_columns.append(month_col)
    if 'Week' in filtered_df.columns:
        base_columns.append('Week')
    elif 'week' in filtered_df.columns:
        base_columns.append('week')
    if 'week_of_month' in filtered_df.columns:
        base_columns.append('week_of_month')

    display_columns = base_columns + \
        [col for col in optimizable_features if col in filtered_df.columns]

    # Apply percentage conversion if toggled
    if st.session_state.percentage_view:
        # For percentage view, calculate percentages using ALL data (base_df_for_percentage)
        # Then filter the results for display
        group_cols = []
        if year_col in base_df_for_percentage.columns:
            group_cols.append(year_col)
        if month_col in base_df_for_percentage.columns:
            group_cols.append(month_col)
        if 'Week' in base_df_for_percentage.columns:
            group_cols.append('Week')
        elif 'week' in base_df_for_percentage.columns:
            group_cols.append('week')
        if 'week_of_month' in base_df_for_percentage.columns:
            group_cols.append('week_of_month')

        # Calculate percentages using ALL data
        base_percentage_df = convert_to_percentage(
            base_df_for_percentage[display_columns], group_cols)

        # Filter the percentage results to match the display filters
        display_df = base_percentage_df.copy()
        if selected_brands and brand_col in display_df.columns:
            display_df = display_df[display_df[brand_col].isin(
                selected_brands)]
        if selected_years and year_col in display_df.columns:
            display_df = display_df[display_df[year_col].isin(selected_years)]
        if selected_months and month_col in display_df.columns:
            display_df = display_df[display_df[month_col].isin(
                selected_months)]

    else:
        display_df = filtered_df[display_columns]

    # Display editable table
    if not display_df.empty:
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                col: st.column_config.NumberColumn(
                    col.replace('_', ' ').title(),
                    help=f"Optimizable feature: {col}",
                    format="%.2f" if st.session_state.percentage_view else "%.0f"
                ) for col in optimizable_features if col in display_df.columns
            },
            key="data_editor"
        )

        # Store edited data for change detection
        if 'edited_data' not in st.session_state:
            st.session_state.edited_data = edited_df.copy()
        else:
            st.session_state.edited_data = edited_df.copy()

    else:
        st.warning("No data available for the selected filters.")

    # Calculate Button
    st.markdown('<div class="section-header">🚀 Calculate</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calculate_button = st.button(
            "🧮 Calculate Brand Power",
            type="primary",
            use_container_width=True,
            help="Generate brand power analysis and forecasts"
        )

    if calculate_button:
        with st.spinner("🔄 Calculating brand power..."):
            # Detect changes in user data
            user_changes = {}

            if 'edited_data' in st.session_state and st.session_state.original_data is not None:
                # Compare edited data with original data
                try:
                    # Check if required columns exist
                    brand_col = 'Brand' if 'Brand' in st.session_state.original_data.columns else 'brand'
                    year_col = 'Year' if 'Year' in st.session_state.original_data.columns else 'year'
                    month_col = 'Month' if 'Month' in st.session_state.original_data.columns else 'month'

                    if brand_col in st.session_state.original_data.columns:
                        # Filter by selected brands
                        if selected_brands:
                            original_subset = st.session_state.original_data[
                                st.session_state.original_data[brand_col].isin(
                                    selected_brands)
                            ]
                        else:
                            original_subset = st.session_state.original_data

                        if year_col in st.session_state.original_data.columns:
                            # Filter by selected years
                            if selected_years:
                                original_subset = original_subset[
                                    original_subset[year_col].isin(
                                        selected_years)
                                ]

                        if month_col in st.session_state.original_data.columns:
                            # Filter by selected months
                            if selected_months:
                                original_subset = original_subset[
                                    original_subset[month_col].isin(
                                        selected_months)
                                ]

                        if not original_subset.empty and not st.session_state.edited_data.empty:
                            # Get the data in the same format as edited_data for comparison
                            if st.session_state.percentage_view:
                                # Calculate group columns for percentage conversion
                                group_cols_for_pct = []
                                if year_col in original_subset.columns:
                                    group_cols_for_pct.append(year_col)
                                if month_col in original_subset.columns:
                                    group_cols_for_pct.append(month_col)
                                if 'Week' in original_subset.columns:
                                    group_cols_for_pct.append('Week')
                                elif 'week' in original_subset.columns:
                                    group_cols_for_pct.append('week')
                                if 'week_of_month' in original_subset.columns:
                                    group_cols_for_pct.append('week_of_month')

                                original_for_comparison = convert_to_percentage(
                                    original_subset[display_columns], group_cols_for_pct)
                            else:
                                original_for_comparison = original_subset[display_columns]

                            # Calculate percentage changes based on sum of values
                            column_changes = {}

                            for col in optimizable_features:
                                if col in original_for_comparison.columns and col in st.session_state.edited_data.columns:
                                    # Calculate sum of original values
                                    original_sum = original_for_comparison[col].sum(
                                    )
                                    # Calculate sum of edited values
                                    edited_sum = st.session_state.edited_data[col].sum(
                                    )

                                    if original_sum != 0:
                                        change_pct = (
                                            (edited_sum - original_sum) / original_sum) * 100
                                        if abs(change_pct) > 0.01:  # Only show changes > 0.01%
                                            column_changes[col] = {
                                                'original_sum': original_sum,
                                                'new_sum': edited_sum,
                                                'change_pct': change_pct
                                            }
                                        user_changes[col] = change_pct
                                    else:
                                        user_changes[col] = 0

                            # Store changes in session state for persistent display
                            st.session_state.column_changes = column_changes
                except Exception as e:
                    st.warning(f"Could not detect changes: {str(e)}")
                    user_changes = {}

            # Apply user changes to simulation
            if st.session_state.baseline_data:
                st.session_state.simulated_data = simulate_outcomes_from_changes(
                    st.session_state.baseline_data, user_changes
                )

            st.session_state.calculation_done = True

        st.success("✅ Calculation completed successfully!")

    # Display marketing channel changes (persistent across reruns)
    if st.session_state.column_changes:
        st.markdown("### 📊 Marketing Channel Changes")
        st.markdown("**Percentage change in total spend by channel:**")

        # Create a DataFrame for better display
        change_data = []
        for col, data in st.session_state.column_changes.items():
            change_data.append({
                'Marketing Channel': col.replace('_', ' ').title(),
                'Original Sum': f"{data['original_sum']:,.0f}",
                'New Sum': f"{data['new_sum']:,.0f}",
                'Change %': f"{data['change_pct']:+.1f}%"
            })

        changes_df = pd.DataFrame(change_data)
        st.dataframe(changes_df, use_container_width=True)
    elif st.session_state.calculation_done:
        st.info("ℹ️ No significant changes detected in marketing channels.")

    # Show results if calculation is done
    if st.session_state.calculation_done and st.session_state.baseline_data and st.session_state.simulated_data:

        # Section: Quarterly Brand Power Table
        st.markdown(
            '<div class="section-header">📊 Quarterly Brand Power Results</div>', unsafe_allow_html=True)

        quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

        # Create results dataframe
        results_data = []
        for brand in brands:
            if brand in st.session_state.baseline_data and brand in st.session_state.simulated_data:
                row = {'Brand': brand}
                for i, quarter in enumerate(quarters):
                    baseline_val = st.session_state.baseline_data[brand][i]
                    simulated_val = st.session_state.simulated_data[brand][i]

                    # Calculate change
                    change = ((simulated_val - baseline_val) /
                              baseline_val) * 100 if baseline_val != 0 else 0

                    # Format with change indicator
                    if abs(change) < 1:  # Less than 1% change
                        status = "unchanged"
                        indicator = "→"
                    elif change > 0:
                        status = "increase"
                        indicator = "↗"
                    else:
                        status = "decrease"
                        indicator = "↘"

                    row[quarter] = f"{simulated_val:.2f} {indicator} ({change:+.1f}%)"

                results_data.append(row)

        results_df = pd.DataFrame(results_data)

        if not results_df.empty:
            # Display results table with custom styling
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    quarter: st.column_config.TextColumn(
                        quarter,
                        help=f"Power value and change for {quarter}"
                    ) for quarter in quarters
                }
            )

            # Legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **↗ Increase** vs baseline")
            with col2:
                st.markdown("🔴 **↘ Decrease** vs baseline")
            with col3:
                st.markdown("⚪ **→ Unchanged** (< 1% change)")

        # Section: Charts
        st.markdown(
            '<div class="section-header">📈 Baseline vs Simulated Comparison</div>', unsafe_allow_html=True)

        # Brand selection for chart
        chart_options = ["All Brands"] + brands
        chart_brand = st.selectbox(
            "Select brand for detailed chart:",
            chart_options,
            help="Choose which brand(s) to display in the comparison chart"
        )

        # Create and display chart
        selected_brand_for_chart = None if chart_brand == "All Brands" else chart_brand

        fig = create_comparison_chart(
            st.session_state.baseline_data,
            st.session_state.simulated_data,
            quarters,
            selected_brand_for_chart
        )

        st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        with st.expander("🔍 Analysis Insights"):
            st.write("### Key Findings:")

            # Calculate overall trends
            total_baseline = sum(
                [sum(values) for values in st.session_state.baseline_data.values()])
            total_simulated = sum(
                [sum(values) for values in st.session_state.simulated_data.values()])
            overall_change = ((total_simulated - total_baseline) /
                              total_baseline) * 100 if total_baseline != 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Performance", f"{overall_change:+.2f}%",
                          delta=f"{total_simulated - total_baseline:.2f}")
            with col2:
                best_brand = max(brands, key=lambda b: sum(
                    st.session_state.simulated_data.get(b, [0])))
                st.metric("Top Performing Brand", best_brand,
                          delta="Highest total power")

            st.write("""
            **Methodology:**
            - Baseline values generated using historical patterns and trends
            - Simulated outcomes incorporate optimization adjustments
            - Results show quarterly power projections for 2024 Q3 through 2025 Q2
            - Color coding indicates performance relative to baseline forecasts
            """)

    # Save Experiment Section
    st.markdown('<div class="section-header">💾 Save Experiment</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        # Generate default experiment name
        existing_names = [exp['name']
                          for exp in st.session_state.saved_experiments]
        default_name = f"experiment_{len(st.session_state.saved_experiments) + 1}"
        counter = 1
        while default_name in existing_names:
            counter += 1
            default_name = f"experiment_{counter}"

        experiment_name = st.text_input(
            "Experiment Name:",
            value=default_name,
            help="Give your experiment a descriptive name"
        )

    with col2:
        save_button = st.button(
            "💾 Save Experiment",
            type="secondary",
            use_container_width=True,
            help="Save current analysis state for later comparison"
        )

    if save_button:
        if not experiment_name.strip():
            st.error("⚠️ Please enter a name for the experiment.")
        elif len(st.session_state.saved_experiments) >= 5:
            st.error(
                "⚠️ Maximum of 5 experiments allowed. Please delete an existing experiment first.")
        elif not st.session_state.calculation_done:
            st.warning("⚠️ Please run the calculation first before saving.")
        else:
            # Save the experiment
            experiment = save_experiment(
                experiment_name.strip(),
                st.session_state.baseline_data,
                st.session_state.simulated_data,
                st.session_state.original_data,
                st.session_state.column_changes
            )
            st.session_state.saved_experiments.append(experiment)
            st.success(f"✅ Experiment '{experiment_name}' saved successfully!")
            st.rerun()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Back to Home"):
            st.switch_page("BrandCompass.py")
    with col2:
        if st.button("🔄 Reset Analysis"):
            st.session_state.calculation_done = False
            st.session_state.baseline_data = None
            st.session_state.simulated_data = None
            st.session_state.baseline_df = None
            st.session_state.original_data = None
            st.session_state.percentage_view = False
            st.session_state.column_changes = None
            # Note: saved_experiments are NOT cleared - only current working state
            if 'edited_data' in st.session_state:
                del st.session_state.edited_data
            st.rerun()


if __name__ == "__main__":
    main()
