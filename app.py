#!/usr/bin/env python3
"""
BrandCompass.ai - Flask Web Application
A comprehensive brand power simulation platform
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_session import Session
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from frontend.utils import (
    get_optimizable_columns,
    roll_data_to_quarter,
    get_brands_from_data,
    lst_id_columns,
    get_channel_groups,
    aggregate_by_channel_groups,
)

app = Flask(__name__)

# Configure server-side session to handle large session data
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'brandcompass:'
app.secret_key = os.urandom(24)

# Initialize the session extension
Session(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Create flask_session directory for server-side sessions
if not os.path.exists('./flask_session/'):
    os.makedirs('./flask_session/')

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_template_csv():
    """Create a template CSV for download"""
    template_data = {
        'Brand': ['BrandA', 'BrandA', 'BrandB', 'BrandB', 'BrandC', 'BrandC'],
        'Year': [2024, 2024, 2024, 2024, 2024, 2024],
        'Month': [7, 8, 7, 8, 7, 8],
        'Week': [1, 1, 1, 1, 1, 1],
        'brand_events': [100, 110, 80, 90, 120, 130],
        'brand_promotion': [200, 210, 180, 190, 220, 230],
        'digitaldisplayandsearch': [150, 160, 140, 150, 170, 180],
        'digitalvideo': [300, 310, 280, 290, 320, 330],
        'influencer': [50, 55, 45, 50, 60, 65],
        'meta': [400, 410, 380, 390, 420, 430],
        'ooh': [75, 80, 70, 75, 85, 90],
        'opentv': [500, 510, 480, 490, 520, 530],
        'others': [25, 30, 20, 25, 35, 40],
        'paytv': [200, 210, 180, 190, 220, 230],
        'radio': [100, 110, 90, 100, 120, 130],
        'sponsorship': [150, 160, 140, 150, 170, 180],
        'streamingaudio': [80, 85, 75, 80, 90, 95],
        'tiktok': [120, 125, 110, 115, 130, 135],
        'twitter': [90, 95, 85, 90, 100, 105],
        'youtube': [250, 260, 240, 250, 270, 280]
    }
    return pd.DataFrame(template_data)


def load_baseline_forecast():
    """Load baseline forecast data from CSV file"""
    baseline_path = os.path.join("frontend", "data", "baseline_forecast.csv")
    alternative_paths = [
        os.path.join("data", "baseline_forecast.csv"),
        "frontend/data/baseline_forecast.csv"
    ]

    try:
        if os.path.exists(baseline_path):
            return pd.read_csv(baseline_path)
    except Exception:
        pass

    for alt_path in alternative_paths:
        try:
            if os.path.exists(alt_path):
                return pd.read_csv(alt_path)
        except Exception:
            continue

    return None


def get_baseline_data_for_brands(baseline_df, brands):
    """Extract baseline data for specific brands and quarters"""
    if baseline_df is None or baseline_df.empty:
        return None

    baseline_data = {}
    quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

    for brand in brands:
        brand_values = []
        for quarter in quarters:
            if quarter == '2024 Q3':
                csv_year, csv_period = 2024, 'Q3'
            elif quarter == '2024 Q4':
                csv_year, csv_period = 2024, 'Q4'
            elif quarter == '2025 Q1':
                csv_year, csv_period = 2025, 'Q1'
            elif quarter == '2025 Q2':
                csv_year, csv_period = 2025, 'Q2'
            else:
                csv_year, csv_period = 2024, 'Q3'

            matching_rows = baseline_df[
                (baseline_df['year'] == csv_year) &
                (baseline_df['period'] == csv_period) &
                (baseline_df['brand'].str.lower() == brand.lower())
            ]

            if not matching_rows.empty:
                brand_values.append(matching_rows['power'].iloc[0])
            else:
                quarter_avg = baseline_df[
                    (baseline_df['year'] == csv_year) &
                    (baseline_df['period'] == csv_period)
                ]['power'].mean()
                brand_values.append(
                    quarter_avg if pd.notna(quarter_avg) else 15.0)

        baseline_data[brand] = brand_values

    return baseline_data


def simulate_outcomes_from_changes(baseline_data, user_changes=None):
    """Simulate outcomes based on user changes to marketing spend"""
    if user_changes is None:
        return baseline_data.copy()

    simulated_data = {}
    for brand, baseline_values in baseline_data.items():
        simulated_values = []
        for value in baseline_values:
            impact_factor = 1.0
            if user_changes:
                total_change = sum(user_changes.values())
                change_count = len(
                    [c for c in user_changes.values() if c != 0])
                if change_count > 0:
                    avg_change = total_change / change_count
                    impact_factor = 1 + (avg_change * 0.001)
            simulated_value = value * impact_factor
            simulated_values.append(max(0, simulated_value))
        simulated_data[brand] = simulated_values

    return simulated_data


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/select_country', methods=['POST'])
def select_country():
    """Handle country selection"""
    country = request.form.get('country')
    session['selected_country'] = country
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Round all numeric columns to 0 decimal points
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Replace infinite values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Fill NaN values with 0
                df[col] = df[col].fillna(0)
                # Round and convert to integer
                df[col] = df[col].round(0).astype(int)

            # Save the preprocessed data back to the file
            df.to_csv(filepath, index=False)

            session['uploaded_file'] = filename
            session['data_shape'] = {'rows': len(df), 'cols': len(df.columns)}

            brand_col = 'Brand' if 'Brand' in df.columns else 'brand'
            if brand_col in df.columns:
                brands = sorted(df[brand_col].unique().tolist())
                session['brands'] = brands

            return jsonify({
                'success': True,
                'filename': filename,
                'rows': len(df),
                'cols': len(df.columns)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/download_template')
def download_template():
    """Download CSV template"""
    template_df = create_template_csv()
    output = io.BytesIO()
    template_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='brandcompass_template.csv'
    )


@app.route('/analysis')
def analysis():
    """Analysis page"""
    if 'uploaded_file' not in session:
        return redirect(url_for('index'))

    filename = session['uploaded_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Store filepath in session for historical data calculation
    session['file_path'] = filepath

    try:
        df = pd.read_csv(filepath)

        year_col = 'Year' if 'Year' in df.columns else 'year'
        month_col = 'Month' if 'Month' in df.columns else 'month'
        brand_col = 'Brand' if 'Brand' in df.columns else 'brand'
        week_col = 'week_of_month' if 'week_of_month' in df.columns else 'Week' if 'Week' in df.columns else 'week'

        if year_col in df.columns and month_col in df.columns:
            df = df[
                (df[year_col] > 2024) |
                ((df[year_col] == 2024) & (df[month_col] >= 7))
            ].copy()

        brands = sorted(df[brand_col].unique().tolist()
                        ) if brand_col in df.columns else []
        years = sorted(df[year_col].unique().tolist()
                       ) if year_col in df.columns else []
        months = sorted(df[month_col].unique().tolist()
                        ) if month_col in df.columns else []
        weeks = sorted(df[week_col].unique().tolist()
                       ) if week_col in df.columns else []

        baseline_df = load_baseline_forecast()
        baseline_data = None
        if baseline_df is not None and brands:
            baseline_data = get_baseline_data_for_brands(baseline_df, brands)

        session['baseline_data'] = baseline_data

        # Build editable table data: ID columns + GROUPED features
        optimizable_features = get_optimizable_columns()
        channel_groups = get_channel_groups()

        present_id_columns = []
        for id_col in lst_id_columns:
            if id_col in df.columns:
                present_id_columns.append(id_col)
            elif id_col.capitalize() in df.columns:
                present_id_columns.append(id_col.capitalize())
            elif id_col.upper() in df.columns:
                present_id_columns.append(id_col.upper())

        # Aggregate by channel groups
        df_aggregated = aggregate_by_channel_groups(df)

        # Use grouped column names instead of individual features
        grouped_feature_columns = list(channel_groups.keys())
        display_columns = present_id_columns + grouped_feature_columns

        table_df = df_aggregated[display_columns].copy(
        ) if display_columns else df_aggregated.copy()
        # limit rows for UI responsiveness
        table_rows = table_df.to_dict(orient='records')[:500]

        # Determine canonical id column names present
        brand_col_name = 'Brand' if 'Brand' in df.columns else 'brand' if 'brand' in df.columns else None
        year_col_name = 'Year' if 'Year' in df.columns else 'year' if 'year' in df.columns else None
        month_col_name = 'Month' if 'Month' in df.columns else 'month' if 'month' in df.columns else None
        week_col_name = 'week_of_month' if 'week_of_month' in df.columns else 'Week' if 'Week' in df.columns else 'week' if 'week' in df.columns else None

        return render_template(
            'analysis.html',
            brands=brands,
            years=years,
            months=months,
            weeks=weeks,
            data_preview=df.head(10).to_html(
                classes='table table-striped', index=False),
            id_columns=present_id_columns,
            feature_columns=grouped_feature_columns,
            table_columns=display_columns,
            table_rows=table_rows,
            brand_col=brand_col_name,
            year_col=year_col_name,
            month_col=month_col_name,
            week_col=week_col_name,
            channel_groups=channel_groups
        )
    except Exception as e:
        return redirect(url_for('index'))


@app.route('/calculate', methods=['POST'])
def calculate():
    """Calculate brand power"""
    data = request.get_json()

    if 'uploaded_file' not in session:
        return jsonify({'error': 'No data uploaded'}), 400

    filename = session['uploaded_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = pd.read_csv(filepath)

        # Canonical column names
        brand_col = 'Brand' if 'Brand' in df.columns else 'brand' if 'brand' in df.columns else None
        year_col = 'Year' if 'Year' in df.columns else 'year' if 'year' in df.columns else None
        month_col = 'Month' if 'Month' in df.columns else 'month' if 'month' in df.columns else None
        week_col = 'week_of_month' if 'week_of_month' in df.columns else 'Week' if 'Week' in df.columns else 'week' if 'week' in df.columns else None

        # Apply standard date filter
        if year_col in df.columns and month_col in df.columns:
            df = df[(df[year_col] > 2024) | (
                (df[year_col] == 2024) & (df[month_col] >= 7))].copy()

        # Filters from client
        filters = data.get('filters', {})
        sel_brands = filters.get('brands', [])
        sel_years = filters.get('years', [])
        sel_months = filters.get('months', [])
        sel_weeks = filters.get('weeks', [])

        df_filt = df.copy()
        if brand_col and sel_brands:
            df_filt = df_filt[df_filt[brand_col].isin(sel_brands)]
        if year_col and sel_years:
            df_filt = df_filt[df_filt[year_col].isin(sel_years)]
        if month_col and sel_months:
            df_filt = df_filt[df_filt[month_col].isin(sel_months)]
        if week_col and sel_weeks:
            df_filt = df_filt[df_filt[week_col].isin(sel_weeks)]

        # Baseline per-brand power - get ALL brands from full dataset, not just filtered
        brands_all = sorted(df[brand_col].unique(
        ).tolist()) if brand_col in df.columns else []
        baseline_df = load_baseline_forecast()
        baseline_data = get_baseline_data_for_brands(
            baseline_df, brands_all) if baseline_df is not None else None

        # Compute user changes from edited rows vs original sums
        optimizable = get_optimizable_columns()
        channel_groups_dict = get_channel_groups()
        edited_rows = data.get('edited_rows', [])
        edited_columns = data.get('columns', [])

        user_changes = {}
        if edited_rows and edited_columns:
            try:
                edited_df = pd.DataFrame(edited_rows, columns=edited_columns)
                # Apply same filters to edited data if those id columns exist
                if brand_col and brand_col in edited_df.columns and sel_brands:
                    edited_df = edited_df[edited_df[brand_col].isin(
                        sel_brands)]
                if year_col and year_col in edited_df.columns and sel_years:
                    edited_df = edited_df[edited_df[year_col].isin(sel_years)]
                if month_col and month_col in edited_df.columns and sel_months:
                    edited_df = edited_df[edited_df[month_col].isin(
                        sel_months)]
                if week_col and week_col in edited_df.columns and sel_weeks:
                    edited_df = edited_df[edited_df[week_col].isin(sel_weeks)]

                # Aggregate original data by channel groups for comparison
                df_filt_aggregated = aggregate_by_channel_groups(df_filt)

                # Check if we're working with grouped or individual columns
                grouped_cols = list(channel_groups_dict.keys())
                is_grouped = any(
                    col in edited_df.columns for col in grouped_cols)

                if is_grouped:
                    # Working with grouped data - compare grouped sums
                    for group_name in grouped_cols:
                        if group_name in edited_df.columns and group_name in df_filt_aggregated.columns:
                            orig_sum = float(
                                df_filt_aggregated[group_name].sum())
                            new_sum = float(edited_df[group_name].sum())
                            if orig_sum != 0:
                                # Store changes at group level - will be applied to all components
                                pct_change = (
                                    (new_sum - orig_sum) / orig_sum) * 100.0
                                user_changes[group_name] = pct_change
                                # Also apply same change to all individual features in this group
                                for feat in channel_groups_dict[group_name]:
                                    if feat in df_filt.columns:
                                        user_changes[feat] = pct_change
                            else:
                                user_changes[group_name] = 0.0
                else:
                    # Working with individual features
                    for feat in optimizable:
                        if feat in df_filt.columns and feat in edited_df.columns:
                            orig_sum = float(df_filt[feat].sum())
                            new_sum = float(edited_df[feat].sum())
                            if orig_sum != 0:
                                user_changes[feat] = (
                                    (new_sum - orig_sum) / orig_sum) * 100.0
                            else:
                                user_changes[feat] = 0.0
            except Exception as e:
                print(f"Error processing edited rows: {e}")
                user_changes = {}
        else:
            user_changes = data.get('changes', {}) or {}

        simulated_data = simulate_outcomes_from_changes(
            baseline_data, user_changes) if baseline_data else None

        # Calculate historical data (rolled up quarterly) from uploaded file
        historical_data = {}
        historical_quarters = []
        try:
            print("\n" + "="*80)
            print("=== HISTORICAL DATA CALCULATION - START ===")
            print("="*80)

            print("\n--- READING COMPLETE UPLOADED FILE (NO FILTERS) ---")
            # Read the complete original file from session (before any filtering)
            file_path = session.get('file_path')
            if not file_path or not os.path.exists(file_path):
                print("ERROR: No uploaded file found in session!")
                raise FileNotFoundError("Uploaded file not accessible")

            print(f"Reading full data from: {file_path}")
            df_full = pd.read_csv(file_path)

            print("\n--- BEFORE ROLLUP: Complete Original Data ---")
            print(f"Complete df shape: {df_full.shape}")
            print(f"Complete df columns: {df_full.columns.tolist()}")
            print(f"Brand column name: '{brand_col}'")
            print(f"Year column name: '{year_col}'")
            print(f"Month column name: '{month_col}'")
            print(f"\nFirst 5 rows of key columns:")
            if brand_col and year_col and month_col:
                print(df_full[[brand_col, year_col, month_col]].head())
            print(
                f"\nUnique years in FULL data: {sorted(df_full[year_col].unique().tolist()) if year_col in df_full.columns else 'N/A'}")
            print(
                f"Unique months in FULL data: {sorted(df_full[month_col].unique().tolist()) if month_col in df_full.columns else 'N/A'}")
            print(
                f"Unique brands in FULL data: {sorted(df_full[brand_col].unique().tolist()) if brand_col in df_full.columns else 'N/A'}")

            # Check date range
            if year_col in df_full.columns and month_col in df_full.columns:
                print(f"\nDate range in FULL data:")
                print(
                    f"  Earliest: {df_full[year_col].min()}-{df_full[month_col].min()}")
                print(
                    f"  Latest: {df_full[year_col].max()}-{df_full[month_col].max()}")

            print("\n--- CALLING roll_data_to_quarter() ON COMPLETE DATA ---")
            # Roll up COMPLETE data quarterly (not the filtered df)
            df_quarterly = roll_data_to_quarter(df_full)

            print("\n--- AFTER ROLLUP: Quarterly Data ---")
            print(f"Quarterly df shape: {df_quarterly.shape}")
            print(f"Quarterly df columns: {df_quarterly.columns.tolist()}")

            # Check for quarter column (could be 'quarter' or 'Quarter')
            quarter_col = 'quarter' if 'quarter' in df_quarterly.columns else 'Quarter' if 'Quarter' in df_quarterly.columns else None
            print(f"Quarter column name: '{quarter_col}'")

            if not df_quarterly.empty:
                print(f"\nFirst 10 rows of quarterly data:")
                print(df_quarterly.head(10))

                if quarter_col:
                    print(
                        f"\nUnique quarters: {sorted(df_quarterly[quarter_col].unique().tolist())}")
                if year_col in df_quarterly.columns:
                    print(
                        f"Unique years after rollup: {sorted(df_quarterly[year_col].unique().tolist())}")

            print("\n--- MERGE LOGIC ---")
            if year_col in df_quarterly.columns and quarter_col and brand_col in df_quarterly.columns:
                print(f"✓ All required columns present for merge:")
                print(f"  - Year column: '{year_col}' - EXISTS")
                print(f"  - Quarter column: '{quarter_col}' - EXISTS")
                print(f"  - Brand column: '{brand_col}' - EXISTS")

                # Create quarter labels
                print(f"\nCreating quarter_label column...")
                df_quarterly['quarter_label'] = df_quarterly[year_col].astype(
                    str) + ' ' + df_quarterly[quarter_col].astype(str)
                print(
                    f"Sample quarter_label values: {df_quarterly['quarter_label'].head().tolist()}")

                historical_quarters = sorted(
                    df_quarterly['quarter_label'].unique().tolist())

                print(f"\n--- HISTORICAL QUARTERS IDENTIFIED ---")
                print(f"Total unique quarters: {len(historical_quarters)}")
                print(f"Quarters list: {historical_quarters}")
                print(f"\n--- BRANDS TO PROCESS ---")
                print(f"Total brands: {len(brands_all)}")
                print(f"Brands list: {brands_all}")

                # Calculate AVERAGE POWER for each brand/quarter from the power column
                print("\n--- CALCULATING AVERAGE BRAND POWER PER QUARTER ---")

                # Find the power column (could be 'power', 'Power', 'brand_power', etc.)
                power_col = None
                for col_name in ['power', 'Power', 'brand_power', 'Brand_Power', 'POWER']:
                    if col_name in df_quarterly.columns:
                        power_col = col_name
                        break

                print(f"\nLooking for power column in quarterly data...")
                print(f"Available columns: {df_quarterly.columns.tolist()}")
                print(
                    f"Power column found: '{power_col}'" if power_col else "ERROR: No power column found!")

                if power_col:
                    print(f"\n{'='*80}")
                    print("PROCESSING EACH BRAND - CALCULATING AVERAGE POWER")
                    print(f"{'='*80}")

                    for idx, brand in enumerate(brands_all, 1):
                        print(
                            f"\n[{idx}/{len(brands_all)}] Processing Brand: '{brand}'")
                        print(
                            f"  Filtering quarterly data where {brand_col} == '{brand}'...")

                        brand_data = df_quarterly[df_quarterly[brand_col] == brand]
                        print(
                            f"  → Found {len(brand_data)} quarterly records for this brand")

                        if len(brand_data) > 0:
                            print(
                                f"  → Quarters present for {brand}: {sorted(brand_data['quarter_label'].unique().tolist())}")

                        historical_values = []
                        for quarter in historical_quarters:
                            # Only include quarters from 2021 Q1 UP TO 2024 Q2 (intervention point)
                            if quarter < '2021 Q1':
                                print(
                                    f"    ⊗ {quarter}: SKIPPING (before 2021 Q1)")
                                continue
                            if quarter > '2024 Q2':
                                print(
                                    f"    ⊗ {quarter}: SKIPPING (after intervention, will use forecast)")
                                continue

                            quarter_data = brand_data[brand_data['quarter_label'] == quarter]
                            if not quarter_data.empty:
                                # Calculate AVERAGE POWER from the power column
                                power_value = quarter_data[power_col].mean()
                                historical_values.append(power_value)
                                print(
                                    f"    ✓ {quarter}: avg power = {power_value:,.2f} (from {len(quarter_data)} rows)")
                            else:
                                print(
                                    f"    ✗ {quarter}: NO DATA (skipping this quarter)")
                        historical_data[brand] = historical_values
                        print(
                            f"  → Total values stored: {len(historical_values)}")

                    print(f"\n{'='*80}")
                    print("=== HISTORICAL POWER TABLE SUMMARY ===")
                    print(f"{'='*80}")

                    # Filter historical_quarters to only include 2021 Q1 to 2024 Q2
                    historical_quarters = [
                        q for q in historical_quarters if '2021 Q1' <= q <= '2024 Q2']
                    print(
                        f"Final historical quarters (2021 Q1 to 2024 Q2): {historical_quarters}")

                    for brand, values in historical_data.items():
                        non_zero = sum(1 for v in values if v > 0)
                        avg_power = sum(values) / len(values) if values else 0
                        print(
                            f"{brand}: {len(values)} quarters ({non_zero} non-zero) | Avg Power: {avg_power:,.2f}")
                else:
                    print("\n" + "!"*80)
                    print("ERROR: No power column found in data!")
                    print("!"*80)
                    print(
                        "Expected column names: 'power', 'Power', 'brand_power', 'Brand_Power', 'POWER'")
                    print(
                        "This means the uploaded data does not contain a power column")
            else:
                print("\n" + "!"*80)
                print("ERROR: Missing required columns after rollup!")
                print("!"*80)
                print(f"Required columns check:")
                print(
                    f"  - {year_col} present: {year_col in df_quarterly.columns if df_quarterly is not None else 'N/A'}")
                print(f"  - quarter column present: {quarter_col is not None}")
                print(
                    f"  - {brand_col} present: {brand_col in df_quarterly.columns if df_quarterly is not None else 'N/A'}")

                if df_quarterly is not None:
                    print(
                        f"\nActual columns in quarterly df: {df_quarterly.columns.tolist()}")
        except FileNotFoundError as e:
            print("\n" + "!"*80)
            print(f"FILE NOT FOUND: Could not read uploaded file")
            print("!"*80)
            historical_data = {}
            historical_quarters = []
        except Exception as e:
            print("\n" + "!"*80)
            print(f"EXCEPTION in historical data calculation: {e}")
            print("!"*80)
            import traceback
            traceback.print_exc()
            historical_data = {}
            historical_quarters = []

        forecast_quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

        results = {
            'baseline': baseline_data if baseline_data else {},
            'simulated': simulated_data if simulated_data else {},
            'quarters': forecast_quarters,
            'historical': historical_data if historical_data else {},
            'historical_quarters': historical_quarters if historical_quarters else []
        }

        print("\n" + "="*80)
        print("=== FINAL RESPONSE SUMMARY ===")
        print("="*80)
        print(f"\n1. BASELINE DATA:")
        print(f"   - Brands count: {len(results['baseline'])}")
        print(f"   - Brands: {list(results['baseline'].keys())[:5]}..." if len(
            results['baseline']) > 5 else f"   - Brands: {list(results['baseline'].keys())}")

        print(f"\n2. SIMULATED DATA:")
        print(f"   - Brands count: {len(results['simulated'])}")
        print(f"   - Brands: {list(results['simulated'].keys())[:5]}..." if len(
            results['simulated']) > 5 else f"   - Brands: {list(results['simulated'].keys())}")

        print(f"\n3. FORECAST PERIODS:")
        print(f"   - Quarters: {results['quarters']}")

        print(f"\n4. HISTORICAL DATA:")
        print(f"   - Brands count: {len(results['historical'])}")
        print(f"   - Brands: {list(results['historical'].keys())[:5]}..." if len(
            results['historical']) > 5 else f"   - Brands: {list(results['historical'].keys())}")
        print(f"   - Historical quarters: {results['historical_quarters']}")

        print(f"\n5. DATA ALIGNMENT CHECK:")
        print(
            f"   - Historical quarters == Forecast quarters? {results['historical_quarters'] == results['quarters']}")
        if results['historical_quarters'] == results['quarters']:
            print(
                f"   ⚠ WARNING: No pre-intervention historical data! Both cover same period.")
        else:
            print(
                f"   ✓ Good: Historical data is from different time period than forecast")

        print("\n" + "="*80)
        print("=== END OF CALCULATION ===")
        print("="*80 + "\n")

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    """Save experiment"""
    data = request.get_json()

    if 'experiments' not in session:
        session['experiments'] = []

    experiment = {
        'name': data.get('name'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_data': data.get('baseline_data'),
        'simulated_data': data.get('simulated_data'),
        'changes': data.get('changes'),
        # Store actual user values
        'user_entered_data': data.get('user_entered_data'),
        'table_columns': data.get('table_columns'),
        'table_rows': data.get('table_rows')
    }

    session['experiments'].append(experiment)
    session.modified = True

    return jsonify({'success': True, 'experiment_count': len(session['experiments'])})


@app.route('/experiments')
def list_experiments():
    """List saved experiments"""
    experiments = session.get('experiments', [])
    return jsonify(experiments)


@app.route('/delete_experiment/<int:index>', methods=['POST'])
def delete_experiment(index: int):
    """Delete an experiment by index"""
    experiments = session.get('experiments', [])
    if 0 <= index < len(experiments):
        experiments.pop(index)
        session['experiments'] = experiments
        session.modified = True
        return jsonify({'success': True, 'experiment_count': len(experiments)})
    return jsonify({'error': 'Invalid index'}), 400


@app.route('/clear_experiments', methods=['POST'])
def clear_experiments():
    """Clear all experiments"""
    session['experiments'] = []
    session.modified = True
    return jsonify({'success': True})


@app.route('/debug_experiments')
def debug_experiments():
    """Debug endpoint to see experiment data structure"""
    experiments = session.get('experiments', [])

    if not experiments:
        return jsonify({'message': 'No experiments found'})

    # Return structure of first experiment
    exp = experiments[0]
    debug_info = {
        'experiment_name': exp.get('name'),
        'has_table_rows': bool(exp.get('table_rows')),
        'table_rows_count': len(exp.get('table_rows', [])),
        'table_columns': exp.get('table_columns', []),
        'sample_table_row': exp.get('table_rows', [{}])[0] if exp.get('table_rows') else {},
        'baseline_brands': list(exp.get('baseline_data', {}).keys())[:5],
        'channel_groups': list(get_channel_groups().keys())
    }

    return jsonify(debug_info)


@app.route('/export_experiments')
def export_experiments():
    """Export all experiments to Excel with quarterly data"""
    experiments = session.get('experiments', [])

    if not experiments:
        return jsonify({'error': 'No experiments to export'}), 400

    output = io.BytesIO()
    channel_groups = get_channel_groups()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, exp in enumerate(experiments):
            # Create sheet name (limit to 31 characters)
            sheet_name = f"Exp_{i+1}_{exp['name'][:20]}"  # Limit name length

            # Get experiment data
            baseline_data = exp.get('baseline_data', {})
            simulated_data = exp.get('simulated_data', {})
            table_rows = exp.get('table_rows', [])
            table_columns = exp.get('table_columns', [])

            # Create quarterly data structure
            quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

            # Prepare data for export
            export_data = []

            # Process each brand and quarter combination
            for brand in baseline_data.keys():
                baseline_values = baseline_data.get(brand, [])
                simulated_values = simulated_data.get(brand, [])

                for q_idx, quarter in enumerate(quarters):
                    row = {
                        'Brand': brand,
                        'Quarter': quarter,
                        'Baseline_Power': round(baseline_values[q_idx], 2) if q_idx < len(baseline_values) else 0,
                        'Simulated_Power': round(simulated_values[q_idx], 2) if q_idx < len(simulated_values) else 0
                    }

                    # Add grouped channel values from user-entered data
                    # Since we're working with grouped data, look for group columns directly
                    for group_name in channel_groups.keys():
                        group_total = 0
                        # Find matching brand rows and sum the group values
                        for table_row in table_rows:
                            # Check if this row is for the current brand
                            row_brand = table_row.get(
                                'brand', '') or table_row.get('Brand', '')
                            if row_brand.upper() == brand.upper():
                                # Get the grouped channel value directly
                                if group_name in table_row:
                                    group_total += float(table_row.get(group_name, 0))

                        row[f'{group_name}_Spend'] = round(group_total, 2)

                    # Debug: Add some debugging info to see what data we have
                    if i == 0 and q_idx == 0:  # Only for first experiment and first quarter
                        print(f"DEBUG Export - Brand: {brand}")
                        print(f"DEBUG Export - Table columns: {table_columns}")
                        print(
                            f"DEBUG Export - Sample table row: {table_rows[0] if table_rows else 'No rows'}")
                        print(
                            f"DEBUG Export - Channel groups: {list(channel_groups.keys())}")
                        print(f"DEBUG Export - Row data: {row}")

                    export_data.append(row)

            # Create DataFrame and export
            if export_data:
                df = pd.DataFrame(export_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Fallback if no data
                empty_df = pd.DataFrame(
                    {'Message': ['No data available for this experiment']})
                empty_df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
