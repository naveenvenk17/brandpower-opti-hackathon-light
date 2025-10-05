#!/usr/bin/env python3
"""
BrandCompass.ai - Flask Web Application
A comprehensive brand power simulation platform
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
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
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

        # Build editable table data: only ID columns and optimizable features present
        optimizable_features = get_optimizable_columns()

        present_id_columns = []
        for id_col in lst_id_columns:
            if id_col in df.columns:
                present_id_columns.append(id_col)
            elif id_col.capitalize() in df.columns:
                present_id_columns.append(id_col.capitalize())
            elif id_col.upper() in df.columns:
                present_id_columns.append(id_col.upper())

        feature_columns = [c for c in optimizable_features if c in df.columns]
        display_columns = present_id_columns + feature_columns

        table_df = df[display_columns].copy() if display_columns else df.copy()
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
            feature_columns=feature_columns,
            table_columns=display_columns,
            table_rows=table_rows,
            brand_col=brand_col_name,
            year_col=year_col_name,
            month_col=month_col_name,
            week_col=week_col_name
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

                for feat in optimizable:
                    if feat in df_filt.columns and feat in edited_df.columns:
                        orig_sum = float(df_filt[feat].sum())
                        new_sum = float(edited_df[feat].sum())
                        if orig_sum != 0:
                            user_changes[feat] = (
                                (new_sum - orig_sum) / orig_sum) * 100.0
                        else:
                            user_changes[feat] = 0.0
            except Exception:
                user_changes = {}
        else:
            user_changes = data.get('changes', {}) or {}

        simulated_data = simulate_outcomes_from_changes(
            baseline_data, user_changes) if baseline_data else None

        quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']

        results = {
            'baseline': baseline_data,
            'simulated': simulated_data,
            'quarters': quarters
        }

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
        'changes': data.get('changes')
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


@app.route('/export_experiments')
def export_experiments():
    """Export all experiments to Excel"""
    experiments = session.get('experiments', [])

    if not experiments:
        return jsonify({'error': 'No experiments to export'}), 400

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for i, exp in enumerate(experiments):
            data = {
                'Experiment': [exp['name']],
                'Timestamp': [exp['timestamp']]
            }
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f"Experiment_{i+1}", index=False)

    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
