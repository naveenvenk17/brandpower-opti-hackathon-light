#!/usr/bin/env python3
"""
BrandCompass.ai - Flask Web Application
A comprehensive brand power simulation platform
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import pandas as pd

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
import requests
import logging

from typing import Optional

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'csv'}

# FastAPI backend configuration (POC local)
FASTAPI_BASE_URL = os.environ.get('FASTAPI_BASE_URL', 'http://127.0.0.1:8010')

# Load available countries and brands from training data
def load_available_data():
    """Load available countries and brands from training data"""
    try:
        import pandas as pd
        df = pd.read_csv('data/brand_power_with_marketing_features.csv')
        countries = sorted(df['country'].unique().tolist())
        brands_by_country = {}
        for country in countries:
            brands_by_country[country] = sorted(df[df['country'] == country]['brand'].unique().tolist())
        return countries, brands_by_country
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return ['brazil', 'colombia', 'us'], {'brazil': ['amstel'], 'colombia': ['aguila'], 'us': ['bud light']}

AVAILABLE_COUNTRIES, BRANDS_BY_COUNTRY = load_available_data()

def get_country_and_brand(brand_name: str, selected_country: str = None):
    """
    Get country and brand for API calls
    Returns: (country, brand) tuple with actual names from data
    """
    # If country is specified in session, use it
    if selected_country and selected_country in AVAILABLE_COUNTRIES:
        country = selected_country
        # Find brand in this country
        if brand_name in BRANDS_BY_COUNTRY.get(country, []):
            return country, brand_name
        # Brand not in this country, use first available brand
        brands = BRANDS_BY_COUNTRY.get(country, [])
        return country, brands[0] if brands else 'aguila'

    # Search for brand across all countries
    for country, brands in BRANDS_BY_COUNTRY.items():
        if brand_name in brands:
            return country, brand_name

    # Default fallback: colombia/aguila
    return 'colombia', 'aguila'


def fastapi_get(path: str, params: Optional[dict] = None):
    url = f"{FASTAPI_BASE_URL}{path}"
    app.logger.info(f"FastAPI GET {url} params={params}")
    resp = requests.get(url, params=params, timeout=30)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        app.logger.error(f"FastAPI GET error {resp.status_code}: {resp.text}")
        raise
    return resp.json()


def fastapi_post(path: str, payload: dict):
    url = f"{FASTAPI_BASE_URL}{path}"
    app.logger.info(f"FastAPI POST {url} body_keys={list(payload.keys())}")
    resp = requests.post(url, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        app.logger.error(f"FastAPI POST error {resp.status_code}: {resp.text}")
        raise
    return resp.json()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_template_csv():
    """Create a template CSV for download with simplified marketing channels"""
    # Use actual data structure that matches backend expectations
    template_data = {
        'country': ['colombia', 'colombia', 'colombia', 'colombia', 'colombia', 'colombia'],
        'brand': ['aguila', 'aguila', 'aguila light', 'aguila light', 'costena bacana', 'costena bacana'],
        'year': [2024, 2024, 2024, 2024, 2024, 2024],
        'quarter': ['Q3', 'Q4', 'Q3', 'Q4', 'Q3', 'Q4'],
        'month': [7, 10, 7, 10, 7, 10],
        'week_of_month': [1, 1, 1, 1, 1, 1],
        # Simplified marketing channels (matches backend)
        'digital_spend': [500000, 520000, 400000, 420000, 300000, 310000],
        'tv_spend': [1000000, 1050000, 800000, 840000, 600000, 630000],
        'traditional_spend': [300000, 315000, 250000, 260000, 200000, 210000],
        'sponsorship_spend': [400000, 420000, 350000, 370000, 250000, 260000],
        'other_spend': [200000, 210000, 150000, 160000, 100000, 110000]
    }
    return pd.DataFrame(template_data)




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

        return render_template(
            'analysis.html',
            brands=brands,
            years=years,
            months=months,
            data_preview=df.head(10).to_html(
                classes='table table-striped', index=False),
            id_columns=present_id_columns,
            feature_columns=feature_columns,
            table_columns=display_columns,
            table_rows=table_rows,
            brand_col=brand_col_name,
            year_col=year_col_name,
            month_col=month_col_name
        )
    except Exception as e:
        return redirect(url_for('index'))


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Calculate brand power by sending the user's edited data to the backend for simulation.
    """
    data = request.get_json()
    if not data or 'edited_rows' not in data or 'columns' not in data:
        return jsonify({'error': 'Invalid request payload. Must include edited_rows and columns.'}), 400

    if 'uploaded_file' not in session:
        return jsonify({'error': 'No data uploaded. Please upload a file first.'}), 400

    filename = session['uploaded_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Determine which brands to process based on frontend filters
        filters = data.get('filters', {})
        sel_brands = filters.get('brands', [])
        
        if not sel_brands:
            # If no brands are selected in the filter, use all brands present in the edited data
            edited_df = pd.DataFrame(data['edited_rows'], columns=data['columns'])
            brand_col = 'Brand' if 'Brand' in edited_df.columns else 'brand'
            if brand_col in edited_df.columns:
                target_brands = sorted(edited_df[brand_col].unique().tolist())
            else:
                # If no brand column, this will likely fail, but we send it anyway
                target_brands = []
        else:
            target_brands = sel_brands

        # Construct the payload for the new backend endpoint
        payload = {
            "edited_rows": data['edited_rows'],
            "columns": data['columns'],
            "target_brands": target_brands,
            "max_horizon": 4
        }

        app.logger.info(f"Sending simulation request for brands: {target_brands}")

        # Call the backend to get the simulation results
        sim_resp = fastapi_post("/api/v1/simulate/scenario", payload)

        # The backend now returns the data in the exact format the frontend UI expects.
        # The response looks like: {'baseline': {'BrandA': [...]}, 'simulated': {'BrandA': [...]}, 'quarters': [...]}
        return jsonify(sim_resp)

    except Exception as e:
        app.logger.error(f"An error occurred in /calculate: {e}")
        # Pass the backend's error message to the frontend if possible
        error_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                pass
        return jsonify({'error': error_detail}), 500


@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    """Save experiment"""
    data = request.get_json()
    name = data.get('name') or f"Experiment {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Get country and brand from session or use defaults
    selected_country = session.get('selected_country', 'colombia')
    country, brand = get_country_and_brand('DEFAULT', selected_country)

    baseline_data = data.get('baseline_data') or {}
    simulated_data = data.get('simulated_data') or {}

    # Build scenarios from provided baseline/simulated maps
    baseline_predictions = next(iter(baseline_data.values()), [15.0, 15.1, 15.2, 15.3])
    marketing_predictions = next(iter(simulated_data.values()), baseline_predictions)

    scenarios = [
        {
            'scenario_name': 'Baseline',
            'allocation': {},
            'results': {
                'baseline_prediction': baseline_predictions,
                'marketing_prediction': baseline_predictions,
                'incremental_lift': [0, 0, 0, 0],
                'total_spend': 0,
                'roi': 0
            }
        },
        {
            'scenario_name': 'Simulated Changes',
            'allocation': data.get('changes', {}) or {},
            'results': {
                'baseline_prediction': baseline_predictions,
                'marketing_prediction': marketing_predictions,
                'incremental_lift': [m - b for m, b in zip(marketing_predictions, baseline_predictions)],
                'total_spend': sum(abs(v) for v in (data.get('changes', {}) or {}).values()),
                'roi': 0
            }
        }
    ]

    try:
        resp = fastapi_post('/api/v1/experiments', {
            'name': name,
            'description': data.get('description'),
            'country': country,
            'brand': brand,
            'scenarios': scenarios
        })
        # Return count via listing
        listing = fastapi_get('/api/v1/experiments')
        count = len(listing.get('experiments', []))
        return jsonify({'success': True, 'experiment_id': resp.get('id'), 'experiment_count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/experiments')
def list_experiments():
    """List saved experiments"""
    try:
        listing = fastapi_get('/api/v1/experiments')
        return jsonify(listing.get('experiments', []))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/delete_experiment/<int:index>', methods=['POST'])
def delete_experiment(index: int):
    """Delete an experiment by index"""
    try:
        listing = fastapi_get('/api/v1/experiments')
        experiments = listing.get('experiments', [])
        if 0 <= index < len(experiments):
            exp_id = experiments[index].get('id')
            if not exp_id:
                return jsonify({'error': 'Experiment id missing'}), 400
            # delete
            url = f"/api/v1/experiments/{exp_id}"
            _ = requests.delete(f"{FASTAPI_BASE_URL}{url}", timeout=30)
            # new count
            listing2 = fastapi_get('/api/v1/experiments')
            return jsonify({'success': True, 'experiment_count': len(listing2.get('experiments', []))})
        return jsonify({'error': 'Invalid index'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/clear_experiments', methods=['POST'])
def clear_experiments():
    """Clear all experiments"""
    try:
        listing = fastapi_get('/api/v1/experiments')
        for exp in listing.get('experiments', []):
            exp_id = exp.get('id')
            if exp_id:
                _ = requests.delete(f"{FASTAPI_BASE_URL}/api/v1/experiments/{exp_id}", timeout=30)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/export_experiments')
def export_experiments():
    """Export all experiments to Excel"""
    try:
        listing = fastapi_get('/api/v1/experiments')
        experiments = listing.get('experiments', [])
        if not experiments:
            return jsonify({'error': 'No experiments to export'}), 400

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for i, exp in enumerate(experiments):
                df = pd.DataFrame({
                    'Experiment': [exp.get('name')],
                    'Created At': [exp.get('created_at')],
                    'Updated At': [exp.get('updated_at')],
                    'Country': [exp.get('country')],
                    'Brand': [exp.get('brand')],
                    'Scenario Count': [len(exp.get('scenarios', []))]
                })
                df.to_excel(writer, sheet_name=f"Experiment_{i+1}", index=False)

        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
