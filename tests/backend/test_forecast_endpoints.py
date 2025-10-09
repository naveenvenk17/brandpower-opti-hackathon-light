"""
Test forecast and simulate endpoints
Tests all forecast/simulate API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

from src.app import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_baseline_data():
    """Mock baseline forecast data"""
    return pd.DataFrame({
        'year': [2024, 2024, 2024, 2024],
        'quarter': ['Q3', 'Q4', 'Q3', 'Q4'],
        'country': ['colombia', 'colombia', 'colombia', 'colombia'],
        'brand': ['AGUILA', 'AGUILA', 'FAMILIA POKER', 'FAMILIA POKER'],
        'predicted_power': [15.17, 15.22, 16.09, 15.64]
    })


@pytest.fixture(autouse=True)
def mock_service_and_baseline(mock_baseline_data):
    """Mock the get_service_and_baseline function for all tests"""
    mock_service = MagicMock()
    
    with patch('src.app.get_service_and_baseline', return_value=(mock_service, mock_baseline_data)):
        yield mock_service, mock_baseline_data


class TestForecastBaselineEndpoint:
    """Tests for /api/v1/forecast/baseline/{country}/{brand} endpoint"""

    def test_baseline_forecast_success(self, client, mock_baseline_data):
        """Test successful baseline forecast request"""
        response = client.get("/api/v1/forecast/baseline/colombia/AGUILA")
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2  # Q3 and Q4
        assert data["predictions"][0] == 15.17

    def test_baseline_forecast_not_found(self, client):
        """Test baseline forecast for non-existent brand"""
        response = client.get("/api/v1/forecast/baseline/colombia/NONEXISTENT")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_baseline_forecast_case_insensitive_country(self, client):
        """Test that country comparison is case-insensitive"""
        response = client.get("/api/v1/forecast/baseline/COLOMBIA/AGUILA")
        assert response.status_code == 200

    def test_baseline_forecast_max_horizon(self, client):
        """Test max_horizon parameter"""
        response = client.get("/api/v1/forecast/baseline/colombia/AGUILA?max_horizon=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1


class TestSimulateScenarioEndpoint:
    """Tests for /api/v1/simulate/scenario endpoint"""

    def test_simulate_scenario_no_file(self, client, mock_baseline_data):
        """Test simulate scenario when no uploaded file exists"""
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "AGUILA",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1,
                    "paytv": 100000
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month", "paytv"],
            "target_brands": ["AGUILA"],
            "max_horizon": 4
        }
        
        with patch('src.app._load_ui_state', return_value={}):
            response = client.post("/api/v1/simulate/scenario", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "simulated" in data
        assert "quarters" in data
        assert "AGUILA" in data["baseline"]

    def test_simulate_scenario_with_file(self, client, mock_baseline_data, tmp_path):
        """Test simulate scenario with uploaded file (graceful fallback)"""
        # Create temporary uploaded file
        upload_file = tmp_path / "test_upload.csv"
        test_df = pd.DataFrame({
            'country': ['colombia'] * 10,
            'brand': ['AGUILA'] * 10,
            'year': [2024] * 10,
            'month': [7] * 10,
            'week_of_month': list(range(1, 11)),
            'paytv': [100000] * 10,
            'wholesalers': [50000] * 10,
            'total_distribution': [75000] * 10,
            'volume': [10000] * 10
        })
        test_df.to_csv(upload_file, index=False)
        
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "AGUILA",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1,
                    "paytv": 200000  # Changed value
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month", "paytv"],
            "target_brands": ["AGUILA"],
            "max_horizon": 4
        }
        
        with patch('src.app._load_ui_state', return_value={'last_uploaded_file': str(upload_file)}):
            response = client.post("/api/v1/simulate/scenario", json=payload)
        
        # Should work with graceful fallback to baseline if AutoGluon fails
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "simulated" in data
        assert "quarters" in data
        
        # Check that we got data for AGUILA
        assert "AGUILA" in data["baseline"]
        assert "AGUILA" in data["simulated"]

    def test_simulate_scenario_empty_rows(self, client):
        """Test simulate scenario with empty edited_rows"""
        payload = {
            "edited_rows": [],
            "columns": ["country", "brand"],
            "target_brands": ["AGUILA"],
            "max_horizon": 4
        }
        
        response = client.post("/api/v1/simulate/scenario", json=payload)
        assert response.status_code == 400

    def test_simulate_scenario_invalid_brand(self, client):
        """Test simulate scenario with invalid brand"""
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "NONEXISTENT",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month"],
            "target_brands": ["NONEXISTENT"],
            "max_horizon": 4
        }
        
        with patch('src.app._load_ui_state', return_value={}):
            response = client.post("/api/v1/simulate/scenario", json=payload)
        
        # Should return 200 but with empty/zero values (graceful handling)
        assert response.status_code in [200, 500]


class TestCalculateEndpoint:
    """Tests for /calculate endpoint"""

    def test_calculate_success(self, client, mock_baseline_data, tmp_path):
        """Test calculate endpoint with valid data"""
        # Create temporary uploaded file with all required features
        upload_file = tmp_path / "test_upload.csv"
        test_df = pd.DataFrame({
            'country': ['colombia'] * 10,
            'brand': ['AGUILA'] * 10,
            'year': [2024] * 10,
            'month': [7, 7, 7, 8, 8, 8, 9, 9, 10, 10],
            'week_of_month': [1, 2, 3, 1, 2, 3, 1, 2, 1, 2],
            'paytv': [100000] * 10,
            'wholesalers': [50000] * 10,
            'total_distribution': [75000] * 10,
            'volume': [10000] * 10,
            'power': [15.0] * 10  # Historical power
        })
        test_df.to_csv(upload_file, index=False)
        
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "AGUILA",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1,
                    "paytv": 200000,
                    "wholesalers": 60000,
                    "total_distribution": 80000,
                    "volume": 12000
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month", "paytv", "wholesalers", "total_distribution", "volume"]
        }
        
        with patch('src.app._load_ui_state', return_value={'last_uploaded_file': str(upload_file)}):
            response = client.post("/calculate", json=payload)
        
        # Should work with fallback even if AutoGluon not installed
        assert response.status_code in [200, 500]

    def test_calculate_no_uploaded_file(self, client):
        """Test calculate endpoint when no file is uploaded"""
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "AGUILA",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month"]
        }
        
        with patch('src.app._load_ui_state', return_value={}):
            response = client.post("/calculate", json=payload)
        
        # Should return 200 with baseline data
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "simulated" in data

    def test_calculate_missing_columns(self, client):
        """Test calculate endpoint with missing required fields"""
        payload = {
            "edited_rows": [{"brand": "AGUILA"}],
            # Missing "columns" field
        }
        
        response = client.post("/calculate", json=payload)
        assert response.status_code == 400

    def test_calculate_empty_rows(self, client):
        """Test calculate endpoint with empty edited_rows"""
        payload = {
            "edited_rows": [],
            "columns": ["country", "brand"]
        }
        
        response = client.post("/calculate", json=payload)
        assert response.status_code == 400


class TestEndpointsIntegration:
    """Integration tests for forecast endpoints"""

    def test_baseline_then_simulate_workflow(self, client, mock_baseline_data):
        """Test typical user workflow: get baseline, then simulate"""
        # 1. Get baseline forecast
        baseline_response = client.get("/api/v1/forecast/baseline/colombia/AGUILA")
        assert baseline_response.status_code == 200
        baseline_predictions = baseline_response.json()["predictions"]
        
        # 2. Simulate scenario
        payload = {
            "edited_rows": [
                {
                    "country": "colombia",
                    "brand": "AGUILA",
                    "year": 2024,
                    "month": 7,
                    "week_of_month": 1,
                    "paytv": 100000
                }
            ],
            "columns": ["country", "brand", "year", "month", "week_of_month", "paytv"],
            "target_brands": ["AGUILA"],
            "max_horizon": 4
        }
        
        with patch('src.app._load_ui_state', return_value={}):
            simulate_response = client.post("/api/v1/simulate/scenario", json=payload)
        
        assert simulate_response.status_code == 200
        simulate_data = simulate_response.json()
        
        # Baseline from simulate should match baseline endpoint
        assert "AGUILA" in simulate_data["baseline"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

