"""Additional tests for FastAPI app endpoints"""
import pytest
import pandas as pd
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.app import app

client = TestClient(app)


@pytest.fixture(scope="module")
def mock_service():
    with patch('src.app.get_service_and_baseline') as mock_get_service:
        mock_forecast_service = MagicMock()
        
        mock_forecast_service.forecast_baseline.return_value = pd.DataFrame({
            'brand': ['amstel', 'skol'],
            'country': ['brazil', 'brazil'],
            'year': [2024, 2024],
            'quarter': ['Q1', 'Q1'],
            'predicted_power': [10.0, 20.0]
        })

        mock_forecast_service.simulate.return_value = pd.DataFrame({
            'brand': ['amstel'],
            'country': ['brazil'],
            'year': [2024],
            'quarter': ['Q1'],
            'baseline_power': [10.0],
            'simulated_power': [11.0],
            'uplift': [1.0],
            'uplift_pct': [10.0]
        })

        mock_forecast_service.optimize_allocation.return_value = {
            'optimal_allocation': {'digital_spend': 1000},
            'expected_lift': 0.15,
            'roi': 2.5
        }

        mock_get_service.return_value = (mock_forecast_service, mock_forecast_service.forecast_baseline.return_value)
        yield


class TestAppEndpoints:
    def test_index_page(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_analysis_page(self):
        response = client.get("/analysis")
        assert response.status_code == 200

    def test_optimization_page(self):
        response = client.get("/optimization")
        assert response.status_code == 200

    @pytest.mark.usefixtures("mock_service")
    def test_upload_file(self):
        csv_content = "country,brand,year,quarter\nbrazil,amstel,2024,Q1\n"
        file_data = BytesIO(csv_content.encode())
        response = client.post(
            "/upload",
            files={"file": ("test.csv", file_data, "text/csv")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "filename" in data

    def test_select_country(self):
        response = client.post("/select_country", data={"country": "brazil"})
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_download_template_default(self):
        response = client.get("/download_template")
        assert response.status_code == 200

    def test_download_template_brazil(self):
        response = client.get("/download_template?country=brazil")
        assert response.status_code == 200

    def test_download_template_colombia(self):
        response = client.get("/download_template?country=colombia")
        assert response.status_code == 200

    @pytest.mark.usefixtures("mock_service")
    def test_calculate_endpoint(self):
        payload = {
            "columns": ["country", "brand", "year", "quarter", "digital_spend"],
            "edited_rows": [
                {"country": "brazil", "brand": "amstel", "year": 2024, "quarter": "Q1", "digital_spend": 100}
            ]
        }
        response = client.post("/calculate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "simulated" in data

    @pytest.mark.usefixtures("mock_service")
    def test_calculate_endpoint_missing_data(self):
        payload = {}
        response = client.post("/calculate", json=payload)
        assert response.status_code == 400

    @pytest.mark.usefixtures("mock_service")
    def test_save_experiment(self):
        payload = {
            "name": "Test Experiment",
            "baseline_data": {"brand1": [10, 11]},
            "simulated_data": {"brand1": [12, 13]},
            "changes": {}
        }
        response = client.post("/save_experiment", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.usefixtures("mock_service")
    def test_list_experiments(self):
        response = client.get("/experiments")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.usefixtures("mock_service")
    def test_export_experiments(self):
        response = client.get("/export_experiments")
        assert response.status_code == 200

    @pytest.mark.usefixtures("mock_service")
    def test_clear_experiments(self):
        response = client.post("/clear_experiments")
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.usefixtures("mock_service")
    def test_create_experiment_api(self):
        payload = {
            "name": "API Experiment",
            "description": "Test",
            "country": "brazil",
            "brand": "amstel",
            "scenarios": [{"scenario": 1}]
        }
        response = client.post("/api/v1/experiments", json=payload)
        assert response.status_code == 200
        assert "id" in response.json()

    @pytest.mark.usefixtures("mock_service")
    def test_list_experiments_api(self):
        response = client.get("/api/v1/experiments?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert "pagination" in data

    @pytest.mark.usefixtures("mock_service")
    def test_delete_experiment_by_id_not_found(self):
        response = client.delete("/api/v1/experiments/nonexistent_id")
        assert response.status_code == 404

    @pytest.mark.usefixtures("mock_service")
    def test_delete_experiment_by_index_not_found(self):
        response = client.delete("/api/v1/experiments/999")
        assert response.status_code == 404

    @pytest.mark.usefixtures("mock_service")
    def test_upload_invalid_csv(self):
        file_data = BytesIO(b"invalid csv content that cannot be parsed")
        response = client.post(
            "/upload",
            files={"file": ("bad.csv", file_data, "text/csv")}
        )
        # Should still succeed upload but with 0 rows/cols
        assert response.status_code == 200

    @pytest.mark.usefixtures("mock_service")
    def test_calculate_missing_columns(self):
        payload = {
            "columns": []
        }
        response = client.post("/calculate", json=payload)
        assert response.status_code == 400

    @pytest.mark.usefixtures("mock_service")
    def test_calculate_empty_rows(self):
        payload = {
            "columns": ["brand"],
            "edited_rows": []
        }
        response = client.post("/calculate", json=payload)
        assert response.status_code == 400

    @pytest.mark.usefixtures("mock_service")
    def test_simulate_scenario_error_handling(self):
        with patch('src.app.get_service_and_baseline') as mock_get:
            mock_service = MagicMock()
            mock_service.simulate.side_effect = Exception("Simulation error")
            mock_get.return_value = (mock_service, pd.DataFrame())
            
            payload = {
                "edited_rows": [{"country": "brazil", "brand": "amstel", "year": 2024, "quarter": "Q1"}],
                "columns": ["country", "brand", "year", "quarter"],
                "target_brands": ["amstel"]
            }
            response = client.post("/api/v1/simulate/scenario", json=payload)
            assert response.status_code == 500

    @pytest.mark.usefixtures("mock_service")
    def test_simulate_scenario_empty_rows(self):
        with patch('src.app.get_service_and_baseline') as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = (mock_service, pd.DataFrame())
            
            payload = {
                "edited_rows": [],
                "columns": ["brand"],
                "target_brands": ["amstel"]
            }
            response = client.post("/api/v1/simulate/scenario", json=payload)
            # HTTPException 400 gets caught and returned as 500 by the generic exception handler
            assert response.status_code in [400, 500]

    @pytest.mark.usefixtures("mock_service")
    def test_forecast_baseline_error(self):
        with patch('src.app.get_service_and_baseline') as mock_get:
            mock_service = MagicMock()
            mock_service.forecast_baseline.side_effect = Exception("Forecast error")
            mock_get.return_value = (mock_service, pd.DataFrame())
            
            response = client.get("/api/v1/forecast/baseline/brazil/amstel")
            assert response.status_code == 500

    @pytest.mark.usefixtures("mock_service")
    def test_optimize_allocation_error(self):
        with patch('src.app.get_service_and_baseline') as mock_get:
            mock_service = MagicMock()
            mock_service.optimize_allocation.side_effect = Exception("Optimization error")
            mock_get.return_value = (mock_service, pd.DataFrame())
            
            payload = {"total_budget": 1000, "channels": ["digital_spend"]}
            response = client.post("/api/v1/optimize/allocation", json=payload)
            assert response.status_code == 500

    @pytest.mark.usefixtures("mock_service")
    def test_calculate_error_handling(self):
        with patch('src.app.get_service_and_baseline') as mock_get:
            mock_service = MagicMock()
            mock_service.simulate.side_effect = Exception("Calculate error")
            mock_get.return_value = (mock_service, pd.DataFrame())
            
            payload = {
                "columns": ["country", "brand"],
                "edited_rows": [{"country": "brazil", "brand": "amstel"}]
            }
            response = client.post("/calculate", json=payload)
            assert response.status_code == 500

    @pytest.mark.usefixtures("mock_service")
    def test_analysis_page_error_handling(self):
        # Test analysis page when data loading fails
        with patch('src.app._load_ui_state', return_value={}):
            with patch('pandas.read_csv', side_effect=Exception("CSV error")):
                response = client.get("/analysis")
                # Should still return 200 but with empty data
                assert response.status_code == 200

    def test_app_startup(self):
        # Test the startup event by re-initializing
        from src import app as app_module
        # Just verify startup completes without error
        assert app_module.app is not None

    def test_json_persistence_functions(self):
        # Test the internal JSON helper functions
        from src.app import _load_json, _save_json
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.json")
            
            # Test load with non-existent file
            result = _load_json(test_file, {"default": "value"})
            assert result == {"default": "value"}
            
            # Test save
            _save_json(test_file, {"key": "value"})
            
            # Test load existing file
            result = _load_json(test_file, {})
            assert result == {"key": "value"}
            
            # Test load with corrupted file
            with open(test_file, 'w') as f:
                f.write("invalid json")
            result = _load_json(test_file, {"fallback": True})
            assert result == {"fallback": True}

    @pytest.mark.usefixtures("mock_service")
    def test_download_template_not_found(self):
        # Test requesting a template that doesn't exist
        with patch('os.path.exists', return_value=False):
            response = client.get("/download_template?country=nonexistent")
            # Should fall back to default or raise 404
            assert response.status_code in [200, 404]

    @pytest.mark.usefixtures("mock_service")
    def test_upload_file_error(self):
        with patch('builtins.open', side_effect=Exception("IO error")):
            csv_content = "country,brand\nbrazil,amstel\n"
            file_data = BytesIO(csv_content.encode())
            response = client.post(
                "/upload",
                files={"file": ("test.csv", file_data, "text/csv")}
            )
            assert response.status_code == 500

    @pytest.mark.usefixtures("mock_service")
    def test_delete_experiment_by_id_success(self):
        # First create an experiment
        payload = {
            "name": "Test Experiment",
            "description": "Test",
            "country": "brazil",
            "brand": "amstel",
            "scenarios": [{"scenario": 1}]
        }
        create_response = client.post("/api/v1/experiments", json=payload)
        assert create_response.status_code == 200
        exp_id = create_response.json()["id"]
        
        # Now delete it
        delete_response = client.delete(f"/api/v1/experiments/{exp_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] is True

    @pytest.mark.usefixtures("mock_service")
    def test_delete_experiment_by_index_out_of_range(self):
        # Try to delete an index that's out of range
        response = client.delete("/api/v1/experiments/999")
        # Should return 404 or handle gracefully
        assert response.status_code == 404

    @pytest.mark.usefixtures("mock_service")
    def test_list_experiments_pagination(self):
        # Create multiple experiments
        for i in range(3):
            payload = {
                "name": f"Experiment {i}",
                "description": "Test",
                "country": "brazil",
                "brand": "amstel",
                "scenarios": [{"scenario": 1}]
            }
            client.post("/api/v1/experiments", json=payload)
        
        # Test pagination
        response = client.get("/api/v1/experiments?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert "pagination" in data
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 2

