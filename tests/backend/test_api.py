
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd

# Import the FastAPI application
from src.app import app

# Create a TestClient instance
client = TestClient(app)

# Mock the ForecastSimulateService for API tests
@pytest.fixture(scope="module")
def mock_service():
    with patch('src.app.get_service_and_baseline') as mock_get_service:
        mock_forecast_service = MagicMock()
        
        # Mock forecast_baseline
        mock_forecast_service.forecast_baseline.return_value = pd.DataFrame({
            'brand': ['amstel', 'amstel', 'skol', 'skol'],
            'country': ['brazil', 'brazil', 'brazil', 'brazil'],
            'year': [2024, 2024, 2024, 2024],
            'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
            'predicted_power': [10.0, 10.5, 20.0, 20.5]
        })

        # Mock simulate
        mock_forecast_service.simulate.return_value = pd.DataFrame({
            'brand': ['amstel', 'amstel'],
            'country': ['brazil', 'brazil'],
            'year': [2024, 2024],
            'quarter': ['Q1', 'Q2'],
            'baseline_power': [10.0, 10.5],
            'simulated_power': [11.0, 11.5],
            'uplift': [1.0, 1.0],
            'uplift_pct': [10.0, 9.52]
        })

        # Mock optimize_allocation
        mock_forecast_service.optimize_allocation.return_value = {
            'optimal_allocation': {'digital_spend': 1000, 'tv_spend': 500},
            'expected_lift': 0.15,
            'roi': 2.5
        }

        mock_get_service.return_value = (mock_forecast_service, mock_forecast_service.forecast_baseline.return_value)
        yield


class TestFastAPIEndpoints:

    def test_health_endpoint(self):
        response = client.get("/health")
        # Health check may return 200 (healthy) or 503 (degraded/unhealthy)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "checks" in data

    @pytest.mark.usefixtures("mock_service")
    def test_forecast_baseline_endpoint(self):
        response = client.get("/api/v1/forecast/baseline/brazil/amstel")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2 # Q1 and Q2 for amstel
        assert data["predictions"][0] == 10.0

    @pytest.mark.usefixtures("mock_service")
    def test_forecast_baseline_endpoint_not_found(self):
        # Mock the service to return empty for a specific query
        with patch('src.app.get_service_and_baseline') as mock_get_service:
            mock_forecast_service = MagicMock()
            mock_forecast_service.forecast_baseline.return_value = pd.DataFrame({
                'brand': [], 'country': [], 'year': [], 'quarter': [], 'predicted_power': []
            })
            mock_get_service.return_value = (mock_forecast_service, mock_forecast_service.forecast_baseline.return_value)
            response = client.get("/api/v1/forecast/baseline/nonexistent/brand")
            assert response.status_code == 404
            assert "detail" in response.json()

    @pytest.mark.usefixtures("mock_service")
    def test_simulate_scenario_endpoint(self):
        payload = {
            "edited_rows": [
                {"country": "brazil", "brand": "amstel", "year": 2024, "quarter": "Q1", "digital_spend": 120},
                {"country": "brazil", "brand": "amstel", "year": 2024, "quarter": "Q2", "digital_spend": 130}
            ],
            "columns": ["country", "brand", "year", "quarter", "digital_spend"],
            "target_brands": ["amstel"]
        }
        response = client.post("/api/v1/simulate/scenario", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "baseline" in data
        assert "simulated" in data
        assert "quarters" in data
        assert data["simulated"]["amstel"][0] == 11.0 # From mock

    @pytest.mark.usefixtures("mock_service")
    def test_optimize_allocation_endpoint(self):
        payload = {"total_budget": 1500, "channels": ["digital_spend", "tv_spend"]}
        response = client.post("/api/v1/optimize/allocation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data['expected_lift'] == 0.15 # From mock
        assert "optimal_allocation" in data

    @pytest.mark.usefixtures("mock_service")
    def test_chat_endpoint(self):
        # Mock the agent executor to return a predefined response
        with patch('src.app.get_agent_executor') as mock_get_agent_executor:
            mock_agent_executor = AsyncMock() # Use AsyncMock here
            mock_agent_executor.ainvoke.return_value = {'output': 'Hello from agent!'}
            mock_get_agent_executor.return_value = mock_agent_executor

            payload = {"message": "Hello", "chat_history": []}
            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Hello from agent!"

