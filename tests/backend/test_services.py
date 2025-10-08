
import pytest
import pandas as pd
import pickle
from unittest.mock import patch, MagicMock

from production_scripts.services.forecast_simulate_service import ForecastSimulateService
from production_scripts.models.marketing.impact_model import MarketingImpactModel
from production_scripts.utils.errors import DataNotFoundError, DataValidationError

# Mock data for testing
@pytest.fixture
def mock_model_artifact():
    return {
        'model': MagicMock(),
        'feature_cols': ['digital_spend', 'tv_spend', 'power_lag_1']
    }

@pytest.pytest.fixture
def mock_baseline_features_df():
    return pd.DataFrame({
        'brand': ['amstel', 'amstel', 'skol', 'skol'],
        'country': ['brazil', 'brazil', 'brazil', 'brazil'],
        'year': [2024, 2024, 2024, 2024],
        'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
        'digital_spend': [100, 110, 200, 210],
        'tv_spend': [50, 55, 100, 105],
        'power_lag_1': [10, 11, 20, 21],
        'predicted_power': [12, 13, 22, 23] # Added for forecast_baseline output
    })

@pytest.fixture
def mock_uploaded_template_df():
    return pd.DataFrame({
        'country': ['brazil', 'brazil'],
        'brand': ['amstel', 'amstel'],
        'year': [2024, 2024],
        'quarter': ['Q1', 'Q2'],
        'month': [1, 4],
        'week_of_month': [1, 1],
        'digital_spend': [120, 130],
        'tv_spend': [60, 65],
        'traditional_spend': [10, 12],
        'sponsorship_spend': [5, 6],
        'other_spend': [2, 3],
        'power': [10, 11] # Ignored
    })

@pytest.fixture
def service(mock_model_artifact, mock_baseline_features_df):
    with patch('builtins.open', MagicMock()), \
         patch('pickle.load', return_value=mock_model_artifact), \
         patch('pandas.read_csv', return_value=mock_baseline_features_df): # Mock read_csv for baseline
        s = ForecastSimulateService(model_path='dummy_model.pkl', forecast_features_path='dummy_features.csv')
        s.model.predict.return_value = [15, 16, 25, 26] # Mock predictions
        return s


class TestForecastSimulateService:

    def test_init(self, service):
        assert service.model is not None
        assert 'digital_spend' in service.feature_cols
        assert not service.baseline_features_df.empty

    def test_forecast_baseline_all(self, service):
        forecast_df = service.forecast_baseline()
        assert not forecast_df.empty
        assert 'predicted_power' in forecast_df.columns
        assert len(forecast_df) == 4 # All rows from mock_baseline_features_df

    def test_forecast_baseline_filtered(self, service):
        forecast_df = service.forecast_baseline(country='brazil', brand='amstel')
        assert not forecast_df.empty
        assert all(forecast_df['brand'] == 'amstel')
        assert all(forecast_df['country'] == 'brazil')
        assert len(forecast_df) == 2

    def test_forecast_baseline_no_data(self, service):
        forecast_df = service.forecast_baseline(country='nonexistent', brand='nonexistent')
        assert forecast_df.empty

    def test_simulate(self, service, mock_uploaded_template_df, mock_baseline_features_df):
        # Ensure baseline_forecast is passed correctly
        baseline_forecast_for_sim = mock_baseline_features_df.copy()
        baseline_forecast_for_sim['predicted_power'] = [12, 13, 22, 23]

        # Mock the re-engineering method as it's complex and internal
        with patch.object(service, '_reengineer_marketing_features', side_effect=lambda df: df) as mock_reengineer:
            simulation_result_df = service.simulate(mock_uploaded_template_df, baseline_forecast_for_sim)

            assert not simulation_result_df.empty
            assert 'simulated_power' in simulation_result_df.columns
            assert 'uplift' in simulation_result_df.columns
            assert 'uplift_pct' in simulation_result_df.columns
            assert len(simulation_result_df) == 2 # Only amstel data
            mock_reengineer.assert_called_once()

    def test_optimize_allocation(self, service):
        # Mock the internal MarketingImpactModel.optimize method
        with patch.object(service.impact_model, 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                'optimal_allocation': {'digital_spend': 1000, 'tv_spend': 500},
                'expected_lift': 0.15,
                'roi': 2.5
            }
            
            result = service.optimize_allocation(total_budget=1500, channels=['digital_spend', 'tv_spend'])
            
            mock_optimize.assert_called_once_with(
                total_budget=1500,
                channels=['digital_spend', 'tv_spend'],
                method='gradient',
                digital_cap=0.99,
                tv_cap=0.5
            )
            assert result['expected_lift'] == 0.15
            assert 'optimal_allocation' in result

