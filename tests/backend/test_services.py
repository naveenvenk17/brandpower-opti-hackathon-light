import pytest
import pandas as pd
import pickle
from unittest.mock import patch, MagicMock

from src.services.simulation.simulation_service import ForecastSimulateService
from src.services.optimization.impact_model import MarketingImpactModel

# Mock data for testing
@pytest.fixture
def mock_model_artifact():
    return {
        'model': MagicMock(),
        'feature_cols': ['digital_spend', 'tv_spend', 'traditional_spend', 'sponsorship_spend', 'other_spend', 'power_lag_1']
    }

@pytest.fixture
def mock_baseline_features_df():
    return pd.DataFrame({
        'brand': ['amstel', 'amstel', 'skol', 'skol', 'familia budweiser', 'familia budweiser', 'familia club colombia'],
        'country': ['brazil', 'brazil', 'brazil', 'brazil', 'colombia', 'colombia', 'colombia'],
        'year': [2024, 2024, 2024, 2024, 2024, 2024, 2024],
        'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2', 'Q2'],
        'digital_spend': [100, 110, 200, 210, 300, 305, 310],
        'tv_spend': [50, 55, 100, 105, 150, 152, 155],
        'traditional_spend': [10, 12, 15, 16, 20, 21, 22],
        'sponsorship_spend': [5, 6, 7, 8, 9, 9, 10],
        'other_spend': [2, 3, 4, 5, 6, 6, 7],
        'power_lag_1': [10, 11, 20, 21, 30, 30, 31],
        'predicted_power': [12, 13, 22, 23, 32, 32, 33] # Added for forecast_baseline output
    })

@pytest.fixture
def mock_uploaded_template_df():
    return pd.DataFrame({
        'country': ['colombia', 'colombia'],
        'brand': ['familia budweiser', 'familia budweiser'],
        'year': [2024, 2024],
        'quarter': ['Q1', 'Q2'],
        'month': [1, 4],
        'week_of_month': [1, 1],
        'digital_spend': [320, 330],
        'tv_spend': [160, 165],
        'traditional_spend': [10, 12],
        'sponsorship_spend': [5, 6],
        'other_spend': [2, 3],
        'power': [30, 31] # Ignored
    })

def predict_side_effect(df):
    if len(df) == 7:
        return [15, 16, 25, 26, 35, 35, 36]
    elif len(df) == 2:
        return [35, 35]
    else:
        return [0] * len(df)

@pytest.fixture(scope="function")
def service(mock_model_artifact, mock_baseline_features_df, monkeypatch):
    with patch('builtins.open', MagicMock()), \
         patch('pickle.load', return_value=mock_model_artifact):
        s = ForecastSimulateService(model_path='dummy_model.pkl', baseline_features_df=mock_baseline_features_df)
        s.model.predict.side_effect = predict_side_effect
        return s


class TestForecastSimulateService:

    def test_init(self, service):
        assert service.model is not None
        assert 'digital_spend' in service.feature_cols
        assert not service.baseline_features_df.empty
        assert len(service.baseline_features_df) == 7

    def test_forecast_baseline_all(self, service):
        forecast_df = service.forecast_baseline()
        assert not forecast_df.empty
        assert 'predicted_power' in forecast_df.columns
        assert len(forecast_df) == 7 # All rows from mock_baseline_features_df

    def test_forecast_baseline_filtered_exact(self, service):
        forecast_df = service.forecast_baseline(country='brazil', brand='amstel')
        assert not forecast_df.empty
        assert all(forecast_df['brand'] == 'amstel')
        assert all(forecast_df['country'] == 'brazil')
        assert len(forecast_df) == 2

    def test_forecast_baseline_filtered_case_insensitive(self, service):
        forecast_df = service.forecast_baseline(country='Brazil', brand='Amstel')
        assert not forecast_df.empty
        assert all(forecast_df['brand'] == 'amstel')
        assert all(forecast_df['country'] == 'brazil')
        assert len(forecast_df) == 2

    def test_forecast_baseline_no_data(self, service):
        forecast_df = service.forecast_baseline(country='nonexistent', brand='nonexistent')
        assert forecast_df.empty

    def test_list_brands_all(self, service):
        brands = service.list_brands()
        assert 'amstel' in brands
        assert 'skol' in brands
        assert 'familia budweiser' in brands
        assert len(brands) == 4 # amstel, skol, familia budweiser, familia club colombia

    def test_list_brands_filtered(self, service):
        brands = service.list_brands(country='colombia')
        assert 'familia budweiser' in brands
        assert 'familia club colombia' in brands
        assert len(brands) == 2

    def test_list_brands_filtered_case_insensitive(self, service):
        print(f"Countries in mock_baseline_features_df: {service.baseline_features_df['country'].unique().tolist()}")
        brands = service.list_brands(country='Colombia')
        print(f"Brands returned by list_brands: {brands}")
        assert 'familia budweiser' in brands
        assert 'familia club colombia' in brands
        assert len(brands) == 2

    def test_simulate(self, service, mock_uploaded_template_df, mock_baseline_features_df):
        # Ensure baseline_forecast is passed correctly
        baseline_forecast_for_sim = mock_baseline_features_df.copy()
        baseline_forecast_for_sim['predicted_power'] = [12, 13, 22, 23, 32, 32, 33]

        # Mock the re-engineering method as it's complex and internal
        with patch.object(service, '_reengineer_marketing_features', side_effect=lambda df: df) as mock_reengineer:
            simulation_result_df = service.simulate(mock_uploaded_template_df, baseline_forecast_for_sim)

            assert not simulation_result_df.empty
            assert 'simulated_power' in simulation_result_df.columns
            assert 'uplift' in simulation_result_df.columns
            assert 'uplift_pct' in simulation_result_df.columns
            assert len(simulation_result_df) == 2 # Only familia budweiser data
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

    def test_init_with_none_baseline(self, mock_model_artifact):
        with patch('builtins.open', MagicMock()), \
             patch('pickle.load', return_value=mock_model_artifact):
            s = ForecastSimulateService(model_path='dummy_model.pkl', baseline_features_df=None)
            assert s.baseline_features_df is None

    def test_init_with_missing_columns(self, mock_model_artifact):
        df = pd.DataFrame({
            'brand': ['test'],
            'country': ['us'],
            'year': [2024],
            'quarter': ['Q1'],
            'digital_spend': [100]
            # Missing tv_spend, traditional_spend, etc.
        })
        with patch('builtins.open', MagicMock()), \
             patch('pickle.load', return_value=mock_model_artifact):
            s = ForecastSimulateService(model_path='dummy_model.pkl', baseline_features_df=df)
            # Should have added missing columns
            assert 'tv_spend' in s.baseline_features_df.columns
            assert 'traditional_spend' in s.baseline_features_df.columns

    def test_forecast_baseline_with_save(self, service, tmp_path):
        save_file = tmp_path / "forecast.csv"
        forecast_df = service.forecast_baseline(save_path=str(save_file))
        assert save_file.exists()
        loaded = pd.read_csv(save_file)
        assert len(loaded) == len(forecast_df)

    def test_simulate_with_none_baseline(self, service, mock_uploaded_template_df):
        with patch.object(service, '_reengineer_marketing_features', side_effect=lambda df: df):
            # Call without passing baseline_forecast (should use internal forecast_baseline)
            result = service.simulate(mock_uploaded_template_df, baseline_forecast=None)
            assert not result.empty
            assert 'simulated_power' in result.columns

    def test_simulation_service_wrapper(self, service):
        from src.services.simulation.simulation_service import SimulationService
        sim_service = SimulationService(forecast_service=service)
        assert sim_service.forecast_service is service
        
        # Test the wrapper simulate method
        mock_template = pd.DataFrame({
            'country': ['brazil'],
            'brand': ['amstel'],
            'year': [2024],
            'quarter': ['Q1'],
            'month': [1],
            'week_of_month': [1],
            'digital_spend': [100],
            'tv_spend': [50],
            'traditional_spend': [10],
            'sponsorship_spend': [5],
            'other_spend': [2],
            'power': [10]
        })
        with patch.object(service, '_reengineer_marketing_features', side_effect=lambda df: df):
            result = sim_service.simulate(mock_template)
            assert not result.empty

    def test_reengineer_marketing_features_default(self, service):
        # Test the default _reengineer_marketing_features returns input unchanged
        test_df = pd.DataFrame({'brand': ['test'], 'country': ['us']})
        result = service._reengineer_marketing_features(test_df)
        assert result.equals(test_df)