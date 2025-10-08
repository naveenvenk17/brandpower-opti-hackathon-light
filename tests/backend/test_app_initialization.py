"""Tests for app initialization and service loading"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os


class TestAppInitialization:
    def test_get_service_and_baseline_fallback_models(self):
        """Test get_service_and_baseline with different model fallbacks"""
        from src.app import get_service_and_baseline, _service_cache, _baseline_forecast_cache
        
        # Reset caches
        import src.app as app_module
        app_module._service_cache = None
        app_module._baseline_forecast_cache = None
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'brand': ['test'], 'country': ['us'], 'year': [2024], 'quarter': ['Q1'],
                'digital_spend': [100], 'tv_spend': [50], 'traditional_spend': [10],
                'sponsorship_spend': [5], 'other_spend': [2], 'power_lag_1': [10]
            })
            
            with patch('os.path.exists') as mock_exists:
                # Simulate unified model doesn't exist, fallback to brand_power_forecaster
                def exists_side_effect(path):
                    if 'unified_brand_power_model' in path:
                        return False
                    elif 'brand_power_forecaster.pkl' in path:
                        return True
                    return False
                
                mock_exists.side_effect = exists_side_effect
                
                with patch('builtins.open', MagicMock()):
                    with patch('pickle.load') as mock_pickle:
                        mock_pickle.return_value = {
                            'model': MagicMock(predict=MagicMock(return_value=[10])),
                            'feature_cols': ['digital_spend', 'tv_spend', 'traditional_spend', 
                                           'sponsorship_spend', 'other_spend', 'power_lag_1']
                        }
                        
                        service, baseline = get_service_and_baseline()
                        assert service is not None
                        assert baseline is not None
                        
                        # Reset for next test
                        app_module._service_cache = None
                        app_module._baseline_forecast_cache = None

    def test_get_service_and_baseline_simple_model_fallback(self):
        """Test fallback to simple_brand_power_forecaster"""
        import src.app as app_module
        app_module._service_cache = None
        app_module._baseline_forecast_cache = None
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'brand': ['test'], 'country': ['us'], 'year': [2024], 'quarter': ['Q1'],
                'digital_spend': [100], 'tv_spend': [50], 'traditional_spend': [10],
                'sponsorship_spend': [5], 'other_spend': [2], 'power_lag_1': [10]
            })
            
            with patch('os.path.exists') as mock_exists:
                def exists_side_effect(path):
                    if 'simple_brand_power_forecaster.pkl' in path:
                        return True
                    return False
                
                mock_exists.side_effect = exists_side_effect
                
                with patch('builtins.open', MagicMock()):
                    with patch('pickle.load') as mock_pickle:
                        mock_pickle.return_value = {
                            'model': MagicMock(predict=MagicMock(return_value=[10])),
                            'feature_cols': ['digital_spend', 'tv_spend', 'traditional_spend',
                                           'sponsorship_spend', 'other_spend', 'power_lag_1']
                        }
                        
                        from src.app import get_service_and_baseline
                        service, baseline = get_service_and_baseline()
                        assert service is not None
                        
                        # Reset
                        app_module._service_cache = None
                        app_module._baseline_forecast_cache = None

    def test_get_service_and_baseline_caching(self):
        """Test that service and baseline are cached after first call"""
        import src.app as app_module
        app_module._service_cache = None
        app_module._baseline_forecast_cache = None
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'brand': ['test'], 'country': ['us'], 'year': [2024], 'quarter': ['Q1'],
                'digital_spend': [100], 'tv_spend': [50], 'traditional_spend': [10],
                'sponsorship_spend': [5], 'other_spend': [2], 'power_lag_1': [10]
            })
            
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', MagicMock()):
                    with patch('pickle.load') as mock_pickle:
                        mock_pickle.return_value = {
                            'model': MagicMock(predict=MagicMock(return_value=[10])),
                            'feature_cols': ['digital_spend', 'tv_spend', 'traditional_spend',
                                           'sponsorship_spend', 'other_spend', 'power_lag_1']
                        }
                        
                        from src.app import get_service_and_baseline
                        service1, baseline1 = get_service_and_baseline()
                        service2, baseline2 = get_service_and_baseline()
                        
                        # Should return cached instances
                        assert service1 is service2
                        assert baseline1 is baseline2
                        
                        # Reset
                        app_module._service_cache = None
                        app_module._baseline_forecast_cache = None

