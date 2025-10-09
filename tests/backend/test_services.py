"""
Tests for forecast service - Updated to use AutoGluonForecastService
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.services.forecast.forecast_service import AutoGluonForecastService


# Mock data for testing
@pytest.fixture
def mock_baseline_csv_data():
    """Mock baseline forecast CSV data"""
    return pd.DataFrame({
        'year': [2024, 2024, 2024, 2024, 2024, 2024, 2024],
        'period': ['Q3', 'Q3', 'Q3', 'Q4', 'Q4', 'Q4', 'Q4'],
        'period_type': ['QTR'] * 7,
        'country': ['colombia', 'colombia', 'colombia', 'colombia', 'colombia', 'colombia', 'colombia'],
        'brand': ['AGUILA', 'FAMILIA POKER', 'FAMILIA CORONA', 'AGUILA', 'FAMILIA POKER', 'FAMILIA CORONA', 'FAMILIA CLUB COLOMBIA'],
        'power': [15.17, 16.09, 14.32, 15.22, 15.64, 14.14, 10.62]
    })


@pytest.fixture
def mock_baseline_csv_path(tmp_path, mock_baseline_csv_data):
    """Create temporary baseline CSV file"""
    csv_path = tmp_path / "baseline_forecast.csv"
    mock_baseline_csv_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def service(tmp_path, mock_baseline_csv_path):
    """Create AutoGluonForecastService with mocked baseline"""
    model_path = str(tmp_path / "models")
    
    # Service will load baseline CSV but won't need AutoGluon until forecast_with_changes is called
    service = AutoGluonForecastService(
        model_path=model_path,
        baseline_forecast_path=mock_baseline_csv_path
    )
    return service


class TestAutoGluonForecastService:
    """Tests for AutoGluonForecastService"""

    def test_init(self, service):
        """Test service initialization"""
        assert service.model_path is not None
        assert service.baseline_forecast is not None
        assert not service.baseline_forecast.empty
        assert len(service.baseline_forecast) == 7

    def test_get_baseline_forecast_all(self, service):
        """Test getting all baseline forecasts"""
        baseline = service.get_baseline_forecast()
        assert not baseline.empty
        assert 'predicted_power' in baseline.columns or 'power' in baseline.columns
        assert len(baseline) == 7

    def test_get_baseline_forecast_filtered_by_country(self, service):
        """Test filtering baseline by country"""
        baseline = service.get_baseline_forecast(country='colombia')
        assert not baseline.empty
        assert all(baseline['country'].str.lower() == 'colombia')

    def test_get_baseline_forecast_filtered_by_brand(self, service):
        """Test filtering baseline by brand"""
        baseline = service.get_baseline_forecast(brand='AGUILA')
        assert not baseline.empty
        assert all(baseline['brand'].str.upper() == 'AGUILA')
        assert len(baseline) == 2  # Q3 and Q4

    def test_get_baseline_forecast_filtered_both(self, service):
        """Test filtering by both country and brand"""
        baseline = service.get_baseline_forecast(country='colombia', brand='AGUILA')
        assert not baseline.empty
        assert all(baseline['country'].str.lower() == 'colombia')
        assert all(baseline['brand'].str.upper() == 'AGUILA')

    def test_list_brands_all(self, service):
        """Test listing all brands"""
        brands = service.list_brands()
        assert 'AGUILA' in brands
        assert 'FAMILIA POKER' in brands
        assert 'FAMILIA CORONA' in brands
        assert len(brands) == 4

    def test_list_brands_filtered(self, service):
        """Test listing brands filtered by country"""
        brands = service.list_brands(country='colombia')
        assert len(brands) > 0
        assert all(brand.isupper() for brand in brands)

    def test_forecast_with_changes_mocked(self, service, mock_baseline_csv_data):
        """Test forecast_with_changes with mocked AutoGluon"""
        # Mock the lazy import function
        mock_forecast_fn = MagicMock(return_value=(
            ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2'],
            {'AGUILA': [15.17, 15.22, 15.36, 15.75]}
        ))
        
        with patch('src.services.forecast.forecast_service._lazy_import_autogluon', 
                   return_value=(mock_forecast_fn, [])):
            
            input_data = pd.DataFrame({
                'country': ['colombia'],
                'brand': ['AGUILA'],
                'year': [2024],
                'month': [7],
                'week_of_month': [1],
                'paytv': [100000],
                'wholesalers': [50000]
            })
            
            quarters, brand_power = service.forecast_with_changes(input_data)
            
            assert quarters == ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
            assert 'AGUILA' in brand_power
            assert len(brand_power['AGUILA']) == 4

    def test_forecast_with_changes_autogluon_not_installed(self, service):
        """Test forecast_with_changes when AutoGluon is not installed"""
        # Mock lazy import to raise ImportError
        with patch('src.services.forecast.forecast_service._lazy_import_autogluon', 
                   side_effect=ImportError("AutoGluon not installed")):
            
            input_data = pd.DataFrame({
                'country': ['colombia'],
                'brand': ['AGUILA']
            })
            
            with pytest.raises(ImportError, match="AutoGluon"):
                service.forecast_with_changes(input_data)

    def test_baseline_forecast_no_file(self, tmp_path):
        """Test service when baseline file doesn't exist"""
        service = AutoGluonForecastService(
            model_path=str(tmp_path / "models"),
            baseline_forecast_path=str(tmp_path / "nonexistent.csv")
        )
        
        assert service.baseline_forecast is None
        assert service.list_brands() == []

    def test_baseline_forecast_normalization(self, service):
        """Test that baseline forecast columns are normalized correctly"""
        baseline = service.get_baseline_forecast()
        
        # Check lowercase country
        if 'country' in baseline.columns:
            assert all(baseline['country'].str.islower())
        
        # Check uppercase brand
        if 'brand' in baseline.columns:
            assert all(baseline['brand'].str.isupper())
        
        # Check predicted_power column exists
        assert 'predicted_power' in baseline.columns or 'power' in baseline.columns


# Backward compatibility tests (for code that might still reference old service)
class TestBackwardCompatibility:
    """Tests for backward compatibility"""

    def test_service_has_list_brands(self, service):
        """Ensure new service has list_brands method"""
        assert hasattr(service, 'list_brands')
        assert callable(service.list_brands)

    def test_service_has_forecast_method(self, service):
        """Ensure new service has forecast_with_changes method"""
        assert hasattr(service, 'forecast_with_changes')
        assert callable(service.forecast_with_changes)

    def test_service_has_baseline_method(self, service):
        """Ensure new service has get_baseline_forecast method"""
        assert hasattr(service, 'get_baseline_forecast')
        assert callable(service.get_baseline_forecast)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
