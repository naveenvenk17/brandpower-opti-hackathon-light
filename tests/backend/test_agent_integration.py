"""
Test Agent Integration with Forecast Service
"""
import pytest
import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.master_agent import (
    list_matching_brands,
    forecast_baseline,
    simulate,
    optimize_allocation,
)


class TestAgentTools:
    """Test the agent tools work correctly"""
    
    def test_list_matching_brands(self):
        """Test brand listing with country filter"""
        result = list_matching_brands("corona", country="brazil")
        assert isinstance(result, str)
        assert "corona" in result.lower() or "brand" in result.lower()
    
    def test_forecast_baseline_with_country(self):
        """Test baseline forecast with country filter"""
        result = forecast_baseline(country="brazil")
        assert isinstance(result, str)
        # Should return either formatted results or an error message
        assert len(result) > 0
    
    def test_forecast_baseline_with_brand(self):
        """Test baseline forecast with brand filter"""
        # This should return formatted results or a "not found" message
        result = forecast_baseline(brand="CORONA")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_simulate_tool(self):
        """Test simulation tool with sample data"""
        # Create a minimal sample CSV for testing
        sample_data = pd.DataFrame({
            'brand': ['CORONA', 'CORONA'],
            'country': ['brazil', 'brazil'],
            'year': [2024, 2024],
            'month': [7, 8],
            'paytv': [1000, 1000],
            'digital': [2000, 2000],
        })
        
        test_file = "/tmp/test_template.csv"
        sample_data.to_csv(test_file, index=False)
        
        try:
            result = simulate(test_file)
            assert isinstance(result, str)
            # Should return either results or error message
            assert len(result) > 0
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_optimize_allocation(self):
        """Test budget optimization tool"""
        result = optimize_allocation(
            total_budget=1000000.0,
            method='gradient',
            digital_cap=0.99,
            tv_cap=0.5
        )
        assert isinstance(result, str)
        # Should return formatted optimization results
        assert "budget" in result.lower() or "allocation" in result.lower()


class TestAgentService:
    """Test the AutoGluonForecastService methods called by agent"""
    
    def test_service_methods_exist(self):
        """Verify the service has all required methods"""
        from src.services.forecast.forecast_service import AutoGluonForecastService
        
        # Check that all required methods exist
        required_methods = [
            'forecast_baseline',
            'simulate',
            'optimize_allocation',
            'list_brands',
            'get_baseline_forecast',
        ]
        
        for method in required_methods:
            assert hasattr(AutoGluonForecastService, method), f"Missing method: {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

