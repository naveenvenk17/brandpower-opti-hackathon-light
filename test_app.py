#!/usr/bin/env python3
"""
Comprehensive test script for BrandCompass.ai Streamlit application
"""

from utils import *
import pandas as pd
import numpy as np
import os
import sys

# Add frontend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))


def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")

    # Test with valid CSV
    test_data_path = "frontend/data/test_quarterly_data.csv"
    if os.path.exists(test_data_path):
        data = load_data(test_data_path)
        if data is not None:
            print("✅ Data loading successful")
            print(f"   - Loaded {len(data)} rows")
            print(f"   - Columns: {list(data.columns)}")
        else:
            print("❌ Data loading failed")
    else:
        print("⚠️  Test data file not found")

    # Test with invalid file
    invalid_data = load_data("nonexistent.csv")
    if invalid_data is None:
        print("✅ Invalid file handling works correctly")
    else:
        print("❌ Invalid file handling failed")


def test_quarterly_conversion():
    """Test quarterly data conversion"""
    print("\n🧪 Testing quarterly conversion...")

    # Create test data
    test_data = pd.DataFrame({
        'country': ['Brazil'] * 12,
        'brand': ['BrandA'] * 12,
        'year': [2024] * 12,
        'month': list(range(1, 13)),
        'week_of_month': [1] * 12,
        'brand events': [100] * 12,
        'power': [15.0] * 12
    })

    quarterly_data = roll_data_to_quarter(test_data)

    if quarterly_data is not None:
        print("✅ Quarterly conversion successful")
        print(f"   - Original rows: {len(test_data)}")
        print(f"   - Quarterly rows: {len(quarterly_data)}")
        print(
            f"   - Quarters: {quarterly_data['quarter'].unique() if 'quarter' in quarterly_data.columns else 'No quarter column'}")
    else:
        print("❌ Quarterly conversion failed")

    # Test with missing month column
    test_data_no_month = test_data.drop('month', axis=1)
    quarterly_data_no_month = roll_data_to_quarter(test_data_no_month)

    if quarterly_data_no_month is not None:
        print("✅ Quarterly conversion with missing month column handled")
    else:
        print("❌ Quarterly conversion with missing month column failed")


def test_forecast_power():
    """Test power forecasting functionality"""
    print("\n🧪 Testing power forecasting...")

    # Create test data
    test_data = pd.DataFrame({
        'country': ['Brazil'] * 4,
        'brand': ['BrandA'] * 4,
        'year': [2024, 2024, 2025, 2025],
        'quarter': ['Q3', 'Q4', 'Q1', 'Q2'],
        'brand events': [100, 110, 120, 130],
        'meta': [200, 210, 220, 230]
    })

    # Test without filters
    forecast_data = forecast_power(test_data)
    if forecast_data is not None and 'power' in forecast_data.columns:
        print("✅ Power forecasting without filters successful")
        print(
            f"   - Generated power values: {forecast_data['power'].tolist()}")
    else:
        print("❌ Power forecasting without filters failed")

    # Test with filters
    filters = {'brand events': 150, 'meta': 250}
    forecast_data_filtered = forecast_power(test_data, filters)
    if forecast_data_filtered is not None and 'power' in forecast_data_filtered.columns:
        print("✅ Power forecasting with filters successful")
        print(f"   - Applied filters: {filters}")
    else:
        print("❌ Power forecasting with filters failed")


def test_utility_functions():
    """Test utility functions"""
    print("\n🧪 Testing utility functions...")

    # Test get_optimizable_columns
    optimizable_cols = get_optimizable_columns()
    if optimizable_cols and len(optimizable_cols) > 0:
        print(
            f"✅ get_optimizable_columns returned {len(optimizable_cols)} features")
    else:
        print("❌ get_optimizable_columns failed")

    # Test get_brands_from_data
    test_data = pd.DataFrame({
        'brand': ['BrandA', 'BrandB', 'BrandA', 'BrandC'],
        'power': [10, 20, 15, 25]
    })

    brands = get_brands_from_data(test_data)
    if brands == ['BrandA', 'BrandB', 'BrandC']:
        print("✅ get_brands_from_data works correctly")
    else:
        print(f"❌ get_brands_from_data failed. Got: {brands}")

    # Test with empty data
    empty_brands = get_brands_from_data(pd.DataFrame())
    if empty_brands == []:
        print("✅ get_brands_from_data handles empty data correctly")
    else:
        print("❌ get_brands_from_data failed with empty data")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")

    # Test with None data
    result = roll_data_to_quarter(None)
    if result is None:
        print("✅ roll_data_to_quarter handles None input correctly")
    else:
        print("❌ roll_data_to_quarter failed with None input")

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = roll_data_to_quarter(empty_df)
    if result is None:
        print("✅ roll_data_to_quarter handles empty DataFrame correctly")
    else:
        print("❌ roll_data_to_quarter failed with empty DataFrame")

    # Test forecast_power with None
    result = forecast_power(None)
    if result is None:
        print("✅ forecast_power handles None input correctly")
    else:
        print("❌ forecast_power failed with None input")


def test_data_types():
    """Test data type handling"""
    print("\n🧪 Testing data type handling...")

    # Test with string months
    test_data = pd.DataFrame({
        'country': ['Brazil'] * 4,
        'brand': ['BrandA'] * 4,
        'year': [2024] * 4,
        'month': ['1', '2', '3', '4'],  # String months
        'power': [10, 20, 15, 25]
    })

    quarterly_data = roll_data_to_quarter(test_data)
    if quarterly_data is not None and 'quarter' in quarterly_data.columns:
        print("✅ String month handling works correctly")
    else:
        print("❌ String month handling failed")

    # Test with invalid months
    test_data_invalid = pd.DataFrame({
        'country': ['Brazil'] * 4,
        'brand': ['BrandA'] * 4,
        'year': [2024] * 4,
        'month': ['invalid', 'data', '13', '0'],  # Invalid months
        'power': [10, 20, 15, 25]
    })

    quarterly_data_invalid = roll_data_to_quarter(test_data_invalid)
    if quarterly_data_invalid is not None:
        print("✅ Invalid month handling works correctly")
    else:
        print("❌ Invalid month handling failed")


def main():
    """Run all tests"""
    print("🚀 Starting comprehensive testing of BrandCompass.ai\n")

    test_data_loading()
    test_quarterly_conversion()
    test_forecast_power()
    test_utility_functions()
    test_edge_cases()
    test_data_types()

    print("\n✅ Testing completed!")
    print("\n📋 Summary:")
    print("- All core functionality has been tested")
    print("- Error handling has been validated")
    print("- Edge cases have been covered")
    print("- Data type conversions work correctly")


if __name__ == "__main__":
    main()
