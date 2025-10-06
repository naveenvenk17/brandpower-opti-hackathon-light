"""Test the forecast pipeline independently"""
from frontend.scaled_forecast import build_brand_quarter_forecast
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_forecast():
    print("="*80)
    print("TESTING FORECAST PIPELINE INDEPENDENTLY")
    print("="*80)

    # Load the uploaded data
    data_path = "uploads/weekly_colombia.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        return

    df = pd.read_csv(data_path)
    print(f"\nLoaded data: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")

    # Filter to 2024 data (simulating Flask filter)
    df_filtered = df[(df['year'] == 2024) & (df['month'] >= 7)].copy()
    print(f"Filtered to 2024 Jul+: {df_filtered.shape}")

    try:
        print("\nCalling build_brand_quarter_forecast...")
        forecast_quarters, brand_to_values = build_brand_quarter_forecast(
            df_filtered)

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Forecast quarters: {forecast_quarters}")
        print(f"Number of brands: {len(brand_to_values)}")
        print(f"First 3 brands:")
        for i, (brand, values) in enumerate(list(brand_to_values.items())[:3]):
            print(f"  {brand}: {values}")

        # Verify structure matches Flask response
        print("\n" + "="*80)
        print("FLASK RESPONSE FORMAT CHECK")
        print("="*80)

        results = {
            'baseline': brand_to_values,  # In real Flask, this comes from baseline CSV
            'simulated': brand_to_values,
            'quarters': forecast_quarters,
            'historical': {},
            'historical_quarters': []
        }

        print(f"Response keys: {list(results.keys())}")
        print(f"baseline type: {type(results['baseline'])}")
        print(f"simulated type: {type(results['simulated'])}")
        print(f"quarters type: {type(results['quarters'])}")
        print(f"Sample baseline brand: {list(results['baseline'].items())[0]}")

        return results

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_forecast()
    if result:
        print("\n" + "="*80)
        print("PIPELINE TEST PASSED")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("PIPELINE TEST FAILED")
        print("="*80)
