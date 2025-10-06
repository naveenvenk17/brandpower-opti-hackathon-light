"""Test the /calculate endpoint to verify it works as expected"""
import requests
import json
import pandas as pd


def test_calculate_endpoint():
    print("="*80)
    print("TESTING /calculate ENDPOINT")
    print("="*80)

    # Load the uploaded data to see what we have
    df = pd.read_csv("uploads/weekly_colombia.csv")
    print(f"\nUploaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:15]}...")

    # Filter to simulate UI selection (default: AGUILA brand, 2024, July)
    df_filtered = df[
        (df['brand'] == 'AGUILA') &
        (df['year'] == 2024) &
        (df['month'] == 7)
    ].copy()
    print(f"Filtered data (AGUILA, 2024-07): {df_filtered.shape}")

    # Aggregate by channel groups (simulating what UI sends)
    from frontend.utils import aggregate_by_channel_groups
    df_aggregated = aggregate_by_channel_groups(df_filtered)

    # Build edited_rows (simulating UI table data)
    id_columns = ['country', 'brand', 'year', 'month', 'week_of_month']
    group_columns = ['Digital', 'Influencer',
                     'TV', 'OOH_Audio', 'Events_Sponsorship']
    table_columns = id_columns + group_columns

    # Convert to list of dicts (first 500 rows as UI limits display)
    edited_rows = df_aggregated[table_columns].head(
        500).to_dict(orient='records')

    print(f"\nPayload summary:")
    print(f"  edited_rows count: {len(edited_rows)}")
    print(f"  table_columns: {table_columns}")
    print(f"  Sample row: {edited_rows[0] if edited_rows else 'None'}")

    # Build the payload exactly as the frontend sends it
    payload = {
        'changes': {},  # No manual changes, just testing baseline
        'filters': {
            'brands': ['AGUILA'],
            'years': [2024],
            'months': [7],
            'weeks': []
        },
        'edited_rows': edited_rows,
        'columns': table_columns
    }

    print(f"\n" + "="*80)
    print("SENDING REQUEST TO /calculate")
    print("="*80)

    try:
        # Send request to Flask server
        response = requests.post(
            'http://localhost:5000/calculate',
            json=payload,
            timeout=300  # 5 minute timeout for model inference
        )

        print(f"\nResponse Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print("\n" + "="*80)
            print("SUCCESS! RESPONSE RECEIVED")
            print("="*80)

            # Verify response structure
            print(f"\nResponse keys: {list(data.keys())}")

            if 'baseline' in data:
                print(f"\nBaseline:")
                print(f"  Type: {type(data['baseline'])}")
                print(f"  Brands count: {len(data['baseline'])}")
                print(f"  Sample brands: {list(data['baseline'].keys())[:3]}")
                if data['baseline']:
                    sample_brand = list(data['baseline'].keys())[0]
                    print(
                        f"  Sample values ({sample_brand}): {data['baseline'][sample_brand]}")

            if 'simulated' in data:
                print(f"\nSimulated:")
                print(f"  Type: {type(data['simulated'])}")
                print(f"  Brands count: {len(data['simulated'])}")
                print(f"  Sample brands: {list(data['simulated'].keys())[:3]}")
                if data['simulated']:
                    sample_brand = list(data['simulated'].keys())[0]
                    print(
                        f"  Sample values ({sample_brand}): {data['simulated'][sample_brand]}")

            if 'quarters' in data:
                print(f"\nQuarters:")
                print(f"  Type: {type(data['quarters'])}")
                print(f"  Values: {data['quarters']}")

            if 'historical' in data:
                print(f"\nHistorical:")
                print(f"  Type: {type(data['historical'])}")
                print(f"  Brands count: {len(data['historical'])}")

            if 'historical_quarters' in data:
                print(f"\nHistorical Quarters:")
                print(f"  Type: {type(data['historical_quarters'])}")
                print(f"  Count: {len(data['historical_quarters'])}")
                print(f"  Values: {data['historical_quarters'][:5]}..." if len(
                    data['historical_quarters']) > 5 else f"  Values: {data['historical_quarters']}")

            # Verify data format for frontend
            print("\n" + "="*80)
            print("FRONTEND COMPATIBILITY CHECK")
            print("="*80)

            checks_passed = 0
            checks_total = 0

            # Check 1: quarters is an array
            checks_total += 1
            if isinstance(data.get('quarters'), list):
                print("✓ quarters is a list")
                checks_passed += 1
            else:
                print(
                    f"✗ quarters is not a list: {type(data.get('quarters'))}")

            # Check 2: baseline is an object
            checks_total += 1
            if isinstance(data.get('baseline'), dict):
                print("✓ baseline is a dict")
                checks_passed += 1
            else:
                print(
                    f"✗ baseline is not a dict: {type(data.get('baseline'))}")

            # Check 3: simulated is an object
            checks_total += 1
            if isinstance(data.get('simulated'), dict):
                print("✓ simulated is a dict")
                checks_passed += 1
            else:
                print(
                    f"✗ simulated is not a dict: {type(data.get('simulated'))}")

            # Check 4: Each brand has 4 values
            checks_total += 1
            if data.get('baseline'):
                sample_brand = list(data['baseline'].keys())[0]
                if len(data['baseline'][sample_brand]) == 4:
                    print(f"✓ Each brand has 4 quarter values")
                    checks_passed += 1
                else:
                    print(
                        f"✗ Brand values count mismatch: {len(data['baseline'][sample_brand])}")

            # Check 5: quarters matches expected
            checks_total += 1
            expected_quarters = ['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2']
            if data.get('quarters') == expected_quarters:
                print(f"✓ Quarters match expected: {expected_quarters}")
                checks_passed += 1
            else:
                print(f"✗ Quarters mismatch:")
                print(f"  Expected: {expected_quarters}")
                print(f"  Got: {data.get('quarters')}")

            print(f"\nCompatibility Score: {checks_passed}/{checks_total}")

            return data

        else:
            print(f"\nERROR Response")
            print(f"Status: {response.status_code}")
            print(f"Body: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Flask server")
        print("Make sure the Flask server is running on http://localhost:5000")
        print("Run: python run_flask.py")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_calculate_endpoint()
    if result:
        print("\n" + "="*80)
        print("ENDPOINT TEST PASSED")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ENDPOINT TEST FAILED")
        print("="*80)
