"""
Test script for forecast and simulate endpoints
"""
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_baseline_forecast():
    """Test /api/v1/forecast/baseline endpoint"""
    print("\n" + "="*80)
    print("TEST 1: /api/v1/forecast/baseline/{country}/{brand}")
    print("="*80)
    
    url = f"{BASE_URL}/api/v1/forecast/baseline/colombia/AGUILA"
    print(f"GET {url}")
    
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Predictions: {data.get('predictions', [])}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start with: uvicorn src.app:app --reload")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_simulate_scenario():
    """Test /api/v1/simulate/scenario endpoint"""
    print("\n" + "="*80)
    print("TEST 2: /api/v1/simulate/scenario")
    print("="*80)
    
    url = f"{BASE_URL}/api/v1/simulate/scenario"
    print(f"POST {url}")
    
    # Sample request payload
    payload = {
        "edited_rows": [
            {
                "country": "colombia",
                "brand": "AGUILA",
                "year": 2024,
                "month": 7,
                "week_of_month": 1,
                "paytv": 100000,
                "wholesalers": 50000
            }
        ],
        "columns": ["country", "brand", "year", "month", "week_of_month", "paytv", "wholesalers"],
        "target_brands": ["AGUILA"],
        "max_horizon": 4
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)[:200]}...")
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"  Baseline: {data.get('baseline', {})}")
            print(f"  Simulated: {data.get('simulated', {})}")
            print(f"  Quarters: {data.get('quarters', [])}")
            if 'warning' in data:
                print(f"  ⚠️  Warning: {data['warning']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start with: uvicorn src.app:app --reload")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_calculate():
    """Test /calculate endpoint"""
    print("\n" + "="*80)
    print("TEST 3: /calculate")
    print("="*80)
    
    url = f"{BASE_URL}/calculate"
    print(f"POST {url}")
    
    # Sample request payload
    payload = {
        "edited_rows": [
            {
                "country": "colombia",
                "brand": "AGUILA",
                "year": 2024,
                "month": 7,
                "week_of_month": 1,
                "paytv": 100000,
                "wholesalers": 50000,
                "total_distribution": 75000,
                "volume": 10000
            }
        ],
        "columns": ["country", "brand", "year", "month", "week_of_month", "paytv", "wholesalers", "total_distribution", "volume"]
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)[:200]}...")
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"  Baseline brands: {list(data.get('baseline', {}).keys())}")
            print(f"  Simulated brands: {list(data.get('simulated', {}).keys())}")
            print(f"  Quarters: {data.get('quarters', [])}")
            print(f"  Has historical data: {bool(data.get('historical', {}))}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running. Start with: uvicorn src.app:app --reload")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "🧪 " + "="*76)
    print("🧪  FORECAST & SIMULATE ENDPOINTS TEST SUITE")
    print("🧪 " + "="*76)
    
    results = {
        "baseline_forecast": test_baseline_forecast(),
        "simulate_scenario": test_simulate_scenario(),
        "calculate": test_calculate()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP (server not running)"
        print(f"{status}: {test_name}")
    
    if None in results.values():
        print("\n💡 To run tests, start the server:")
        print("   uvicorn src.app:app --reload")
    
    print("="*80)


if __name__ == "__main__":
    main()

