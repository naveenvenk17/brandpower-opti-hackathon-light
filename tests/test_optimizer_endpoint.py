"""
Quick test of the Brand Power Optimizer API endpoint
"""
import requests
import json

# Test the endpoint
url = "http://localhost:8000/api/v1/optimize/brand-power"

payload = {
    "total_budget": 1000000000,
    "brands": ["AGUILA", "FAMILIA POKER", "FAMILIA CORONA", "FAMILIA CLUB COLOMBIA"],
    "quarters": ["2024 Q3", "2024 Q4", "2025 Q1", "2025 Q2"],
    "mode": "all_brands",
    "method": "gradient",
    "constraints": {
        "paytv_max_pct": 0.5
    }
}

print("Testing Brand Power Optimizer API...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print("\nSending request...")

try:
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS!")
        print(f"\nPower Uplift: +{data['total_uplift_pct']:.2f}%")
        print(f"Total Baseline Power: {data['total_baseline_power']:.2f}")
        print(f"Total Optimized Power: {data['total_optimized_power']:.2f}")
        print(f"Constraints Satisfied: {data['constraints_satisfied']}")
        
        print("\nBudget Allocation:")
        for brand, budget in data['budget_allocation'].items():
            allocation = data['optimal_allocation'][brand]
            paytv_pct = (allocation['paytv'] / budget * 100) if budget > 0 else 0
            print(f"  {brand}: ${budget:,.0f}")
            print(f"    - PayTV: {paytv_pct:.1f}%")
            print(f"    - Wholesalers: {100-paytv_pct:.1f}%")
        
    else:
        print(f"\n❌ ERROR:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to server")
    print("Make sure the app is running: python -m uvicorn src.app:app --reload")
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")

