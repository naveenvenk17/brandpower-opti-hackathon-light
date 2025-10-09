#!/usr/bin/env python3
"""
Verify that the app loads correctly with all endpoints registered
"""
import sys

print("="*80)
print("VERIFYING APP CONFIGURATION")
print("="*80)
print()

print("Step 1: Importing app...")
try:
    from src.app import app
    print("✅ App imported successfully")
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    sys.exit(1)

print()
print("Step 2: Checking optimizer imports...")
try:
    from src.services.optimization import (
        BrandPowerOptimizer,
        PowerPredictor,
        OptimizationRequest,
        optimize_weekly_spend
    )
    print("✅ All optimizer components available")
except Exception as e:
    print(f"❌ Failed to import optimizer: {e}")
    sys.exit(1)

print()
print("Step 3: Listing registered endpoints...")
routes = []
for route in app.routes:
    if hasattr(route, 'path') and hasattr(route, 'methods'):
        routes.append((route.path, list(route.methods)))

# Group by prefix
api_routes = [(p, m) for p, m in routes if p.startswith('/api/')]
ui_routes = [(p, m) for p, m in routes if not p.startswith('/api/') and p != '/']

print()
print(f"✅ Found {len(routes)} total routes:")
print(f"   - {len(api_routes)} API endpoints")
print(f"   - {len(ui_routes)} UI routes")

print()
print("Step 4: Checking optimizer endpoints...")
optimizer_endpoints = [
    '/api/v1/optimize/brand-power',
    '/api/optimizer/ga',
    '/api/v1/optimize/allocation'
]

missing = []
for endpoint in optimizer_endpoints:
    found = any(p == endpoint for p, _ in routes)
    status = "✅" if found else "❌"
    print(f"  {status} {endpoint}")
    if not found:
        missing.append(endpoint)

if missing:
    print()
    print(f"⚠️  WARNING: {len(missing)} endpoint(s) missing!")
    print("   This may indicate an import or registration error.")
else:
    print()
    print("✅ All optimizer endpoints registered correctly!")

print()
print("Step 5: Testing endpoint function...")
try:
    # Import the endpoint function directly
    from src.app import api_brand_power_optimizer
    print("✅ Endpoint function is defined and importable")
except Exception as e:
    print(f"❌ Cannot import endpoint function: {e}")
    sys.exit(1)

print()
print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print()

if not missing:
    print("✅ App is correctly configured!")
    print()
    print("To start the server:")
    print("  python src/app.py")
    print()
    print("Then test with:")
    print("  python test_endpoint.py")
else:
    print("⚠️  App has configuration issues.")
    print("   Please check the errors above.")

print()

