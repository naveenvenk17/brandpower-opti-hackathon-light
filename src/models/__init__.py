"""
Models loader - Simple .pkl file loading
Only contains model loading logic, no business logic
"""
import pickle
from pathlib import Path

# Base path for models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Load brand power forecasting model
with open(MODELS_DIR / 'brand_power_forecaster.pkl', 'rb') as f:
    brand_power_forecaster = pickle.load(f)

# Load unified model (if needed)
with open(MODELS_DIR / 'unified_brand_power_model.pkl', 'rb') as f:
    unified_model = pickle.load(f)

__all__ = ['brand_power_forecaster', 'unified_model', 'MODELS_DIR']
