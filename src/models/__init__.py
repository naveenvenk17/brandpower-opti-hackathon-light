"""
Models loader - Only uses AutoGluon model from models/ directory
All model references point to models/colombia_round_1_weekly_model (AutoGluon)
"""
from pathlib import Path

# Base path for models directory - contains only AutoGluon model
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# AutoGluon model path (colombia_round_1_weekly_model)
AUTOGLUON_MODEL_PATH = MODELS_DIR

__all__ = ['MODELS_DIR', 'AUTOGLUON_MODEL_PATH']
