"""
Optimization module - Unified Marketing Budget Optimization

Production implementation with consolidated optimizer and GA fallback.
"""
from .unified_optimizer import (
    BrandPowerOptimizer,
    PowerPredictor,
    OptimizationRequest,
    OptimizationResult,
    OptimizationConstraints,
    OptimizationMode,
    OptimizationMethod,
    ConstraintValidator,
    hill_saturation,
)

# Keep GA optimizer for fallback functionality
from .optimizer_ga import optimize_weekly_spend

__all__ = [
    'BrandPowerOptimizer',
    'PowerPredictor',
    'OptimizationRequest',
    'OptimizationResult',
    'OptimizationConstraints',
    'OptimizationMode',
    'OptimizationMethod',
    'ConstraintValidator',
    'hill_saturation',
    'optimize_weekly_spend',
]

