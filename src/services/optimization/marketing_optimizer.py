"""
Marketing Optimizer - Local implementation using src.models.impact_model
"""
from typing import Dict, List, Optional, Any

from src.services.optimization.impact_model import MarketingImpactModel


class MarketingOptimizer:
    def __init__(self, impact_model: Optional[MarketingImpactModel] = None):
        self.impact_model = impact_model or MarketingImpactModel(channel_coefficients={})

    def optimize(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = 'gradient',
        digital_cap: float = 0.99,
        tv_cap: float = 0.5
    ) -> Dict[str, Any]:
        if self.impact_model is None:
            raise ValueError("MarketingImpactModel not initialized")
        return self.impact_model.optimize(
            total_budget=total_budget,
            channels=channels,
            method=method,
            digital_cap=digital_cap,
            tv_cap=tv_cap,
        )


__all__ = ['MarketingOptimizer']

