"""
Lightweight Marketing Impact Model for optimization (no production_scripts deps)

This module contains a self-contained implementation of the marketing impact
model and the saturation utility it depends on. It is API-compatible with the
previous implementation used under production_scripts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import logging


logger = logging.getLogger(__name__)


# --- Saturation utility (replaces production_scripts.data.feature_engineering) ---
SATURATION_PARAMS: Dict[str, Any] = {
    "alpha": 0.5,     # curvature of the saturation curve
    "K_scale": 1_000_000.0,  # half-saturation point scaling
}


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """Hill-type saturation transform applied element-wise.

    f(x) = x^alpha / (K^alpha + x^alpha)
    """
    x = np.asarray(x, dtype=float)
    x_alpha = np.power(np.maximum(x, 0.0), alpha)
    denom = np.power(K, alpha) + x_alpha
    # Avoid division by zero, though K>0 makes denom>0
    denom = np.where(denom == 0.0, 1.0, denom)
    return x_alpha / denom


class MarketingImpactModel:
    """Lightweight marketing impact model for fast optimization."""

    def __init__(
        self,
        channel_coefficients: Dict[str, float],
        saturation_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.coefs: Dict[str, float] = dict(channel_coefficients)
        self.saturation_params: Dict[str, Any] = {
            **SATURATION_PARAMS,
            **(saturation_params or {}),
        }
        logger.info("MarketingImpactModel initialized with %d channels", len(self.coefs))

    @classmethod
    def from_unified_forecaster(cls, unified_forecaster: Any) -> "MarketingImpactModel":
        """Compat constructor kept for API parity; not used in this project runtime."""
        channel_coefs: Dict[str, float] = {}
        # If unified_forecaster has attributes, try to extract saturated channel coefs
        ridge = getattr(unified_forecaster, "ridge", None)
        features = getattr(unified_forecaster, "features", [])
        if ridge is not None and hasattr(ridge, "coef_") and features:
            for i, feature in enumerate(features):
                if "_saturated" in feature and "lag" not in feature:
                    channel = feature.replace("_saturated", "")
                    try:
                        channel_coefs[channel] = float(ridge.coef_[i])
                    except Exception:
                        continue
        return cls(channel_coefficients=channel_coefs, saturation_params=getattr(unified_forecaster, "saturation_params", None))

    def predict_lift(self, allocation: Dict[str, float]) -> float:
        total_lift = 0.0
        alpha = float(self.saturation_params["alpha"])
        K = float(self.saturation_params["K_scale"])
        for channel, spend in allocation.items():
            coef = self.coefs.get(channel, 0.0)
            if coef == 0.0:
                continue
            saturated = hill_saturation(np.array([spend], dtype=float), alpha=alpha, K=K)[0]
            total_lift += coef * saturated
        return float(total_lift)

    def optimize(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = "gradient",
        digital_cap: float = 0.99,
        tv_cap: float = 0.5,
    ) -> Dict[str, Any]:
        if channels is None:
            channels = [ch for ch, coef in self.coefs.items() if coef > 0]

        digital_channels = [
            "digitaldisplayandsearch",
            "digitalvideo",
            "meta",
            "twitter",
            "youtube",
            "tiktok",
            "streamingaudio",
        ]
        tv_channels = ["opentv", "paytv"]

        digital_indices = [i for i, ch in enumerate(channels) if ch in digital_channels]
        tv_indices = [i for i, ch in enumerate(channels) if ch in tv_channels]

        if method == "gradient":
            return self._optimize_gradient(total_budget, channels, digital_indices, tv_indices, digital_cap, tv_cap)
        elif method == "evolutionary":
            return self._optimize_evolutionary(total_budget, channels, digital_indices, tv_indices, digital_cap, tv_cap)
        else:
            raise ValueError(f"Unknown method: {method}")

    # --- Optimization backends ---
    def _optimize_gradient(
        self,
        total_budget: float,
        channels: List[str],
        digital_indices: List[int],
        tv_indices: List[int],
        digital_cap: float,
        tv_cap: float,
    ) -> Dict[str, Any]:
        def objective(x: np.ndarray) -> float:
            allocation = {ch: spend for ch, spend in zip(channels, x)}
            return -self.predict_lift(allocation)

        constraints: List[Dict[str, Any]] = [
            {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
        ]
        if digital_indices:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: digital_cap * total_budget - np.sum([x[i] for i in digital_indices]),
            })
        if tv_indices:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: tv_cap * total_budget - np.sum([x[i] for i in tv_indices]),
            })

        bounds = [(0.0, total_budget) for _ in channels]
        x0 = np.array([total_budget / max(len(channels), 1)] * len(channels))

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        optimal_allocation = {ch: float(spend) for ch, spend in zip(channels, result.x)}
        expected_lift = float(self.predict_lift(optimal_allocation))
        roi = expected_lift / (total_budget / 1_000_000.0) if total_budget > 0 else 0.0
        return {
            "optimal_allocation": optimal_allocation,
            "expected_lift": expected_lift,
            "roi": roi,
            "method": "gradient",
            "total_budget": float(total_budget),
        }

    def _optimize_evolutionary(
        self,
        total_budget: float,
        channels: List[str],
        digital_indices: List[int],
        tv_indices: List[int],
        digital_cap: float,
        tv_cap: float,
    ) -> Dict[str, Any]:
        def objective(x: np.ndarray) -> float:
            x_sum = float(np.sum(x)) or 1.0
            x_normalized = x / x_sum * total_budget

            penalty = 0.0
            if digital_indices:
                digital_spend = float(np.sum([x_normalized[i] for i in digital_indices]))
                if digital_spend > digital_cap * total_budget:
                    penalty += 1000.0 * (digital_spend - digital_cap * total_budget)
            if tv_indices:
                tv_spend = float(np.sum([x_normalized[i] for i in tv_indices]))
                if tv_spend > tv_cap * total_budget:
                    penalty += 1000.0 * (tv_spend - tv_cap * total_budget)

            allocation = {ch: float(spend) for ch, spend in zip(channels, x_normalized)}
            lift = self.predict_lift(allocation)
            return -lift + penalty

        bounds = [(0.0, total_budget) for _ in channels]
        result = differential_evolution(objective, bounds, maxiter=50, seed=42)
        x_sum = float(np.sum(result.x)) or 1.0
        x_normalized = result.x / x_sum * total_budget
        optimal_allocation = {ch: float(spend) for ch, spend in zip(channels, x_normalized)}
        expected_lift = float(self.predict_lift(optimal_allocation))
        roi = expected_lift / (total_budget / 1_000_000.0) if total_budget > 0 else 0.0
        return {
            "optimal_allocation": optimal_allocation,
            "expected_lift": expected_lift,
            "roi": roi,
            "method": "evolutionary",
            "total_budget": float(total_budget),
        }


__all__ = ["MarketingImpactModel", "hill_saturation", "SATURATION_PARAMS"]


