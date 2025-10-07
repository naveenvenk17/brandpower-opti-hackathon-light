"""
Lightweight Marketing Impact Model for Fast Optimization

This model provides clean marketing lift calculation without baseline contamination.
It's 60x faster than full unified forecaster for optimization purposes.

Architecture:
- Extracts Ridge coefficients from trained unified model
- Uses only marketing features (saturated channels)
- No recursive forecasting overhead
- Clean gradient computation for SLSQP

Usage:
    # Extract from trained model
    impact_model = MarketingImpactModel.from_unified_forecaster(unified_model)

    # Fast optimization
    optimal_allocation = impact_model.optimize(total_budget=5_000_000)

    # Clean lift calculation
    lift = impact_model.predict_lift({'opentv': 1_500_000, 'digital': 800_000})
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.optimize import minimize, differential_evolution
import logging

from production_scripts.data.feature_engineering import hill_saturation, SATURATION_PARAMS

logger = logging.getLogger(__name__)


class MarketingImpactModel:
    """
    Lightweight marketing impact model for fast optimization

    Provides:
    - Clean lift calculation (no baseline contamination)
    - Fast optimization (60x faster than unified forecaster)
    - Smooth gradients (only Ridge component, no LGBM noise)
    """

    def __init__(
        self,
        channel_coefficients: Dict[str, float],
        saturation_params: Optional[Dict[str, any]] = None
    ):
        """
        Initialize marketing impact model

        Args:
            channel_coefficients: Dict of {channel: ridge_coefficient}
            saturation_params: Saturation curve parameters (alpha, K_scale)
        """
        self.coefs = channel_coefficients
        self.saturation_params = saturation_params or SATURATION_PARAMS

        logger.info(f"MarketingImpactModel initialized with {len(self.coefs)} channels")

    @classmethod
    def from_unified_forecaster(cls, unified_forecaster):
        """
        Extract marketing impact model from trained unified forecaster

        Args:
            unified_forecaster: Trained UnifiedForecaster instance

        Returns:
            MarketingImpactModel with extracted coefficients
        """
        # Extract Ridge coefficients for marketing features
        ridge = unified_forecaster.ridge
        features = unified_forecaster.features

        channel_coefs = {}
        for i, feature in enumerate(features):
            if '_saturated' in feature and 'lag' not in feature:
                channel = feature.replace('_saturated', '')
                channel_coefs[channel] = ridge.coef_[i]

        logger.info(f"Extracted {len(channel_coefs)} channel coefficients from unified forecaster")

        return cls(
            channel_coefficients=channel_coefs,
            saturation_params=unified_forecaster.saturation_params
        )

    def predict_lift(self, allocation: Dict[str, float]) -> float:
        """
        Predict pure marketing lift (no baseline contamination)

        Args:
            allocation: Dict of {channel: spend}

        Returns:
            Total marketing lift (power increase)
        """
        total_lift = 0.0

        for channel, spend in allocation.items():
            if channel in self.coefs:
                # Apply saturation
                saturated = hill_saturation(
                    np.array([spend]),
                    alpha=self.saturation_params['alpha'],
                    K=self.saturation_params['K_scale']
                )[0]

                # Apply coefficient
                lift = self.coefs[channel] * saturated
                total_lift += lift

        return total_lift

    def optimize(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        method: str = 'gradient',
        digital_cap: float = 0.99,
        tv_cap: float = 0.5
    ) -> Dict[str, any]:
        """
        Optimize marketing allocation (FAST - no full model calls)

        Args:
            total_budget: Total marketing budget
            channels: List of channels to optimize (default: all with non-zero coefs)
            method: 'gradient' or 'evolutionary'
            digital_cap: Maximum fraction of budget for digital (default 99%)
            tv_cap: Maximum fraction of budget for TV (default 50%)

        Returns:
            {
                'optimal_allocation': Dict[str, float],
                'expected_lift': float,
                'roi': float,
                'method': str
            }
        """
        # Default to channels with positive coefficients
        if channels is None:
            channels = [ch for ch, coef in self.coefs.items() if coef > 0]

        # Channel classification
        digital_channels = ['digitaldisplayandsearch', 'digitalvideo', 'meta', 'twitter',
                           'youtube', 'tiktok', 'streamingaudio']
        tv_channels = ['opentv', 'paytv']

        digital_indices = [i for i, ch in enumerate(channels) if ch in digital_channels]
        tv_indices = [i for i, ch in enumerate(channels) if ch in tv_channels]

        if method == 'gradient':
            return self._optimize_gradient(
                total_budget, channels, digital_indices, tv_indices, digital_cap, tv_cap
            )
        elif method == 'evolutionary':
            return self._optimize_evolutionary(
                total_budget, channels, digital_indices, tv_indices, digital_cap, tv_cap
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _optimize_gradient(
        self,
        total_budget: float,
        channels: List[str],
        digital_indices: List[int],
        tv_indices: List[int],
        digital_cap: float,
        tv_cap: float
    ) -> Dict[str, any]:
        """Gradient-based optimization using SLSQP"""

        def objective(x):
            allocation = {ch: spend for ch, spend in zip(channels, x)}
            lift = self.predict_lift(allocation)
            return -lift  # Minimize negative lift

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
        ]

        if digital_indices:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: digital_cap * total_budget - np.sum([x[i] for i in digital_indices])
            })

        if tv_indices:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: tv_cap * total_budget - np.sum([x[i] for i in tv_indices])
            })

        bounds = [(0, total_budget) for _ in range(len(channels))]
        x0 = np.array([total_budget / len(channels)] * len(channels))

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        optimal_allocation = {ch: spend for ch, spend in zip(channels, result.x)}
        expected_lift = self.predict_lift(optimal_allocation)
        roi = expected_lift / (total_budget / 1_000_000) if total_budget > 0 else 0

        logger.info(f"Gradient optimization: Lift={expected_lift:.4f}, ROI={roi:.2f}x")

        return {
            'optimal_allocation': optimal_allocation,
            'expected_lift': expected_lift,
            'roi': roi,
            'method': 'gradient',
            'total_budget': total_budget
        }

    def _optimize_evolutionary(
        self,
        total_budget: float,
        channels: List[str],
        digital_indices: List[int],
        tv_indices: List[int],
        digital_cap: float,
        tv_cap: float
    ) -> Dict[str, any]:
        """Evolutionary optimization using differential evolution"""

        def objective(x):
            # Normalize to budget
            x_normalized = x / np.sum(x) * total_budget

            # Penalty for constraint violations
            penalty = 0

            if digital_indices:
                digital_spend = np.sum([x_normalized[i] for i in digital_indices])
                if digital_spend > digital_cap * total_budget:
                    penalty += 1000 * (digital_spend - digital_cap * total_budget)

            if tv_indices:
                tv_spend = np.sum([x_normalized[i] for i in tv_indices])
                if tv_spend > tv_cap * total_budget:
                    penalty += 1000 * (tv_spend - tv_cap * total_budget)

            allocation = {ch: spend for ch, spend in zip(channels, x_normalized)}
            lift = self.predict_lift(allocation)

            return -lift + penalty

        bounds = [(0, total_budget) for _ in range(len(channels))]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            seed=42
        )

        # Normalize to exact budget
        x_normalized = result.x / np.sum(result.x) * total_budget
        optimal_allocation = {ch: spend for ch, spend in zip(channels, x_normalized)}
        expected_lift = self.predict_lift(optimal_allocation)
        roi = expected_lift / (total_budget / 1_000_000) if total_budget > 0 else 0

        logger.info(f"Evolutionary optimization: Lift={expected_lift:.4f}, ROI={roi:.2f}x")

        return {
            'optimal_allocation': optimal_allocation,
            'expected_lift': expected_lift,
            'roi': roi,
            'method': 'evolutionary',
            'total_budget': total_budget
        }

    def get_marginal_roi(self, channel: str, current_spend: float, delta: float = 100_000) -> float:
        """
        Calculate marginal ROI for a channel at current spend level

        Args:
            channel: Channel name
            current_spend: Current spend on channel
            delta: Incremental spend to test (default $100k)

        Returns:
            Marginal ROI (lift per $1M incremental spend)
        """
        current_lift = self.predict_lift({channel: current_spend})
        new_lift = self.predict_lift({channel: current_spend + delta})

        marginal_lift = new_lift - current_lift
        marginal_roi = marginal_lift / (delta / 1_000_000)

        return marginal_roi

    def get_channel_saturation_curve(
        self,
        channel: str,
        spend_range: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get saturation curve for a channel

        Args:
            channel: Channel name
            spend_range: Array of spend values to evaluate (default: 0 to 5M)

        Returns:
            DataFrame with columns: spend, saturated, lift, marginal_roi
        """
        if spend_range is None:
            spend_range = np.linspace(0, 5_000_000, 100)

        results = []
        for spend in spend_range:
            saturated = hill_saturation(
                np.array([spend]),
                alpha=self.saturation_params['alpha'],
                K=self.saturation_params['K_scale']
            )[0]

            lift = self.coefs.get(channel, 0) * saturated
            marginal_roi = self.get_marginal_roi(channel, spend, delta=100_000)

            results.append({
                'spend': spend,
                'saturated': saturated,
                'lift': lift,
                'marginal_roi': marginal_roi
            })

        return pd.DataFrame(results)
