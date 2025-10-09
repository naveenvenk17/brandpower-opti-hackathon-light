"""
Genetic Algorithm (GA) based marketing spend optimizer
Migrated from hackathon_website_lightweight/production_scripts/services/optimizer_ga.py

This module implements a genetic algorithm to optimize weekly marketing spend allocation
across brands and channel groups to maximize total brand power.
"""
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.services.utils.channel_utils import (
    get_channel_groups,
    calculate_brand_power,
    colombia_megabrands as DEFAULT_MEGABRANDS
)


def _normalize_nonnegative(vector: np.ndarray) -> np.ndarray:
    """
    Ensure non-negative and sum to 1; fall back to uniform if all zeros.

    Args:
        vector: Input array to normalize

    Returns:
        Normalized array that sums to 1
    """
    vector = np.clip(vector, 0.0, None)
    s = vector.sum()
    if s <= 0:
        return np.full_like(vector, 1.0 / len(vector))
    return vector / s


def _build_weekly_plan_from_vector(
    vec: np.ndarray,
    brands: List[str],
    groups: List[str],
    weekly_budget: float,
    num_weeks: int = 48,
) -> pd.DataFrame:
    """
    Expand a brand-group share vector into a 48-week spend plan.

    Args:
        vec: Allocation vector (brand x group shares)
        brands: List of brand names
        groups: List of channel group names
        weekly_budget: Budget per week
        num_weeks: Number of weeks to plan (default: 48)

    Returns:
        DataFrame with columns: ['brand', 'week', 'channel', 'optimized_spend']
    """
    num_cells = len(brands) * len(groups)
    assert len(vec) == num_cells, f"Vector length {len(vec)} != expected {num_cells}"

    # Shares across brand-group cells for a single week
    shares = _normalize_nonnegative(vec)

    records = []
    for w in range(1, num_weeks + 1):
        for b_idx, brand in enumerate(brands):
            for g_idx, group in enumerate(groups):
                idx = b_idx * len(groups) + g_idx
                spend = weekly_budget * float(shares[idx])
                records.append({
                    'brand': brand,
                    'week': w,
                    'channel': group,
                    'optimized_spend': spend,
                })
    return pd.DataFrame.from_records(records)


def _plan_to_feature_frame(plan_df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    """
    Pivot plan into wide format with group columns for power calculation.

    Args:
        plan_df: Long-format plan DataFrame
        groups: List of channel group names

    Returns:
        Wide-format DataFrame with one row per brand-week
    """
    # One row per brand-week with group columns
    wide = plan_df.pivot_table(
        index=['brand', 'week'],
        columns='channel',
        values='optimized_spend',
        aggfunc='sum',
        fill_value=0.0,
    ).reset_index()

    # Ensure all expected group columns exist
    for g in groups:
        if g not in wide.columns:
            wide[g] = 0.0

    # Order columns roughly
    cols = ['brand', 'week'] + groups
    wide = wide[cols]
    return wide


def _evaluate_power_proxy(plan_df: pd.DataFrame, groups: List[str]) -> float:
    """
    Fast proxy objective using calculate_brand_power utility.

    Args:
        plan_df: Plan DataFrame in long format
        groups: List of channel group names

    Returns:
        Total brand power score (objective to maximize)
    """
    wide = _plan_to_feature_frame(plan_df, groups)
    # calculate_brand_power expects feature columns; returns a df with 'power'
    scored = calculate_brand_power(wide)
    if scored is None or 'power' not in scored.columns:
        return -1e9
    # Objective: maximize total power across all brand-weeks
    return float(scored['power'].sum())


def optimize_weekly_spend(
    total_spend: float,
    historical_spend_df: pd.DataFrame,
    megabrands: Optional[List[str]] = None,
    num_weeks: int = 48,
    pop_size: int = 18,
    generations: int = 14,
    mutation_rate: float = 0.15,
    elite_frac: float = 0.25,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Optimize weekly allocation shares across brand x group using a genetic algorithm.

    The GA optimizes the distribution of marketing spend across brands and channel groups
    to maximize total brand power over a specified planning horizon.

    Args:
        total_spend: Total budget to allocate over all weeks
        historical_spend_df: Historical spending data (not currently used, for future enhancement)
        megabrands: List of brands to optimize for (defaults to Colombia megabrands)
        num_weeks: Planning horizon in weeks (default: 48)
        pop_size: Population size for GA (default: 18)
        generations: Number of GA generations (default: 14)
        mutation_rate: Probability of mutation (default: 0.15)
        elite_frac: Fraction of population to preserve as elites (default: 0.25)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame with columns: ['brand', 'week', 'channel', 'optimized_spend']
        representing the optimal weekly spend allocation plan
    """
    rng = np.random.default_rng(random_seed)

    channel_groups = get_channel_groups()  # dict: group -> [components]
    groups = list(channel_groups.keys())
    brands = megabrands if megabrands else list(DEFAULT_MEGABRANDS)

    # Weekly budget: distribute total evenly across weeks
    weekly_budget = float(total_spend) / float(num_weeks)

    dim = len(brands) * len(groups)

    # Initialize population using Dirichlet distribution (ensures sum to 1)
    population = rng.dirichlet(alpha=np.ones(dim), size=pop_size)

    def fitness(vec: np.ndarray) -> float:
        """Evaluate fitness of a solution vector"""
        plan = _build_weekly_plan_from_vector(vec, brands, groups, weekly_budget, num_weeks)
        return _evaluate_power_proxy(plan, groups)

    # Evaluate initial population
    fitness_scores = np.array([fitness(ind) for ind in population], dtype=float)

    elites_n = max(1, int(pop_size * elite_frac))

    # GA main loop
    for generation in range(generations):
        # Select elites (best individuals)
        elite_indices = np.argsort(fitness_scores)[-elites_n:]
        elites = population[elite_indices]

        # Produce offspring by blend crossover of elites
        offspring = []
        while len(offspring) < pop_size - elites_n:
            p1, p2 = elites[rng.integers(0, len(elites))], elites[rng.integers(0, len(elites))]
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2

            # Mutation: add gaussian noise then renormalize
            if rng.random() < mutation_rate:
                noise = rng.normal(loc=0.0, scale=0.05, size=dim)
                child = child + noise

            child = _normalize_nonnegative(child)
            offspring.append(child)

        # Form new population: elites + offspring
        population = np.vstack([elites] + [np.array(offspring)])
        fitness_scores = np.array([fitness(ind) for ind in population], dtype=float)

    # Best individual from final population
    best_idx = int(np.argmax(fitness_scores))
    best_vec = population[best_idx]
    best_plan = _build_weekly_plan_from_vector(best_vec, brands, groups, weekly_budget, num_weeks)

    return best_plan


__all__ = ['optimize_weekly_spend']
