from __future__ import annotations

from typing import Tuple

import pandas as pd


RANDOM_SEED = 42


def build_core_balanced_dataset(valid_series_df: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build `core_balanced` profile: all RU and equally sized US sample.

    Args:
        valid_series_df: Series-level dataframe with at least `series_id`, `ticker`, `market`.
        random_seed: Random seed for deterministic US sampling.

    Returns:
        Profile membership dataframe with `dataset_profile` column.
    """
    ru = valid_series_df[valid_series_df["market"] == "RU"].copy()
    us = valid_series_df[valid_series_df["market"] == "US"].copy()

    sample_size = min(len(ru), len(us))
    us_sample = us.sample(n=sample_size, random_state=random_seed) if sample_size > 0 else us.iloc[0:0].copy()

    profile = pd.concat([ru, us_sample], ignore_index=True)
    profile = profile.assign(dataset_profile="core_balanced")
    return profile


def build_extended_dataset(
    valid_series_df: pd.DataFrame,
    us_target: int = 800,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Build `extended_us_heavy` profile: all RU and up to `us_target` US sample."""
    ru = valid_series_df[valid_series_df["market"] == "RU"].copy()
    us = valid_series_df[valid_series_df["market"] == "US"].copy()

    us_take = min(us_target, len(us))
    us_sample = us.sample(n=us_take, random_state=random_seed) if us_take > 0 else us.iloc[0:0].copy()

    profile = pd.concat([ru, us_sample], ignore_index=True)
    profile = profile.assign(dataset_profile="extended_us_heavy")
    return profile


def build_dataset_profiles(
    valid_series_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
    extended_us_target: int = 800,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build both dataset profiles from valid series metadata."""
    core = build_core_balanced_dataset(valid_series_df, random_seed=random_seed)
    extended = build_extended_dataset(
        valid_series_df,
        us_target=extended_us_target,
        random_seed=random_seed,
    )
    return core, extended
