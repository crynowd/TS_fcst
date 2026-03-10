from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset_profiles import build_core_balanced_dataset, build_extended_dataset
from src.data.log_returns import apply_series_length_policy, compute_log_returns


def test_compute_log_returns_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [100.0, 110.0, 121.0],
        }
    )

    out = compute_log_returns(df)

    expected = np.array([np.log(110.0 / 100.0), np.log(121.0 / 110.0)])
    np.testing.assert_allclose(out["log_return"].to_numpy(), expected)
    assert len(out) == 2


def test_apply_series_length_policy() -> None:
    long_df = pd.DataFrame({"log_return": np.ones(2100)})
    selected_long, meta_long = apply_series_length_policy(long_df, target_length=2000, min_length=1500)
    assert len(selected_long) == 2000
    assert meta_long["status"] == "valid_target"
    assert meta_long["short_series"] is False

    short_df = pd.DataFrame({"log_return": np.ones(1700)})
    selected_short, meta_short = apply_series_length_policy(short_df, target_length=2000, min_length=1500)
    assert len(selected_short) == 1700
    assert meta_short["status"] == "valid_min_only"
    assert meta_short["short_series"] is True

    excluded_df = pd.DataFrame({"log_return": np.ones(1499)})
    selected_excluded, meta_excluded = apply_series_length_policy(
        excluded_df,
        target_length=2000,
        min_length=1500,
    )
    assert selected_excluded.empty
    assert meta_excluded["status"] == "excluded_too_short"


def test_core_sampling_is_reproducible_seed_42() -> None:
    valid_series_df = pd.DataFrame(
        [
            {"series_id": "RU:AAA", "ticker": "AAA", "market": "RU"},
            {"series_id": "RU:BBB", "ticker": "BBB", "market": "RU"},
            {"series_id": "US:A", "ticker": "A", "market": "US"},
            {"series_id": "US:B", "ticker": "B", "market": "US"},
            {"series_id": "US:C", "ticker": "C", "market": "US"},
            {"series_id": "US:D", "ticker": "D", "market": "US"},
            {"series_id": "US:E", "ticker": "E", "market": "US"},
        ]
    )

    out1 = build_core_balanced_dataset(valid_series_df, random_seed=42)
    out2 = build_core_balanced_dataset(valid_series_df, random_seed=42)

    us1 = sorted(out1.loc[out1["market"] == "US", "series_id"].tolist())
    us2 = sorted(out2.loc[out2["market"] == "US", "series_id"].tolist())
    assert us1 == us2


def test_dataset_profile_sizes() -> None:
    ru_rows = [
        {"series_id": f"RU:R{i}", "ticker": f"R{i}", "market": "RU"}
        for i in range(3)
    ]
    us_rows = [
        {"series_id": f"US:U{i}", "ticker": f"U{i}", "market": "US"}
        for i in range(1000)
    ]
    valid_series_df = pd.DataFrame(ru_rows + us_rows)

    core = build_core_balanced_dataset(valid_series_df, random_seed=42)
    extended = build_extended_dataset(valid_series_df, us_target=800, random_seed=42)

    assert len(core) == 6
    assert int((core["market"] == "RU").sum()) == 3
    assert int((core["market"] == "US").sum()) == 3

    assert len(extended) == 803
    assert int((extended["market"] == "RU").sum()) == 3
    assert int((extended["market"] == "US").sum()) == 800
