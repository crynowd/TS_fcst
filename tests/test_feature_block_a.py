from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.block_a_dependence import (
    compute_acf_grid,
    compute_ljung_box,
    hurst_dfa,
    hurst_rs,
    integrated_acf,
    run_feature_block_a_pipeline,
    variance_ratio,
)


def _make_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def test_hurst_rs_on_simple_series() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, 3000)
    h, flags = hurst_rs(x)
    assert np.isfinite(h)
    assert 0.2 < h < 0.9
    assert "hurst_rs_too_short" not in flags


def test_hurst_dfa_on_simple_series() -> None:
    rng = np.random.default_rng(321)
    x = rng.normal(0.0, 1.0, 3000)
    h, flags = hurst_dfa(x)
    assert np.isfinite(h)
    assert 0.2 < h < 0.9
    assert "hurst_dfa_too_short" not in flags


def test_iacf_computation() -> None:
    white = np.random.default_rng(11).normal(0.0, 1.0, 2000)
    ar1 = _make_ar1(2000, phi=0.65, seed=22)
    lags = list(range(1, 21))

    acf_white = compute_acf_grid(white, lags)
    acf_ar1 = compute_acf_grid(ar1, lags)

    iacf_white = integrated_acf(acf_white, 20)
    iacf_ar1 = integrated_acf(acf_ar1, 20)

    assert np.isfinite(iacf_white)
    assert np.isfinite(iacf_ar1)
    assert iacf_ar1 > iacf_white


def test_ljung_box_output_ranges() -> None:
    x = np.random.default_rng(7).normal(0.0, 1.0, 1500)
    out = compute_ljung_box(x, [5, 10, 20, 50])
    for lag, (stat, p) in out.items():
        assert np.isfinite(stat), f"lag={lag}"
        assert np.isfinite(p), f"lag={lag}"
        assert 0.0 <= p <= 1.0, f"lag={lag}"


def test_variance_ratio_outputs() -> None:
    x = np.random.default_rng(9).normal(0.0, 1.0, 3000)
    for q in [2, 5, 10]:
        vr, flags = variance_ratio(x, q=q)
        assert np.isfinite(vr)
        assert 0.5 < vr < 1.5
        assert flags == []


def test_feature_block_a_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    artifacts_dir = root / "artifacts"
    processed_dir = artifacts_dir / "processed"
    features_dir = artifacts_dir / "features"
    reports_dir = artifacts_dir / "reports"
    logs_dir = artifacts_dir / "logs"
    manifests_dir = artifacts_dir / "manifests"

    configs_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    white = np.random.default_rng(101).normal(0.0, 1.0, n)
    ar1 = _make_ar1(n, phi=0.7, seed=202)

    returns_df = pd.DataFrame(
        {
            "series_id": ["RU:WN"] * n + ["US:AR1"] * n,
            "ticker": ["WN"] * n + ["AR1"] * n,
            "market": ["RU"] * n + ["US"] * n,
            "date": list(dates) + list(dates),
            "log_return": np.concatenate([white, ar1]),
            "dataset_profile": ["core_balanced"] * (2 * n),
        }
    )
    profiles_df = pd.DataFrame(
        [
            {"series_id": "RU:WN", "ticker": "WN", "market": "RU", "dataset_profile": "core_balanced"},
            {"series_id": "US:AR1", "ticker": "AR1", "market": "US", "dataset_profile": "core_balanced"},
        ]
    )

    log_returns_path = processed_dir / "log_returns_v1.parquet"
    profiles_path = processed_dir / "dataset_profiles_v1.parquet"
    returns_df.to_parquet(log_returns_path, index=False)
    profiles_df.to_parquet(profiles_path, index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "interim": str(artifacts_dir / "interim"),
            "processed": str(processed_dir),
            "features": str(features_dir),
            "reports": str(reports_dir),
            "logs": str(logs_dir),
            "manifests": str(manifests_dir),
        },
    }
    (configs_dir / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    stage_cfg = {
        "run_name": "feature_block_A_v1",
        "stage": "feature_block_A",
        "input": {
            "log_returns_parquet": "artifacts/processed/log_returns_v1.parquet",
            "dataset_profiles_parquet": "artifacts/processed/dataset_profiles_v1.parquet",
        },
        "dataset_profile": "core_balanced",
        "metrics": {
            "acf_lags": list(range(1, 21)) + [25, 30, 40, 50, 75, 100, 125, 150, 200, 250],
            "integrated_acf_windows": [5, 10, 20, 50, 100, 250],
            "ljung_box_lags": [5, 10, 20, 50],
            "variance_ratio_q": [2, 5, 10],
            "hurst_rs": {"min_window": 8, "max_window": 64, "num_windows": 8, "min_scales": 3},
            "hurst_dfa": {"min_window": 8, "max_window": 64, "num_windows": 8, "min_scales": 3},
        },
        "output": {"parquet_name": "features_block_A_v1.parquet", "excel_name": "features_block_A_v1.xlsx"},
    }
    cfg_path = configs_dir / "features_block_A_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_block_a_pipeline(str(cfg_path))

    out_df = pd.read_parquet(result.parquet_path)
    assert len(out_df) == 2
    assert {"series_id", "feature_status", "hurst_rs", "hurst_dfa", "vr_q2", "iacf_abs_1_20"}.issubset(out_df.columns)
    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()

    wn_iacf = float(out_df.loc[out_df["series_id"] == "RU:WN", "iacf_abs_1_20"].iloc[0])
    ar_iacf = float(out_df.loc[out_df["series_id"] == "US:AR1", "iacf_abs_1_20"].iloc[0])
    assert ar_iacf > wn_iacf
