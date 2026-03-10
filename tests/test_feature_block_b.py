from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.block_b_spectrum import (
    lz_complexity,
    noise_fn,
    permutation_entropy,
    run_feature_block_b_pipeline,
    sample_entropy,
    spectral_entropy,
    spectral_flatness,
)


def _default_spectral_cfg() -> dict:
    return {
        "method": "welch",
        "welch_nperseg": 128,
        "regression_freq_min_ratio": 0.02,
        "regression_freq_max_ratio": 1.0,
        "min_regression_points": 8,
        "entropy_normalized": True,
    }


def _make_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def test_spectral_entropy_range() -> None:
    x = np.random.default_rng(1).normal(0.0, 1.0, 2048)
    val, flags = spectral_entropy(x - x.mean(), _default_spectral_cfg())
    assert flags == []
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_spectral_flatness_nonnegative() -> None:
    x = np.random.default_rng(2).normal(0.0, 1.0, 2048)
    val, flags = spectral_flatness(x - x.mean(), _default_spectral_cfg())
    assert flags == []
    assert np.isfinite(val)
    assert val >= 0.0


def test_noise_fn_simple_series() -> None:
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    val, flags = noise_fn(x)
    expected = np.std(np.diff(x), ddof=1) / np.std(x, ddof=1)
    assert flags == []
    assert np.isfinite(val)
    assert np.isclose(val, expected)


def test_permutation_entropy_range_or_validity() -> None:
    x = np.random.default_rng(3).normal(0.0, 1.0, 1024)
    val, flags = permutation_entropy(x, order=3, delay=1, normalized=True)
    assert flags == []
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_lz_complexity_validity() -> None:
    x = np.random.default_rng(4).normal(0.0, 1.0, 1024)
    val, flags = lz_complexity(x, binarization="median", normalized=True)
    assert flags == []
    assert np.isfinite(val)
    assert val >= 0.0


def test_sample_entropy_validity() -> None:
    x = np.random.default_rng(5).normal(0.0, 1.0, 1500)
    val, flags = sample_entropy(x, m=2, r_ratio=0.2)
    assert flags == []
    assert np.isfinite(val)
    assert val >= 0.0


def test_block_b_synthetic_sanity_white_vs_sine() -> None:
    n = 2048
    t = np.arange(n)
    white = np.random.default_rng(6).normal(0.0, 1.0, n)
    sine = np.sin(2.0 * np.pi * t / 64.0)

    flat_white, _ = spectral_flatness(white - white.mean(), _default_spectral_cfg())
    flat_sine, _ = spectral_flatness(sine - sine.mean(), _default_spectral_cfg())

    ent_white, _ = spectral_entropy(white - white.mean(), _default_spectral_cfg())
    ent_sine, _ = spectral_entropy(sine - sine.mean(), _default_spectral_cfg())

    assert np.isfinite(flat_white) and np.isfinite(flat_sine)
    assert np.isfinite(ent_white) and np.isfinite(ent_sine)
    assert flat_white > flat_sine
    assert ent_white > ent_sine


def test_feature_block_b_small_e2e(tmp_path: Path) -> None:
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

    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
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
        "run_name": "feature_block_B_v1",
        "stage": "feature_block_B",
        "input": {
            "log_returns_parquet": "artifacts/processed/log_returns_v1.parquet",
            "dataset_profiles_parquet": "artifacts/processed/dataset_profiles_v1.parquet",
        },
        "dataset_profile": "core_balanced",
        "metrics": {
            "spectral": _default_spectral_cfg(),
            "permutation_entropy": {"order": 3, "delay": 1, "normalized": True},
            "lz_complexity": {"binarization": "median", "normalized": True},
            "sample_entropy": {"m": 2, "r_ratio": 0.2},
        },
        "output": {
            "parquet_name": "features_block_B_v1.parquet",
            "excel_name": "features_block_B_v1.xlsx",
            "metric_columns": [
                "spectral_slope_beta",
                "spectral_entropy",
                "spectral_flatness",
                "noise_fn",
                "permutation_entropy",
                "lz_complexity",
                "sample_entropy",
            ],
        },
    }
    cfg_path = configs_dir / "features_block_B_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_block_b_pipeline(str(cfg_path))

    out_df = pd.read_parquet(result.parquet_path)
    assert len(out_df) == 2
    assert {
        "series_id",
        "feature_status",
        "spectral_slope_beta",
        "spectral_entropy",
        "spectral_flatness",
        "noise_fn",
        "permutation_entropy",
        "lz_complexity",
        "sample_entropy",
    }.issubset(out_df.columns)

    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()

    wn_flat = float(out_df.loc[out_df["series_id"] == "RU:WN", "spectral_flatness"].iloc[0])
    ar_flat = float(out_df.loc[out_df["series_id"] == "US:AR1", "spectral_flatness"].iloc[0])
    assert wn_flat > ar_flat
