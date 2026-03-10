from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.block_d_chaos import (
    compute_lyapunov_time,
    run_feature_block_d_pipeline,
    select_delay_tau,
    select_embedding_dimension_fnn,
)


def _make_logistic_map(n: int, r: float = 3.9, x0: float = 0.12345) -> np.ndarray:
    x = np.empty(n, dtype=float)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def test_delay_selection_returns_positive_tau() -> None:
    n = 800
    t = np.arange(n, dtype=float)
    x = np.sin(2.0 * np.pi * t / 32.0)
    tau, method, _, flags = select_delay_tau(
        x,
        {
            "tau_min": 1,
            "tau_max": 40,
            "ami_bins": 20,
        },
    )
    assert isinstance(tau, int)
    assert tau > 0
    assert method in {"ami", "acf_zero", "acf_einv", "fallback_default"}
    assert "tau_selection_series_too_short" not in flags


def test_fnn_returns_dimension_in_range() -> None:
    x = _make_logistic_map(1500)
    m, success, ratios, flags = select_embedding_dimension_fnn(
        x,
        tau=1,
        emb_cfg={
            "m_min": 2,
            "m_max": 8,
            "fnn_threshold": 0.05,
            "fnn_rtol": 10.0,
            "fnn_atol_std": 2.0,
        },
    )
    assert np.isfinite(m), flags
    assert 2 <= int(m) <= 8
    assert isinstance(success, bool)
    assert len(ratios) > 0


def test_lyapunov_time_nan_for_nonpositive_exponent() -> None:
    val_zero, flags_zero = compute_lyapunov_time(0.0)
    val_neg, flags_neg = compute_lyapunov_time(-0.1)
    assert np.isnan(val_zero)
    assert np.isnan(val_neg)
    assert "lyapunov_time_nonpositive_lle" in flags_zero
    assert "lyapunov_time_nonpositive_lle" in flags_neg


def test_block_D_pipeline_small_dataset(tmp_path: Path) -> None:
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

    n = 900
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    white = rng.normal(0.0, 1.0, n)
    sine = np.sin(2.0 * np.pi * np.arange(n) / 36.0)
    logistic = _make_logistic_map(n)

    returns_df = pd.DataFrame(
        {
            "series_id": ["RU:WHITE"] * n + ["US:SINE"] * n + ["US:LOGI"] * n,
            "ticker": ["WHITE"] * n + ["SINE"] * n + ["LOGI"] * n,
            "market": ["RU"] * n + ["US"] * n + ["US"] * n,
            "date": list(dates) + list(dates) + list(dates),
            "log_return": np.concatenate([white, sine, logistic]),
            "dataset_profile": ["core_balanced"] * (3 * n),
        }
    )
    profiles_df = pd.DataFrame(
        [
            {"series_id": "RU:WHITE", "ticker": "WHITE", "market": "RU", "dataset_profile": "core_balanced"},
            {"series_id": "US:SINE", "ticker": "SINE", "market": "US", "dataset_profile": "core_balanced"},
            {"series_id": "US:LOGI", "ticker": "LOGI", "market": "US", "dataset_profile": "core_balanced"},
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
        "run_name": "feature_block_D_v1",
        "stage": "feature_block_D",
        "input": {
            "log_returns_parquet": "artifacts/processed/log_returns_v1.parquet",
            "dataset_profiles_parquet": "artifacts/processed/dataset_profiles_v1.parquet",
        },
        "dataset_profile": "core_balanced",
        "metrics": {
            "minimum_series_length": 300,
            "delay_selection": {
                "method": "ami_with_fallbacks",
                "tau_min": 1,
                "tau_max": 40,
                "ami_bins": 20,
            },
            "embedding": {
                "m_min": 2,
                "m_max": 8,
                "fnn_threshold": 0.05,
                "fnn_rtol": 10.0,
                "fnn_atol_std": 2.0,
            },
            "correlation_dimension": {
                "enabled": True,
                "radius_grid_size": 20,
                "min_scaling_points": 5,
                "min_r2": 0.85,
                "radius_quantile_low": 0.05,
                "radius_quantile_high": 0.8,
                "max_points": 500,
            },
            "lyapunov": {
                "method": "rosenstein",
                "theiler_window": 10,
                "neighbor_search_k": 40,
                "trajectory_len": 18,
                "fit_start": 1,
                "fit_end": 8,
                "min_pairs": 15,
                "min_fit_points": 4,
            },
            "metric_columns": [
                "selected_delay_tau",
                "tau_selection_method",
                "fnn_success_flag",
                "embedding_dimension",
                "correlation_dimension",
                "largest_lyapunov_exponent",
                "lyapunov_time",
            ],
        },
        "output": {
            "parquet_name": "features_block_D_v1.parquet",
            "excel_name": "features_block_D_v1.xlsx",
        },
    }
    cfg_path = configs_dir / "features_block_D_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_block_d_pipeline(str(cfg_path))
    out_df = pd.read_parquet(result.parquet_path)

    assert len(out_df) == 3
    assert {
        "series_id",
        "feature_status",
        "selected_delay_tau",
        "tau_selection_method",
        "fnn_success_flag",
        "embedding_dimension",
        "correlation_dimension",
        "largest_lyapunov_exponent",
        "lyapunov_time",
    }.issubset(out_df.columns)

    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()

    logi = out_df[out_df["series_id"] == "US:LOGI"].iloc[0]
    sine_row = out_df[out_df["series_id"] == "US:SINE"].iloc[0]
    white_row = out_df[out_df["series_id"] == "RU:WHITE"].iloc[0]

    assert int(logi["selected_delay_tau"]) >= 1
    assert str(logi["tau_selection_method"]) in {"ami", "acf_zero", "acf_einv", "fallback_default"}

    # Qualitative sanity: chaotic logistic map should not collapse to identical diagnostics.
    diff_sine = (
        (pd.to_numeric(logi["embedding_dimension"], errors="coerce") != pd.to_numeric(sine_row["embedding_dimension"], errors="coerce"))
        or (
            abs(
                pd.to_numeric(logi["largest_lyapunov_exponent"], errors="coerce")
                - pd.to_numeric(sine_row["largest_lyapunov_exponent"], errors="coerce")
            )
            > 1e-9
        )
    )
    diff_white = (
        (pd.to_numeric(logi["embedding_dimension"], errors="coerce") != pd.to_numeric(white_row["embedding_dimension"], errors="coerce"))
        or (
            abs(
                pd.to_numeric(logi["largest_lyapunov_exponent"], errors="coerce")
                - pd.to_numeric(white_row["largest_lyapunov_exponent"], errors="coerce")
            )
            > 1e-9
        )
    )
    assert diff_sine or diff_white
