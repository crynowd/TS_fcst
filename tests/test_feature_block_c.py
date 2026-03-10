from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.block_c_tails import (
    fisher_kurtosis,
    hill_tail_index,
    quantile_tail_ratios,
    run_feature_block_c_pipeline,
)


def test_kurtosis_normal_distribution() -> None:
    x = np.random.default_rng(42).normal(0.0, 1.0, 12000)
    val, flags = fisher_kurtosis(x)
    assert flags == []
    assert np.isfinite(val)
    assert -0.3 < val < 0.3


def test_tail_ratio_heavy_tail_distribution() -> None:
    rng = np.random.default_rng(123)
    normal = rng.normal(0.0, 1.0, 15000)
    student_t = rng.standard_t(df=3, size=15000)

    tr_n, flags_n = quantile_tail_ratios(normal)
    tr_t, flags_t = quantile_tail_ratios(student_t)

    assert flags_n == []
    assert flags_t == []
    assert np.isfinite(tr_n["tail_ratio_symmetric"]) and np.isfinite(tr_t["tail_ratio_symmetric"])
    assert np.isfinite(tr_n["tail_ratio_upper"]) and np.isfinite(tr_t["tail_ratio_upper"])
    assert np.isfinite(tr_n["tail_ratio_lower"]) and np.isfinite(tr_t["tail_ratio_lower"])

    assert tr_t["tail_ratio_symmetric"] > tr_n["tail_ratio_symmetric"]
    assert tr_t["tail_ratio_upper"] > tr_n["tail_ratio_upper"]
    assert tr_t["tail_ratio_lower"] > tr_n["tail_ratio_lower"]


def test_hill_estimator_behavior() -> None:
    rng = np.random.default_rng(7)
    normal_abs = np.abs(rng.normal(0.0, 1.0, 12000))
    student_abs = np.abs(rng.standard_t(df=3, size=12000))

    hill_n, flags_n = hill_tail_index(normal_abs, k_fraction=0.05)
    hill_t, flags_t = hill_tail_index(student_abs, k_fraction=0.05)

    assert np.isfinite(hill_n)
    assert np.isfinite(hill_t)
    assert flags_n == []
    assert flags_t == []

    # Heavier tails -> smaller tail index alpha.
    assert hill_t < hill_n


def test_block_C_pipeline_small_dataset(tmp_path: Path) -> None:
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

    n = 1000
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(999)
    normal = rng.normal(0.0, 1.0, n)
    student_t = rng.standard_t(df=3, size=n)

    returns_df = pd.DataFrame(
        {
            "series_id": ["RU:NORM"] * n + ["US:T3"] * n,
            "ticker": ["NORM"] * n + ["T3"] * n,
            "market": ["RU"] * n + ["US"] * n,
            "date": list(dates) + list(dates),
            "log_return": np.concatenate([normal, student_t]),
            "dataset_profile": ["core_balanced"] * (2 * n),
        }
    )
    profiles_df = pd.DataFrame(
        [
            {"series_id": "RU:NORM", "ticker": "NORM", "market": "RU", "dataset_profile": "core_balanced"},
            {"series_id": "US:T3", "ticker": "T3", "market": "US", "dataset_profile": "core_balanced"},
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
        "run_name": "feature_block_C_v1",
        "stage": "feature_block_C",
        "input": {
            "log_returns_parquet": "artifacts/processed/log_returns_v1.parquet",
            "dataset_profiles_parquet": "artifacts/processed/dataset_profiles_v1.parquet",
        },
        "dataset_profile": "core_balanced",
        "metrics": {
            "quantiles": [0.01, 0.25, 0.75, 0.99],
            "hill_k_fraction": 0.05,
            "use_hill": True,
        },
        "output": {
            "parquet_name": "features_block_C_v1.parquet",
            "excel_name": "features_block_C_v1.xlsx",
            "metric_columns": [
                "kurtosis",
                "robust_kurtosis",
                "tail_ratio_symmetric",
                "tail_ratio_upper",
                "tail_ratio_lower",
                "hill_tail_index",
            ],
        },
    }
    cfg_path = configs_dir / "features_block_C_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_block_c_pipeline(str(cfg_path))

    out_df = pd.read_parquet(result.parquet_path)
    assert len(out_df) == 2
    assert {
        "series_id",
        "feature_status",
        "kurtosis",
        "robust_kurtosis",
        "tail_ratio_symmetric",
        "tail_ratio_upper",
        "tail_ratio_lower",
        "hill_tail_index",
    }.issubset(out_df.columns)

    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()

    norm_tail = float(out_df.loc[out_df["series_id"] == "RU:NORM", "tail_ratio_symmetric"].iloc[0])
    t3_tail = float(out_df.loc[out_df["series_id"] == "US:T3", "tail_ratio_symmetric"].iloc[0])
    norm_hill = float(out_df.loc[out_df["series_id"] == "RU:NORM", "hill_tail_index"].iloc[0])
    t3_hill = float(out_df.loc[out_df["series_id"] == "US:T3", "hill_tail_index"].iloc[0])

    assert t3_tail > norm_tail
    assert t3_hill < norm_hill
