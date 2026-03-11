from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.master_table import build_master_feature_table
from src.features.quality_screening import (
    build_screening_shortlist,
    compute_feature_quality_metrics,
    compute_high_correlation_pairs,
    run_feature_screening_pipeline,
)


def _base_keys() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"series_id": "S1", "ticker": "AAA", "market": "RU", "dataset_profile": "core_balanced"},
            {"series_id": "S2", "ticker": "BBB", "market": "US", "dataset_profile": "core_balanced"},
            {"series_id": "S3", "ticker": "CCC", "market": "US", "dataset_profile": "core_balanced"},
        ]
    )


def _screen_cfg() -> dict:
    return {
        "high_missing_threshold": 0.20,
        "low_variance_threshold": 1e-6,
        "near_constant_unique_threshold": 2,
        "high_correlation_threshold": 0.90,
        "warning_heavy_threshold": 0.30,
    }


def test_master_merge_preserves_row_count() -> None:
    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        hurst_rs=[0.4, 0.5, 0.6],
    )
    block_b = keys.assign(
        feature_status="success",
        feature_warning_flags="warn_b",
        n_obs_used=100,
        spectral_entropy=[0.8, 0.7, 0.6],
    )
    block_c = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        kurtosis=[1.0, 2.0, 3.0],
    )
    block_d = keys.assign(
        feature_status="warning",
        feature_warning_flags="tau_fallback_default",
        n_obs_used=100,
        largest_lyapunov_exponent=[0.01, 0.02, 0.03],
    )

    result = build_master_feature_table({"A": block_a, "B": block_b, "C": block_c, "D": block_d})
    assert len(result.master_df) == len(keys)
    assert result.master_df["series_id"].nunique() == len(keys)
    assert {"block_A_status", "block_B_status", "block_C_status", "block_D_status"}.issubset(result.master_df.columns)
    assert {
        "block_A_warning_flags",
        "block_B_warning_flags",
        "block_C_warning_flags",
        "block_D_warning_flags",
    }.issubset(result.master_df.columns)


def test_quality_metrics_detect_missingness() -> None:
    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        feat_missing=[1.0, np.nan, np.nan],
    )
    master = build_master_feature_table({"A": block_a})

    quality_df = compute_feature_quality_metrics(master.master_df, master.feature_to_block, _screen_cfg())
    row = quality_df.loc[quality_df["feature_name"] == "feat_missing"].iloc[0]
    assert row["missing_count"] == 2
    assert np.isclose(row["missing_rate"], 2 / 3)
    assert bool(row["high_missing_flag"]) is True


def test_low_variance_flag_behavior() -> None:
    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        feat_constant=[0.001, 0.001, 0.001],
    )
    master = build_master_feature_table({"A": block_a})
    quality_df = compute_feature_quality_metrics(master.master_df, master.feature_to_block, _screen_cfg())
    row = quality_df.loc[quality_df["feature_name"] == "feat_constant"].iloc[0]
    assert bool(row["low_variance_flag"]) is True
    assert bool(row["near_constant_flag"]) is True


def test_high_correlation_pairs_detection() -> None:
    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        x=[1.0, 2.0, 3.0],
    )
    block_b = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        y=[2.0, 4.0, 6.0],
    )
    master = build_master_feature_table({"A": block_a, "B": block_b})
    pairs_df = compute_high_correlation_pairs(master.master_df, master.feature_to_block, threshold=0.90)
    assert len(pairs_df) == 1
    assert set(pairs_df.iloc[0][["feature_1", "feature_2"]]) == {"x", "y"}


def test_screening_status_assignment() -> None:
    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=100,
        feat_high_missing=[1.0, np.nan, np.nan],
        feat_low_var=[0.1, 0.1, 0.1],
    )
    block_d = keys.assign(
        feature_status="warning",
        feature_warning_flags=["warn_1;warn_2", "warn_1;warn_2", ""],
        n_obs_used=100,
        feat_d=[0.1, 0.2, 0.3],
    )
    master = build_master_feature_table({"A": block_a, "D": block_d})
    quality_df = compute_feature_quality_metrics(master.master_df, master.feature_to_block, _screen_cfg())
    corr_pairs_df = compute_high_correlation_pairs(master.master_df, master.feature_to_block, threshold=0.99)
    shortlist = build_screening_shortlist(quality_df, corr_pairs_df, _screen_cfg())

    statuses = dict(zip(shortlist["feature_name"], shortlist["screening_status"]))
    assert statuses["feat_high_missing"] == "review_high_missing"
    assert statuses["feat_low_var"] == "review_low_variance"
    assert statuses["feat_d"] == "reserve_candidate"


def test_feature_screening_pipeline_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    artifacts_dir = root / "artifacts"
    features_dir = artifacts_dir / "features"
    reports_dir = artifacts_dir / "reports"
    logs_dir = artifacts_dir / "logs"
    manifests_dir = artifacts_dir / "manifests"
    processed_dir = artifacts_dir / "processed"
    for d in [configs_dir, features_dir, reports_dir, logs_dir, manifests_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    keys = _base_keys()
    block_a = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=120,
        feat_a1=[1.0, 2.0, np.nan],
        feat_a2=[0.2, 0.2, 0.2],
    )
    block_b = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=120,
        feat_b1=[1.0, 2.0, 3.0],
    )
    block_c = keys.assign(
        feature_status="success",
        feature_warning_flags="",
        n_obs_used=120,
        feat_c1=[2.0, 4.0, 6.0],
    )
    block_d = keys.assign(
        feature_status="warning",
        feature_warning_flags=["warn_x", "warn_x", ""],
        n_obs_used=120,
        feat_d1=[0.4, 0.5, 0.6],
    )

    block_a.to_parquet(features_dir / "features_block_A_v1.parquet", index=False)
    block_b.to_parquet(features_dir / "features_block_B_v1.parquet", index=False)
    block_c.to_parquet(features_dir / "features_block_C_v1.parquet", index=False)
    block_d.to_parquet(features_dir / "features_block_D_v1.parquet", index=False)
    keys.to_parquet(processed_dir / "dataset_profiles_v1.parquet", index=False)

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
        "run_name": "feature_screening_v1",
        "stage": "feature_screening",
        "input": {
            "features_block_A_parquet": "artifacts/features/features_block_A_v1.parquet",
            "features_block_B_parquet": "artifacts/features/features_block_B_v1.parquet",
            "features_block_C_parquet": "artifacts/features/features_block_C_v1.parquet",
            "features_block_D_parquet": "artifacts/features/features_block_D_v1.parquet",
            "dataset_profiles_parquet": "artifacts/processed/dataset_profiles_v1.parquet",
        },
        "dataset_profile": "core_balanced",
        "screening": {
            "high_missing_threshold": 0.2,
            "low_variance_threshold": 1e-6,
            "near_constant_unique_threshold": 2,
            "high_correlation_threshold": 0.9,
            "warning_heavy_threshold": 0.3,
            "master_preview_rows": 10,
        },
        "output": {
            "master_parquet_name": "features_master_v1.parquet",
            "excel_name": "feature_screening_v1.xlsx",
            "high_correlation_csv_name": "high_correlation_pairs_v1.csv",
            "shortlist_csv_name": "screening_shortlist_v1.csv",
        },
    }
    cfg_path = configs_dir / "feature_screening_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_screening_pipeline(str(cfg_path))
    assert Path(result.master_parquet_path).exists()
    assert Path(result.excel_path).exists()
    assert Path(result.correlation_pairs_csv_path).exists()
    assert Path(result.shortlist_csv_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()

    master_df = pd.read_parquet(result.master_parquet_path)
    shortlist_df = pd.read_csv(result.shortlist_csv_path)
    corr_df = pd.read_csv(result.correlation_pairs_csv_path)
    assert len(master_df) == 3
    assert {"feat_a1", "feat_b1", "feat_c1", "feat_d1"}.issubset(master_df.columns)
    assert "screening_status" in shortlist_df.columns
    assert len(corr_df) >= 1
