from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.analysis.cluster_forecasting import (
    append_relative_to_baseline,
    build_best_model_by_cluster,
    build_cluster_model_performance,
    join_forecasting_with_cluster_labels,
)
from src.analysis.stat_tests import run_kruskal_tests
from src.analysis.utility_metrics import build_clustering_utility
from src.cli.run_cluster_forecasting_analysis import run_cluster_forecasting_analysis_pipeline


def _series_metrics_fixture() -> pd.DataFrame:
    rows = []
    for series_id, cluster in [("S1", 0), ("S2", 0), ("S3", 1), ("S4", 1)]:
        for model_name, mae in [("naive_zero", 1.0 + 0.1 * cluster), ("ridge_lag", 0.7 + 0.05 * cluster)]:
            rows.append(
                {
                    "run_id": "r1",
                    "model_name": model_name,
                    "series_id": series_id,
                    "ticker": series_id,
                    "market": "US",
                    "horizon": 1,
                    "n_folds": 2,
                    "mae_mean": mae,
                    "mae_median": mae,
                    "rmse_mean": mae + 0.1,
                    "rmse_median": mae + 0.1,
                    "mase_mean": mae + 0.2,
                    "mase_median": mae + 0.2,
                    "smape_mean": 100.0,
                    "directional_accuracy_mean": 0.5 if model_name == "naive_zero" else 0.7,
                    "huber_mean": 0.1,
                    "medae_mean": 0.2,
                    "bias_mean": 0.0,
                    "r2_mean": 0.1,
                    "r2_oos_mean": 0.05,
                }
            )
    return pd.DataFrame(rows)


def _assignments_fixture() -> pd.DataFrame:
    rows = []
    for cfg in ["cfg_0185", "cfg_0443"]:
        for series_id, cluster in [("S1", 0), ("S2", 0), ("S3", 1), ("S4", 1)]:
            rows.append(
                {
                    "config_id": cfg,
                    "feature_set": "base",
                    "scaler": "identity",
                    "space_type": "pca",
                    "algorithm": "gmm",
                    "n_clusters": 2,
                    "series_id": series_id,
                    "ticker": series_id,
                    "market": "US",
                    "dataset_profile": "core_balanced",
                    "cluster_label": cluster,
                    "assignment_confidence": 0.8,
                }
            )
    return pd.DataFrame(rows)


def test_join_forecasting_and_clusters() -> None:
    series_metrics = _series_metrics_fixture()
    fold_metrics = pd.DataFrame(
        {
            "run_id": ["r1"],
            "model_name": ["naive_zero"],
            "series_id": ["S1"],
            "ticker": ["S1"],
            "market": ["US"],
            "horizon": [1],
            "fold_id": [1],
            "mae": [1.0],
        }
    )
    raw_predictions = pd.DataFrame(
        {
            "run_id": ["r1"],
            "model_name": ["naive_zero"],
            "series_id": ["S1"],
            "ticker": ["S1"],
            "market": ["US"],
            "horizon": [1],
            "fold_id": [1],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "y_true": [0.0],
            "y_pred": [0.0],
            "status": ["success"],
            "y_train_mean": [0.0],
            "y_naive_baseline": [0.0],
        }
    )
    joined = join_forecasting_with_cluster_labels(
        series_metrics_df=series_metrics,
        fold_metrics_df=fold_metrics,
        raw_predictions_df=raw_predictions,
        cluster_assignments_df=_assignments_fixture(),
        available_configs=["cfg_0185", "cfg_0443"],
    )

    assert not joined["series_joined"].empty
    assert "cluster_label" in joined["series_joined"].columns
    assert joined["series_joined"]["clustering_config"].nunique() == 2


def test_best_model_by_cluster_selection() -> None:
    joined_series = _series_metrics_fixture().merge(
        _assignments_fixture()[["config_id", "series_id", "cluster_label"]].rename(columns={"config_id": "clustering_config"}),
        on="series_id",
        how="inner",
    )
    perf = build_cluster_model_performance(joined_series)
    best = build_best_model_by_cluster(perf)

    assert not best.empty
    assert (best["best_model_mae"] == "ridge_lag").all()
    assert (best["best_model_rmse"] == "ridge_lag").all()


def test_relative_to_baseline_metrics() -> None:
    joined_series = _series_metrics_fixture().merge(
        _assignments_fixture()[["config_id", "series_id", "cluster_label"]].rename(columns={"config_id": "clustering_config"}),
        on="series_id",
        how="inner",
    )
    perf = build_cluster_model_performance(joined_series)
    rel, missing = append_relative_to_baseline(perf, baseline_model="naive_zero")

    assert missing.empty
    assert "mae_vs_naive_zero_diff" in rel.columns
    ridge_rows = rel[rel["model_name"] == "ridge_lag"]
    assert (ridge_rows["mae_vs_naive_zero_diff"] < 0).all()
    assert (ridge_rows["mae_vs_naive_zero_ratio"] < 1).all()


def test_clustering_utility_computation() -> None:
    joined_series = _series_metrics_fixture().merge(
        _assignments_fixture()[["config_id", "series_id", "cluster_label"]].rename(columns={"config_id": "clustering_config"}),
        on="series_id",
        how="inner",
    )
    perf = build_cluster_model_performance(joined_series)
    util = build_clustering_utility(perf, metrics=["mae", "mase", "directional_accuracy"])

    assert not util.empty
    assert {"regret_improvement", "best_model_diversity_count", "model_rank_variance"}.issubset(util.columns)


def test_kruskal_output_schema() -> None:
    joined_series = _series_metrics_fixture().merge(
        _assignments_fixture()[["config_id", "series_id", "cluster_label"]].rename(columns={"config_id": "clustering_config"}),
        on="series_id",
        how="inner",
    )
    tests_df = run_kruskal_tests(joined_series, metrics=["mae", "directional_accuracy"])

    assert not tests_df.empty
    expected_cols = {
        "clustering_config",
        "model_name",
        "horizon",
        "metric_name",
        "test_name",
        "statistic",
        "p_value",
        "notes",
    }
    assert expected_cols.issubset(set(tests_df.columns))


def test_cluster_forecasting_analysis_stage_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    artifacts_dir = root / "artifacts"
    forecasting_dir = artifacts_dir / "forecasting"
    clustering_dir = artifacts_dir / "clustering"
    reports_dir = artifacts_dir / "reports"
    logs_dir = artifacts_dir / "logs"
    manifests_dir = artifacts_dir / "manifests"
    analysis_dir = artifacts_dir / "analysis"

    for d in [configs_dir, forecasting_dir, clustering_dir, reports_dir, logs_dir, manifests_dir, analysis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    series_metrics = _series_metrics_fixture()
    series_metrics.to_parquet(forecasting_dir / "series_metrics_smoke_v1.parquet", index=False)

    fold_metrics = pd.DataFrame(
        {
            "run_id": ["r1"],
            "model_name": ["naive_zero"],
            "series_id": ["S1"],
            "ticker": ["S1"],
            "market": ["US"],
            "horizon": [1],
            "fold_id": [1],
            "mae": [1.0],
            "rmse": [1.1],
            "mase": [1.2],
            "directional_accuracy": [0.5],
        }
    )
    fold_metrics.to_parquet(forecasting_dir / "fold_metrics_smoke_v1.parquet", index=False)

    raw_predictions = pd.DataFrame(
        {
            "run_id": ["r1"],
            "model_name": ["naive_zero"],
            "series_id": ["S1"],
            "ticker": ["S1"],
            "market": ["US"],
            "horizon": [1],
            "fold_id": [1],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "y_true": [0.1],
            "y_pred": [0.0],
            "status": ["success"],
            "y_train_mean": [0.0],
            "y_naive_baseline": [0.0],
        }
    )
    raw_predictions.to_parquet(forecasting_dir / "raw_predictions_smoke_v1.parquet", index=False)

    assignments = _assignments_fixture()
    assignments.to_parquet(clustering_dir / "cluster_assignments_v1.parquet", index=False)

    selected_configs = pd.DataFrame(
        {
            "config_id": ["cfg_0191", "cfg_0185", "cfg_0193", "cfg_0443", "cfg_0444", "cfg_0445"],
            "feature_set": ["base"] * 6,
            "ranking_score": [0.1] * 6,
        }
    )
    selected_configs.to_csv(clustering_dir / "selected_configs_v1.csv", index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "reports": str(reports_dir),
            "logs": str(logs_dir),
            "manifests": str(manifests_dir),
            "analysis": str(analysis_dir),
        },
    }
    (configs_dir / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    stage_cfg = {
        "run_name": "cluster_forecasting_analysis_smoke_v1",
        "stage": "cluster_forecasting_analysis_smoke",
        "forecasting_inputs": {
            "series_metrics_path": "artifacts/forecasting/series_metrics_smoke_v1.parquet",
            "fold_metrics_path": "artifacts/forecasting/fold_metrics_smoke_v1.parquet",
            "raw_predictions_path": "artifacts/forecasting/raw_predictions_smoke_v1.parquet",
        },
        "clustering_inputs": {
            "cluster_assignments_path": "artifacts/clustering/cluster_assignments_v1.parquet",
            "selected_configs_path": "artifacts/clustering/selected_configs_v1.csv",
        },
        "shortlist_configs": ["cfg_0191", "cfg_0185", "cfg_0193", "cfg_0443", "cfg_0444", "cfg_0445"],
        "baselines": {"primary": "naive_zero", "secondary": "naive_mean"},
        "metrics_for_comparison": ["mae", "rmse", "mase", "directional_accuracy"],
        "stat_tests": {"use_kruskal": True, "use_pairwise_posthoc": False},
        "outputs": {
            "run_name": "cluster_forecasting_analysis_smoke_v1",
            "cluster_model_performance_path": "artifacts/analysis/cluster_model_performance_v1.parquet",
            "cluster_model_performance_csv_path": "artifacts/analysis/cluster_model_performance_v1.csv",
            "best_model_by_cluster_path": "artifacts/analysis/best_model_by_cluster_v1.csv",
            "clustering_utility_path": "artifacts/analysis/clustering_utility_v1.csv",
            "cluster_metric_tests_path": "artifacts/analysis/cluster_metric_tests_v1.csv",
            "tidy_cluster_model_metrics_path": "artifacts/analysis/tidy_cluster_model_metrics_v1.parquet",
            "tidy_series_cluster_metrics_path": "artifacts/analysis/tidy_series_cluster_metrics_v1.parquet",
            "excel_report_path": "artifacts/reports/cluster_forecasting_analysis_v1.xlsx",
        },
    }
    cfg_path = configs_dir / "cluster_forecasting_analysis_smoke_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_cluster_forecasting_analysis_pipeline(str(cfg_path))

    assert Path(result["outputs"]["cluster_model_performance_path"]).exists()
    assert Path(result["outputs"]["best_model_by_cluster_path"]).exists()
    assert Path(result["outputs"]["clustering_utility_path"]).exists()
    assert Path(result["outputs"]["cluster_metric_tests_path"]).exists()
    assert Path(result["outputs"]["tidy_cluster_model_metrics_path"]).exists()
    assert Path(result["outputs"]["tidy_series_cluster_metrics_path"]).exists()
    assert Path(result["excel_path"]).exists()
    assert Path(result["manifest_path"]).exists()
