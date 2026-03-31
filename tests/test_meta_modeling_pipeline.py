from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.meta_modeling.pipeline import (
    build_meta_dataset_from_existing_artifacts,
    build_task_dataset,
    compute_best_single_baseline,
    run_meta_modeling_pipeline,
)


def _build_synthetic_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    feature_rows = []
    for i in range(1, 9):
        feature_rows.append(
            {
                "series_id": f"S{i}",
                "ticker": f"S{i}",
                "market": "US",
                "dataset_profile": "core_balanced",
                "feat_a": float(i),
                "feat_b": float(i % 3),
                "feat_c": float(i * 0.5),
            }
        )
    features_df = pd.DataFrame(feature_rows)

    model_names = ["m_a", "m_b", "m_c"]
    horizons = [1, 5]
    metric_rows = []
    for sid_idx in range(1, 9):
        sid = f"S{sid_idx}"
        for h in horizons:
            for m_idx, model_name in enumerate(model_names):
                rmse = 1.0 + 0.1 * m_idx + 0.01 * sid_idx + 0.001 * h
                da = 0.5 + 0.03 * (2 - m_idx) - 0.001 * sid_idx + 0.0005 * h
                metric_rows.append(
                    {
                        "run_id": "r_meta",
                        "model_name": model_name,
                        "series_id": sid,
                        "ticker": sid,
                        "market": "US",
                        "horizon": h,
                        "rmse_mean": rmse,
                        "directional_accuracy_mean": da,
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)

    features_path = tmp_path / "features.parquet"
    metrics_path = tmp_path / "series_metrics.parquet"
    features_df.to_parquet(features_path, index=False)
    metrics_df.to_parquet(metrics_path, index=False)
    return features_path, metrics_path


def _base_cfg(tmp_path: Path, features_path: Path, metrics_path: Path) -> dict:
    return {
        "stage": "meta_modeling",
        "run_name": "meta_modeling_test",
        "inputs": {
            "features_path": str(features_path),
            "forecasting_series_metrics_path": str(metrics_path),
            "forecasting_manifest_path": "",
        },
        "dataset_filter": "",
        "join_keys": ["series_id", "ticker"],
        "target_metrics": ["rmse", "directional_accuracy"],
        "metric_columns": {"rmse": "rmse_mean", "directional_accuracy": "directional_accuracy_mean"},
        "meta_models": ["ridge", "random_forest"],
        "model_overrides": {
            "ridge": {"alpha": 1.0},
            "random_forest": {"n_estimators": 50, "max_depth": 4, "n_jobs": -1},
        },
        "split": {"test_size": 0.25, "validation_size": 0.125, "random_seed": 42},
        "outputs": {
            "meta_dataset_summary_csv_path": str(tmp_path / "meta_dataset_summary.csv"),
            "meta_dataset_summary_parquet_path": str(tmp_path / "meta_dataset_summary.parquet"),
            "model_order_mapping_csv_path": str(tmp_path / "model_order_mapping.csv"),
            "routing_rows_parquet_path": str(tmp_path / "routing_rows.parquet"),
            "routing_rows_csv_path": str(tmp_path / "routing_rows.csv"),
            "task_results_parquet_path": str(tmp_path / "task_results.parquet"),
            "task_results_csv_path": str(tmp_path / "task_results.csv"),
            "split_assignments_csv_path": str(tmp_path / "split_assignments.csv"),
            "repeat_aggregated_results_csv_path": str(tmp_path / "repeat_aggregated_results.csv"),
            "best_single_baseline_by_repeat_csv_path": str(tmp_path / "best_single_baseline_by_repeat.csv"),
            "forecasting_mean_by_model_csv_path": str(tmp_path / "forecasting_mean_by_model.csv"),
            "forecasting_model_ranking_csv_path": str(tmp_path / "forecasting_model_ranking.csv"),
            "best_single_global_baseline_csv_path": str(tmp_path / "best_single_global_baseline.csv"),
            "forecasting_model_wins_csv_path": str(tmp_path / "forecasting_model_wins.csv"),
            "feature_list_csv_path": str(tmp_path / "feature_list.csv"),
            "feature_manifest_json_path": str(tmp_path / "feature_manifest.json"),
            "excel_report_path": str(tmp_path / "meta_modeling.xlsx"),
        },
        "artifacts": {"manifests": str(tmp_path)},
        "meta": {
            "config_path": "synthetic_meta_cfg",
            "project_root": str(tmp_path),
            "run_id": "meta_modeling_test_run",
            "log_path": str(tmp_path / "meta_modeling.log"),
        },
    }


def test_meta_dataset_builds_from_existing_artifacts(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _base_cfg(tmp_path, features_path, metrics_path)
    meta_long_df, summary = build_meta_dataset_from_existing_artifacts(cfg)

    assert not meta_long_df.empty
    assert summary["join_key"] == "series_id"
    assert set(summary["target_metrics"]) == {"rmse", "directional_accuracy"}
    assert set(summary["horizons"]) == {1, 5}
    assert set(summary["models"]) == {"m_a", "m_b", "m_c"}


def test_model_order_is_stable_and_saved(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _base_cfg(tmp_path, features_path, metrics_path)

    logger = logging.getLogger("meta_model_test_stable")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    run_meta_modeling_pipeline(cfg=cfg, logger=logger)

    mapping_df = pd.read_csv(tmp_path / "model_order_mapping.csv")
    grouped = mapping_df.groupby(["horizon", "target_metric"], sort=True)
    for _, sdf in grouped:
        ordered = sdf.sort_values("output_dim", kind="stable")["model_name"].tolist()
        assert ordered == sorted(ordered)


def test_multioutput_targets_align_with_forecasting_models(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _base_cfg(tmp_path, features_path, metrics_path)
    meta_long_df, summary = build_meta_dataset_from_existing_artifacts(cfg)

    task = build_task_dataset(
        meta_long_df=meta_long_df,
        join_key=summary["join_key"],
        feature_cols=summary["feature_cols"],
        horizon=1,
        target_metric="rmse",
    )
    assert task.y.shape[1] == len(task.model_order)
    assert task.model_order == sorted(task.model_order)


def test_best_single_baseline_is_computed_per_horizon_and_metric(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _base_cfg(tmp_path, features_path, metrics_path)
    meta_long_df, summary = build_meta_dataset_from_existing_artifacts(cfg)

    task = build_task_dataset(
        meta_long_df=meta_long_df,
        join_key=summary["join_key"],
        feature_cols=summary["feature_cols"],
        horizon=5,
        target_metric="directional_accuracy",
    )
    split_point = 5
    baseline = compute_best_single_baseline(
        y_train=task.y[:split_point],
        y_test=task.y[split_point:],
        model_order=task.model_order,
        direction="max",
    )
    assert baseline["baseline_model"] in set(task.model_order)
    assert baseline["baseline_values"].shape[0] == task.y[split_point:].shape[0]


def test_small_smoke_one_horizon_one_target_metric(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _base_cfg(tmp_path, features_path, metrics_path)
    cfg["target_metrics"] = ["rmse"]
    cfg["meta_models"] = ["ridge"]

    logger = logging.getLogger("meta_model_test_smoke")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    result = run_meta_modeling_pipeline(cfg=cfg, logger=logger)

    assert result["tasks_total"] > 0
    assert Path(result["manifest_path"]).exists()
    assert Path(cfg["outputs"]["task_results_csv_path"]).exists()
    assert Path(cfg["outputs"]["routing_rows_csv_path"]).exists()
    assert Path(cfg["outputs"]["repeat_aggregated_results_csv_path"]).exists()
    assert Path(cfg["outputs"]["forecasting_model_ranking_csv_path"]).exists()
