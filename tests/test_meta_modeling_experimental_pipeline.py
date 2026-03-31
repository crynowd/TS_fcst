from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.meta_modeling.experimental_pipeline import run_meta_modeling_experiments


def _build_synthetic_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    feature_rows = []
    for i in range(1, 11):
        feature_rows.append(
            {
                "series_id": f"S{i}",
                "ticker": f"S{i}",
                "market": "US",
                "dataset_profile": "core_balanced",
                "feat_a": float(i),
                "feat_b": float(i % 4),
                "feat_c": float(i * 0.3),
                "feat_dup": float(i),
            }
        )
    features_df = pd.DataFrame(feature_rows)

    model_names = ["m_a", "m_b", "m_c", "m_d"]
    horizons = [1, 5]
    metric_rows = []
    for sid_idx in range(1, 11):
        sid = f"S{sid_idx}"
        for h in horizons:
            for m_idx, model_name in enumerate(model_names):
                rmse = 1.0 + 0.15 * m_idx + 0.01 * sid_idx + 0.001 * h
                da = 0.50 + 0.04 * (3 - m_idx) - 0.001 * sid_idx + 0.0003 * h
                metric_rows.append(
                    {
                        "run_id": "r_meta_exp",
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


def _cfg(tmp_path: Path, features_path: Path, metrics_path: Path) -> dict:
    return {
        "run_name": "meta_modeling_experiments_test",
        "stage": "meta_modeling_experiments",
        "inputs": {
            "features_path": str(features_path),
            "forecasting_series_metrics_path": str(metrics_path),
            "forecasting_manifest_path": "",
        },
        "dataset_filter": "",
        "join_keys": ["series_id", "ticker"],
        "target_metrics": ["rmse", "directional_accuracy"],
        "metric_columns": {"rmse": "rmse_mean", "directional_accuracy": "directional_accuracy_mean"},
        "methods": ["regression", "classification", "scoring"],
        "meta_models": ["ridge", "random_forest"],
        "classification_models": ["logistic_regression", "random_forest_classifier"],
        "scoring_models": ["ridge", "random_forest"],
        "model_overrides": {"random_forest": {"n_estimators": 20, "max_depth": 4, "n_jobs": -1}, "random_forest_classifier": {"n_estimators": 20, "max_depth": 4, "n_jobs": -1}},
        "feature_pruning": {"corr_threshold": 0.9},
        "candidate_selection": {"top_k_values": [3], "closeness_tolerance": 0.05},
        "split": {"test_size": 0.2, "validation_size": 0.1, "random_seed": 42, "n_repeats": 2, "random_seeds": []},
        "prediction_examples_per_task": 3,
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
            "feature_list_csv_path": str(tmp_path / "feature_list.csv"),
            "pruned_feature_list_csv_path": str(tmp_path / "pruned_feature_list.csv"),
            "dropped_correlation_pairs_csv_path": str(tmp_path / "dropped_correlation_pairs.csv"),
            "feature_manifest_json_path": str(tmp_path / "feature_manifest.json"),
            "comparison_table_csv_path": str(tmp_path / "comparison_table.csv"),
            "candidate_models_csv_path": str(tmp_path / "candidate_models.csv"),
            "feature_importance_csv_path": str(tmp_path / "feature_importance.csv"),
            "prediction_examples_csv_path": str(tmp_path / "prediction_examples.csv"),
            "classification_probabilities_csv_path": str(tmp_path / "classification_probabilities.csv"),
            "excel_report_path": str(tmp_path / "meta_modeling_experiments.xlsx"),
        },
        "artifacts": {"manifests": str(tmp_path)},
        "meta": {"config_path": "synthetic_experiment_cfg", "project_root": str(tmp_path), "run_id": "meta_modeling_experiments_test_run", "log_path": str(tmp_path / "meta_experiments.log")},
    }


def test_meta_modeling_experiments_smoke(tmp_path: Path) -> None:
    features_path, metrics_path = _build_synthetic_artifacts(tmp_path)
    cfg = _cfg(tmp_path, features_path, metrics_path)
    logger = logging.getLogger("meta_modeling_experiments_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    result = run_meta_modeling_experiments(cfg=cfg, logger=logger)

    assert result["tasks_total"] > 0
    assert Path(result["manifest_path"]).exists()
    assert Path(cfg["outputs"]["comparison_table_csv_path"]).exists()
    assert Path(cfg["outputs"]["candidate_models_csv_path"]).exists()
    assert Path(cfg["outputs"]["feature_importance_csv_path"]).exists()
    assert Path(cfg["outputs"]["classification_probabilities_csv_path"]).exists()
