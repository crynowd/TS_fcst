from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.data import build_series_lookup, select_series
from src.forecasting.io import FOLD_METRICS_COLUMNS, RAW_PREDICTION_COLUMNS, SPLIT_METADATA_COLUMNS, ensure_table_schema
from src.forecasting.registry import build_model, get_model_specs
from src.forecasting.runners import run_forecasting_benchmark
from src.forecasting.targets import build_direct_horizon_target
from src.forecasting.windowing import build_leakage_safe_fold, build_rolling_origin_folds, build_supervised_windows


def test_target_construction_for_multiple_horizons() -> None:
    returns = np.array([0.10, -0.20, 0.30, 0.40, -0.10], dtype=np.float64)

    y_h1 = build_direct_horizon_target(returns, horizon=1)
    y_h2 = build_direct_horizon_target(returns, horizon=2)

    np.testing.assert_allclose(y_h1, np.array([-0.20, 0.30, 0.40, -0.10]))
    np.testing.assert_allclose(y_h2, np.array([0.10, 0.70, 0.30]))


def test_window_builder_shapes() -> None:
    n = 60
    df = pd.DataFrame(
        {
            "series_id": ["S1"] * n,
            "ticker": ["S1"] * n,
            "market": ["US"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "log_return": np.linspace(-0.01, 0.01, n),
            "dataset_profile": ["core_balanced"] * n,
        }
    )

    data = build_supervised_windows(df, horizon=5, window_size=8)
    assert data.X.ndim == 2
    assert data.X.shape[1] == 8
    assert data.y.ndim == 1
    assert data.X.shape[0] == data.y.shape[0] == data.timestamps.shape[0]
    assert np.all(data.feature_end_idx < data.target_start_idx)
    assert np.all((data.target_end_idx - data.target_start_idx + 1) == 5)

    folds = build_rolling_origin_folds(n_samples=len(data.y), n_folds=3)
    assert len(folds) == 3
    assert folds[0].train_idx.max() < folds[0].test_idx.min()


@pytest.mark.parametrize(("horizon", "window_size"), [(1, 64), (5, 32), (20, 16)])
def test_leakage_safe_fold_boundaries_and_causal_inputs(horizon: int, window_size: int) -> None:
    n = 400
    returns = np.arange(n, dtype=np.float64) / 10_000.0
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "log_return": returns,
        }
    )
    supervised = build_supervised_windows(df, horizon=horizon, window_size=window_size)

    for sample_idx in range(len(supervised.y)):
        origin = int(supervised.feature_end_idx[sample_idx])
        feature_positions = range(origin - window_size + 1, origin + 1)
        expected = np.array(
            [returns[pos - horizon + 1 : pos + 1].sum() for pos in feature_positions],
            dtype=np.float64,
        )
        np.testing.assert_allclose(supervised.X[sample_idx], expected)

    for fold in build_rolling_origin_folds(len(supervised.y), n_folds=3):
        safe = build_leakage_safe_fold(supervised, fold)
        test_origin = int(supervised.feature_end_idx[safe.test_idx[0]])
        validation_origin = int(supervised.feature_end_idx[safe.validation_idx[0]])

        assert int(supervised.target_end_idx[safe.outer_train_idx].max()) <= test_origin
        assert int(supervised.target_end_idx[safe.fit_idx].max()) <= validation_origin
        assert int(supervised.target_end_idx[safe.validation_idx].max()) <= test_origin
        np.testing.assert_array_equal(safe.test_idx, fold.test_idx)

        expected_boundary_purge = horizon - 1
        assert len(safe.outer_train_idx_before_purge) - len(safe.outer_train_idx) == expected_boundary_purge
        assert len(safe.fit_idx_before_purge) - len(safe.fit_idx) == expected_boundary_purge
        assert len(safe.validation_idx_before_purge) - len(safe.validation_idx) == 0
        if horizon == 1:
            np.testing.assert_array_equal(safe.outer_train_idx, safe.outer_train_idx_before_purge)
            np.testing.assert_array_equal(safe.fit_idx, safe.fit_idx_before_purge)
        elif horizon == 5:
            assert expected_boundary_purge <= 4
        else:
            assert expected_boundary_purge <= 19


def test_split_metadata_schema_contains_target_aware_boundaries() -> None:
    required = {
        "split_policy",
        "outer_train_first_forecast_origin_idx",
        "outer_train_last_target_end_idx",
        "outer_train_n_samples_before_purge",
        "outer_train_n_samples_after_purge",
        "outer_train_n_samples_removed",
        "fit_first_forecast_origin_idx",
        "fit_last_target_end_idx",
        "fit_n_samples_removed",
        "validation_first_forecast_origin_idx",
        "validation_last_target_end_idx",
        "validation_n_samples_removed",
        "test_first_forecast_origin_idx",
        "test_last_target_end_idx",
        "test_n_samples_removed",
    }
    assert required.issubset(SPLIT_METADATA_COLUMNS)


def test_series_lookup_respects_dataset_profile() -> None:
    df = pd.DataFrame(
        [
            {
                "series_id": "S1",
                "ticker": "S1",
                "market": "US",
                "date": pd.Timestamp("2020-01-01"),
                "log_return": 0.01,
                "dataset_profile": "core_balanced",
            },
            {
                "series_id": "S1",
                "ticker": "S1",
                "market": "US",
                "date": pd.Timestamp("2020-01-01"),
                "log_return": 0.02,
                "dataset_profile": "other",
            },
        ]
    )
    selected = select_series(df, dataset_profile="core_balanced")
    lookup = build_series_lookup(df, selected, dataset_profile="core_balanced")
    assert len(lookup["S1"]) == 1
    assert lookup["S1"]["dataset_profile"].unique().tolist() == ["core_balanced"]


def test_model_registry_contains_expected_models() -> None:
    specs = get_model_specs()
    expected = {
        "naive_zero",
        "naive_mean",
        "ridge_lag",
        "esn",
        "chaotic_esn",
        "transient_chaotic_esn",
        "vanilla_mlp",
        "chaotic_mlp",
        "chaotic_logistic_net",
        "lstm_forecast",
        "chaotic_lstm_forecast",
    }
    assert expected.issubset(set(specs.keys()))


def test_raw_predictions_table_schema() -> None:
    df = pd.DataFrame([{"run_id": "x", "model_name": "naive_zero"}])
    out = ensure_table_schema(df, RAW_PREDICTION_COLUMNS)
    assert out.columns.tolist() == RAW_PREDICTION_COLUMNS


def test_fold_metrics_table_schema() -> None:
    df = pd.DataFrame([{"run_id": "x", "model_name": "naive_zero"}])
    out = ensure_table_schema(df, FOLD_METRICS_COLUMNS)
    assert out.columns.tolist() == FOLD_METRICS_COLUMNS


def test_small_e2e_smoke_benchmark_on_synthetic_dataset(tmp_path) -> None:
    rows = []
    n_series = 2
    n_points = 120
    for sid in range(n_series):
        series_id = f"S{sid}"
        base = np.sin(np.linspace(0, 8, n_points)) * 0.01 + sid * 0.001
        for i in range(n_points):
            rows.append(
                {
                    "series_id": series_id,
                    "ticker": series_id,
                    "market": "US",
                    "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
                    "log_return": float(base[i]),
                    "dataset_profile": "core_balanced",
                }
            )
    df = pd.DataFrame(rows)
    source_path = tmp_path / "synthetic.parquet"
    df.to_parquet(source_path, index=False)

    cfg = {
        "data": {
            "source_path": str(source_path),
            "dataset_profile": "core_balanced",
            "max_series": 2,
            "series_selection_mode": "first_n",
        },
        "models": {
            "active": ["naive_zero", "ridge_lag"],
            "inactive_but_supported": ["lstm_forecast", "chaotic_lstm_forecast"],
        },
        "horizons": [1],
        "window_sizes": {1: 16},
        "validation": {"n_folds": 2},
        "timeouts": {"max_train_seconds_per_task": 10, "max_predict_seconds_per_task": 10},
        "training": {"max_epochs": 2, "early_stopping_patience": 1, "batch_size": 32, "learning_rate": 1e-3},
        "model_overrides": {},
        "filters": {"active_models": [], "horizons": [], "series_ids": [], "resume_failed_only": False},
        "outputs": {
            "run_name": "test_smoke",
            "raw_predictions_path": str(tmp_path / "raw.parquet"),
            "fold_metrics_path": str(tmp_path / "fold.parquet"),
            "series_metrics_path": str(tmp_path / "series.parquet"),
            "task_audit_path": str(tmp_path / "audit.parquet"),
            "excel_report_path": str(tmp_path / "report.xlsx"),
        },
        "artifacts": {
            "manifests": str(tmp_path),
        },
        "meta": {
            "config_path": "synthetic",
            "project_root": str(tmp_path),
            "run_id": "synthetic_run",
            "log_path": str(tmp_path / "synthetic.log"),
        },
    }

    logger = logging.getLogger("forecasting_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    result = run_forecasting_benchmark(cfg=cfg, logger=logger)

    assert result["n_tasks"] > 0
    assert result["success_count"] == result["n_tasks"]
    assert (tmp_path / "raw.parquet").exists()
    assert (tmp_path / "fold.parquet").exists()
    assert (tmp_path / "series.parquet").exists()
    assert (tmp_path / "audit.parquet").exists()
    assert (tmp_path / "report.xlsx").exists()


def test_v2_resume_merges_split_metadata_across_disjoint_series_batches(tmp_path: Path) -> None:
    rows = []
    for series_id in ["S1", "S2"]:
        for i in range(120):
            rows.append(
                {
                    "series_id": series_id,
                    "ticker": series_id,
                    "market": "US",
                    "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
                    "log_return": float(i) / 10_000.0,
                    "dataset_profile": "core_balanced",
                }
            )
    source_path = tmp_path / "synthetic.parquet"
    pd.DataFrame(rows).to_parquet(source_path, index=False)
    run_dir = tmp_path / "clean_batched"
    outputs = {
        "output_dir": str(run_dir),
        "metrics_long_path": str(run_dir / "metrics_long.parquet"),
        "metrics_long_csv_path": str(run_dir / "metrics_long.csv"),
        "split_metadata_path": str(run_dir / "split_metadata.parquet"),
        "errors_csv_path": str(run_dir / "errors.csv"),
        "run_manifest_path": str(run_dir / "run_manifest.json"),
        "config_snapshot_path": str(run_dir / "config_snapshot.yaml"),
        "predictions_path": str(run_dir / "predictions.parquet"),
        "task_audit_path": str(run_dir / "task_audit.parquet"),
        "fold_metrics_path": str(run_dir / "metrics_long.parquet"),
        "raw_predictions_path": str(run_dir / "predictions.parquet"),
        "excel_report_path": str(tmp_path / "report.xlsx"),
    }
    base_cfg = {
        "stage": "forecasting_benchmark_v2",
        "data": {"source_path": str(source_path), "dataset_profile": "core_balanced", "max_series": 2, "series_selection_mode": "first_n"},
        "models": {"active": ["naive_zero"], "inactive_but_supported": []},
        "horizons": [1],
        "window_sizes": {1: 16},
        "validation": {"n_folds": 2},
        "timeouts": {"max_train_seconds_per_task": 10, "max_predict_seconds_per_task": 10},
        "training": {"max_epochs": 2, "early_stopping_patience": 1, "batch_size": 32, "learning_rate": 1e-3},
        "model_overrides": {},
        "filters": {"active_models": [], "horizons": [], "series_ids": [], "resume_failed_only": False},
        "outputs": outputs,
        "artifacts": {"manifests": str(tmp_path)},
        "resume": True,
        "save_predictions": True,
        "random_seed": 2026,
        "dataset_version": "synthetic",
        "meta": {"config_path": "synthetic", "project_root": str(tmp_path), "run_id": "clean_batched", "log_path": str(tmp_path / "synthetic.log")},
    }
    logger = logging.getLogger("forecasting_batched_resume_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    for series_id in ["S1", "S2"]:
        cfg = copy.deepcopy(base_cfg)
        cfg["filters"]["series_ids"] = [series_id]
        run_forecasting_benchmark(cfg=cfg, logger=logger)

    split_df = pd.read_parquet(outputs["split_metadata_path"])
    audit_df = pd.read_parquet(outputs["task_audit_path"])
    assert set(split_df["series_id"]) == {"S1", "S2"}
    assert len(split_df) == 4
    assert len(audit_df) == 4
    assert split_df.duplicated(["series_id", "horizon", "fold_id", "split_policy"]).sum() == 0


def test_selected_architectures_are_loaded_into_forecasting_pipeline() -> None:
    cfg = load_forecasting_benchmark_config("configs/forecasting_benchmark_smoke_v1.yaml")

    selected_path = Path("configs/forecasting_selected_architectures_v1.yaml").resolve()
    payload = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    selected_models = payload["selected_models"]
    expected_model_names = [str(x["model_name"]) for x in selected_models]

    active_models = list(cfg["models"]["active"])
    for baseline in ["naive_zero", "naive_mean", "ridge_lag"]:
        assert baseline in active_models
    for model_name in expected_model_names:
        assert model_name in active_models

    selected_meta = cfg["meta"]["selected_architectures"]
    assert selected_meta["applied"] is True
    assert set(selected_meta["resolved_candidate_ids"]) == set(x["candidate_id"] for x in selected_models)
    assert selected_meta["missing_candidate_ids"] == []


def test_lstm_family_enabled_in_smoke_config() -> None:
    cfg = load_forecasting_benchmark_config("configs/forecasting_benchmark_smoke_v1.yaml")
    active_models = set(cfg["models"]["active"])
    assert "lstm_forecast" in active_models
    assert "chaotic_lstm_forecast" in active_models
    assert "lstm_forecast" not in set(cfg["models"]["inactive_but_supported"])
    assert "chaotic_lstm_forecast" not in set(cfg["models"]["inactive_but_supported"])


def test_model_materialization_matches_shortlist() -> None:
    cfg = load_forecasting_benchmark_config("configs/forecasting_benchmark_smoke_v1.yaml")
    selected_meta = cfg["meta"]["selected_architectures"]
    model_metadata = selected_meta["model_metadata"]

    selected_path = Path("configs/forecasting_selected_architectures_v1.yaml").resolve()
    payload = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    by_model = {str(x["model_name"]): x for x in payload["selected_models"]}

    for model_name, row in by_model.items():
        assert model_name in cfg["model_overrides"]
        assert model_name in model_metadata
        assert model_metadata[model_name]["candidate_id"] == row["candidate_id"]
        assert model_metadata[model_name]["selection_role"] == row["selection_role"]

        merged_expected = {}
        merged_expected.update(row.get("model_params", {}))
        merged_expected.update(row.get("runtime_params", {}))
        assert cfg["model_overrides"][model_name] == merged_expected

        model = build_model(model_name=model_name, config=cfg, logger=None)
        assert model.get_model_name() == model_name
