from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.architecture_tuning.benchmark import (
    BEST_CANDIDATE_SUMMARY_COLUMNS,
    CANDIDATE_LEVEL_COLUMNS,
    PAIR_COMPARISON_COLUMNS,
    build_best_by_model_table,
    build_best_candidate_summary_table,
    build_pair_comparison_table,
    run_architecture_tuning_benchmark,
)
from src.architecture_tuning.dataset import (
    SELECTED_SERIES_COLUMNS,
    list_external_csv_files,
    sample_external_series,
    selected_series_to_frame,
)
from src.forecasting.architectures.mlp import ChaoticMLP, VanillaMLP


def _write_synthetic_external_csvs(tmp_path, n_files: int = 30, n_rows: int = 160) -> str:
    base_dir = tmp_path / "etfs"
    base_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
        close = 100 + np.cumsum(np.sin(np.linspace(0, 10, n_rows)) * 0.2 + 0.05 + i * 1e-3)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": np.full(n_rows, 1000 + i),
                "OpenInt": np.zeros(n_rows),
            }
        )
        df.to_csv(base_dir / f"ETF_{i:03d}.csv", index=False)
    return str(base_dir)


def test_reproducible_external_series_sampling(tmp_path) -> None:
    data_dir = _write_synthetic_external_csvs(tmp_path, n_files=30, n_rows=120)
    files = list_external_csv_files(data_dir)

    sampled_a = sample_external_series(files=files, sample_size=25, random_seed=777)
    sampled_b = sample_external_series(files=files, sample_size=25, random_seed=777)

    ids_a = [s.series_id for s in sampled_a]
    ids_b = [s.series_id for s in sampled_b]
    assert ids_a == ids_b


def test_selected_series_schema(tmp_path) -> None:
    data_dir = _write_synthetic_external_csvs(tmp_path, n_files=30, n_rows=120)
    files = list_external_csv_files(data_dir)
    sampled = sample_external_series(files=files, sample_size=25, random_seed=42)
    selected_df = selected_series_to_frame(sampled)

    assert selected_df.columns.tolist() == SELECTED_SERIES_COLUMNS
    assert len(selected_df) == 25
    assert selected_df["series_id"].notna().all()


def test_candidate_results_schema() -> None:
    df = pd.DataFrame([{c: pd.NA for c in CANDIDATE_LEVEL_COLUMNS}])
    assert df.columns.tolist() == CANDIDATE_LEVEL_COLUMNS
    required = {"model_name", "candidate_id", "compare_group_id", "hidden_dims", "depth", "fit_time", "predict_time", "status"}
    assert required.issubset(set(df.columns))


def test_best_by_model_summary() -> None:
    candidate_level_df = pd.DataFrame(
        [
            {
                "model_name": "esn",
                "candidate_id": "esn_1",
                "horizon": 1,
                "mae_mean": 0.10,
                "rmse_mean": 0.20,
                "mase_mean": 0.30,
                "directional_accuracy_mean": 0.51,
                "fit_time_sec_mean": 0.4,
                "predict_time_sec_mean": 0.1,
                "status": "success",
            },
            {
                "model_name": "esn",
                "candidate_id": "esn_2",
                "horizon": 1,
                "mae_mean": 0.08,
                "rmse_mean": 0.21,
                "mase_mean": 0.29,
                "directional_accuracy_mean": 0.50,
                "fit_time_sec_mean": 0.5,
                "predict_time_sec_mean": 0.1,
                "status": "success",
            },
        ]
    )
    best_df = build_best_by_model_table(candidate_level_df)
    assert len(best_df) == 1
    assert best_df.loc[0, "best_candidate_id"] == "esn_2"


def test_esn_compare_group_pairing_in_config() -> None:
    cfg_path = Path("configs/architecture_tuning_esn_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    candidates = cfg["models"]["candidates"]
    groups_by_model = {}
    for model_name in ["esn", "chaotic_esn", "transient_chaotic_esn"]:
        groups_by_model[model_name] = {str(item["compare_group_id"]) for item in candidates[model_name]}

    assert groups_by_model["esn"] == groups_by_model["chaotic_esn"] == groups_by_model["transient_chaotic_esn"]


def test_paired_esn_candidates_share_non_chaotic_params() -> None:
    cfg_path = Path("configs/architecture_tuning_esn_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    candidates = cfg["models"]["candidates"]
    non_chaotic_keys = ["n_reservoir", "spectral_radius", "input_scale", "leak_rate", "ridge_alpha", "seed"]
    base_by_group = {
        str(item["compare_group_id"]): dict(item["model_params"])
        for item in candidates["esn"]
    }

    for model_name in ["chaotic_esn", "transient_chaotic_esn"]:
        for item in candidates[model_name]:
            group_id = str(item["compare_group_id"])
            model_params = dict(item["model_params"])
            base_params = base_by_group[group_id]
            for key in non_chaotic_keys:
                assert model_params.get(key) == base_params.get(key), f"{model_name}/{group_id} key mismatch: {key}"


def test_mlp_compare_group_pairing_in_config() -> None:
    cfg_path = Path("configs/architecture_tuning_mlp_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    candidates = cfg["models"]["candidates"]
    vanilla_groups = {str(item["compare_group_id"]) for item in candidates["vanilla_mlp"]}
    chaotic_groups = {str(item["compare_group_id"]) for item in candidates["chaotic_mlp"]}
    assert vanilla_groups == chaotic_groups
    assert len(vanilla_groups) == 4


def test_mlp_paired_candidates_identical_except_activation() -> None:
    cfg_path = Path("configs/architecture_tuning_mlp_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    candidates = cfg["models"]["candidates"]
    vanilla_by_group = {str(item["compare_group_id"]): item for item in candidates["vanilla_mlp"]}
    chaotic_by_group = {str(item["compare_group_id"]): item for item in candidates["chaotic_mlp"]}
    assert set(vanilla_by_group) == set(chaotic_by_group)

    for group_id in sorted(vanilla_by_group.keys()):
        vanilla = vanilla_by_group[group_id]
        chaotic = chaotic_by_group[group_id]
        assert dict(vanilla["model_params"]) == dict(chaotic["model_params"])
        assert dict(vanilla["runtime_params"]) == dict(chaotic["runtime_params"])


def test_vanilla_and_chaotic_mlp_architecture_match_except_activation() -> None:
    torch.manual_seed(123)
    vanilla = VanillaMLP(input_size=16, hidden_dims=[64, 32])
    torch.manual_seed(123)
    chaotic = ChaoticMLP(input_size=16, hidden_dims=[64, 32])

    assert len(vanilla.hidden_layers) == len(chaotic.hidden_layers)
    vanilla_shapes = [(int(layer.in_features), int(layer.out_features)) for layer in vanilla.hidden_layers]
    chaotic_shapes = [(int(layer.in_features), int(layer.out_features)) for layer in chaotic.hidden_layers]
    assert vanilla_shapes == chaotic_shapes == [(16, 64), (64, 32)]
    assert int(vanilla.output_layer.in_features) == int(chaotic.output_layer.in_features) == 32
    assert int(vanilla.output_layer.out_features) == int(chaotic.output_layer.out_features) == 1
    assert vanilla.act.__class__.__name__ == "ReLU"
    assert hasattr(chaotic, "r")


def test_candidate_level_summary_schemas() -> None:
    candidate_level_df = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "model_name": "esn",
                "candidate_id": "esn_g001",
                "compare_group_id": "grp1",
                "horizon": 1,
                "mae_mean": 0.10,
                "rmse_mean": 0.20,
                "mase_mean": 0.30,
                "directional_accuracy_mean": 0.52,
                "fit_time_sec_mean": 0.4,
                "predict_time_sec_mean": 0.1,
                "total_runtime_sec": 0.5,
                "status": "success",
            },
            {
                "run_id": "run_1",
                "model_name": "chaotic_esn",
                "candidate_id": "chaotic_esn_g001",
                "compare_group_id": "grp1",
                "horizon": 1,
                "mae_mean": 0.09,
                "rmse_mean": 0.19,
                "mase_mean": 0.29,
                "directional_accuracy_mean": 0.53,
                "fit_time_sec_mean": 0.5,
                "predict_time_sec_mean": 0.1,
                "total_runtime_sec": 0.6,
                "status": "success",
            },
            {
                "run_id": "run_1",
                "model_name": "transient_chaotic_esn",
                "candidate_id": "transient_chaotic_esn_g001",
                "compare_group_id": "grp1",
                "horizon": 1,
                "mae_mean": 0.08,
                "rmse_mean": 0.18,
                "mase_mean": 0.28,
                "directional_accuracy_mean": 0.54,
                "fit_time_sec_mean": 0.6,
                "predict_time_sec_mean": 0.1,
                "total_runtime_sec": 0.7,
                "status": "success",
            },
        ]
    )

    pair_df = build_pair_comparison_table(candidate_level_df)
    best_summary_df = build_best_candidate_summary_table(candidate_level_df)
    assert pair_df.columns.tolist() == PAIR_COMPARISON_COLUMNS
    assert best_summary_df.columns.tolist() == BEST_CANDIDATE_SUMMARY_COLUMNS
    assert np.isclose(float(pair_df.loc[0, "chaotic_esn_delta_mae_vs_esn"]), -0.01)


def test_small_e2e_architecture_tuning_benchmark_esn_family(tmp_path) -> None:
    data_dir = _write_synthetic_external_csvs(tmp_path, n_files=30, n_rows=140)
    cfg = {
        "stage": "architecture_tuning_benchmark",
        "run_name": "architecture_tuning_test",
        "external_data_dir": data_dir,
        "file_pattern": "*.csv",
        "sample_size": 3,
        "random_seed": 123,
        "selected_models": ["esn", "chaotic_esn", "transient_chaotic_esn"],
        "candidates": {
            "esn": [
                {
                    "candidate_id": "esn_small",
                    "compare_group_id": "grp_smoke",
                    "model_params": {
                        "n_reservoir": 8,
                        "spectral_radius": 0.9,
                        "input_scale": 0.1,
                        "leak_rate": 1.0,
                        "ridge_alpha": 1.0e-4,
                        "seed": 42,
                    },
                    "runtime_params": {},
                }
            ],
            "chaotic_esn": [
                {
                    "candidate_id": "chaotic_esn_small",
                    "compare_group_id": "grp_smoke",
                    "model_params": {
                        "n_reservoir": 8,
                        "spectral_radius": 0.9,
                        "chaotic_spectral_radius": 1.2,
                        "input_scale": 0.1,
                        "leak_rate": 1.0,
                        "ridge_alpha": 1.0e-4,
                        "seed": 42,
                    },
                    "runtime_params": {},
                }
            ],
            "transient_chaotic_esn": [
                {
                    "candidate_id": "transient_chaotic_esn_small",
                    "compare_group_id": "grp_smoke",
                    "model_params": {
                        "n_reservoir": 8,
                        "spectral_radius": 0.9,
                        "input_scale": 0.1,
                        "leak_rate": 1.0,
                        "ridge_alpha": 1.0e-4,
                        "g_start": 1.2,
                        "g_end": 0.95,
                        "seed": 42,
                    },
                    "runtime_params": {},
                }
            ],
        },
        "horizons": [1],
        "window_sizes": {"1": 12},
        "validation": {"method": "rolling_origin", "n_folds": 2},
        "training": {"max_epochs": 2, "early_stopping_patience": 1, "batch_size": 16, "learning_rate": 1e-3},
        "timeouts": {"max_train_seconds_per_task": 10, "max_predict_seconds_per_task": 10},
        "device": "cpu",
        "data_transforms": {"date_column": "Date", "close_column": "Close", "min_history": 80},
        "outputs": {
            "selected_series_csv": str(tmp_path / "selected_series_architecture_tuning_v1.csv"),
            "candidate_level_results_parquet": str(tmp_path / "candidate_level_results_v1.parquet"),
            "candidate_level_results_csv": str(tmp_path / "candidate_level_results_v1.csv"),
            "series_level_results_parquet": str(tmp_path / "series_level_results_v1.parquet"),
            "pair_comparison_csv": str(tmp_path / "pair_comparison_summary_v1.csv"),
            "best_candidate_summary_csv": str(tmp_path / "best_candidate_summary_v1.csv"),
            "excel_report_path": str(tmp_path / "architecture_tuning_benchmark_v1.xlsx"),
        },
        "artifacts": {"manifests": str(tmp_path), "logs": str(tmp_path)},
        "meta": {
            "config_path": "synthetic",
            "project_root": str(tmp_path),
            "run_id": "architecture_tuning_test_run",
            "log_path": str(tmp_path / "run.log"),
        },
    }

    logger = logging.getLogger("architecture_tuning_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    result = run_architecture_tuning_benchmark(cfg=cfg, logger=logger)

    assert result["n_candidates"] == 3
    assert (tmp_path / "selected_series_architecture_tuning_v1.csv").exists()
    assert (tmp_path / "candidate_level_results_v1.parquet").exists()
    assert (tmp_path / "candidate_level_results_v1.csv").exists()
    assert (tmp_path / "series_level_results_v1.parquet").exists()
    assert (tmp_path / "pair_comparison_summary_v1.csv").exists()
    assert (tmp_path / "best_candidate_summary_v1.csv").exists()
    assert (tmp_path / "architecture_tuning_benchmark_v1.xlsx").exists()

    candidate_df = pd.read_parquet(tmp_path / "candidate_level_results_v1.parquet")
    assert candidate_df.columns.tolist() == CANDIDATE_LEVEL_COLUMNS


def test_small_e2e_architecture_tuning_benchmark_mlp_family(tmp_path) -> None:
    data_dir = _write_synthetic_external_csvs(tmp_path, n_files=20, n_rows=140)
    cfg = {
        "stage": "architecture_tuning_benchmark",
        "run_name": "architecture_tuning_test_mlp",
        "external_data_dir": data_dir,
        "file_pattern": "*.csv",
        "sample_size": 2,
        "random_seed": 123,
        "selected_models": ["vanilla_mlp", "chaotic_mlp"],
        "candidates": {
            "vanilla_mlp": [
                {
                    "candidate_id": "vanilla_mlp_small",
                    "compare_group_id": "mlp_grp_smoke",
                    "model_params": {"hidden_dims": [16], "seed": 42},
                    "runtime_params": {},
                }
            ],
            "chaotic_mlp": [
                {
                    "candidate_id": "chaotic_mlp_small",
                    "compare_group_id": "mlp_grp_smoke",
                    "model_params": {"hidden_dims": [16], "seed": 42},
                    "runtime_params": {},
                }
            ],
        },
        "horizons": [1],
        "window_sizes": {"1": 12},
        "validation": {"method": "rolling_origin", "n_folds": 2},
        "training": {"max_epochs": 2, "early_stopping_patience": 1, "batch_size": 16, "learning_rate": 1e-3},
        "timeouts": {"max_train_seconds_per_task": 10, "max_predict_seconds_per_task": 10},
        "device": "cpu",
        "data_transforms": {"date_column": "Date", "close_column": "Close", "min_history": 80},
        "outputs": {
            "selected_series_csv": str(tmp_path / "selected_series_architecture_tuning_mlp_v1.csv"),
            "candidate_level_results_parquet": str(tmp_path / "candidate_level_results_mlp_v1.parquet"),
            "candidate_level_results_csv": str(tmp_path / "candidate_level_results_mlp_v1.csv"),
            "series_level_results_parquet": str(tmp_path / "series_level_results_mlp_v1.parquet"),
            "pair_comparison_csv": str(tmp_path / "pair_comparison_summary_mlp_v1.csv"),
            "best_candidate_summary_csv": str(tmp_path / "best_candidate_summary_mlp_v1.csv"),
            "excel_report_path": str(tmp_path / "architecture_tuning_benchmark_mlp_v1.xlsx"),
        },
        "artifacts": {"manifests": str(tmp_path), "logs": str(tmp_path)},
        "meta": {
            "config_path": "synthetic_mlp",
            "project_root": str(tmp_path),
            "run_id": "architecture_tuning_test_mlp_run",
            "log_path": str(tmp_path / "run_mlp.log"),
        },
    }

    logger = logging.getLogger("architecture_tuning_mlp_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    result = run_architecture_tuning_benchmark(cfg=cfg, logger=logger)
    assert result["n_candidates"] == 2

    candidate_df = pd.read_parquet(tmp_path / "candidate_level_results_mlp_v1.parquet")
    assert candidate_df.columns.tolist() == CANDIDATE_LEVEL_COLUMNS
    for column in ["hidden_dims", "depth", "fit_time", "predict_time"]:
        assert column in candidate_df.columns
