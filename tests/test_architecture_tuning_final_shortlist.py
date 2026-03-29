from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.architecture_tuning.final_shortlist import (
    FINAL_CONFIG_REQUIRED_KEYS,
    UNIFIED_COLUMNS,
    assign_selection_role,
    build_unified_tuning_summary_table,
    check_shortlist_candidate_presence,
    export_final_shortlist_config_yaml,
    run_architecture_tuning_final_shortlist,
)


def _candidate_row(
    model_name: str,
    candidate_id: str,
    compare_group_id: str,
    mae: float,
    fit: float,
    runtime: float,
    params_json: str,
) -> dict:
    return {
        "run_id": "run_x",
        "model_name": model_name,
        "candidate_id": candidate_id,
        "compare_group_id": compare_group_id,
        "mae_mean": mae,
        "rmse_mean": mae + 0.01,
        "mase_mean": mae + 0.5,
        "directional_accuracy_mean": 0.51,
        "fit_time_sec_mean": fit,
        "predict_time_sec_mean": 0.01,
        "total_runtime_sec": runtime,
        "status": "success",
        "notes": "",
        "params_json": params_json,
    }


def _build_shortlist_metadata() -> dict[str, dict[str, str]]:
    return {
        "esn_g004": {"family": "esn", "model_name": "esn", "selection_role": "practical_main", "rationale": "r"},
        "chaotic_esn_g003": {"family": "esn", "model_name": "chaotic_esn", "selection_role": "chaotic_counterpart", "rationale": "r"},
        "transient_chaotic_esn_g002": {
            "family": "esn",
            "model_name": "transient_chaotic_esn",
            "selection_role": "exploratory_chaotic",
            "rationale": "r",
        },
        "vanilla_mlp_g004": {"family": "mlp", "model_name": "vanilla_mlp", "selection_role": "practical_main", "rationale": "r"},
        "chaotic_mlp_g003": {"family": "mlp", "model_name": "chaotic_mlp", "selection_role": "chaotic_counterpart", "rationale": "r"},
        "logistic_g006": {"family": "logistic", "model_name": "chaotic_logistic_net", "selection_role": "nonlinear_specialist", "rationale": "r"},
        "lstm_g001": {"family": "lstm", "model_name": "lstm_forecast", "selection_role": "practical_main", "rationale": "r"},
        "chaotic_lstm_g003": {"family": "lstm", "model_name": "chaotic_lstm_forecast", "selection_role": "chaotic_counterpart", "rationale": "r"},
    }


def _build_all_candidates_df() -> pd.DataFrame:
    rows = [
        _candidate_row(
            "esn",
            "esn_g004",
            "esn_grp_004",
            mae=0.03,
            fit=0.40,
            runtime=20.0,
            params_json='{"model_params":{"n_reservoir":160,"spectral_radius":0.92,"input_scale":0.12,"leak_rate":0.9,"ridge_alpha":0.001},"runtime_params":{}}',
        ),
        _candidate_row(
            "chaotic_esn",
            "chaotic_esn_g003",
            "esn_grp_003",
            mae=0.031,
            fit=0.42,
            runtime=22.0,
            params_json='{"model_params":{"n_reservoir":128,"spectral_radius":0.98,"input_scale":0.08,"leak_rate":0.7,"ridge_alpha":0.0005},"runtime_params":{}}',
        ),
        _candidate_row(
            "transient_chaotic_esn",
            "transient_chaotic_esn_g002",
            "esn_grp_002",
            mae=0.032,
            fit=0.44,
            runtime=24.0,
            params_json='{"model_params":{"n_reservoir":96,"spectral_radius":0.95,"input_scale":0.1,"leak_rate":0.8,"ridge_alpha":0.0001},"runtime_params":{}}',
        ),
        _candidate_row(
            "vanilla_mlp",
            "vanilla_mlp_g004",
            "mlp_grp_004",
            mae=0.026,
            fit=0.14,
            runtime=4.8,
            params_json='{"model_params":{"hidden_dims":[128,64]},"runtime_params":{}}',
        ),
        _candidate_row(
            "chaotic_mlp",
            "chaotic_mlp_g003",
            "mlp_grp_003",
            mae=0.029,
            fit=0.16,
            runtime=6.1,
            params_json='{"model_params":{"hidden_dims":[64,32]},"runtime_params":{}}',
        ),
        _candidate_row(
            "chaotic_logistic_net",
            "logistic_g006",
            "logistic_alt64",
            mae=0.031,
            fit=0.44,
            runtime=16.1,
            params_json='{"model_params":{"hidden_size":64,"train_r":false,"beta":0.08,"r_min":3.55,"r_max":3.8},"runtime_params":{}}',
        ),
        _candidate_row(
            "lstm_forecast",
            "lstm_g001",
            "lstm_grp_001",
            mae=0.026,
            fit=0.42,
            runtime=15.2,
            params_json='{"model_params":{"hidden_size":32,"num_layers":1,"dropout":0.0},"runtime_params":{}}',
        ),
        _candidate_row(
            "chaotic_lstm_forecast",
            "chaotic_lstm_g003",
            "lstm_grp_003",
            mae=0.027,
            fit=2.38,
            runtime=86.3,
            params_json='{"model_params":{"hidden_size":64,"num_layers":2,"dropout":0.1},"runtime_params":{}}',
        ),
    ]
    return pd.DataFrame(rows)


def test_unified_tuning_summary_schema() -> None:
    df = _build_all_candidates_df()
    shortlist_ids = list(_build_shortlist_metadata().keys())
    unified_df, _, _ = build_unified_tuning_summary_table(
        all_candidates_df=df,
        shortlist_candidate_ids=shortlist_ids,
        shortlist_metadata=_build_shortlist_metadata(),
    )
    assert unified_df.columns.tolist() == UNIFIED_COLUMNS
    assert len(unified_df) == 8


def test_shortlist_candidate_presence_check() -> None:
    df = _build_all_candidates_df().copy()
    shortlist_ids = ["esn_g004", "missing_x"]
    found, missing = check_shortlist_candidate_presence(df, shortlist_ids)
    assert found == ["esn_g004"]
    assert missing == ["missing_x"]


def test_final_shortlist_config_schema(tmp_path) -> None:
    out_path = export_final_shortlist_config_yaml(
        output_path=tmp_path / "forecasting_selected_architectures_v1.yaml",
        run_id="run_1",
        selected_models_cfg=[
            {
                "candidate_id": "esn_g004",
                "model_name": "esn",
                "family": "esn",
                "source_run_name": "architecture_tuning_esn_v1",
                "compare_group_id": "esn_grp_004",
                "model_params": {"n_reservoir": 160},
                "runtime_params": {},
                "selection_role": "practical_main",
                "selected_for_main_run": True,
            }
        ],
        missing_candidate_ids=[],
    )
    payload = yaml.safe_load(Path(out_path).read_text(encoding="utf-8"))
    assert "selected_models" in payload
    assert len(payload["selected_models"]) == 1
    model = payload["selected_models"][0]
    assert all(k in model for k in FINAL_CONFIG_REQUIRED_KEYS)


def test_selection_role_assignment() -> None:
    metadata = _build_shortlist_metadata()
    assert assign_selection_role("esn_g004", metadata) == "practical_main"
    assert assign_selection_role("chaotic_esn_g003", metadata) == "chaotic_counterpart"
    assert assign_selection_role("transient_chaotic_esn_g002", metadata) == "exploratory_chaotic"
    assert assign_selection_role("logistic_g006", metadata) == "nonlinear_specialist"


def test_small_e2e_final_shortlist_stage(tmp_path) -> None:
    base = tmp_path / "artifacts" / "architecture_tuning"
    esn_dir = base / "esn_v1"
    mlp_dir = base / "mlp_v1"
    logistic_dir = base / "logistic_v1"
    lstm_dir = base / "lstm_v1"
    for p in [esn_dir, mlp_dir, logistic_dir, lstm_dir]:
        p.mkdir(parents=True, exist_ok=True)

    all_df = _build_all_candidates_df()
    all_df[all_df["model_name"].isin(["esn", "chaotic_esn", "transient_chaotic_esn"])].to_csv(
        esn_dir / "candidate_level_results_esn_v1.csv",
        index=False,
    )
    all_df[all_df["model_name"].isin(["vanilla_mlp", "chaotic_mlp"])].to_csv(
        mlp_dir / "candidate_level_results_mlp_v1.csv",
        index=False,
    )
    all_df[all_df["model_name"] == "chaotic_logistic_net"].to_csv(
        logistic_dir / "candidate_level_results_logistic_v1.csv",
        index=False,
    )
    all_df[all_df["model_name"].isin(["lstm_forecast", "chaotic_lstm_forecast"])].to_csv(
        lstm_dir / "candidate_level_results_lstm_v1.csv",
        index=False,
    )

    cfg = {
        "run_name": "architecture_tuning_final_shortlist_v1",
        "shortlist_candidate_ids": list(_build_shortlist_metadata().keys()),
        "shortlist_metadata": _build_shortlist_metadata(),
        "input_families": [
            {
                "family": "esn",
                "source_run_name": "architecture_tuning_esn_v1",
                "candidate_level_csv": str(esn_dir / "candidate_level_results_esn_v1.csv"),
            },
            {
                "family": "mlp",
                "source_run_name": "architecture_tuning_mlp_v1",
                "candidate_level_csv": str(mlp_dir / "candidate_level_results_mlp_v1.csv"),
            },
            {
                "family": "logistic",
                "source_run_name": "architecture_tuning_logistic_v1",
                "candidate_level_csv": str(logistic_dir / "candidate_level_results_logistic_v1.csv"),
            },
            {
                "family": "lstm",
                "source_run_name": "architecture_tuning_lstm_v1",
                "candidate_level_csv": str(lstm_dir / "candidate_level_results_lstm_v1.csv"),
            },
        ],
        "outputs": {
            "unified_summary_csv": str(tmp_path / "unified_tuning_summary_v1.csv"),
            "unified_summary_parquet": str(tmp_path / "unified_tuning_summary_v1.parquet"),
            "family_summary_csv": str(tmp_path / "family_summary_v1.csv"),
            "selected_architectures_yaml": str(tmp_path / "forecasting_selected_architectures_v1.yaml"),
            "excel_report_path": str(tmp_path / "architecture_tuning_final_shortlist_v1.xlsx"),
        },
        "artifacts": {
            "manifests": str(tmp_path),
            "logs": str(tmp_path),
        },
        "meta": {
            "config_path": "synthetic_final_shortlist",
            "project_root": str(tmp_path),
            "run_id": "architecture_tuning_final_shortlist_test_run",
            "log_path": str(tmp_path / "run.log"),
        },
    }

    logger = logging.getLogger("architecture_tuning_final_shortlist_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    result = run_architecture_tuning_final_shortlist(cfg=cfg, logger=logger)
    assert result["missing_candidate_ids"] == []
    assert len(result["shortlist_candidate_ids"]) == 8
    assert Path(result["unified_summary_csv"]).exists()
    assert Path(result["unified_summary_parquet"]).exists()
    assert Path(result["family_summary_csv"]).exists()
    assert Path(result["selected_architectures_yaml"]).exists()
    assert Path(result["excel_report_path"]).exists()
    assert Path(result["manifest_path"]).exists()
