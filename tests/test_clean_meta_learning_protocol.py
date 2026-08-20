from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import src.meta_modeling.classification_pipeline as clean_pipeline


MODEL_ORDER = [
    "chaotic_esn",
    "chaotic_logistic_net",
    "chaotic_lstm_forecast",
    "chaotic_mlp",
    "esn",
    "lstm_forecast",
    "naive_mean",
    "naive_zero",
    "ridge_lag",
    "transient_chaotic_esn",
    "vanilla_mlp",
]


def _write_clean_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    feature_names = [f"feat_{idx:02d}" for idx in range(25)]
    wide_rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | int | str]] = []
    for series_idx in range(12):
        series_id = f"S{series_idx:02d}"
        for fold_id in (1, 2, 3):
            features = {name: float(series_idx * 100 + fold_id + idx / 100) for idx, name in enumerate(feature_names)}
            feature_rows.append({"series_id": series_id, "horizon": 1, "fold_id": fold_id, **features})
            wide_row: dict[str, float | int | str] = {"series_id": series_id, "horizon": 1, "fold_id": fold_id, **features}
            winner = (series_idx + fold_id) % 4
            for model_idx, model_name in enumerate(MODEL_ORDER):
                rmse = 0.1 + abs(model_idx - winner) * 0.02 + series_idx * 1e-4 + fold_id * 1e-5
                da = 0.8 - abs(model_idx - winner) * 0.03 - series_idx * 1e-4
                wide_row[f"rmse__{model_name}"] = rmse
                wide_row[f"da__{model_name}"] = da
                metric_rows.append(
                    {
                        "series_id": series_id,
                        "horizon": 1,
                        "fold_id": fold_id,
                        "model_name": model_name,
                        "status": "success",
                        "rmse": rmse,
                        "directional_accuracy": da,
                    }
                )
            wide_rows.append(wide_row)
    meta_path = tmp_path / "clean_fold_aware_meta_inputs.parquet"
    features_path = tmp_path / "final_train_only_features_by_fold.parquet"
    metrics_path = tmp_path / "metrics_long.parquet"
    pd.DataFrame(wide_rows).to_parquet(meta_path, index=False)
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(metric_rows).to_parquet(metrics_path, index=False)
    return meta_path, features_path, metrics_path


def _clean_cfg(tmp_path: Path, meta_path: Path, features_path: Path, metrics_path: Path) -> dict:
    return {
        "run_name": "clean_protocol_test",
        "stage": "clean_meta_learning_v1",
        "feature_scope": "fold_aware_train_only",
        "inputs": {
            "meta_inputs_path": str(meta_path),
            "features_path": str(features_path),
            "metrics_path": str(metrics_path),
        },
        "expected_observations": 36,
        "expected_feature_count": 25,
        "expected_model_count": 11,
        "horizons": [1],
        "metrics": ["rmse"],
        "classification_models": ["fake_classifier"],
        "balancing_modes": ["default"],
        "feature_sets": ["full"],
        "candidate_selection": {"top_k_values": [3, 4], "closeness_tolerance": 0.05},
        "decision_rules": ["top_1", "confidence_fallback"],
        "confidence_thresholds": [0.80],
        "random_seed": 42,
        "n_repeats": 1,
        "split": {"test_size": 0.2, "validation_size": 0.2, "random_seed": 42, "n_repeats": 1, "random_seeds": []},
        "output_dir": str(tmp_path / "clean_outputs"),
        "report_dir": str(tmp_path / "clean_reports"),
        "outputs": {},
        "artifacts": {"manifests": str(tmp_path / "manifests"), "meta_modeling": str(tmp_path / "clean_outputs")},
        "meta": {"config_path": "clean_protocol_test.yaml", "project_root": str(tmp_path), "run_id": "clean_protocol_test_run"},
    }


class _FakeClassifier:
    def __init__(self, fitted_X: list[np.ndarray]) -> None:
        self.fitted_X = fitted_X
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeClassifier":
        self.fitted_X.append(np.asarray(X).copy())
        self.classes_ = np.unique(y).astype(int)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        probs = np.full((len(X), len(self.classes_)), 0.1 / max(1, len(self.classes_) - 1), dtype=float)
        probs[:, 0] = 0.9
        return probs


def test_clean_protocol_end_to_end_and_train_only_fit(tmp_path: Path, monkeypatch) -> None:
    meta_path, features_path, metrics_path = _write_clean_inputs(tmp_path)
    cfg = _clean_cfg(tmp_path, meta_path, features_path, metrics_path)
    fitted_X: list[np.ndarray] = []
    monkeypatch.setattr(clean_pipeline, "build_meta_classifier", lambda **_: _FakeClassifier(fitted_X))
    result = clean_pipeline.run_meta_modeling_experiments(cfg, logging.getLogger("clean_protocol_test"))

    split_df = pd.read_csv(cfg["outputs"]["split_assignments_csv_path"])
    by_split = {name: set(g["series_id"].astype(str)) for name, g in split_df.groupby("split")}
    assert not (by_split["train"] & by_split["validation"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["validation"] & by_split["test"])
    assert split_df.groupby(["repeat_id", "series_id"])["split"].nunique().max() == 1
    assert split_df.groupby(["repeat_id", "series_id"])["fold_id"].nunique().eq(3).all()

    train_numeric_ids = {float(int(sid[1:]) * 100 + fold) for sid in by_split["train"] for fold in (1, 2, 3)}
    assert fitted_X
    assert all(set(X[:, 0]).issubset(train_numeric_ids) for X in fitted_X)

    validation_df = pd.read_csv(cfg["outputs"]["task_results_csv_path"])
    selected_df = pd.read_csv(cfg["outputs"]["best_config_per_task_csv_path"])
    test_df = pd.read_csv(cfg["outputs"]["selected_test_results_csv_path"])
    assert set(validation_df["evaluation_partition"]) == {"validation"}
    assert int(validation_df["selected_by_validation"].sum()) == 1
    assert len(selected_df) == 1 and not bool(selected_df.iloc[0]["test_used_for_selection"])
    assert len(test_df) == 1 and set(test_df["evaluation_partition"]) == {"test"}
    assert int(test_df.iloc[0]["test_evaluations_for_task"]) == 1

    dataset_df = pd.read_csv(cfg["outputs"]["meta_dataset_summary_csv_path"])
    assert set(dataset_df["top_k_ranking_scope"]) == {"train_only"}
    assert set(dataset_df["fixed_model_scope"]) == {"train_only"}
    assert set(dataset_df["classifier_fit_scope"]) == {"train_only"}
    diagnostics = pd.read_csv(cfg["outputs"]["winner_label_diagnostics_csv_path"])
    assert {"best_score_tie_flag", "tied_best_count", "first_second_score_margin", "final_winner_label"}.issubset(diagnostics.columns)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["inputs_used"] == {
        "meta_inputs_path": str(meta_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "features_path": str(features_path.resolve()),
    }
    assert manifest["selection_protocol"]["test_used_for_selection"] is False


def test_top_k_and_fixed_model_ignore_non_train_scores() -> None:
    train_rows = []
    heldout_rows = []
    for object_idx in range(6):
        for model_name, train_score, heldout_score in (("a", 1.0, 100.0), ("b", 2.0, 0.1), ("c", 3.0, 0.2)):
            train_rows.append({"object_id": f"train_{object_idx}", "horizon": 1, "model_name": model_name, "rmse": train_score})
            heldout_rows.append({"object_id": f"heldout_{object_idx}", "horizon": 1, "model_name": model_name, "rmse": heldout_score})
    candidates = clean_pipeline._candidate_set_from_train_v2(pd.DataFrame(train_rows), horizon=1, metric="rmse", top_k=2)
    assert candidates.sort_values("candidate_rank")["model_name"].tolist() == ["a", "b"]
    y_train = np.array([[1.0, 2.0], [1.1, 2.1]])
    y_test = np.array([[100.0, 0.1]])
    fixed = clean_pipeline.compute_best_single_baseline(y_train, y_test, ["a", "b"], "min")
    assert fixed["baseline_model"] == "a"


def test_validation_selection_uses_metric_direction_and_deterministic_tie_break() -> None:
    rows = pd.DataFrame(
        [
            {"status": "success", "achieved_metric": 0.20, "config_order": 3, "test_metric": 0.01},
            {"status": "success", "achieved_metric": 0.10, "config_order": 2, "test_metric": 9.00},
            {"status": "success", "achieved_metric": 0.10, "config_order": 1, "test_metric": 8.00},
        ]
    )
    assert int(clean_pipeline._select_validation_configuration(rows, "min")["config_order"]) == 1
    da_rows = rows.assign(achieved_metric=[0.70, 0.80, 0.80])
    assert int(clean_pipeline._select_validation_configuration(da_rows, "max")["config_order"]) == 1


def test_winner_diagnostics_preserve_first_model_tie_break() -> None:
    diag = clean_pipeline._winner_diagnostics(np.array([0.8, 0.8, 0.7]), ["a", "b", "c"], "max")
    assert diag["best_score_tie_flag"] == 1
    assert diag["tied_best_count"] == 2
    assert diag["first_second_score_margin"] == 0.0
    assert diag["final_winner_label"] == "a"


def test_predeclared_feature_family_sets_are_fixed_and_validated() -> None:
    available = ["a", "b", "c", "d"]
    cfg = {
        "feature_sets": ["full_4", "without_d", "nonlinear"],
        "feature_set_definitions": {
            "full_4": ["a", "b", "c", "d"],
            "without_d": ["a", "b", "c"],
            "nonlinear": ["c", "d"],
        },
    }
    assert clean_pipeline._resolve_clean_feature_sets(cfg, available) == cfg["feature_set_definitions"]
