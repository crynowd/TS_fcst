from __future__ import annotations

import json
import math
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.meta_modeling.experimental_pipeline import (
    METRIC_DIRECTIONS,
    _evaluate_predictions,
    _normalize_metric_alias,
    _pick_best_index,
    _train_only_feature_selection,
    build_candidate_sets,
    build_meta_dataset,
    build_task,
    compute_best_single_baseline,
    repeat_seeds,
    split_by_series_ids,
)
from src.meta_modeling.models import build_meta_classifier
from src.reporting.excel_export import export_meta_modeling_excel
from src.utils.manifest import get_git_commit, write_manifest


JOIN_KEYS_V2 = ["series_id", "horizon", "fold_id"]


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    return [value]


def _metric_direction(metric: str) -> str:
    metric = _normalize_metric_alias(metric)
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(f"Unsupported target metric for meta-modeling v2: {metric}")
    return METRIC_DIRECTIONS[metric]


def _finite_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _catboost_available() -> bool:
    return importlib.util.find_spec("catboost") is not None


def _build_v2_output_defaults(cfg: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(str(cfg.get("output_dir", cfg.get("outputs", {}).get("output_dir", "artifacts/meta_modeling")))).resolve()
    report_dir = Path(str(cfg.get("report_dir", cfg.get("outputs", {}).get("report_dir", "artifacts/reports/forecasting_audit_v2")))).resolve()
    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v2"))
    defaults = {
        "meta_dataset_summary_csv_path": output_dir / "meta_dataset_summary_v2.csv",
        "meta_dataset_summary_parquet_path": output_dir / "meta_dataset_summary_v2.parquet",
        "model_order_mapping_csv_path": output_dir / "model_order_mapping_v2.csv",
        "routing_rows_parquet_path": output_dir / "routing_rows_v2.parquet",
        "routing_rows_csv_path": output_dir / "routing_rows_v2.csv",
        "task_results_parquet_path": output_dir / "task_results_v2.parquet",
        "task_results_csv_path": output_dir / "task_results_v2.csv",
        "split_assignments_csv_path": output_dir / "split_assignments_v2.csv",
        "repeat_aggregated_results_csv_path": output_dir / "repeat_aggregated_results_v2.csv",
        "best_single_baseline_by_repeat_csv_path": output_dir / "best_single_baseline_by_repeat_v2.csv",
        "feature_list_csv_path": output_dir / "feature_list_v2.csv",
        "feature_manifest_json_path": output_dir / "feature_manifest_v2.json",
        "comparison_table_csv_path": output_dir / "comparison_table_v2.csv",
        "candidate_models_csv_path": output_dir / "candidate_models_v2.csv",
        "classification_probabilities_csv_path": output_dir / "classification_probabilities_v2.csv",
        "per_class_metrics_csv_path": output_dir / "per_class_metrics_v2.csv",
        "confusion_matrix_csv_path": output_dir / "confusion_matrix_v2.csv",
        "class_distribution_csv_path": output_dir / "class_distribution_v2.csv",
        "confidence_summary_csv_path": output_dir / "confidence_summary_v2.csv",
        "selected_features_csv_path": output_dir / "selected_features_v2.csv",
        "candidate_filtering_summary_csv_path": output_dir / "candidate_filtering_summary_v2.csv",
        "feature_regime_summary_csv_path": output_dir / "feature_regime_summary_v2.csv",
        "best_config_per_task_csv_path": output_dir / "best_config_per_task_v2.csv",
        "confident_examples_csv_path": output_dir / "confident_examples_v2.csv",
        "coverage_checks_json_path": report_dir / "metamodeling_v2_coverage_checks.json",
        "resolved_config_snapshot_path": report_dir / f"{run_name}_resolved_config_snapshot.yaml",
        "excel_report_path": report_dir / "meta_modeling_experiments_v2.xlsx",
    }
    configured = {}
    if str(cfg.get("feature_scope", "")).strip() != "fold_aware_train_only":
        configured = dict(cfg.get("outputs", {}))
    return {k: str(Path(str(configured.get(k, v))).resolve()) for k, v in defaults.items()}


def _feature_cols_v2(features_df: pd.DataFrame) -> list[str]:
    non_features = set(JOIN_KEYS_V2) | {
        "train_start",
        "train_end",
        "n_train",
        "feature_status",
        "feature_warning_flags",
        "ticker",
        "market",
    }
    cols = [c for c in features_df.columns if c not in non_features and pd.api.types.is_numeric_dtype(features_df[c])]
    if not cols:
        raise ValueError("No numeric fold-aware feature columns found")
    return cols


def _coverage_checks_v2(
    metrics_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    horizons: list[int],
    metrics: list[str],
    expected_n_folds: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    required_metrics_cols = set(JOIN_KEYS_V2 + ["model_name", *metrics])
    required_feature_cols = set(JOIN_KEYS_V2)
    missing_metric_cols = sorted(required_metrics_cols - set(metrics_df.columns))
    missing_feature_cols = sorted(required_feature_cols - set(features_df.columns))
    if missing_metric_cols or missing_feature_cols:
        raise ValueError(f"Missing v2 input columns: metrics={missing_metric_cols}, features={missing_feature_cols}")

    m = metrics_df[metrics_df["horizon"].astype(int).isin(horizons)].copy()
    f = features_df[features_df["horizon"].astype(int).isin(horizons)].copy()
    m_keys = m[JOIN_KEYS_V2].drop_duplicates()
    f_keys = f[JOIN_KEYS_V2].drop_duplicates()
    missing_features = m_keys.merge(f_keys, on=JOIN_KEYS_V2, how="left", indicator=True)
    missing_features = missing_features[missing_features["_merge"] == "left_only"]
    feature_only = f_keys.merge(m_keys, on=JOIN_KEYS_V2, how="left", indicator=True)
    feature_only = feature_only[feature_only["_merge"] == "left_only"]

    fold_counts = f.groupby(["series_id", "horizon"], dropna=False)["fold_id"].nunique()
    bad_fold_counts = fold_counts[fold_counts != int(expected_n_folds)]
    feature_cols = _feature_cols_v2(f)
    feature_values = f[feature_cols].to_numpy(dtype=np.float64)
    non_finite_count = int((~np.isfinite(feature_values)).sum())

    metric_dupes = int(m.duplicated(subset=JOIN_KEYS_V2 + ["model_name"]).sum())
    per_object_models = m.groupby(JOIN_KEYS_V2, dropna=False)["model_name"].nunique()
    expected_models = int(per_object_models.max()) if len(per_object_models) else 0
    incomplete_objects = int((per_object_models != expected_models).sum())
    metric_nan_counts = {metric: int(pd.to_numeric(m[metric], errors="coerce").isna().sum()) for metric in metrics}

    checks.update(
        {
            "metrics_rows": int(len(m)),
            "feature_rows": int(len(f)),
            "metrics_objects": int(len(m_keys)),
            "feature_objects": int(len(f_keys)),
            "missing_feature_matches": int(len(missing_features)),
            "feature_rows_without_metrics_filtered": int(len(feature_only)),
            "expected_n_folds": int(expected_n_folds),
            "bad_series_horizon_fold_counts": int(len(bad_fold_counts)),
            "numeric_feature_count": int(len(feature_cols)),
            "non_finite_feature_values": non_finite_count,
            "metric_duplicate_model_rows": metric_dupes,
            "expected_models_per_object": expected_models,
            "incomplete_metric_objects": incomplete_objects,
            "metric_nan_counts": metric_nan_counts,
        }
    )
    failures = []
    if len(missing_features):
        failures.append("metrics rows without matching fold-aware features")
    if len(bad_fold_counts):
        failures.append("unexpected number of folds")
    if non_finite_count:
        failures.append("NaN/inf in features")
    if metric_dupes:
        failures.append("duplicate metric rows by series_id/horizon/fold_id/model_name")
    if incomplete_objects:
        failures.append("missing model metrics for some series_id/horizon/fold_id objects")
    if any(v > 0 for v in metric_nan_counts.values()):
        failures.append("NaN target metric values")
    checks["status"] = "passed" if not failures else "failed"
    checks["failures"] = failures
    if failures:
        raise ValueError(f"Meta-modeling v2 coverage checks failed: {failures}")
    return checks


def _load_meta_dataset_v2(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = dict(cfg.get("inputs", {}))
    metrics_path = str(inputs.get("metrics_path") or inputs.get("forecasting_series_metrics_path") or cfg.get("metrics_path", "")).strip()
    features_path = str(inputs.get("features_path") or cfg.get("features_path", "")).strip()
    if not metrics_path or not features_path:
        raise ValueError("meta-modeling v2 requires metrics_path and features_path")

    metrics_df = pd.read_parquet(metrics_path).copy()
    features_df = pd.read_parquet(features_path).copy()
    horizons = [int(x) for x in cfg.get("horizons", cfg.get("target_horizons", [1, 5, 20]))]
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("metrics", cfg.get("target_metrics", ["rmse", "directional_accuracy"]))]
    expected_n_folds = int(cfg.get("expected_n_folds", 3))
    coverage = _coverage_checks_v2(metrics_df, features_df, horizons=horizons, metrics=target_metrics, expected_n_folds=expected_n_folds)

    feature_cols = _feature_cols_v2(features_df)
    f = features_df[features_df["horizon"].astype(int).isin(horizons)][JOIN_KEYS_V2 + feature_cols].copy()
    m = metrics_df[metrics_df["horizon"].astype(int).isin(horizons)][JOIN_KEYS_V2 + ["model_name", *target_metrics]].copy()
    for c in ["horizon", "fold_id"]:
        f[c] = pd.to_numeric(f[c], errors="raise").astype(int)
        m[c] = pd.to_numeric(m[c], errors="raise").astype(int)
    m["model_name"] = m["model_name"].astype(str)
    merged = m.merge(f, on=JOIN_KEYS_V2, how="inner", validate="many_to_one")
    merged["object_id"] = (
        merged["series_id"].astype(str)
        + "|h"
        + merged["horizon"].astype(str)
        + "|f"
        + merged["fold_id"].astype(str)
    )
    summary = {
        "join_key": "object_id",
        "join_keys": JOIN_KEYS_V2,
        "feature_cols": feature_cols,
        "feature_source_path": str(Path(features_path).resolve()),
        "forecasting_series_metrics_path": str(Path(metrics_path).resolve()),
        "forecasting_manifest_path": "",
        "n_rows": int(len(merged)),
        "n_series": int(merged["series_id"].nunique()),
        "horizons": horizons,
        "models": sorted(merged["model_name"].unique().tolist()),
        "coverage_checks": coverage,
    }
    return merged, summary


def _split_object_indices_by_series(object_df: pd.DataFrame, cfg: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    unique_ids = object_df["series_id"].astype(str).drop_duplicates().to_numpy()
    split = split_by_series_ids(unique_ids, cfg=cfg, random_seed=int(seed))
    train_ids = set(unique_ids[split.train].astype(str).tolist())
    val_ids = set(unique_ids[split.validation].astype(str).tolist())
    test_ids = set(unique_ids[split.test].astype(str).tolist())
    train_idx = np.where(object_df["series_id"].astype(str).isin(train_ids).to_numpy())[0]
    val_idx = np.where(object_df["series_id"].astype(str).isin(val_ids).to_numpy())[0]
    test_idx = np.where(object_df["series_id"].astype(str).isin(test_ids).to_numpy())[0]
    overlap = (train_ids & test_ids) | (train_ids & val_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Series split leakage detected: {sorted(overlap)[:5]}")
    meta = {
        "train_series": len(train_ids),
        "validation_series": len(val_ids),
        "test_series": len(test_ids),
        "overlap_series": len(overlap),
    }
    return train_idx, val_idx, test_idx, meta


def _candidate_set_from_train_v2(
    train_long_df: pd.DataFrame,
    *,
    horizon: int,
    metric: str,
    top_k: int,
) -> pd.DataFrame:
    direction = _metric_direction(metric)
    sdf = train_long_df[train_long_df["horizon"].astype(int) == int(horizon)][["object_id", "model_name", metric]].dropna().copy()
    mean_tbl = sdf.groupby("model_name", sort=False)[metric].mean().reset_index(name="mean_metric")
    winners = []
    for _, g in sdf.groupby("object_id", sort=False):
        asc = direction == "min"
        winners.append(str(g.sort_values([metric, "model_name"], ascending=[asc, True], kind="stable").iloc[0]["model_name"]))
    win_counts = pd.Series(winners).value_counts().rename_axis("model_name").reset_index(name="win_count") if winners else pd.DataFrame(columns=["model_name", "win_count"])
    hwork = mean_tbl.merge(win_counts, on="model_name", how="left")
    hwork["win_count"] = hwork["win_count"].fillna(0).astype(int)
    best_value = float(hwork["mean_metric"].min()) if direction == "min" else float(hwork["mean_metric"].max())
    hwork["gap_to_best"] = (hwork["mean_metric"] - best_value) if direction == "min" else (best_value - hwork["mean_metric"])
    hwork = hwork.sort_values(["gap_to_best", "win_count", "model_name"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    k_eff = min(max(1, int(top_k)), len(hwork))
    selected = hwork.head(k_eff).copy()
    selected["horizon"] = int(horizon)
    selected["target_metric"] = metric
    selected["candidate_set"] = f"top_{k_eff}"
    selected["top_k"] = int(k_eff)
    selected["candidate_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["ranking_scope"] = "meta_train_only"
    return selected[["horizon", "target_metric", "candidate_set", "top_k", "candidate_rank", "model_name", "mean_metric", "win_count", "gap_to_best", "ranking_scope"]]


def _build_task_arrays_v2(
    meta_long_df: pd.DataFrame,
    *,
    horizon: int,
    metric: str,
    feature_cols: list[str],
    candidate_models: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    sub = meta_long_df[
        (meta_long_df["horizon"].astype(int) == int(horizon))
        & (meta_long_df["model_name"].astype(str).isin(candidate_models))
    ].copy()
    model_order = sorted(sub["model_name"].astype(str).unique().tolist())
    pivot = sub.pivot_table(index="object_id", columns="model_name", values=metric, aggfunc="mean").reindex(columns=model_order)
    objects = (
        sub[["object_id", "series_id", "horizon", "fold_id", *feature_cols]]
        .drop_duplicates(subset=["object_id"], keep="first")
        .set_index("object_id")
    )
    aligned = objects.join(pivot, how="inner").dropna(axis=0, how="any").reset_index()
    if aligned.empty:
        raise ValueError(f"Empty v2 task: h={horizon}, metric={metric}, candidates={candidate_models}")
    return aligned, aligned[feature_cols].to_numpy(dtype=np.float64), aligned[model_order].to_numpy(dtype=np.float64), model_order


def _evaluate_v2(
    *,
    run_id: str,
    repeat_id: int,
    seed: int,
    model_name: str,
    horizon: int,
    metric: str,
    feature_set: str,
    candidate_set: str,
    object_df: pd.DataFrame,
    y: np.ndarray,
    model_order: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    selected_idx: np.ndarray,
    class_probs: np.ndarray,
    top2_idx: np.ndarray,
    confidence: np.ndarray,
    balancing: str,
    decision_rule: str,
    threshold: float | None,
    selected_feature_count: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    direction = _metric_direction(metric)
    y_train = y[train_idx]
    y_test = y[test_idx]
    baseline = compute_best_single_baseline(y_train=y_train, y_test=y_test, model_order=model_order, direction=direction)
    true_cls = np.array([_pick_best_index(row, direction) for row in y_test], dtype=int)
    rows: list[dict[str, Any]] = []
    for pos, obj_idx in enumerate(test_idx):
        actual = y[obj_idx]
        pred_idx = int(selected_idx[pos])
        oracle_idx = int(true_cls[pos])
        achieved = float(actual[pred_idx])
        oracle = float(actual[oracle_idx])
        base = float(baseline["baseline_values"][pos])
        improvement = (base - achieved) if direction == "min" else (achieved - base)
        gap = (achieved - oracle) if direction == "min" else (oracle - achieved)
        obj = object_df.iloc[int(obj_idx)]
        rows.append(
            {
                "run_id": run_id,
                "repeat_id": int(repeat_id),
                "split_seed": int(seed),
                "method": "classification",
                "model": model_name,
                "horizon": int(horizon),
                "target_metric": metric,
                "feature_set": feature_set,
                "candidate_set": candidate_set,
                "selected_feature_count": int(selected_feature_count),
                "object_id": str(obj["object_id"]),
                "series_id": str(obj["series_id"]),
                "fold_id": int(obj["fold_id"]),
                "balancing_mode": balancing,
                "decision_rule": decision_rule,
                "confidence_threshold": float(threshold) if threshold is not None else np.nan,
                "selected_model": model_order[pred_idx],
                "selected_model_top2": model_order[int(top2_idx[pos])],
                "oracle_model": model_order[oracle_idx],
                "best_single_model": baseline["baseline_model"],
                "achieved_metric": achieved,
                "oracle_metric": oracle,
                "baseline_metric": base,
                "improvement_vs_best_single": float(improvement),
                "gap_to_oracle": float(gap),
                "routing_hit_oracle": int(pred_idx == oracle_idx),
                "top2_hit_oracle": int(oracle_idx in {pred_idx, int(top2_idx[pos])}),
                "confidence": float(confidence[pos]),
                "fallback_applied": int(threshold is not None and float(confidence[pos]) < float(threshold)),
                "predicted_scores_json": json.dumps([float(x) for x in class_probs[pos]]),
                "actual_metric_vector_json": json.dumps([float(x) for x in actual]),
                "model_order_json": json.dumps(model_order),
                "class_probability_json": json.dumps({m: float(v) for m, v in zip(model_order, class_probs[pos])}),
            }
        )
    routing = pd.DataFrame(rows)
    pred_cls = np.asarray(selected_idx, dtype=int)
    summary = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(seed),
        "method": "classification",
        "model": model_name,
        "horizon": int(horizon),
        "target_metric": metric,
        "feature_set": feature_set,
        "candidate_set": candidate_set,
        "selected_feature_count": int(selected_feature_count),
        "balancing_mode": balancing,
        "decision_rule": decision_rule,
        "confidence_threshold": float(threshold) if threshold is not None else np.nan,
        "n_test_series": int(object_df.iloc[test_idx]["series_id"].nunique()),
        "n_test_objects": int(len(test_idx)),
        "achieved_metric": float(routing["achieved_metric"].mean()),
        "best_single_metric": float(routing["baseline_metric"].mean()),
        "oracle_metric": float(routing["oracle_metric"].mean()),
        "improvement_vs_best_single": float(routing["improvement_vs_best_single"].mean()),
        "gap_to_oracle": float(routing["gap_to_oracle"].mean()),
        "routing_hit_oracle_rate": float(routing["routing_hit_oracle"].mean()),
        "classification_accuracy": float(accuracy_score(true_cls, pred_cls)),
        "balanced_accuracy": float(balanced_accuracy_score(true_cls, pred_cls)),
        "top2_hit_rate": float(routing["top2_hit_oracle"].mean()),
        "fallback_rate": float(routing["fallback_applied"].mean()),
        "status": "success",
        "notes": "",
    }
    baseline_row = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(seed),
        "horizon": int(horizon),
        "target_metric": metric,
        "feature_set": feature_set,
        "candidate_set": candidate_set,
        "selected_feature_count": int(selected_feature_count),
        "balancing_mode": balancing,
        "decision_rule": decision_rule,
        "confidence_threshold": float(threshold) if threshold is not None else np.nan,
        "best_single_model": baseline["baseline_model"],
        "best_single_train_mean_metric": float(baseline["train_mean_by_model"][baseline["baseline_idx"]]),
        "best_single_test_mean_metric": summary["best_single_metric"],
    }
    return routing, summary, baseline_row


def _run_meta_modeling_experiments_v2(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v2"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    started = time.perf_counter()
    start_ts = datetime.now(timezone.utc)
    output_cfg = _build_v2_output_defaults(cfg)
    cfg["outputs"] = output_cfg
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("metrics", cfg.get("target_metrics", ["rmse", "directional_accuracy"]))]
    horizons = [int(x) for x in cfg.get("horizons", [1, 5, 20])]
    classifiers = [str(x) for x in cfg.get("classification_models", ["logistic_regression", "random_forest_classifier"])]
    if "catboost_classifier" in classifiers and not _catboost_available():
        logger.warning("CatBoost not installed, skipping catboost_classifier")
        classifiers = [x for x in classifiers if x != "catboost_classifier"]
    balancing_modes = [str(x) for x in cfg.get("balancing_modes", ["default"])]
    decision_rules = [str(x) for x in cfg.get("decision_rules", ["top_1"])]
    thresholds = [float(x) for x in cfg.get("confidence_thresholds", [])]
    top_k_values = [int(x) for x in cfg.get("candidate_selection", {}).get("top_k_values", cfg.get("candidate_top_k", [3]))]
    seeds = repeat_seeds({"random_seed": int(cfg.get("random_seed", cfg.get("split", {}).get("random_seed", 42))), "n_repeats": int(cfg.get("n_repeats", cfg.get("split", {}).get("n_repeats", 1))), "random_seeds": cfg.get("split", {}).get("random_seeds", [])})
    cfg["split"] = {**dict(cfg.get("split", {})), "random_seed": int(cfg.get("random_seed", cfg.get("split", {}).get("random_seed", 42))), "n_repeats": len(seeds)}
    catboost_root = Path(str(cfg.get("artifacts", {}).get("meta_modeling", "artifacts/meta_modeling"))).resolve() / "catboost_info" / run_id
    catboost_root.mkdir(parents=True, exist_ok=True)

    meta_long_df, ds = _load_meta_dataset_v2(cfg)
    feature_cols = list(ds["feature_cols"])
    feature_sets = {"full": feature_cols}
    if any(x in {"selected", "selected_features"} for x in _as_list(cfg.get("feature_sets"), [])):
        feature_sets["selected"] = feature_cols
    rule_specs: list[tuple[str, float | None]] = [("top_1", None)] if "top_1" in decision_rules else []
    for threshold in thresholds:
        rule = f"confidence_fallback_{threshold:.2f}"
        if rule in decision_rules or "confidence_fallback" in decision_rules:
            rule_specs.append((rule, threshold))
    if not rule_specs:
        rule_specs = [("top_1", None)]

    total = len(feature_sets) * len(horizons) * len(target_metrics) * len(seeds) * len(top_k_values) * len(classifiers) * len(balancing_modes) * len(rule_specs)
    processed = completed = failed = skipped = 0
    routing_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_single_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    probs_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    for feature_set_name, base_feature_cols in feature_sets.items():
        for horizon in horizons:
            horizon_df = meta_long_df[meta_long_df["horizon"].astype(int) == int(horizon)].copy()
            object_base = horizon_df[["object_id", "series_id", "horizon", "fold_id", *base_feature_cols]].drop_duplicates("object_id").reset_index(drop=True)
            for metric in target_metrics:
                for repeat_id, seed in enumerate(seeds, start=1):
                    train_obj_idx, val_obj_idx, test_obj_idx, split_meta = _split_object_indices_by_series(object_base, cfg, int(seed))
                    split_ids = object_base["series_id"].astype(str)
                    if set(split_ids.iloc[train_obj_idx]) & set(split_ids.iloc[test_obj_idx]):
                        raise ValueError("Series leakage between meta-train and meta-test")
                    marker = np.full(len(object_base), "test", dtype=object)
                    marker[train_obj_idx] = "train"
                    marker[val_obj_idx] = "validation"
                    for obj_i, row in object_base.iterrows():
                        split_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "object_id": str(row["object_id"]), "series_id": str(row["series_id"]), "fold_id": int(row["fold_id"]), "split": str(marker[obj_i])})
                    train_object_ids = set(object_base.iloc[train_obj_idx]["object_id"].astype(str))
                    train_long_df = horizon_df[horizon_df["object_id"].astype(str).isin(train_object_ids)]
                    for top_k in top_k_values:
                        try:
                            cdf = _candidate_set_from_train_v2(train_long_df, horizon=horizon, metric=metric, top_k=top_k)
                            cdf["run_id"] = run_id
                            cdf["repeat_id"] = int(repeat_id)
                            cdf["split_seed"] = int(seed)
                            candidate_rows.append(cdf)
                            candidate_models = cdf.sort_values("candidate_rank", kind="stable")["model_name"].astype(str).tolist()
                            object_df, X, y, model_order = _build_task_arrays_v2(meta_long_df, horizon=horizon, metric=metric, feature_cols=base_feature_cols, candidate_models=candidate_models)
                            train_idx, val_idx, test_idx, _ = _split_object_indices_by_series(object_df, cfg, int(seed))
                            direction = _metric_direction(metric)
                            y_train_cls = np.array([_pick_best_index(row, direction) for row in y[train_idx]], dtype=int)
                            feat_idx = np.arange(len(base_feature_cols), dtype=int)
                            selected_names = list(base_feature_cols)
                            if feature_set_name in {"selected", "selected_features"}:
                                feat_idx, selected_names, _ = _train_only_feature_selection(X[train_idx], y_train_cls, base_feature_cols, cfg)
                            Xtr = X[train_idx][:, feat_idx]
                            Xte = X[test_idx][:, feat_idx]
                            baseline_idx = int(compute_best_single_baseline(y_train=y[train_idx], y_test=y[test_idx], model_order=model_order, direction=direction)["baseline_idx"])
                            dataset_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "candidate_set": str(cdf["candidate_set"].iloc[0]), "candidate_size": int(len(model_order)), "n_objects": int(len(object_df)), "n_series": int(object_df["series_id"].nunique()), "n_train_objects": int(len(train_idx)), "n_test_objects": int(len(test_idx)), "n_features": int(len(selected_names)), "split_overlap_series": int(split_meta["overlap_series"]), "feature_columns_json": json.dumps(selected_names), "model_order_json": json.dumps(model_order)})
                            for class_idx, model_label in enumerate(model_order):
                                mapping_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "candidate_set": str(cdf["candidate_set"].iloc[0]), "class_idx": int(class_idx), "model_name": model_label})
                            selected_feature_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "candidate_set": str(cdf["candidate_set"].iloc[0]), "selected_feature_count": int(len(selected_names)), "selected_features_json": json.dumps(selected_names)})
                            for clf in classifiers:
                                for balancing in balancing_modes:
                                    context = f"horizon={horizon} metric={metric} classifier={clf} candidate_set={str(cdf['candidate_set'].iloc[0])} feature_set={feature_set_name} balancing={balancing} repeat={repeat_id}"
                                    try:
                                        config_started = time.perf_counter()
                                        train_dir = str(catboost_root / f"{clf}_{balancing}_h{horizon}_{metric}_r{repeat_id}") if clf == "catboost_classifier" else None
                                        model = build_meta_classifier(model_name=clf, cfg=cfg, balancing_mode=balancing, catboost_train_dir=train_dir)
                                        if len(np.unique(y_train_cls)) < 2:
                                            class_probs = np.zeros((Xte.shape[0], len(model_order)), dtype=np.float64)
                                            class_probs[:, int(y_train_cls[0])] = 1.0
                                        else:
                                            model.fit(Xtr, y_train_cls)
                                            class_probs = np.asarray(model.predict_proba(Xte), dtype=np.float64)
                                            classes_ = np.asarray(getattr(model, "classes_", np.arange(class_probs.shape[1]))).astype(int)
                                            if class_probs.shape[1] != len(model_order):
                                                aligned = np.zeros((class_probs.shape[0], len(model_order)), dtype=np.float64)
                                                for src, cls_idx in enumerate(classes_):
                                                    if 0 <= int(cls_idx) < len(model_order):
                                                        aligned[:, int(cls_idx)] = class_probs[:, src]
                                                class_probs = aligned
                                        top1 = np.argmax(class_probs, axis=1).astype(int)
                                        top2 = np.argsort(class_probs, axis=1)[:, -2] if class_probs.shape[1] >= 2 else top1
                                        conf = np.max(class_probs, axis=1)
                                        for pos, obj_idx in enumerate(test_idx):
                                            obj = object_df.iloc[int(obj_idx)]
                                            probs_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "method": "classification", "model": clf, "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "candidate_set": str(cdf["candidate_set"].iloc[0]), "balancing_mode": balancing, "object_id": str(obj["object_id"]), "series_id": str(obj["series_id"]), "fold_id": int(obj["fold_id"]), "predicted_top1_model": model_order[int(top1[pos])], "predicted_top2_model": model_order[int(top2[pos])], "confidence": float(conf[pos]), "class_probability_json": json.dumps({m: float(v) for m, v in zip(model_order, class_probs[pos])})})
                                        for rule, threshold in rule_specs:
                                            selected = top1 if threshold is None else np.where(conf >= float(threshold), top1, baseline_idx).astype(int)
                                            rdf, summ, brow = _evaluate_v2(run_id=run_id, repeat_id=repeat_id, seed=int(seed), model_name=clf, horizon=horizon, metric=metric, feature_set=feature_set_name, candidate_set=str(cdf["candidate_set"].iloc[0]), object_df=object_df, y=y, model_order=model_order, train_idx=train_idx, test_idx=test_idx, selected_idx=selected, class_probs=class_probs, top2_idx=top2, confidence=conf, balancing=balancing, decision_rule=rule, threshold=threshold, selected_feature_count=len(selected_names))
                                            summ["elapsed_seconds"] = float(time.perf_counter() - config_started)
                                            routing_rows.append(rdf)
                                            summary_rows.append(summ)
                                            best_single_rows.append(brow)
                                            completed += 1
                                            processed += 1
                                            elapsed = time.perf_counter() - started
                                            avg = elapsed / max(processed, 1)
                                            remaining = avg * max(total - processed, 0)
                                            logger.info("progress processed=%d/%d completed=%d failed=%d skipped=%d current={%s decision_rule=%s} elapsed=%s avg_sec_per_config=%.3f eta=%s", processed, total, completed, failed, skipped, context, rule, _finite_duration(elapsed), avg, _finite_duration(remaining))
                                    except Exception as exc:
                                        failed += len(rule_specs)
                                        processed += len(rule_specs)
                                        elapsed = time.perf_counter() - started
                                        logger.exception("meta_modeling_v2 configuration failed processed=%d/%d current={%s} elapsed=%s error=%s", processed, total, context, _finite_duration(elapsed), exc)
                        except Exception as exc:
                            skipped_count = len(classifiers) * len(balancing_modes) * len(rule_specs)
                            skipped += skipped_count
                            processed += skipped_count
                            summary_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "method": "classification", "model": "", "horizon": int(horizon), "target_metric": metric, "feature_set": feature_set_name, "candidate_set": f"top_{top_k}", "status": "skipped", "notes": f"{exc.__class__.__name__}: {exc}"})

    task_results_df = pd.DataFrame(summary_rows)
    routing_df = pd.concat(routing_rows, ignore_index=True) if routing_rows else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    candidate_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    dataset_df = pd.DataFrame(dataset_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    best_single_df = pd.DataFrame(best_single_rows).drop_duplicates() if best_single_rows else pd.DataFrame()
    probs_df = pd.DataFrame(probs_rows)
    selected_features_df = pd.DataFrame(selected_feature_rows)
    success_df = task_results_df[task_results_df["status"].astype(str) == "success"].copy() if not task_results_df.empty else pd.DataFrame()
    repeat_agg_df = pd.DataFrame()
    if not success_df.empty:
        repeat_agg_df = success_df.groupby(["model", "horizon", "target_metric", "candidate_set", "feature_set", "balancing_mode", "decision_rule", "confidence_threshold"], dropna=False, sort=True).agg(
            n_repeats=("repeat_id", "nunique"),
            best_single_mean=("best_single_metric", "mean"),
            best_single_std=("best_single_metric", "std"),
            achieved_mean=("achieved_metric", "mean"),
            achieved_std=("achieved_metric", "std"),
            oracle_mean=("oracle_metric", "mean"),
            oracle_std=("oracle_metric", "std"),
            improvement_mean=("improvement_vs_best_single", "mean"),
            improvement_std=("improvement_vs_best_single", "std"),
            gap_mean=("gap_to_oracle", "mean"),
            gap_std=("gap_to_oracle", "std"),
            distance_to_oracle_mean=("gap_to_oracle", "mean"),
            distance_to_oracle_std=("gap_to_oracle", "std"),
            classification_accuracy_mean=("classification_accuracy", "mean"),
            classification_accuracy_std=("classification_accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            fallback_rate_mean=("fallback_rate", "mean"),
            fallback_rate_std=("fallback_rate", "std"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
        ).reset_index()
    comparison_df = repeat_agg_df.copy()
    best_cfg_df = repeat_agg_df.sort_values(["horizon", "target_metric", "improvement_mean", "gap_mean"], ascending=[True, True, False, True], kind="stable").groupby(["horizon", "target_metric"], sort=False).head(1).reset_index(drop=True) if not repeat_agg_df.empty else pd.DataFrame()

    for value in output_cfg.values():
        Path(str(value)).resolve().parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_cfg["meta_dataset_summary_csv_path"], index=False)
    dataset_df.to_parquet(output_cfg["meta_dataset_summary_parquet_path"], index=False)
    mapping_df.to_csv(output_cfg["model_order_mapping_csv_path"], index=False)
    routing_df.to_csv(output_cfg["routing_rows_csv_path"], index=False)
    routing_df.to_parquet(output_cfg["routing_rows_parquet_path"], index=False)
    task_results_df.to_csv(output_cfg["task_results_csv_path"], index=False)
    task_results_df.to_parquet(output_cfg["task_results_parquet_path"], index=False)
    split_df.to_csv(output_cfg["split_assignments_csv_path"], index=False)
    repeat_agg_df.to_csv(output_cfg["repeat_aggregated_results_csv_path"], index=False)
    best_single_df.to_csv(output_cfg["best_single_baseline_by_repeat_csv_path"], index=False)
    comparison_df.to_csv(output_cfg["comparison_table_csv_path"], index=False)
    candidate_df.to_csv(output_cfg["candidate_models_csv_path"], index=False)
    probs_df.to_csv(output_cfg["classification_probabilities_csv_path"], index=False)
    pd.DataFrame({"feature_name": feature_cols, "feature_set": "full"}).to_csv(output_cfg["feature_list_csv_path"], index=False)
    pd.DataFrame().to_csv(output_cfg["per_class_metrics_csv_path"], index=False)
    pd.DataFrame().to_csv(output_cfg["confusion_matrix_csv_path"], index=False)
    pd.DataFrame().to_csv(output_cfg["class_distribution_csv_path"], index=False)
    pd.DataFrame().to_csv(output_cfg["confidence_summary_csv_path"], index=False)
    selected_features_df.to_csv(output_cfg["selected_features_csv_path"], index=False)
    candidate_df.groupby(["repeat_id", "split_seed", "horizon", "target_metric", "candidate_set"], dropna=False).size().reset_index(name="n_models").to_csv(output_cfg["candidate_filtering_summary_csv_path"], index=False)
    dataset_df.groupby(["horizon", "target_metric", "feature_set"], dropna=False).agg(n_objects=("n_objects", "max"), n_features=("n_features", "max")).reset_index().to_csv(output_cfg["feature_regime_summary_csv_path"], index=False)
    best_cfg_df.to_csv(output_cfg["best_config_per_task_csv_path"], index=False)
    (routing_df.head(100) if not routing_df.empty else pd.DataFrame()).to_csv(output_cfg["confident_examples_csv_path"], index=False)
    with Path(output_cfg["feature_manifest_json_path"]).open("w", encoding="utf-8") as f:
        json.dump({"feature_scope": "fold_aware_train_only", "full_features": feature_cols}, f, ensure_ascii=False, indent=2)
    with Path(output_cfg["coverage_checks_json_path"]).open("w", encoding="utf-8") as f:
        json.dump(ds["coverage_checks"], f, ensure_ascii=False, indent=2)
    snapshot = {
        **cfg,
        "resolved_run_id": run_id,
        "resolved_timestamp_utc": start_ts.isoformat(),
        "resolved_git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "resolved_inputs": {
            "metrics_path": ds["forecasting_series_metrics_path"],
            "features_path": ds["feature_source_path"],
        },
        "resolved_outputs": output_cfg,
    }
    with Path(output_cfg["resolved_config_snapshot_path"]).open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=False)

    excel_path = export_meta_modeling_excel(
        excel_path=output_cfg["excel_report_path"],
        summary_df=repeat_agg_df,
        task_results_df=task_results_df,
        routing_df=routing_df.head(200000),
        model_order_df=mapping_df,
        dataset_summary_df=dataset_df,
        split_df=split_df,
        repeat_agg_df=repeat_agg_df,
        feature_list_df=pd.read_csv(output_cfg["feature_list_csv_path"]),
        best_single_repeat_df=best_single_df,
        comparison_df=comparison_df,
        candidates_df=candidate_df,
        prediction_examples_df=routing_df.head(100) if not routing_df.empty else pd.DataFrame(),
        classification_probabilities_df=probs_df.head(200000),
    )
    manifest = {
        "run_id": run_id,
        "stage": str(cfg.get("stage", "meta_modeling_experiments_v2")),
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg["meta"]["config_path"],
        "inputs_used": {"metrics_path": ds["forecasting_series_metrics_path"], "features_path": ds["feature_source_path"]},
        "outputs": output_cfg,
        "split": cfg.get("split", {}),
        "candidate_selection": {"scope": "meta_train_only", "top_k_values": top_k_values},
        "features": {"feature_scope": "fold_aware_train_only", "feature_sets": list(feature_sets.keys())},
        "summary": {"tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "tasks_failed": int(failed), "tasks_skipped": int(skipped), "elapsed_seconds": float(time.perf_counter() - started)},
    }
    manifest_path = write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)
    logger.info("meta_modeling_v2 finished run_id=%s processed=%d/%d completed=%d failed=%d skipped=%d elapsed=%s", run_id, processed, total, completed, failed, skipped, _finite_duration(time.perf_counter() - started))
    return {"run_id": run_id, "manifest_path": str(manifest_path), "comparison_table_path": output_cfg["comparison_table_csv_path"], "candidate_models_path": output_cfg["candidate_models_csv_path"], "classification_probabilities_path": output_cfg["classification_probabilities_csv_path"], "excel_report_path": str(excel_path), "tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "best_config_path": output_cfg["best_config_per_task_csv_path"], "dataset_summary": ds, "elapsed_seconds": float(time.perf_counter() - started)}


def run_meta_modeling_experiments(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    if str(cfg.get("feature_scope", "")).strip() == "fold_aware_train_only":
        return _run_meta_modeling_experiments_v2(cfg=cfg, logger=logger)

    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("target_metrics", ["rmse", "directional_accuracy"])]
    classifiers = [str(x) for x in cfg.get("classification_models", ["logistic_regression", "random_forest_classifier", "catboost_classifier"])]
    balancing_modes = [str(x) for x in cfg.get("balancing_modes", ["default", "balanced"])]
    thresholds = [float(x) for x in cfg.get("confidence_thresholds", [0.50, 0.60, 0.70, 0.80])]

    meta_long_df, ds = build_meta_dataset(cfg)
    join_key = ds["join_key"]
    seeds = repeat_seeds(cfg.get("split", {}))

    full_features = list(ds["feature_cols"])
    basic_features = list(ds.get("basic_feature_cols", []))
    full_plus_basic = list(dict.fromkeys(full_features + basic_features))
    feature_sets = {
        "full_features": full_features,
        "full_plus_basic_features": full_plus_basic,
        "selected_features": full_plus_basic,
    }

    candidate_df = build_candidate_sets(
        meta_long_df,
        join_key=join_key,
        target_metrics=target_metrics,
        top_k_values=[int(x) for x in cfg.get("candidate_selection", {}).get("top_k_values", [3, 4, 5, 6])],
        closeness_tolerance=float(cfg.get("candidate_selection", {}).get("closeness_tolerance", 0.05)),
    )

    routing_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_single_rows: list[dict[str, Any]] = []
    probs_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    class_dist_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    catboost_root = Path(str(cfg.get("artifacts", {}).get("meta_modeling", "artifacts/meta_modeling"))).resolve() / "catboost_info" / run_id
    catboost_root.mkdir(parents=True, exist_ok=True)

    for feature_set_name, feature_cols in feature_sets.items():
        for horizon in sorted(int(x) for x in ds["horizons"]):
            for metric in target_metrics:
                csub = candidate_df[(candidate_df["horizon"] == int(horizon)) & (candidate_df["target_metric"] == metric)].copy()
                for cset, g in csub.groupby("candidate_set", sort=True):
                    models = g.sort_values("candidate_rank", kind="stable")["model_name"].astype(str).tolist()
                    task = build_task(
                        meta_long_df,
                        join_key=join_key,
                        feature_cols=feature_cols,
                        horizon=horizon,
                        target_metric=metric,
                        feature_set=feature_set_name,
                        candidate_set=str(cset),
                        candidate_models=models,
                    )
                    dataset_rows.append(
                        {
                            "run_id": run_id,
                            "horizon": int(task.horizon),
                            "target_metric": task.target_metric,
                            "feature_set": task.feature_set,
                            "candidate_set": task.candidate_set,
                            "candidate_size": int(len(task.model_order)),
                            "n_series": int(task.X.shape[0]),
                            "n_features": int(task.X.shape[1]),
                            "n_models": int(task.y.shape[1]),
                            "feature_columns_json": json.dumps(task.feature_cols),
                            "model_order_json": json.dumps(task.model_order),
                        }
                    )
                    for i, m in enumerate(task.model_order):
                        mapping_rows.append({"horizon": task.horizon, "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "class_idx": i, "model_name": m})

                    for repeat_id, seed in enumerate(seeds, start=1):
                        split = split_by_series_ids(task.row_ids, cfg=cfg, random_seed=int(seed))
                        marker = np.full(task.row_ids.shape[0], "test", dtype=object)
                        marker[split.train] = "train"
                        marker[split.validation] = "validation"
                        for i, sid in enumerate(task.row_ids):
                            split_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "series_id": str(sid), "split": str(marker[i])})

                        X_train = task.X[split.train]
                        y_train = task.y[split.train]
                        X_test = task.X[split.test]
                        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
                            continue
                        direction = METRIC_DIRECTIONS[task.target_metric]
                        y_train_cls = np.array([_pick_best_index(row, direction) for row in y_train], dtype=int)
                        feat_idx = np.arange(len(task.feature_cols), dtype=int)
                        selected_names = list(task.feature_cols)
                        meta_sel = {"method": "none", "top_n": len(selected_names), "min_features": 0}
                        if feature_set_name == "selected_features":
                            feat_idx, selected_names, meta_sel = _train_only_feature_selection(X_train, y_train_cls, task.feature_cols, cfg)
                        Xtr = X_train[:, feat_idx]
                        Xte = X_test[:, feat_idx]
                        selected_feature_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "selection_method": str(meta_sel.get("method", "")), "selection_top_n": int(meta_sel.get("top_n", len(selected_names))), "selection_min_features": int(meta_sel.get("min_features", 0)), "selected_feature_count": int(len(selected_names)), "selected_features_json": json.dumps(selected_names)})
                        baseline_idx = int(compute_best_single_baseline(y_train=y_train, y_test=task.y[split.test], model_order=task.model_order, direction=direction)["baseline_idx"])

                        for clf in classifiers:
                            for balancing in balancing_modes:
                                model = build_meta_classifier(model_name=clf, cfg=cfg, balancing_mode=balancing, catboost_train_dir=str(catboost_root / f"{clf}_{balancing}_h{task.horizon}_{task.target_metric}"))
                                if len(np.unique(y_train_cls)) < 2:
                                    class_probs = np.zeros((Xte.shape[0], len(task.model_order)), dtype=np.float64)
                                    class_probs[:, int(y_train_cls[0])] = 1.0
                                else:
                                    model.fit(Xtr, y_train_cls)
                                    class_probs = np.asarray(model.predict_proba(Xte), dtype=np.float64)
                                    classes_ = np.asarray(getattr(model, "classes_", np.arange(class_probs.shape[1]))).astype(int)
                                    if class_probs.shape[1] != len(task.model_order):
                                        aligned = np.zeros((class_probs.shape[0], len(task.model_order)), dtype=np.float64)
                                        for src, cls_idx in enumerate(classes_):
                                            if 0 <= int(cls_idx) < len(task.model_order):
                                                aligned[:, int(cls_idx)] = class_probs[:, src]
                                        class_probs = aligned
                                top1 = np.argmax(class_probs, axis=1).astype(int)
                                top2 = np.argsort(class_probs, axis=1)[:, -2] if class_probs.shape[1] >= 2 else top1
                                conf = np.max(class_probs, axis=1)
                                prob_json = [json.dumps({m: float(v) for m, v in zip(task.model_order, row)}) for row in class_probs]
                                for i, sid in enumerate(task.row_ids[split.test]):
                                    probs_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "method": "classification", "model": clf, "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "balancing_mode": balancing, "series_id": str(sid), "predicted_top1_model": task.model_order[int(top1[i])], "predicted_top2_model": task.model_order[int(top2[i])], "confidence": float(conf[i]), "class_probability_json": prob_json[i]})
                                for rule, thr, sel in [("top_1", None, top1)] + [(f"confidence_fallback_{t:.2f}", float(t), np.where(conf >= float(t), top1, baseline_idx).astype(int)) for t in thresholds]:
                                    rdf, summ, brow, pclass, cmat, cdist, cconf = _evaluate_predictions(run_id=run_id, repeat_id=repeat_id, split_seed=int(seed), method="classification", model_name=clf, task=task, split=split, selected_idx=sel, predicted_scores=class_probs if direction == "max" else -class_probs, top2_idx=top2, confidence=conf, class_probability_json=prob_json, balancing_mode=balancing, decision_rule=rule, confidence_threshold=thr, selected_feature_count=int(len(selected_names)))
                                    routing_rows.append(rdf)
                                    summary_rows.append(summ)
                                    best_single_rows.append(brow)
                                    per_class_rows.extend(pclass)
                                    confusion_rows.extend(cmat)
                                    class_dist_rows.extend(cdist)
                                    confidence_rows.append(cconf)

    task_results_df = pd.DataFrame(summary_rows)
    routing_df = pd.concat(routing_rows, ignore_index=True) if routing_rows else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    best_single_df = pd.DataFrame(best_single_rows).drop_duplicates(subset=["repeat_id", "split_seed", "horizon", "target_metric", "feature_set", "candidate_set", "balancing_mode", "decision_rule", "best_single_model"], keep="first")
    probs_df = pd.DataFrame(probs_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    class_dist_df = pd.DataFrame(class_dist_rows)
    confidence_df = pd.DataFrame(confidence_rows)
    selected_features_df = pd.DataFrame(selected_feature_rows)
    success_df = task_results_df[task_results_df["status"].astype(str) == "success"].copy() if not task_results_df.empty else pd.DataFrame()
    repeat_agg_df = success_df.groupby(["model", "horizon", "target_metric", "candidate_set", "feature_set", "balancing_mode", "decision_rule", "confidence_threshold"], dropna=False, sort=True).agg(n_repeats=("repeat_id", "nunique"), achieved_metric_mean=("achieved_metric", "mean"), achieved_metric_std=("achieved_metric", "std"), best_single_metric_mean=("best_single_metric", "mean"), oracle_metric_mean=("oracle_metric", "mean"), improvement_mean=("improvement_vs_best_single", "mean"), gap_mean=("gap_to_oracle", "mean"), routing_hit_oracle_rate_mean=("routing_hit_oracle_rate", "mean"), accuracy_mean=("classification_accuracy", "mean"), balanced_accuracy_mean=("balanced_accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"), weighted_f1_mean=("weighted_f1", "mean"), top2_hit_rate_mean=("top2_hit_rate", "mean"), fallback_rate_mean=("fallback_rate", "mean")).reset_index() if not success_df.empty else pd.DataFrame()
    comparison_df = repeat_agg_df.copy()
    best_cfg_df = repeat_agg_df.sort_values(["horizon", "target_metric", "improvement_mean", "gap_mean"], ascending=[True, True, False, True], kind="stable").groupby(["horizon", "target_metric"], sort=False).head(1).reset_index(drop=True) if not repeat_agg_df.empty else pd.DataFrame()

    output_cfg = cfg["outputs"]
    for k, v in output_cfg.items():
        if k.endswith("_path"):
            Path(str(v)).resolve().parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_cfg["meta_dataset_summary_csv_path"], index=False)
    dataset_df.to_parquet(output_cfg["meta_dataset_summary_parquet_path"], index=False)
    mapping_df.to_csv(output_cfg["model_order_mapping_csv_path"], index=False)
    routing_df.to_csv(output_cfg["routing_rows_csv_path"], index=False)
    routing_df.to_parquet(output_cfg["routing_rows_parquet_path"], index=False)
    task_results_df.to_csv(output_cfg["task_results_csv_path"], index=False)
    task_results_df.to_parquet(output_cfg["task_results_parquet_path"], index=False)
    split_df.to_csv(output_cfg["split_assignments_csv_path"], index=False)
    repeat_agg_df.to_csv(output_cfg["repeat_aggregated_results_csv_path"], index=False)
    best_single_df.to_csv(output_cfg["best_single_baseline_by_repeat_csv_path"], index=False)
    comparison_df.to_csv(output_cfg["comparison_table_csv_path"], index=False)
    candidate_df.to_csv(output_cfg["candidate_models_csv_path"], index=False)
    probs_df.to_csv(output_cfg["classification_probabilities_csv_path"], index=False)
    pd.DataFrame({"feature_name": full_features, "feature_set": "full_features"}).to_csv(output_cfg["feature_list_csv_path"], index=False)
    pd.DataFrame({"feature_name": full_plus_basic, "feature_set": "full_plus_basic_features"}).to_csv(output_cfg["pruned_feature_list_csv_path"], index=False)
    per_class_df.to_csv(output_cfg["per_class_metrics_csv_path"], index=False)
    confusion_df.to_csv(output_cfg["confusion_matrix_csv_path"], index=False)
    class_dist_df.to_csv(output_cfg["class_distribution_csv_path"], index=False)
    confidence_df.to_csv(output_cfg["confidence_summary_csv_path"], index=False)
    selected_features_df.to_csv(output_cfg["selected_features_csv_path"], index=False)
    candidate_df.groupby(["horizon", "target_metric", "candidate_set"], dropna=False).size().reset_index(name="n_models").to_csv(output_cfg["candidate_filtering_summary_csv_path"], index=False)
    dataset_df.groupby(["horizon", "target_metric", "feature_set"], dropna=False).agg(n_series=("n_series", "max"), n_features=("n_features", "max")).reset_index().to_csv(output_cfg["feature_regime_summary_csv_path"], index=False)
    best_cfg_df.to_csv(output_cfg["best_config_per_task_csv_path"], index=False)
    confident_examples_df = pd.concat([routing_df[(routing_df["routing_hit_oracle"] == 1) & (pd.to_numeric(routing_df["confidence"], errors="coerce") >= 0.7)].head(50), routing_df[(routing_df["routing_hit_oracle"] == 0) & (pd.to_numeric(routing_df["confidence"], errors="coerce") >= 0.7)].head(50)], ignore_index=True) if not routing_df.empty else pd.DataFrame()
    confident_examples_df.to_csv(output_cfg["confident_examples_csv_path"], index=False)
    with Path(output_cfg["feature_manifest_json_path"]).open("w", encoding="utf-8") as f:
        json.dump({"full_features": full_features, "basic_features": basic_features, "full_plus_basic_features": full_plus_basic, "selection_method": str(cfg.get("feature_selection", {}).get("method", "mutual_info"))}, f, ensure_ascii=False, indent=2)

    def _preview(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty or len(df) <= n:
            return df
        return df.head(n).copy()

    excel_path = export_meta_modeling_excel(
        excel_path=output_cfg["excel_report_path"],
        summary_df=_preview(repeat_agg_df, 50000),
        task_results_df=_preview(task_results_df, 200000),
        routing_df=_preview(routing_df, 200000),
        model_order_df=mapping_df,
        dataset_summary_df=dataset_df,
        split_df=_preview(split_df, 200000),
        repeat_agg_df=_preview(repeat_agg_df, 50000),
        feature_list_df=pd.read_csv(output_cfg["feature_list_csv_path"]) if Path(output_cfg["feature_list_csv_path"]).exists() else None,
        best_single_repeat_df=_preview(best_single_df, 200000),
        comparison_df=_preview(comparison_df, 50000),
        candidates_df=candidate_df,
        prediction_examples_df=confident_examples_df,
        classification_probabilities_df=_preview(probs_df, 200000),
        pruned_feature_list_df=pd.read_csv(output_cfg["pruned_feature_list_csv_path"]) if Path(output_cfg["pruned_feature_list_csv_path"]).exists() else None,
    )

    manifest = {"run_id": run_id, "stage": str(cfg.get("stage", "meta_modeling_experiments")), "timestamp_start": datetime.now(timezone.utc).isoformat(), "timestamp_end": datetime.now(timezone.utc).isoformat(), "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])), "config_path": cfg["meta"]["config_path"], "inputs_used": {"features_path": ds["feature_source_path"], "basic_feature_source_path": ds.get("basic_feature_source_path", ""), "forecasting_series_metrics_path": ds.get("forecasting_series_metrics_path", ""), "forecasting_manifest_path": ds.get("forecasting_manifest_path", "")}, "outputs": output_cfg, "summary": {"n_rows_meta_long": int(ds["n_rows"]), "n_series": int(ds["n_series"]), "horizons": ds["horizons"], "target_metrics": target_metrics, "candidate_set_sizes": sorted(candidate_df["candidate_set"].astype(str).unique().tolist()) if not candidate_df.empty else [], "classifiers": classifiers, "balancing_modes": balancing_modes, "feature_regimes": list(feature_sets.keys()), "confidence_thresholds": thresholds, "repeat_splits": len(seeds), "repeat_seeds": seeds, "tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "tasks_failed": int((task_results_df.get("status", pd.Series(dtype=str)) == "failed").sum())}}
    manifest_path = write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)
    logger.info("classification meta-modeling run_id=%s tasks_total=%d success=%d", run_id, len(task_results_df), int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()))
    return {"run_id": run_id, "manifest_path": str(manifest_path), "comparison_table_path": output_cfg["comparison_table_csv_path"], "candidate_models_path": output_cfg["candidate_models_csv_path"], "classification_probabilities_path": output_cfg["classification_probabilities_csv_path"], "excel_report_path": str(excel_path), "tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "best_config_path": output_cfg["best_config_per_task_csv_path"], "dataset_summary": ds}
