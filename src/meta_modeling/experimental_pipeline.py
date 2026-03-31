from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.meta_modeling.models import (
    MetaModelError,
    build_meta_classifier,
    build_meta_model,
    build_scoring_model,
)
from src.reporting.excel_export import export_meta_modeling_excel
from src.utils.manifest import get_git_commit, write_manifest


METRIC_DIRECTIONS = {
    "rmse": "min",
    "directional_accuracy": "max",
}


@dataclass
class TaskData:
    horizon: int
    target_metric: str
    feature_set: str
    candidate_set: str
    feature_cols: list[str]
    model_order: list[str]
    row_ids: np.ndarray
    X: np.ndarray
    y: np.ndarray


@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _normalize_metric_alias(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "rmse": "rmse",
        "rmse_mean": "rmse",
        "directional_accuracy": "directional_accuracy",
        "directional_accuracy_mean": "directional_accuracy",
        "da": "directional_accuracy",
    }
    return aliases.get(key, key)


def _resolve_metric_columns(forecasting_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str]:
    requested = [_normalize_metric_alias(x) for x in cfg.get("target_metrics", ["rmse", "directional_accuracy"])]
    metric_col_cfg = {str(k).lower(): str(v) for k, v in dict(cfg.get("metric_columns", {})).items()}
    resolved: dict[str, str] = {}
    for metric in requested:
        if metric in metric_col_cfg and metric_col_cfg[metric] in forecasting_df.columns:
            resolved[metric] = metric_col_cfg[metric]
            continue
        candidates = [metric, f"{metric}_mean"]
        found = next((c for c in candidates if c in forecasting_df.columns), None)
        if found:
            resolved[metric] = found
            continue
        raise ValueError(f"Target metric '{metric}' is not available in forecasting results")
    return resolved


def _pick_join_key(features_df: pd.DataFrame, forecasting_df: pd.DataFrame, cfg: dict[str, Any]) -> str:
    preferred = [str(x) for x in cfg.get("join_keys", ["series_id", "ticker"])]
    for key in preferred:
        if key in features_df.columns and key in forecasting_df.columns:
            return key
    raise ValueError(f"Cannot find common join key. Tried: {preferred}")


def _feature_columns(features_df: pd.DataFrame, join_key: str) -> list[str]:
    non_features = {join_key, "ticker", "market", "dataset_profile", "asset", "series_name"}
    cols = []
    for c in features_df.columns:
        if c in non_features:
            continue
        if pd.api.types.is_numeric_dtype(features_df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found")
    return cols


def build_meta_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_df = pd.read_parquet(str(cfg["inputs"]["features_path"])).copy()
    forecasting_df = pd.read_parquet(str(cfg["inputs"]["forecasting_series_metrics_path"])).copy()

    join_key = _pick_join_key(features_df, forecasting_df, cfg)
    metric_cols = _resolve_metric_columns(forecasting_df, cfg)

    dataset_filter = str(cfg.get("dataset_filter", "")).strip()
    if dataset_filter and "dataset_profile" in features_df.columns:
        features_df = features_df[features_df["dataset_profile"].astype(str) == dataset_filter].copy()

    feature_cols = _feature_columns(features_df, join_key=join_key)
    keep_cols = [join_key] + ([c for c in ["ticker", "market", "dataset_profile"] if c in features_df.columns]) + feature_cols
    features_dedup = features_df[keep_cols].drop_duplicates(subset=[join_key], keep="first").copy()

    merged = forecasting_df.merge(features_dedup, on=join_key, how="inner")
    merged = merged.dropna(subset=["horizon", "model_name"])
    merged["horizon"] = pd.to_numeric(merged["horizon"], errors="coerce").astype("Int64")
    merged = merged[merged["horizon"].notna()].copy()
    merged["horizon"] = merged["horizon"].astype(int)
    merged["model_name"] = merged["model_name"].astype(str)
    for metric, col in metric_cols.items():
        merged[metric] = pd.to_numeric(merged[col], errors="coerce")

    summary = {
        "join_key": join_key,
        "feature_cols": feature_cols,
        "feature_source_path": str(cfg["inputs"]["features_path"]),
        "metric_columns": metric_cols,
        "n_rows": int(len(merged)),
        "n_series": int(merged[join_key].nunique()),
        "horizons": sorted(merged["horizon"].unique().tolist()),
        "models": sorted(merged["model_name"].unique().tolist()),
    }
    return merged, summary


def compute_pruned_features(meta_long_df: pd.DataFrame, join_key: str, feature_cols: list[str], threshold: float) -> tuple[list[str], pd.DataFrame]:
    base = meta_long_df[[join_key, *feature_cols]].drop_duplicates(subset=[join_key], keep="first").set_index(join_key)
    corr = base[feature_cols].corr(method="pearson").abs().fillna(0.0)
    mean_corr = corr.mean(axis=1)

    dropped: set[str] = set()
    rows: list[dict[str, Any]] = []
    ordered = sorted(feature_cols)
    for i, fa in enumerate(ordered):
        for fb in ordered[i + 1 :]:
            cval = float(corr.loc[fa, fb])
            if cval <= threshold:
                continue
            if fa in dropped or fb in dropped:
                continue
            sa = float(mean_corr.loc[fa])
            sb = float(mean_corr.loc[fb])
            if sa > sb:
                drop, keep = fa, fb
            elif sb > sa:
                drop, keep = fb, fa
            else:
                drop, keep = (fa, fb) if fa > fb else (fb, fa)
            dropped.add(drop)
            rows.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "abs_corr": cval,
                    "dropped_feature": drop,
                    "kept_feature": keep,
                }
            )
    kept = [c for c in feature_cols if c not in dropped]
    return kept, pd.DataFrame(rows)


def build_candidate_sets(
    meta_long_df: pd.DataFrame,
    *,
    join_key: str,
    target_metrics: list[str],
    top_k_values: list[int],
    closeness_tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    k_values = sorted(set(int(x) for x in top_k_values if int(x) > 0))
    for metric in target_metrics:
        direction = METRIC_DIRECTIONS[metric]
        sdf = meta_long_df[[join_key, "horizon", "model_name", metric]].copy().dropna(subset=[metric])
        mean_tbl = (
            sdf.groupby(["horizon", "model_name"], sort=False, dropna=False)[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "mean_metric"})
        )
        win_rows: list[dict[str, Any]] = []
        for horizon, hdf in sdf.groupby("horizon", sort=True):
            winners = []
            for _, g in hdf.groupby(join_key, sort=False):
                g2 = g.sort_values([metric, "model_name"], ascending=[direction == "min", True], kind="stable")
                winners.append(str(g2.iloc[0]["model_name"]))
            if winners:
                cnt = pd.Series(winners).value_counts()
                for model_name, win_count in cnt.items():
                    win_rows.append({"horizon": int(horizon), "model_name": str(model_name), "win_count": int(win_count)})
        wins_df = pd.DataFrame(win_rows)

        for horizon, hdf in mean_tbl.groupby("horizon", sort=True):
            hwork = hdf.copy()
            best_value = float(hwork["mean_metric"].min()) if direction == "min" else float(hwork["mean_metric"].max())
            denom = max(abs(best_value), 1e-12)
            if direction == "min":
                hwork["gap_to_best"] = hwork["mean_metric"] - best_value
            else:
                hwork["gap_to_best"] = best_value - hwork["mean_metric"]
            hwork["relative_gap_to_best"] = hwork["gap_to_best"] / denom
            hwork["is_close_to_best"] = (hwork["relative_gap_to_best"] <= float(closeness_tolerance)).astype(int)
            hwork = hwork.merge(
                wins_df[wins_df["horizon"] == int(horizon)][["model_name", "win_count"]],
                on="model_name",
                how="left",
            )
            hwork["win_count"] = hwork["win_count"].fillna(0).astype(int)
            hwork = hwork.sort_values(
                ["is_close_to_best", "gap_to_best", "win_count", "model_name"],
                ascending=[False, True, False, True],
                kind="stable",
            ).reset_index(drop=True)

            for k in k_values:
                k_eff = min(k, len(hwork))
                if k_eff < 2 and len(hwork) >= 2:
                    k_eff = 2
                selected = hwork.head(k_eff).copy()
                selected["candidate_set"] = f"top_{k_eff}"
                selected["top_k"] = int(k_eff)
                selected["candidate_rank"] = np.arange(1, len(selected) + 1, dtype=int)
                selected["target_metric"] = metric
                rows.extend(
                    selected[
                        [
                            "horizon",
                            "target_metric",
                            "candidate_set",
                            "top_k",
                            "candidate_rank",
                            "model_name",
                            "mean_metric",
                            "win_count",
                            "gap_to_best",
                            "relative_gap_to_best",
                            "is_close_to_best",
                        ]
                    ].to_dict("records")
                )
    return pd.DataFrame(rows)


def build_task(
    meta_long_df: pd.DataFrame,
    *,
    join_key: str,
    feature_cols: list[str],
    horizon: int,
    target_metric: str,
    feature_set: str,
    candidate_set: str,
    candidate_models: list[str],
) -> TaskData:
    sub = meta_long_df[(meta_long_df["horizon"] == int(horizon)) & (meta_long_df["model_name"].astype(str).isin(candidate_models))].copy()
    task_df = sub[[join_key, "model_name", target_metric, *feature_cols]].dropna(subset=[target_metric]).copy()
    model_order = sorted(task_df["model_name"].astype(str).unique().tolist())
    pivot = task_df.pivot_table(index=join_key, columns="model_name", values=target_metric, aggfunc="mean").reindex(columns=model_order)
    pivot = pivot.dropna(axis=0, how="any")
    base = task_df[[join_key, *feature_cols]].drop_duplicates(subset=[join_key], keep="first").set_index(join_key)
    aligned = base.join(pivot, how="inner").dropna(axis=0, how="any")
    if aligned.empty:
        raise ValueError(f"Empty task: h={horizon}, metric={target_metric}, features={feature_set}, candidates={candidate_set}")
    return TaskData(
        horizon=int(horizon),
        target_metric=str(target_metric),
        feature_set=str(feature_set),
        candidate_set=str(candidate_set),
        feature_cols=list(feature_cols),
        model_order=model_order,
        row_ids=aligned.index.to_numpy(),
        X=aligned[feature_cols].to_numpy(dtype=np.float64),
        y=aligned[model_order].to_numpy(dtype=np.float64),
    )


def repeat_seeds(split_cfg: dict[str, Any]) -> list[int]:
    n_repeats = int(split_cfg.get("n_repeats", 1))
    base_seed = int(split_cfg.get("random_seed", 42))
    explicit = [int(x) for x in split_cfg.get("random_seeds", []) if str(x).strip()]
    if explicit:
        return (explicit + [base_seed + i * 9973 for i in range(n_repeats)])[:n_repeats]
    return [base_seed + i * 9973 for i in range(n_repeats)]


def split_by_series_ids(series_ids: np.ndarray, cfg: dict[str, Any], random_seed: int) -> SplitIndices:
    unique_ids = np.unique(series_ids.astype(str))
    if len(unique_ids) < 3:
        raise ValueError("Need at least 3 unique series for split")
    split_cfg = cfg.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    validation_size = float(split_cfg.get("validation_size", 0.1))
    rng = np.random.default_rng(int(random_seed))
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_size)))
    n_val = max(0, int(round(n_total * validation_size)))
    if n_total - n_test - n_val < 1:
        n_val = max(0, n_total - n_test - 1)
    if n_total - n_test - n_val < 1:
        n_test = max(1, n_total - n_val - 1)
    test_ids = set(shuffled[:n_test].tolist())
    val_ids = set(shuffled[n_test : n_test + n_val].tolist())
    train_ids = set(shuffled[n_test + n_val :].tolist())
    ids = series_ids.astype(str)
    return SplitIndices(
        train=np.where(np.isin(ids, list(train_ids)))[0],
        validation=np.where(np.isin(ids, list(val_ids)))[0],
        test=np.where(np.isin(ids, list(test_ids)))[0],
    )


def _pick_best_index(values: np.ndarray, direction: str) -> int:
    return int(np.argmin(values)) if direction == "min" else int(np.argmax(values))


def compute_best_single_baseline(y_train: np.ndarray, y_test: np.ndarray, model_order: list[str], direction: str) -> dict[str, Any]:
    train_mean = np.nanmean(y_train, axis=0)
    baseline_idx = _pick_best_index(train_mean, direction=direction)
    return {
        "baseline_idx": int(baseline_idx),
        "baseline_model": str(model_order[baseline_idx]),
        "baseline_values": y_test[:, baseline_idx],
        "train_mean_by_model": train_mean,
    }


def _extract_feature_importances(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    vector: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        vector = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "_models") and isinstance(model._models, list) and model._models:
        vals = []
        for m in model._models:
            getter = getattr(m, "get_feature_importance", None)
            if callable(getter):
                vals.append(np.asarray(getter(), dtype=np.float64))
        if vals:
            vector = np.nanmean(np.vstack(vals), axis=0)
    if vector is None or vector.ndim != 1 or len(vector) != len(feature_names):
        return []
    return [{"feature_name": str(f), "importance": float(v)} for f, v in zip(feature_names, vector)]


def _evaluate_predictions(
    *,
    run_id: str,
    repeat_id: int,
    split_seed: int,
    method: str,
    model_name: str,
    task: TaskData,
    split: SplitIndices,
    selected_idx: np.ndarray,
    predicted_scores: np.ndarray,
    top2_idx: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    class_probability_json: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    direction = METRIC_DIRECTIONS[task.target_metric]
    y_test = task.y[split.test]
    y_train = task.y[split.train]
    row_ids_test = task.row_ids[split.test]
    baseline = compute_best_single_baseline(y_train=y_train, y_test=y_test, model_order=task.model_order, direction=direction)
    baseline_values = np.asarray(baseline["baseline_values"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    routing_hits: list[int] = []
    top2_hits: list[int] = []
    for i in range(y_test.shape[0]):
        actual_row = y_test[i]
        pred_idx = int(selected_idx[i])
        oracle_idx = _pick_best_index(actual_row, direction=direction)
        achieved = float(actual_row[pred_idx])
        oracle_metric = float(actual_row[oracle_idx])
        baseline_metric = float(baseline_values[i])
        if direction == "min":
            improvement = baseline_metric - achieved
            gap = achieved - oracle_metric
        else:
            improvement = achieved - baseline_metric
            gap = oracle_metric - achieved
        hit = int(pred_idx == oracle_idx)
        routing_hits.append(hit)
        top2_hit = np.nan
        top2_model = ""
        if top2_idx is not None:
            t2 = int(top2_idx[i])
            top2_model = task.model_order[t2]
            top2_hit = int(oracle_idx in {pred_idx, t2})
            top2_hits.append(int(top2_hit))
        rows.append(
            {
                "run_id": run_id,
                "repeat_id": int(repeat_id),
                "split_seed": int(split_seed),
                "method": method,
                "model": model_name,
                "horizon": int(task.horizon),
                "target_metric": task.target_metric,
                "feature_set": task.feature_set,
                "candidate_set": task.candidate_set,
                "series_id": str(row_ids_test[i]),
                "selected_model": task.model_order[pred_idx],
                "selected_model_top2": top2_model,
                "oracle_model": task.model_order[oracle_idx],
                "best_single_model": baseline["baseline_model"],
                "achieved_metric": achieved,
                "oracle_metric": oracle_metric,
                "baseline_metric": baseline_metric,
                "improvement_vs_best_single": float(improvement),
                "gap_to_oracle": float(gap),
                "routing_hit_oracle": hit,
                "top2_hit_oracle": top2_hit,
                "confidence": float(confidence[i]) if confidence is not None else np.nan,
                "predicted_scores_json": json.dumps([float(x) for x in predicted_scores[i]]),
                "actual_metric_vector_json": json.dumps([float(x) for x in actual_row]),
                "model_order_json": json.dumps(task.model_order),
                "class_probability_json": (class_probability_json[i] if class_probability_json is not None else ""),
            }
        )

    routing_df = pd.DataFrame(rows)
    summary = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(split_seed),
        "method": method,
        "model": model_name,
        "horizon": int(task.horizon),
        "target_metric": task.target_metric,
        "feature_set": task.feature_set,
        "candidate_set": task.candidate_set,
        "n_test_series": int(len(routing_df)),
        "achieved_metric": float(pd.to_numeric(routing_df["achieved_metric"], errors="coerce").mean()),
        "best_single_metric": float(pd.to_numeric(routing_df["baseline_metric"], errors="coerce").mean()),
        "oracle_metric": float(pd.to_numeric(routing_df["oracle_metric"], errors="coerce").mean()),
        "improvement_vs_best_single": float(pd.to_numeric(routing_df["improvement_vs_best_single"], errors="coerce").mean()),
        "gap_to_oracle": float(pd.to_numeric(routing_df["gap_to_oracle"], errors="coerce").mean()),
        "routing_hit_oracle_rate": float(np.mean(routing_hits)) if routing_hits else np.nan,
        "classification_accuracy": float(np.mean(routing_hits)) if method == "classification" else np.nan,
        "top2_hit_rate": float(np.mean(top2_hits)) if top2_hits else np.nan,
        "status": "success",
        "notes": "",
    }
    baseline_row = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(split_seed),
        "horizon": int(task.horizon),
        "target_metric": task.target_metric,
        "feature_set": task.feature_set,
        "candidate_set": task.candidate_set,
        "best_single_model": baseline["baseline_model"],
        "best_single_train_mean_metric": float(baseline["train_mean_by_model"][baseline["baseline_idx"]]),
        "best_single_test_mean_metric": summary["best_single_metric"],
    }
    return routing_df, summary, baseline_row


def run_meta_modeling_experiments(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    start_ts = datetime.now(timezone.utc)

    methods = [str(x) for x in cfg.get("methods", ["regression", "classification", "scoring"])]
    regression_models = [str(x) for x in cfg.get("meta_models", ["ridge", "random_forest", "catboost"])]
    classification_models = [str(x) for x in cfg.get("classification_models", ["logistic_regression", "random_forest_classifier", "catboost_classifier"])]
    scoring_models = [str(x) for x in cfg.get("scoring_models", ["ridge", "random_forest", "catboost"])]

    meta_long_df, ds = build_meta_dataset(cfg)
    join_key = ds["join_key"]
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("target_metrics", ["rmse", "directional_accuracy"])]
    seeds = repeat_seeds(cfg.get("split", {}))
    full_features = list(ds["feature_cols"])
    pruned_features, dropped_pairs_df = compute_pruned_features(
        meta_long_df,
        join_key=join_key,
        feature_cols=full_features,
        threshold=float(cfg.get("feature_pruning", {}).get("corr_threshold", 0.90)),
    )
    feature_sets = {"full_features": full_features, "pruned_features": pruned_features}

    candidate_df = build_candidate_sets(
        meta_long_df,
        join_key=join_key,
        target_metrics=target_metrics,
        top_k_values=[int(x) for x in cfg.get("candidate_selection", {}).get("top_k_values", [5])],
        closeness_tolerance=float(cfg.get("candidate_selection", {}).get("closeness_tolerance", 0.05)),
    )

    logger.info("meta-modeling experiments run_id=%s methods=%s repeats=%d", run_id, methods, len(seeds))

    routing_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_single_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for feature_set_name, feature_cols in feature_sets.items():
        for horizon in sorted(int(x) for x in ds["horizons"]):
            for metric in target_metrics:
                csub = candidate_df[(candidate_df["horizon"] == int(horizon)) & (candidate_df["target_metric"] == metric)].copy()
                if csub.empty:
                    models_all = sorted(meta_long_df.loc[meta_long_df["horizon"] == int(horizon), "model_name"].astype(str).unique().tolist())
                    cdefs = [("all_models", models_all)]
                else:
                    cdefs = []
                    for cset, g in csub.groupby("candidate_set", sort=True):
                        cdefs.append((str(cset), g.sort_values("candidate_rank", kind="stable")["model_name"].astype(str).tolist()))

                for candidate_set, candidate_models in cdefs:
                    try:
                        task = build_task(
                            meta_long_df,
                            join_key=join_key,
                            feature_cols=feature_cols,
                            horizon=horizon,
                            target_metric=metric,
                            feature_set=feature_set_name,
                            candidate_set=candidate_set,
                            candidate_models=candidate_models,
                        )
                    except Exception as exc:
                        logger.warning("skip task h=%d metric=%s features=%s candidates=%s err=%s", horizon, metric, feature_set_name, candidate_set, exc)
                        continue

                    dataset_rows.append(
                        {
                            "run_id": run_id,
                            "horizon": int(task.horizon),
                            "target_metric": task.target_metric,
                            "feature_set": task.feature_set,
                            "candidate_set": task.candidate_set,
                            "n_series": int(task.X.shape[0]),
                            "n_features": int(task.X.shape[1]),
                            "n_models": int(task.y.shape[1]),
                            "feature_columns_json": json.dumps(task.feature_cols),
                            "model_order_json": json.dumps(task.model_order),
                        }
                    )
                    for midx, mname in enumerate(task.model_order):
                        mapping_rows.append(
                            {
                                "horizon": int(task.horizon),
                                "target_metric": task.target_metric,
                                "feature_set": task.feature_set,
                                "candidate_set": task.candidate_set,
                                "output_dim": int(midx),
                                "model_name": str(mname),
                            }
                        )

                    for repeat_id, seed in enumerate(seeds, start=1):
                        split = split_by_series_ids(task.row_ids, cfg=cfg, random_seed=int(seed))
                        marker = np.full(task.row_ids.shape[0], "test", dtype=object)
                        marker[split.train] = "train"
                        marker[split.validation] = "validation"
                        for i, sid in enumerate(task.row_ids):
                            split_rows.append(
                                {
                                    "run_id": run_id,
                                    "repeat_id": int(repeat_id),
                                    "split_seed": int(seed),
                                    "horizon": int(task.horizon),
                                    "target_metric": task.target_metric,
                                    "feature_set": task.feature_set,
                                    "candidate_set": task.candidate_set,
                                    "series_id": str(sid),
                                    "split": str(marker[i]),
                                }
                            )

                        X_train = task.X[split.train]
                        y_train = task.y[split.train]
                        X_test = task.X[split.test]
                        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
                            continue

                        if "regression" in methods:
                            for model_name in regression_models:
                                try:
                                    model = build_meta_model(model_name=model_name, cfg=cfg)
                                    model.fit(X_train, y_train)
                                    y_pred = np.asarray(model.predict(X_test), dtype=np.float64)
                                    direction = METRIC_DIRECTIONS[task.target_metric]
                                    selected_idx = np.array([_pick_best_index(row, direction) for row in y_pred], dtype=int)
                                    rdf, summ, brow = _evaluate_predictions(
                                        run_id=run_id,
                                        repeat_id=repeat_id,
                                        split_seed=int(seed),
                                        method="regression",
                                        model_name=model_name,
                                        task=task,
                                        split=split,
                                        selected_idx=selected_idx,
                                        predicted_scores=y_pred,
                                    )
                                    routing_rows.append(rdf)
                                    summary_rows.append(summ)
                                    best_single_rows.append(brow)
                                    if "random_forest" in model_name or "catboost" in model_name:
                                        for row in _extract_feature_importances(model, task.feature_cols):
                                            row.update(
                                                {
                                                    "run_id": run_id,
                                                    "repeat_id": int(repeat_id),
                                                    "split_seed": int(seed),
                                                    "method": "regression",
                                                    "model": model_name,
                                                    "horizon": int(task.horizon),
                                                    "target_metric": task.target_metric,
                                                    "feature_set": task.feature_set,
                                                    "candidate_set": task.candidate_set,
                                                }
                                            )
                                            importance_rows.append(row)
                                except Exception as exc:
                                    summary_rows.append(
                                        {
                                            "run_id": run_id,
                                            "repeat_id": int(repeat_id),
                                            "split_seed": int(seed),
                                            "method": "regression",
                                            "model": model_name,
                                            "horizon": int(task.horizon),
                                            "target_metric": task.target_metric,
                                            "feature_set": task.feature_set,
                                            "candidate_set": task.candidate_set,
                                            "status": "failed",
                                            "notes": f"{exc.__class__.__name__}: {exc}",
                                        }
                                    )

                        if "classification" in methods:
                            for model_name in classification_models:
                                try:
                                    direction = METRIC_DIRECTIONS[task.target_metric]
                                    y_train_cls = np.array([_pick_best_index(row, direction) for row in y_train], dtype=int)
                                    if len(np.unique(y_train_cls)) < 2:
                                        class_probs = np.zeros((X_test.shape[0], len(task.model_order)), dtype=np.float64)
                                        class_probs[:, int(y_train_cls[0])] = 1.0
                                        model = None
                                    else:
                                        model = build_meta_classifier(model_name=model_name, cfg=cfg)
                                        model.fit(X_train, y_train_cls)
                                        class_probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)
                                        classes_ = np.asarray(getattr(model, "classes_", np.arange(class_probs.shape[1]))).astype(int)
                                        if class_probs.shape[1] != len(task.model_order):
                                            aligned = np.zeros((class_probs.shape[0], len(task.model_order)), dtype=np.float64)
                                            for src_col, cls_idx in enumerate(classes_):
                                                if 0 <= int(cls_idx) < len(task.model_order):
                                                    aligned[:, int(cls_idx)] = class_probs[:, src_col]
                                            class_probs = aligned
                                    top1 = np.argmax(class_probs, axis=1).astype(int)
                                    sorted_idx = np.argsort(class_probs, axis=1)
                                    top2 = sorted_idx[:, -2] if class_probs.shape[1] >= 2 else top1
                                    conf = np.max(class_probs, axis=1)
                                    prob_json = [json.dumps({m: float(v) for m, v in zip(task.model_order, row)}) for row in class_probs]
                                    score_for_selection = class_probs if direction == "max" else -class_probs
                                    rdf, summ, brow = _evaluate_predictions(
                                        run_id=run_id,
                                        repeat_id=repeat_id,
                                        split_seed=int(seed),
                                        method="classification",
                                        model_name=model_name,
                                        task=task,
                                        split=split,
                                        selected_idx=top1,
                                        predicted_scores=score_for_selection,
                                        top2_idx=top2,
                                        confidence=conf,
                                        class_probability_json=prob_json,
                                    )
                                    routing_rows.append(rdf)
                                    summary_rows.append(summ)
                                    best_single_rows.append(brow)
                                    for i, sid in enumerate(task.row_ids[split.test]):
                                        probability_rows.append(
                                            {
                                                "run_id": run_id,
                                                "repeat_id": int(repeat_id),
                                                "split_seed": int(seed),
                                                "method": "classification",
                                                "model": model_name,
                                                "horizon": int(task.horizon),
                                                "target_metric": task.target_metric,
                                                "feature_set": task.feature_set,
                                                "candidate_set": task.candidate_set,
                                                "series_id": str(sid),
                                                "predicted_top1_model": task.model_order[int(top1[i])],
                                                "predicted_top2_model": task.model_order[int(top2[i])],
                                                "confidence": float(conf[i]),
                                                "class_probability_json": prob_json[i],
                                            }
                                        )
                                    if model is not None and ("random_forest" in model_name or "catboost" in model_name):
                                        for row in _extract_feature_importances(model, task.feature_cols):
                                            row.update(
                                                {
                                                    "run_id": run_id,
                                                    "repeat_id": int(repeat_id),
                                                    "split_seed": int(seed),
                                                    "method": "classification",
                                                    "model": model_name,
                                                    "horizon": int(task.horizon),
                                                    "target_metric": task.target_metric,
                                                    "feature_set": task.feature_set,
                                                    "candidate_set": task.candidate_set,
                                                }
                                            )
                                            importance_rows.append(row)
                                except Exception as exc:
                                    summary_rows.append(
                                        {
                                            "run_id": run_id,
                                            "repeat_id": int(repeat_id),
                                            "split_seed": int(seed),
                                            "method": "classification",
                                            "model": model_name,
                                            "horizon": int(task.horizon),
                                            "target_metric": task.target_metric,
                                            "feature_set": task.feature_set,
                                            "candidate_set": task.candidate_set,
                                            "status": "failed",
                                            "notes": f"{exc.__class__.__name__}: {exc}",
                                        }
                                    )

                        if "scoring" in methods:
                            for model_name in scoring_models:
                                try:
                                    n_models = len(task.model_order)
                                    train_rows, train_targets = [], []
                                    for i in split.train:
                                        for midx in range(n_models):
                                            ohe = np.zeros(n_models, dtype=np.float64)
                                            ohe[midx] = 1.0
                                            train_rows.append(np.concatenate([task.X[i], ohe]))
                                            train_targets.append(float(task.y[i, midx]))
                                    test_rows = []
                                    test_map = []
                                    for i in split.test:
                                        sid = str(task.row_ids[i])
                                        for midx in range(n_models):
                                            ohe = np.zeros(n_models, dtype=np.float64)
                                            ohe[midx] = 1.0
                                            test_rows.append(np.concatenate([task.X[i], ohe]))
                                            test_map.append((sid, midx))
                                    Xtr = np.asarray(train_rows, dtype=np.float64)
                                    ytr = np.asarray(train_targets, dtype=np.float64)
                                    Xte = np.asarray(test_rows, dtype=np.float64)
                                    model = build_scoring_model(model_name=model_name, cfg=cfg)
                                    model.fit(Xtr, ytr)
                                    pred_long = np.asarray(model.predict(Xte), dtype=np.float64).reshape(-1)
                                    sid_order = task.row_ids[split.test].astype(str)
                                    sid_pos = {sid: idx for idx, sid in enumerate(sid_order)}
                                    pred_matrix = np.full((len(sid_order), n_models), np.nan, dtype=np.float64)
                                    for idx_pred, (sid, midx) in enumerate(test_map):
                                        pred_matrix[sid_pos[sid], int(midx)] = float(pred_long[idx_pred])
                                    direction = METRIC_DIRECTIONS[task.target_metric]
                                    selected = np.array([_pick_best_index(row, direction) for row in pred_matrix], dtype=int)
                                    rdf, summ, brow = _evaluate_predictions(
                                        run_id=run_id,
                                        repeat_id=repeat_id,
                                        split_seed=int(seed),
                                        method="scoring",
                                        model_name=model_name,
                                        task=task,
                                        split=split,
                                        selected_idx=selected,
                                        predicted_scores=pred_matrix,
                                    )
                                    routing_rows.append(rdf)
                                    summary_rows.append(summ)
                                    best_single_rows.append(brow)
                                    if "random_forest" in model_name or "catboost" in model_name:
                                        long_names = list(task.feature_cols) + [f"model__{m}" for m in task.model_order]
                                        for row in _extract_feature_importances(model, long_names):
                                            row.update(
                                                {
                                                    "run_id": run_id,
                                                    "repeat_id": int(repeat_id),
                                                    "split_seed": int(seed),
                                                    "method": "scoring",
                                                    "model": model_name,
                                                    "horizon": int(task.horizon),
                                                    "target_metric": task.target_metric,
                                                    "feature_set": task.feature_set,
                                                    "candidate_set": task.candidate_set,
                                                }
                                            )
                                            importance_rows.append(row)
                                except Exception as exc:
                                    summary_rows.append(
                                        {
                                            "run_id": run_id,
                                            "repeat_id": int(repeat_id),
                                            "split_seed": int(seed),
                                            "method": "scoring",
                                            "model": model_name,
                                            "horizon": int(task.horizon),
                                            "target_metric": task.target_metric,
                                            "feature_set": task.feature_set,
                                            "candidate_set": task.candidate_set,
                                            "status": "failed",
                                            "notes": f"{exc.__class__.__name__}: {exc}",
                                        }
                                    )

    task_results_df = pd.DataFrame(summary_rows)
    routing_df = pd.concat(routing_rows, ignore_index=True) if routing_rows else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    best_single_df = pd.DataFrame(best_single_rows).drop_duplicates(
        subset=["repeat_id", "split_seed", "horizon", "target_metric", "feature_set", "candidate_set", "best_single_model"],
        keep="first",
    )
    importance_df = pd.DataFrame(importance_rows)
    probs_df = pd.DataFrame(probability_rows)

    success_df = task_results_df[task_results_df["status"].astype(str) == "success"].copy() if not task_results_df.empty else pd.DataFrame()
    comparison_df = pd.DataFrame()
    repeat_agg_df = pd.DataFrame()
    if not success_df.empty:
        metric_cols = [
            "achieved_metric",
            "best_single_metric",
            "oracle_metric",
            "improvement_vs_best_single",
            "gap_to_oracle",
            "routing_hit_oracle_rate",
            "classification_accuracy",
            "top2_hit_rate",
        ]
        for c in metric_cols:
            success_df[c] = pd.to_numeric(success_df[c], errors="coerce")
        repeat_agg_df = (
            success_df.groupby(["method", "model", "horizon", "target_metric", "feature_set", "candidate_set"], dropna=False, sort=True)
            .agg(
                n_repeats=("repeat_id", "nunique"),
                achieved_metric_mean=("achieved_metric", "mean"),
                achieved_metric_std=("achieved_metric", "std"),
                best_single_metric_mean=("best_single_metric", "mean"),
                oracle_metric_mean=("oracle_metric", "mean"),
                improvement_mean=("improvement_vs_best_single", "mean"),
                improvement_std=("improvement_vs_best_single", "std"),
                gap_mean=("gap_to_oracle", "mean"),
                gap_std=("gap_to_oracle", "std"),
                routing_hit_oracle_rate_mean=("routing_hit_oracle_rate", "mean"),
                routing_hit_oracle_rate_std=("routing_hit_oracle_rate", "std"),
                classification_accuracy_mean=("classification_accuracy", "mean"),
                classification_accuracy_std=("classification_accuracy", "std"),
                top2_hit_rate_mean=("top2_hit_rate", "mean"),
                top2_hit_rate_std=("top2_hit_rate", "std"),
            )
            .reset_index()
        )
        comparison_df = repeat_agg_df.rename(
            columns={
                "target_metric": "metric",
                "achieved_metric_mean": "achieved_metric",
                "improvement_mean": "improvement",
                "gap_mean": "gap_to_oracle",
            }
        )

    output_cfg = cfg["outputs"]
    for key, value in output_cfg.items():
        if key.endswith("_path"):
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
    importance_df.to_csv(output_cfg["feature_importance_csv_path"], index=False)
    probs_df.to_csv(output_cfg["classification_probabilities_csv_path"], index=False)
    pd.DataFrame({"feature_name": full_features, "feature_set": "full_features"}).to_csv(output_cfg["feature_list_csv_path"], index=False)
    pd.DataFrame({"feature_name": pruned_features, "feature_set": "pruned_features"}).to_csv(output_cfg["pruned_feature_list_csv_path"], index=False)
    dropped_pairs_df.to_csv(output_cfg["dropped_correlation_pairs_csv_path"], index=False)

    prediction_examples_df = pd.DataFrame()
    if not routing_df.empty:
        n_examples = int(cfg.get("prediction_examples_per_task", 5))
        prediction_examples_df = (
            routing_df.sort_values(["method", "model", "horizon", "target_metric", "feature_set", "candidate_set", "repeat_id", "series_id"], kind="stable")
            .groupby(["method", "model", "horizon", "target_metric", "feature_set", "candidate_set"], sort=False)
            .head(n_examples)
            .reset_index(drop=True)
        )
    prediction_examples_df.to_csv(output_cfg["prediction_examples_csv_path"], index=False)

    excel_path = export_meta_modeling_excel(
        excel_path=output_cfg["excel_report_path"],
        summary_df=repeat_agg_df,
        task_results_df=task_results_df,
        routing_df=routing_df,
        model_order_df=mapping_df,
        dataset_summary_df=dataset_df,
        split_df=split_df,
        repeat_agg_df=repeat_agg_df,
        feature_list_df=pd.read_csv(output_cfg["feature_list_csv_path"]) if Path(output_cfg["feature_list_csv_path"]).exists() else None,
        best_single_repeat_df=best_single_df,
        comparison_df=comparison_df,
        candidates_df=candidate_df,
        feature_importance_df=importance_df,
        prediction_examples_df=prediction_examples_df,
        classification_probabilities_df=probs_df,
        pruned_feature_list_df=pd.read_csv(output_cfg["pruned_feature_list_csv_path"]) if Path(output_cfg["pruned_feature_list_csv_path"]).exists() else None,
        dropped_correlation_pairs_df=dropped_pairs_df,
    )

    feature_manifest = {
        "full_features": full_features,
        "pruned_features": pruned_features,
        "corr_threshold": float(cfg.get("feature_pruning", {}).get("corr_threshold", 0.90)),
    }
    with Path(output_cfg["feature_manifest_json_path"]).open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2)

    manifest = {
        "run_id": run_id,
        "stage": str(cfg.get("stage", "meta_modeling_experiments")),
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg["meta"]["config_path"],
        "outputs": output_cfg,
        "methods": methods,
        "summary": {
            "tasks_total": int(len(task_results_df)),
            "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()),
            "tasks_failed": int((task_results_df.get("status", pd.Series(dtype=str)) == "failed").sum()),
        },
    }
    manifest_path = write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "comparison_table_path": output_cfg["comparison_table_csv_path"],
        "candidate_models_path": output_cfg["candidate_models_csv_path"],
        "feature_importance_path": output_cfg["feature_importance_csv_path"],
        "prediction_examples_path": output_cfg["prediction_examples_csv_path"],
        "classification_probabilities_path": output_cfg["classification_probabilities_csv_path"],
        "excel_report_path": str(excel_path),
        "tasks_total": int(len(task_results_df)),
        "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()),
    }
