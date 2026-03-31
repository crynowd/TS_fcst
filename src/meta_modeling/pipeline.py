from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.meta_modeling.models import MetaModelError, build_meta_model
from src.reporting.excel_export import export_meta_modeling_excel
from src.utils.manifest import get_git_commit, write_manifest


METRIC_DIRECTIONS = {
    "rmse": "min",
    "directional_accuracy": "max",
}


@dataclass
class MetaTaskData:
    horizon: int
    target_metric: str
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
    raise ValueError(f"Cannot find common join key between features and forecasting results. Tried: {preferred}")


def _feature_columns(features_df: pd.DataFrame, join_key: str) -> list[str]:
    non_features = {join_key, "ticker", "market", "dataset_profile", "asset", "series_name"}
    cols = []
    for c in features_df.columns:
        if c in non_features:
            continue
        if pd.api.types.is_numeric_dtype(features_df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found in features artifact")
    return cols


def _infer_feature_set_kind(features_path: str) -> str:
    name = Path(features_path).name.lower()
    if name.startswith("final_"):
        return "final_curated_feature_set"
    return "numeric_columns_from_selected_artifact"


def build_meta_dataset_from_existing_artifacts(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_path = str(cfg["inputs"]["features_path"])
    features_df = pd.read_parquet(features_path).copy()
    forecasting_df = pd.read_parquet(cfg["inputs"]["forecasting_series_metrics_path"]).copy()

    join_key = _pick_join_key(features_df, forecasting_df, cfg)
    metric_cols = _resolve_metric_columns(forecasting_df, cfg)

    dataset_filter = str(cfg.get("dataset_filter", "")).strip()
    if dataset_filter and "dataset_profile" in features_df.columns:
        features_df = features_df[features_df["dataset_profile"].astype(str) == dataset_filter].copy()

    feature_cols = _feature_columns(features_df, join_key=join_key)
    feature_dtype_map = {c: str(features_df[c].dtype) for c in feature_cols}

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
        "feature_dtype_map": feature_dtype_map,
        "feature_source_path": features_path,
        "feature_set_kind": _infer_feature_set_kind(features_path),
        "metric_columns": metric_cols,
        "n_rows": int(len(merged)),
        "n_series": int(merged[join_key].nunique()),
        "horizons": sorted(merged["horizon"].unique().tolist()),
        "models": sorted(merged["model_name"].unique().tolist()),
        "target_metrics": sorted(metric_cols.keys()),
    }
    return merged, summary


def build_task_dataset(
    meta_long_df: pd.DataFrame,
    *,
    join_key: str,
    feature_cols: list[str],
    horizon: int,
    target_metric: str,
) -> MetaTaskData:
    sub = meta_long_df[meta_long_df["horizon"] == int(horizon)].copy()
    if sub.empty:
        raise ValueError(f"No rows for horizon={horizon}")

    task_df = sub[[join_key, "horizon", "model_name", target_metric, *feature_cols]].copy()
    task_df = task_df.dropna(subset=[target_metric])
    model_order = sorted(task_df["model_name"].astype(str).unique().tolist())
    if not model_order:
        raise ValueError(f"No models available for horizon={horizon}, metric={target_metric}")

    pivot = task_df.pivot_table(index=join_key, columns="model_name", values=target_metric, aggfunc="mean")
    pivot = pivot.reindex(columns=model_order)
    pivot = pivot.dropna(axis=0, how="any")

    feature_base = task_df[[join_key, *feature_cols]].drop_duplicates(subset=[join_key], keep="first").set_index(join_key)
    aligned = feature_base.join(pivot, how="inner").dropna(axis=0, how="any")
    if aligned.empty:
        raise ValueError(f"Task dataset is empty after alignment for h={horizon}, metric={target_metric}")

    X = aligned[feature_cols].to_numpy(dtype=np.float64)
    y = aligned[model_order].to_numpy(dtype=np.float64)
    row_ids = aligned.index.to_numpy()
    return MetaTaskData(
        horizon=int(horizon),
        target_metric=str(target_metric),
        feature_cols=list(feature_cols),
        model_order=model_order,
        row_ids=row_ids,
        X=X,
        y=y,
    )


def _repeat_seeds(split_cfg: dict[str, Any]) -> list[int]:
    n_repeats = int(split_cfg.get("n_repeats", 1))
    base_seed = int(split_cfg.get("random_seed", 42))
    explicit = [int(x) for x in split_cfg.get("random_seeds", []) if str(x).strip()]
    if explicit:
        if len(explicit) >= n_repeats:
            return explicit[:n_repeats]
        out = list(explicit)
        step = 9973
        while len(out) < n_repeats:
            out.append(base_seed + len(out) * step)
        return out
    return [base_seed + i * 9973 for i in range(n_repeats)]


def split_by_series_ids(series_ids: np.ndarray, cfg: dict[str, Any], random_seed: int | None = None) -> SplitIndices:
    unique_ids = np.unique(series_ids.astype(str))
    if len(unique_ids) < 3:
        raise ValueError("Need at least 3 unique series for train/validation/test split")

    split_cfg = cfg.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    validation_size = float(split_cfg.get("validation_size", 0.1))
    seed = int(split_cfg.get("random_seed", 42) if random_seed is None else random_seed)

    if test_size <= 0 or validation_size < 0 or test_size + validation_size >= 1.0:
        raise ValueError("Invalid split sizes")

    rng = np.random.default_rng(seed)
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
    if not train_ids:
        raise ValueError("Empty train split after split_by_series_ids")

    ids = series_ids.astype(str)
    train_idx = np.where(np.isin(ids, list(train_ids)))[0]
    val_idx = np.where(np.isin(ids, list(val_ids)))[0]
    test_idx = np.where(np.isin(ids, list(test_ids)))[0]
    return SplitIndices(train=train_idx, validation=val_idx, test=test_idx)


def _pick_best_index(values: np.ndarray, direction: str) -> int:
    if direction == "min":
        return int(np.argmin(values))
    return int(np.argmax(values))


def compute_best_single_baseline(y_train: np.ndarray, y_test: np.ndarray, model_order: list[str], direction: str) -> dict[str, Any]:
    train_mean = np.nanmean(y_train, axis=0)
    baseline_idx = _pick_best_index(train_mean, direction=direction)
    baseline_model = model_order[baseline_idx]
    baseline_values = y_test[:, baseline_idx]
    return {
        "baseline_model": baseline_model,
        "baseline_idx": int(baseline_idx),
        "baseline_values": baseline_values,
        "train_mean_by_model": train_mean,
        "selection_rule": (
            "minimum train mean metric" if direction == "min" else "maximum train mean metric"
        ),
    }


def _regression_aux_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse

def _compute_routing_rows(
    *,
    run_id: str,
    repeat_id: int,
    split_seed: int,
    meta_model_name: str,
    task: MetaTaskData,
    split: SplitIndices,
    y_pred_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    direction = METRIC_DIRECTIONS[task.target_metric]
    y_test = task.y[split.test]
    y_train = task.y[split.train]
    row_ids_test = task.row_ids[split.test]

    baseline = compute_best_single_baseline(y_train=y_train, y_test=y_test, model_order=task.model_order, direction=direction)
    baseline_model = str(baseline["baseline_model"])
    baseline_values = np.asarray(baseline["baseline_values"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for i in range(y_test.shape[0]):
        pred_row = y_pred_test[i]
        actual_row = y_test[i]
        selected_idx = _pick_best_index(pred_row, direction=direction)
        oracle_idx = _pick_best_index(actual_row, direction=direction)

        selected_model = task.model_order[selected_idx]
        oracle_model = task.model_order[oracle_idx]
        achieved = float(actual_row[selected_idx])
        oracle_metric = float(actual_row[oracle_idx])
        baseline_metric = float(baseline_values[i])

        if direction == "min":
            regret = achieved - oracle_metric
            improvement_vs_baseline = baseline_metric - achieved
        else:
            regret = oracle_metric - achieved
            improvement_vs_baseline = achieved - baseline_metric

        rows.append(
            {
                "run_id": run_id,
                "repeat_id": int(repeat_id),
                "split_seed": int(split_seed),
                "horizon": int(task.horizon),
                "target_metric": task.target_metric,
                "meta_model": meta_model_name,
                "series_id": str(row_ids_test[i]),
                "selected_model": str(selected_model),
                "oracle_model": str(oracle_model),
                "best_single_model": baseline_model,
                "achieved_metric": achieved,
                "oracle_metric": oracle_metric,
                "baseline_metric": baseline_metric,
                "regret_to_oracle": float(regret),
                "improvement_vs_best_single": float(improvement_vs_baseline),
                "routing_hit_oracle": int(selected_idx == oracle_idx),
                "predicted_metric_vector": json.dumps([float(v) for v in pred_row]),
                "actual_metric_vector": json.dumps([float(v) for v in actual_row]),
                "target_model_order": json.dumps(task.model_order),
                "count_models": int(len(task.model_order)),
                "count_features": int(len(task.feature_cols)),
            }
        )

    routing_df = pd.DataFrame(rows)
    aux_mae, aux_rmse = _regression_aux_metrics(y_true=y_test, y_pred=y_pred_test)
    summary = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(split_seed),
        "horizon": int(task.horizon),
        "target_metric": task.target_metric,
        "meta_model": meta_model_name,
        "n_test_series": int(len(routing_df)),
        "best_single_model": baseline_model,
        "achieved_mean_metric": float(pd.to_numeric(routing_df["achieved_metric"], errors="coerce").mean()),
        "oracle_mean_metric": float(pd.to_numeric(routing_df["oracle_metric"], errors="coerce").mean()),
        "best_single_mean_metric": float(pd.to_numeric(routing_df["baseline_metric"], errors="coerce").mean()),
        "regret_to_oracle_mean": float(pd.to_numeric(routing_df["regret_to_oracle"], errors="coerce").mean()),
        "improvement_vs_best_single_mean": float(pd.to_numeric(routing_df["improvement_vs_best_single"], errors="coerce").mean()),
        "routing_hit_oracle_rate": float(pd.to_numeric(routing_df["routing_hit_oracle"], errors="coerce").mean()),
        "target_vector_mae": float(aux_mae),
        "target_vector_rmse": float(aux_rmse),
        "status": "success",
        "notes": "",
    }
    baseline_row = {
        "run_id": run_id,
        "repeat_id": int(repeat_id),
        "split_seed": int(split_seed),
        "horizon": int(task.horizon),
        "target_metric": task.target_metric,
        "best_single_model": baseline_model,
        "selection_rule": str(baseline["selection_rule"]),
        "best_single_train_mean_metric": float(baseline["train_mean_by_model"][baseline["baseline_idx"]]),
        "best_single_test_mean_metric": float(pd.to_numeric(routing_df["baseline_metric"], errors="coerce").mean()),
    }
    return routing_df, summary, baseline_row


def _build_task_matrix_records(task: MetaTaskData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(task.model_order):
        rows.append(
            {
                "horizon": int(task.horizon),
                "target_metric": task.target_metric,
                "output_dim": int(idx),
                "model_name": str(model_name),
            }
        )
    return rows


def _compute_forecasting_baselines(meta_long_df: pd.DataFrame, *, join_key: str, target_metrics: list[str]) -> dict[str, pd.DataFrame]:
    mean_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []

    for metric in target_metrics:
        direction = METRIC_DIRECTIONS[metric]
        sdf = meta_long_df[[join_key, "horizon", "model_name", metric]].copy().dropna(subset=[metric])

        mean_tbl = (
            sdf.groupby(["horizon", "model_name"], dropna=False, sort=False)[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "mean_metric"})
        )
        mean_tbl["target_metric"] = metric
        mean_tbl["opt_direction"] = direction
        mean_tbl["n_series"] = (
            sdf.groupby(["horizon", "model_name"], dropna=False, sort=False)[join_key]
            .nunique()
            .reset_index(drop=True)
        )
        mean_rows.extend(mean_tbl.to_dict("records"))

        for horizon, hdf in mean_tbl.groupby("horizon", sort=True):
            ascending = direction == "min"
            ranked = hdf.sort_values(["mean_metric", "model_name"], ascending=[ascending, True], kind="stable").copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
            rank_rows.extend(ranked.to_dict("records"))
            best = ranked.iloc[0]
            best_rows.append(
                {
                    "horizon": int(horizon),
                    "target_metric": metric,
                    "best_single_model": str(best["model_name"]),
                    "best_single_mean_metric": float(best["mean_metric"]),
                    "selection_rule": (
                        "minimum global mean metric across series" if direction == "min" else "maximum global mean metric across series"
                    ),
                    "n_series": int(best["n_series"]),
                }
            )

        for horizon, hdf in sdf.groupby("horizon", sort=True):
            winners = []
            for _, g in hdf.groupby(join_key, sort=False):
                g2 = g.sort_values([metric, "model_name"], ascending=[direction == "min", True], kind="stable")
                winners.append(str(g2.iloc[0]["model_name"]))
            if winners:
                cnt = pd.Series(winners).value_counts().reset_index()
                cnt.columns = ["model_name", "win_count"]
                for r in cnt.itertuples(index=False):
                    win_rows.append(
                        {
                            "horizon": int(horizon),
                            "target_metric": metric,
                            "model_name": str(r.model_name),
                            "win_count": int(r.win_count),
                        }
                    )

    mean_df = pd.DataFrame(mean_rows)
    rank_df = pd.DataFrame(rank_rows)
    best_df = pd.DataFrame(best_rows)
    wins_df = pd.DataFrame(win_rows)
    return {
        "forecasting_mean_by_model": mean_df,
        "forecasting_model_ranking": rank_df,
        "best_single_global_baseline": best_df,
        "forecasting_model_wins": wins_df,
    }


def _aggregate_across_repeats(task_summary_df: pd.DataFrame) -> pd.DataFrame:
    if task_summary_df.empty:
        return pd.DataFrame()
    work = task_summary_df[task_summary_df["status"].astype(str) == "success"].copy()
    if work.empty:
        return pd.DataFrame()

    metrics = [
        "achieved_mean_metric",
        "oracle_mean_metric",
        "best_single_mean_metric",
        "regret_to_oracle_mean",
        "improvement_vs_best_single_mean",
        "routing_hit_oracle_rate",
        "target_vector_mae",
        "target_vector_rmse",
    ]
    for col in metrics:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = (
        work.groupby(["horizon", "target_metric", "meta_model"], dropna=False, sort=True)
        .agg(
            n_repeats=("repeat_id", "nunique"),
            achieved_mean_metric_mean=("achieved_mean_metric", "mean"),
            achieved_mean_metric_std=("achieved_mean_metric", "std"),
            oracle_mean_metric_mean=("oracle_mean_metric", "mean"),
            oracle_mean_metric_std=("oracle_mean_metric", "std"),
            best_single_mean_metric_mean=("best_single_mean_metric", "mean"),
            best_single_mean_metric_std=("best_single_mean_metric", "std"),
            regret_to_oracle_mean_mean=("regret_to_oracle_mean", "mean"),
            regret_to_oracle_mean_std=("regret_to_oracle_mean", "std"),
            improvement_vs_best_single_mean_mean=("improvement_vs_best_single_mean", "mean"),
            improvement_vs_best_single_mean_std=("improvement_vs_best_single_mean", "std"),
            routing_hit_oracle_rate_mean=("routing_hit_oracle_rate", "mean"),
            routing_hit_oracle_rate_std=("routing_hit_oracle_rate", "std"),
            target_vector_mae_mean=("target_vector_mae", "mean"),
            target_vector_rmse_mean=("target_vector_rmse", "mean"),
        )
        .reset_index()
    )
    return grouped


def _build_feature_artifacts(dataset_summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = list(dataset_summary["feature_cols"])
    feature_list_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "dtype": [dataset_summary["feature_dtype_map"].get(c, "") for c in feature_cols],
            "source_artifact": dataset_summary["feature_source_path"],
            "feature_set_kind": dataset_summary["feature_set_kind"],
        }
    )
    feature_manifest = {
        "feature_source_path": dataset_summary["feature_source_path"],
        "feature_set_kind": dataset_summary["feature_set_kind"],
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "feature_dtypes": dataset_summary["feature_dtype_map"],
    }
    return feature_list_df, feature_manifest

def run_meta_modeling_pipeline(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    stage_name = str(cfg.get("stage", "meta_modeling"))
    run_name = str(cfg.get("run_name", "meta_modeling_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    start_ts = datetime.now(timezone.utc)

    meta_long_df, dataset_summary = build_meta_dataset_from_existing_artifacts(cfg)
    join_key = dataset_summary["join_key"]
    feature_cols = dataset_summary["feature_cols"]
    horizons = sorted(int(h) for h in dataset_summary["horizons"])
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("target_metrics", ["rmse", "directional_accuracy"])]
    meta_models = [str(x) for x in cfg.get("meta_models", ["ridge", "random_forest", "catboost"])]
    repeat_seeds = _repeat_seeds(cfg.get("split", {}))

    logger.info(
        "meta_modeling start run_id=%s rows=%d series=%d horizons=%d models=%d repeats=%d",
        run_id,
        dataset_summary["n_rows"],
        dataset_summary["n_series"],
        len(horizons),
        len(dataset_summary["models"]),
        len(repeat_seeds),
    )

    forecasting_baselines = _compute_forecasting_baselines(meta_long_df, join_key=join_key, target_metrics=target_metrics)

    dataset_out_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    routing_rows_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_single_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        for target_metric in target_metrics:
            task = build_task_dataset(
                meta_long_df=meta_long_df,
                join_key=join_key,
                feature_cols=feature_cols,
                horizon=horizon,
                target_metric=target_metric,
            )
            dataset_out_rows.append(
                {
                    "run_id": run_id,
                    "horizon": int(task.horizon),
                    "target_metric": task.target_metric,
                    "n_series": int(task.X.shape[0]),
                    "n_features": int(task.X.shape[1]),
                    "n_models": int(task.y.shape[1]),
                    "feature_columns_json": json.dumps(task.feature_cols),
                    "target_model_order_json": json.dumps(task.model_order),
                }
            )
            mapping_rows.extend(_build_task_matrix_records(task))

            for repeat_idx, seed in enumerate(repeat_seeds, start=1):
                split = split_by_series_ids(task.row_ids, cfg=cfg, random_seed=seed)
                split_mark = np.full(task.row_ids.shape[0], "test", dtype=object)
                split_mark[split.train] = "train"
                split_mark[split.validation] = "validation"
                for idx, sid in enumerate(task.row_ids):
                    split_rows.append(
                        {
                            "run_id": run_id,
                            "repeat_id": int(repeat_idx),
                            "split_seed": int(seed),
                            "horizon": int(task.horizon),
                            "target_metric": task.target_metric,
                            "series_id": str(sid),
                            "split": str(split_mark[idx]),
                        }
                    )

                X_train = task.X[split.train]
                y_train = task.y[split.train]
                X_test = task.X[split.test]

                if X_train.shape[0] < 2 or X_test.shape[0] < 1:
                    for model_name in meta_models:
                        summary_rows.append(
                            {
                                "run_id": run_id,
                                "repeat_id": int(repeat_idx),
                                "split_seed": int(seed),
                                "horizon": int(task.horizon),
                                "target_metric": task.target_metric,
                                "meta_model": model_name,
                                "status": "failed",
                                "notes": "insufficient train/test rows for this task",
                            }
                        )
                    continue

                for model_name in meta_models:
                    try:
                        model = build_meta_model(model_name=model_name, cfg=cfg)
                        model.fit(X_train, y_train)
                        y_pred_test = np.asarray(model.predict(X_test), dtype=np.float64)
                        if y_pred_test.ndim != 2:
                            raise MetaModelError("Predicted target matrix must be 2-dimensional")
                        if y_pred_test.shape[1] != y_train.shape[1]:
                            raise MetaModelError("Predicted target dimension does not match model order")

                        routing_df, summary, best_row = _compute_routing_rows(
                            run_id=run_id,
                            repeat_id=repeat_idx,
                            split_seed=seed,
                            meta_model_name=model_name,
                            task=task,
                            split=split,
                            y_pred_test=y_pred_test,
                        )
                        routing_rows_all.append(routing_df)
                        summary_rows.append(summary)
                        best_single_rows.append(best_row)
                    except Exception as exc:
                        summary_rows.append(
                            {
                                "run_id": run_id,
                                "repeat_id": int(repeat_idx),
                                "split_seed": int(seed),
                                "horizon": int(task.horizon),
                                "target_metric": task.target_metric,
                                "meta_model": model_name,
                                "status": "failed",
                                "notes": f"{exc.__class__.__name__}: {exc}",
                            }
                        )
                        logger.warning(
                            "meta_model failed model=%s horizon=%d metric=%s repeat=%d seed=%d err=%s",
                            model_name,
                            task.horizon,
                            task.target_metric,
                            repeat_idx,
                            seed,
                            exc,
                        )

    dataset_summary_df = pd.DataFrame(dataset_out_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    routing_df = pd.concat(routing_rows_all, ignore_index=True) if routing_rows_all else pd.DataFrame()
    task_summary_df = pd.DataFrame(summary_rows)
    split_df = pd.DataFrame(split_rows)
    best_single_by_repeat_df = pd.DataFrame(best_single_rows).drop_duplicates(
        subset=["repeat_id", "split_seed", "horizon", "target_metric", "best_single_model"], keep="first"
    )
    repeat_agg_df = _aggregate_across_repeats(task_summary_df)
    feature_list_df, feature_manifest = _build_feature_artifacts(dataset_summary)

    output_cfg = cfg["outputs"]
    for key, path_str in output_cfg.items():
        if key.endswith("_path"):
            Path(path_str).resolve().parent.mkdir(parents=True, exist_ok=True)

    dataset_summary_df.to_csv(output_cfg["meta_dataset_summary_csv_path"], index=False)
    dataset_summary_df.to_parquet(output_cfg["meta_dataset_summary_parquet_path"], index=False)
    mapping_df.to_csv(output_cfg["model_order_mapping_csv_path"], index=False)

    if not routing_df.empty:
        routing_df.to_parquet(output_cfg["routing_rows_parquet_path"], index=False)
        routing_df.to_csv(output_cfg["routing_rows_csv_path"], index=False)
    else:
        pd.DataFrame().to_parquet(output_cfg["routing_rows_parquet_path"], index=False)
        pd.DataFrame().to_csv(output_cfg["routing_rows_csv_path"], index=False)

    task_summary_df.to_csv(output_cfg["task_results_csv_path"], index=False)
    task_summary_df.to_parquet(output_cfg["task_results_parquet_path"], index=False)
    split_df.to_csv(output_cfg["split_assignments_csv_path"], index=False)
    repeat_agg_df.to_csv(output_cfg["repeat_aggregated_results_csv_path"], index=False)
    best_single_by_repeat_df.to_csv(output_cfg["best_single_baseline_by_repeat_csv_path"], index=False)

    forecasting_baselines["forecasting_mean_by_model"].to_csv(output_cfg["forecasting_mean_by_model_csv_path"], index=False)
    forecasting_baselines["forecasting_model_ranking"].to_csv(output_cfg["forecasting_model_ranking_csv_path"], index=False)
    forecasting_baselines["best_single_global_baseline"].to_csv(output_cfg["best_single_global_baseline_csv_path"], index=False)
    forecasting_baselines["forecasting_model_wins"].to_csv(output_cfg["forecasting_model_wins_csv_path"], index=False)

    feature_list_df.to_csv(output_cfg["feature_list_csv_path"], index=False)
    with Path(output_cfg["feature_manifest_json_path"]).open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2)

    excel_path = export_meta_modeling_excel(
        excel_path=output_cfg["excel_report_path"],
        summary_df=repeat_agg_df,
        task_results_df=task_summary_df,
        routing_df=routing_df,
        model_order_df=mapping_df,
        dataset_summary_df=dataset_summary_df,
        split_df=split_df,
        forecasting_mean_df=forecasting_baselines["forecasting_mean_by_model"],
        forecasting_ranking_df=forecasting_baselines["forecasting_model_ranking"],
        best_single_global_df=forecasting_baselines["best_single_global_baseline"],
        model_wins_df=forecasting_baselines["forecasting_model_wins"],
        repeat_agg_df=repeat_agg_df,
        feature_list_df=feature_list_df,
        best_single_repeat_df=best_single_by_repeat_df,
    )

    end_ts = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg["meta"]["config_path"],
        "input_sources": {
            "features_path": cfg["inputs"]["features_path"],
            "forecasting_series_metrics_path": cfg["inputs"]["forecasting_series_metrics_path"],
            "forecasting_manifest_path": cfg["inputs"].get("forecasting_manifest_path", ""),
        },
        "feature_manifest": feature_manifest,
        "outputs": {
            "meta_dataset_summary_csv": output_cfg["meta_dataset_summary_csv_path"],
            "meta_dataset_summary_parquet": output_cfg["meta_dataset_summary_parquet_path"],
            "model_order_mapping_csv": output_cfg["model_order_mapping_csv_path"],
            "routing_rows_parquet": output_cfg["routing_rows_parquet_path"],
            "routing_rows_csv": output_cfg["routing_rows_csv_path"],
            "task_results_csv": output_cfg["task_results_csv_path"],
            "task_results_parquet": output_cfg["task_results_parquet_path"],
            "split_assignments_csv": output_cfg["split_assignments_csv_path"],
            "repeat_aggregated_results_csv": output_cfg["repeat_aggregated_results_csv_path"],
            "best_single_baseline_by_repeat_csv": output_cfg["best_single_baseline_by_repeat_csv_path"],
            "forecasting_mean_by_model_csv": output_cfg["forecasting_mean_by_model_csv_path"],
            "forecasting_model_ranking_csv": output_cfg["forecasting_model_ranking_csv_path"],
            "best_single_global_baseline_csv": output_cfg["best_single_global_baseline_csv_path"],
            "forecasting_model_wins_csv": output_cfg["forecasting_model_wins_csv_path"],
            "feature_list_csv": output_cfg["feature_list_csv_path"],
            "feature_manifest_json": output_cfg["feature_manifest_json_path"],
            "excel_report": str(excel_path),
            "log": cfg.get("meta", {}).get("log_path", ""),
        },
        "summary": {
            "n_rows_meta_long": int(dataset_summary["n_rows"]),
            "n_series": int(dataset_summary["n_series"]),
            "horizons": horizons,
            "models": dataset_summary["models"],
            "target_metrics": target_metrics,
            "meta_models": meta_models,
            "repeat_splits": len(repeat_seeds),
            "repeat_seeds": repeat_seeds,
            "tasks_total": int(len(task_summary_df)),
            "tasks_success": int((task_summary_df.get("status", pd.Series(dtype=str)) == "success").sum()),
            "tasks_failed": int((task_summary_df.get("status", pd.Series(dtype=str)) == "failed").sum()),
        },
    }
    manifest_path = write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "task_results_path": output_cfg["task_results_csv_path"],
        "routing_rows_path": output_cfg["routing_rows_csv_path"],
        "model_order_mapping_path": output_cfg["model_order_mapping_csv_path"],
        "repeat_aggregated_results_path": output_cfg["repeat_aggregated_results_csv_path"],
        "forecasting_ranking_path": output_cfg["forecasting_model_ranking_csv_path"],
        "feature_list_path": output_cfg["feature_list_csv_path"],
        "excel_report_path": str(excel_path),
        "tasks_total": int(len(task_summary_df)),
        "tasks_success": int((task_summary_df.get("status", pd.Series(dtype=str)) == "success").sum()),
        "detected_horizons": horizons,
        "detected_target_metrics": target_metrics,
        "detected_models": dataset_summary["models"],
        "repeat_seeds": repeat_seeds,
        "feature_set_kind": dataset_summary["feature_set_kind"],
    }
