from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "rmse": "min",
    "directional_accuracy": "max",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _metric_source_columns(metrics_df: pd.DataFrame, target_metrics: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for metric in target_metrics:
        candidates = [metric, f"{metric}_mean"]
        found = next((c for c in candidates if c in metrics_df.columns), None)
        if found is None:
            raise ValueError(f"Metric column not found for '{metric}'. Tried: {candidates}")
        resolved[metric] = found
    return resolved


def aggregate_forecasting_metrics(
    metrics_df: pd.DataFrame,
    *,
    target_metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"series_id", "model_name", "horizon"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise ValueError(f"Forecasting metrics are missing required columns: {missing}")

    work = metrics_df.copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str).str.lower().eq("success")].copy()

    metric_cols = _metric_source_columns(work, target_metrics)
    for metric, col in metric_cols.items():
        work[metric] = pd.to_numeric(work[col], errors="coerce")

    group_cols = ["series_id", "model_name", "horizon"]
    agg = work.groupby(group_cols, dropna=False, sort=True).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in target_metrics},
        n_metric_rows=("model_name", "size"),
        n_folds=("fold_id", "nunique") if "fold_id" in work.columns else ("model_name", "size"),
    )
    agg = agg.reset_index()

    summary = {
        "input_rows": int(len(metrics_df)),
        "success_rows": int(len(work)),
        "aggregated_rows": int(len(agg)),
        "n_series": int(agg["series_id"].nunique()),
        "n_models": int(agg["model_name"].nunique()),
        "horizons": [int(x) for x in sorted(agg["horizon"].dropna().unique().tolist())],
        "metric_source_columns": metric_cols,
        "fold_id_present": bool("fold_id" in metrics_df.columns),
        "n_folds_values": [int(x) for x in sorted(agg["n_folds"].dropna().unique().tolist())],
    }
    return agg, summary


def _winner_distribution(
    aggregated_df: pd.DataFrame,
    *,
    target_metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon, hdf in aggregated_df.groupby("horizon", sort=True):
        for metric in target_metrics:
            metric_col = f"{metric}_mean"
            direction = METRIC_DIRECTIONS[metric]
            winners: list[str] = []
            missing_metric_series = 0
            for _, sdf in hdf.groupby("series_id", sort=True):
                g = sdf.dropna(subset=[metric_col]).copy()
                if g.empty:
                    missing_metric_series += 1
                    continue
                ascending = direction == "min"
                g = g.sort_values([metric_col, "model_name"], ascending=[ascending, True], kind="stable")
                winners.append(str(g.iloc[0]["model_name"]))

            counts = pd.Series(winners, dtype="object").value_counts().sort_index()
            total = int(counts.sum())
            majority_share = float(counts.max() / total) if total else np.nan
            rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "n_series_with_label": total,
                    "missing_metric_series": int(missing_metric_series),
                    "majority_class": str(counts.idxmax()) if total else "",
                    "majority_share": majority_share,
                    "class_counts": {str(k): int(v) for k, v in counts.items()},
                }
            )
    return rows


def _coverage_by_model(aggregated_df: pd.DataFrame, *, target_metrics: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon, hdf in aggregated_df.groupby("horizon", sort=True):
        expected_series = int(hdf["series_id"].nunique())
        for model_name, mdf in hdf.groupby("model_name", sort=True):
            row: dict[str, Any] = {
                "horizon": int(horizon),
                "model_name": str(model_name),
                "expected_series": expected_series,
                "observed_series": int(mdf["series_id"].nunique()),
                "missing_series": int(expected_series - mdf["series_id"].nunique()),
                "n_folds_min": int(mdf["n_folds"].min()) if "n_folds" in mdf.columns and not mdf.empty else None,
                "n_folds_max": int(mdf["n_folds"].max()) if "n_folds" in mdf.columns and not mdf.empty else None,
            }
            for metric in target_metrics:
                col = f"{metric}_mean"
                row[f"{metric}_missing_values"] = int(mdf[col].isna().sum()) if col in mdf.columns else None
            rows.append(row)
    return rows


def _candidate_sets(
    aggregated_df: pd.DataFrame,
    *,
    target_metrics: list[str],
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top_ks = sorted({int(k) for k in top_k_values if int(k) > 0})
    for horizon, hdf in aggregated_df.groupby("horizon", sort=True):
        expected_series = int(hdf["series_id"].nunique())
        for metric in target_metrics:
            col = f"{metric}_mean"
            direction = METRIC_DIRECTIONS[metric]
            means = (
                hdf.dropna(subset=[col])
                .groupby("model_name", sort=True)
                .agg(mean_metric=(col, "mean"), observed_series=("series_id", "nunique"))
                .reset_index()
            )
            means["complete_coverage"] = means["observed_series"].eq(expected_series)
            ascending = direction == "min"
            means = means.sort_values(["mean_metric", "model_name"], ascending=[ascending, True], kind="stable")
            for k in top_ks:
                selected = means.head(min(k, len(means))).copy()
                rows.append(
                    {
                        "horizon": int(horizon),
                        "metric": metric,
                        "candidate_set": f"top_{len(selected)}",
                        "requested_top_k": int(k),
                        "models": selected["model_name"].astype(str).tolist(),
                        "incomplete_models": selected.loc[~selected["complete_coverage"], "model_name"].astype(str).tolist(),
                    }
                )
    return rows


def _feature_diagnostics(features_df: pd.DataFrame, series_ids: set[str]) -> dict[str, Any]:
    if "series_id" not in features_df.columns:
        raise ValueError("Features table must contain 'series_id'")

    feature_series = set(features_df["series_id"].astype(str))
    numeric_cols = [
        c
        for c in features_df.columns
        if c not in {"series_id", "ticker", "market", "dataset_profile", "asset", "series_name"}
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    numeric = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan) if numeric_cols else pd.DataFrame()
    constant_cols = []
    if not numeric.empty:
        constant_cols = [c for c in numeric_cols if numeric[c].nunique(dropna=True) <= 1]

    return {
        "feature_rows": int(len(features_df)),
        "feature_series": int(len(feature_series)),
        "numeric_feature_count": int(len(numeric_cols)),
        "missing_series_in_features": sorted(series_ids - feature_series),
        "extra_feature_series": sorted(feature_series - series_ids),
        "nan_or_inf_cells": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        "features_with_nan_or_inf": [c for c in numeric_cols if numeric[c].isna().any()],
        "constant_features": constant_cols,
    }


def build_metamodeling_diagnostic(
    *,
    metrics_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    target_metrics: list[str] | None = None,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    metrics = target_metrics or ["rmse", "directional_accuracy"]
    top_ks = top_k_values or [3, 4, 5, 6]
    metrics_df = _read_table(metrics_path)
    features_df = _read_table(features_path)

    aggregated, aggregation_summary = aggregate_forecasting_metrics(metrics_df, target_metrics=metrics)
    series_ids = set(aggregated["series_id"].astype(str))
    diagnostic = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "metrics_path": str(Path(metrics_path).resolve()),
            "features_path": str(Path(features_path).resolve()),
        },
        "aggregation": aggregation_summary,
        "winner_distribution": _winner_distribution(aggregated, target_metrics=metrics),
        "coverage_by_model": _coverage_by_model(aggregated, target_metrics=metrics),
        "candidate_sets": _candidate_sets(aggregated, target_metrics=metrics, top_k_values=top_ks),
        "features": _feature_diagnostics(features_df, series_ids),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostic
