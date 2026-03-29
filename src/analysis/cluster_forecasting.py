from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REQUIRED_SHORTLIST_CONFIGS = [
    "cfg_0191",
    "cfg_0185",
    "cfg_0193",
    "cfg_0443",
    "cfg_0444",
    "cfg_0445",
]

_SERIES_METRIC_COLUMNS = [
    "mae_mean",
    "rmse_mean",
    "mase_mean",
    "smape_mean",
    "directional_accuracy_mean",
    "huber_mean",
    "medae_mean",
    "bias_mean",
    "r2_mean",
    "r2_oos_mean",
    "mae_median",
    "rmse_median",
    "mase_median",
]


def validate_shortlist_configs(
    shortlist_configs: Sequence[str],
    selected_configs_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
) -> dict[str, list[str]]:
    selected_ids = set(selected_configs_df.get("config_id", pd.Series(dtype=str)).astype(str).unique())
    assignment_ids = set(cluster_assignments_df.get("config_id", pd.Series(dtype=str)).astype(str).unique())

    missing_in_selected = sorted(set(shortlist_configs) - selected_ids)
    missing_in_assignments = sorted(set(shortlist_configs) - assignment_ids)
    available = sorted(set(shortlist_configs).intersection(assignment_ids))

    return {
        "available_configs": available,
        "missing_in_selected": missing_in_selected,
        "missing_in_assignments": missing_in_assignments,
    }


def _build_config_mapping(cluster_assignments_df: pd.DataFrame, configs: Iterable[str]) -> pd.DataFrame:
    mapping = cluster_assignments_df[
        cluster_assignments_df["config_id"].astype(str).isin([str(c) for c in configs])
    ][["config_id", "series_id", "cluster_label"]].copy()
    mapping = mapping.drop_duplicates(subset=["config_id", "series_id"], keep="last")
    mapping = mapping.rename(columns={"config_id": "clustering_config"})
    return mapping.reset_index(drop=True)


def join_forecasting_with_cluster_labels(
    series_metrics_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    raw_predictions_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
    available_configs: Sequence[str],
) -> dict[str, pd.DataFrame]:
    mapping = _build_config_mapping(cluster_assignments_df, available_configs)

    series_joined = series_metrics_df.merge(mapping, on="series_id", how="inner")
    fold_joined = fold_metrics_df.merge(mapping, on="series_id", how="inner")
    raw_joined = raw_predictions_df.merge(mapping, on="series_id", how="inner") if not raw_predictions_df.empty else raw_predictions_df

    return {
        "mapping": mapping,
        "series_joined": series_joined,
        "fold_joined": fold_joined,
        "raw_joined": raw_joined,
    }


def build_cluster_model_performance(series_joined_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["clustering_config", "horizon", "cluster_label", "model_name"]
    working = series_joined_df.copy()
    for col in _SERIES_METRIC_COLUMNS:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    agg_spec: dict[str, tuple[str, str]] = {
        "n_series": ("series_id", "nunique"),
        "mae_mean": ("mae_mean", "mean"),
        "mae_median": ("mae_mean", "median"),
        "rmse_mean": ("rmse_mean", "mean"),
        "rmse_median": ("rmse_mean", "median"),
        "mase_mean": ("mase_mean", "mean"),
        "mase_median": ("mase_mean", "median"),
        "smape_mean": ("smape_mean", "mean"),
        "directional_accuracy_mean": ("directional_accuracy_mean", "mean"),
        "huber_mean": ("huber_mean", "mean"),
        "medae_mean": ("medae_mean", "mean"),
        "bias_mean": ("bias_mean", "mean"),
        "r2_mean": ("r2_mean", "mean"),
        "r2_oos_mean": ("r2_oos_mean", "mean"),
    }

    out = (
        working.groupby(group_cols, dropna=False, sort=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )
    return out


def append_relative_to_baseline(
    cluster_model_performance_df: pd.DataFrame,
    baseline_model: str = "naive_zero",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["clustering_config", "horizon", "cluster_label"]
    metric_pairs = ["mae", "rmse", "mase"]

    baseline = cluster_model_performance_df[
        cluster_model_performance_df["model_name"].astype(str) == str(baseline_model)
    ][keys + [f"{m}_mean" for m in metric_pairs]].copy()
    baseline = baseline.rename(columns={f"{m}_mean": f"{m}_baseline" for m in metric_pairs})

    merged = cluster_model_performance_df.merge(baseline, on=keys, how="left")

    for metric in metric_pairs:
        metric_col = f"{metric}_mean"
        base_col = f"{metric}_baseline"
        diff_col = f"{metric}_vs_{baseline_model}_diff"
        ratio_col = f"{metric}_vs_{baseline_model}_ratio"

        merged[diff_col] = merged[metric_col] - merged[base_col]
        merged[ratio_col] = np.where(
            pd.to_numeric(merged[base_col], errors="coerce") != 0,
            merged[metric_col] / merged[base_col],
            np.nan,
        )

    missing_baseline = merged[merged[[f"{m}_baseline" for m in metric_pairs]].isna().any(axis=1)][keys].drop_duplicates()
    merged = merged.drop(columns=[f"{m}_baseline" for m in metric_pairs])
    return merged, missing_baseline


def _pick_best_model(group_df: pd.DataFrame, metric_col: str, maximize: bool) -> str:
    order = [True, True]
    sort_cols = [metric_col, "model_name"]
    if maximize:
        ranked = group_df.sort_values(sort_cols, ascending=[False, True], kind="stable")
    else:
        ranked = group_df.sort_values(sort_cols, ascending=order, kind="stable")
    return str(ranked.iloc[0]["model_name"]) if not ranked.empty else ""


def build_best_model_by_cluster(cluster_model_performance_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["clustering_config", "horizon", "cluster_label"]
    rows = []
    for key_vals, gdf in cluster_model_performance_df.groupby(keys, sort=True):
        row = dict(zip(keys, key_vals))
        row["best_model_mae"] = _pick_best_model(gdf, "mae_mean", maximize=False)
        row["best_model_rmse"] = _pick_best_model(gdf, "rmse_mean", maximize=False)
        row["best_model_mase"] = _pick_best_model(gdf, "mase_mean", maximize=False)
        row["best_model_direction"] = _pick_best_model(gdf, "directional_accuracy_mean", maximize=True)
        row["n_series_in_cluster"] = int(pd.to_numeric(gdf["n_series"], errors="coerce").max()) if not gdf.empty else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def build_tidy_cluster_model_metrics(cluster_model_performance_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["clustering_config", "horizon", "cluster_label", "model_name"]
    metric_cols = [
        c
        for c in cluster_model_performance_df.columns
        if c not in id_cols and c != "n_series" and pd.api.types.is_numeric_dtype(cluster_model_performance_df[c])
    ]
    tidy = cluster_model_performance_df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="metric_value",
    )
    return tidy.sort_values([*id_cols, "metric_name"], kind="stable").reset_index(drop=True)


def build_tidy_series_cluster_metrics(series_joined_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["series_id", "clustering_config", "cluster_label", "horizon", "model_name"]
    metric_cols = [c for c in series_joined_df.columns if c.endswith("_mean") or c.endswith("_median") or c.endswith("_std")]
    keep = [c for c in base_cols + metric_cols if c in series_joined_df.columns]
    out = series_joined_df[keep].copy()
    return out.sort_values(base_cols, kind="stable").reset_index(drop=True)
