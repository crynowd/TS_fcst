from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_LOW_IS_BETTER = {"mae": True, "rmse": True, "mase": True, "directional_accuracy": False}


def _metric_column(metric_name: str) -> str:
    if metric_name == "directional_accuracy":
        return "directional_accuracy_mean"
    return f"{metric_name}_mean"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna()
    if not mask.any() or float(w[mask].sum()) == 0.0:
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _rank_variance(gdf: pd.DataFrame, metric_col: str, low_is_better: bool) -> float:
    df = gdf[["cluster_label", "model_name", metric_col]].copy()
    if low_is_better:
        df["rank"] = df.groupby("cluster_label")[metric_col].rank(method="average", ascending=True)
    else:
        df["rank"] = df.groupby("cluster_label")[metric_col].rank(method="average", ascending=False)
    var_by_model = df.groupby("model_name", sort=True)["rank"].var(ddof=0)
    return float(var_by_model.mean()) if not var_by_model.empty else float("nan")


def build_clustering_utility(
    cluster_model_performance_df: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    rows = []
    keys = ["clustering_config", "horizon"]
    for (cfg, horizon), gdf in cluster_model_performance_df.groupby(keys, sort=True):
        for metric_name in metrics:
            metric_col = _metric_column(str(metric_name))
            if metric_col not in gdf.columns:
                continue

            low_is_better = _LOW_IS_BETTER.get(str(metric_name), True)
            mg_rows = []
            for model_name, mdf in gdf.groupby("model_name", sort=True):
                mg_rows.append(
                    {
                        "model_name": model_name,
                        "global_score": _weighted_mean(mdf[metric_col], mdf["n_series"]),
                    }
                )
            model_global = pd.DataFrame(mg_rows)
            model_global = model_global.dropna(subset=["global_score"])
            if model_global.empty:
                continue

            model_global = model_global.sort_values(
                ["global_score", "model_name"],
                ascending=[True, True] if low_is_better else [False, True],
                kind="stable",
            )
            global_best_model = str(model_global.iloc[0]["model_name"])
            global_best_score = float(model_global.iloc[0]["global_score"])

            cluster_best_rows = []
            for _, cdf in gdf.groupby("cluster_label", sort=True):
                ranked = cdf.sort_values(
                    [metric_col, "model_name"],
                    ascending=[True, True] if low_is_better else [False, True],
                    kind="stable",
                )
                best = ranked.iloc[0]
                cluster_best_rows.append(
                    {
                        "cluster_label": best["cluster_label"],
                        "best_model": best["model_name"],
                        "best_score": best[metric_col],
                        "n_series": best["n_series"],
                    }
                )
            cluster_best_df = pd.DataFrame(cluster_best_rows)
            cluster_best_strategy_score = _weighted_mean(cluster_best_df["best_score"], cluster_best_df["n_series"])
            diversity = int(cluster_best_df["best_model"].nunique()) if not cluster_best_df.empty else 0

            global_model_cluster_scores = gdf[gdf["model_name"] == global_best_model]
            global_strategy_score = _weighted_mean(global_model_cluster_scores[metric_col], global_model_cluster_scores["n_series"])

            if low_is_better:
                regret_improvement = global_strategy_score - cluster_best_strategy_score
            else:
                regret_improvement = cluster_best_strategy_score - global_strategy_score

            rows.append(
                {
                    "clustering_config": cfg,
                    "horizon": int(horizon),
                    "metric_name": str(metric_name),
                    "global_best_model": global_best_model,
                    "cluster_best_strategy_score": cluster_best_strategy_score,
                    "global_best_strategy_score": global_strategy_score,
                    "regret_improvement": regret_improvement,
                    "best_model_diversity_count": diversity,
                    "model_rank_variance": _rank_variance(gdf, metric_col=metric_col, low_is_better=low_is_better),
                    "notes": "higher_regret_improvement_is_better",
                }
            )
    return pd.DataFrame(rows).sort_values(["clustering_config", "horizon", "metric_name"], kind="stable").reset_index(drop=True)
