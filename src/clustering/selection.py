from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    lo = float(vals.min())
    hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return (vals - lo) / (hi - lo)


def add_ranking_score(grid_df: pd.DataFrame) -> pd.DataFrame:
    out = grid_df.copy()
    valid = out["silhouette_score"].notna() & out["davies_bouldin_score"].notna() & out["calinski_harabasz_score"].notna()
    out["ranking_score"] = np.nan
    if valid.any():
        dbi_scaled = _minmax(out.loc[valid, "davies_bouldin_score"])
        ch_scaled = _minmax(out.loc[valid, "calinski_harabasz_score"])
        size_penalty = pd.to_numeric(out.loc[valid, "cluster_size_ratio_max"], errors="coerce").fillna(1.0)
        tiny_penalty = (
            pd.to_numeric(out.loc[valid, "n_small_clusters"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out.loc[valid, "n_singleton_clusters"], errors="coerce").fillna(0.0)
        ) / pd.to_numeric(out.loc[valid, "n_clusters_actual"], errors="coerce").replace(0, np.nan).fillna(1.0)
        sil = pd.to_numeric(out.loc[valid, "silhouette_score"], errors="coerce").fillna(-1.0)
        score = sil + 0.35 * ch_scaled - 0.35 * dbi_scaled - 0.50 * size_penalty - 0.30 * tiny_penalty
        out.loc[valid, "ranking_score"] = score
    return out


def select_top_configurations(
    grid_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    top_n = int(cfg["selection"]["top_n_per_algorithm_per_feature_set"])
    max_cluster_ratio = float(cfg["selection"].get("max_cluster_ratio", 0.90))

    ranked = add_ranking_score(grid_df)
    eligible = ranked[
        ranked["ranking_score"].notna()
        & (pd.to_numeric(ranked["n_clusters_actual"], errors="coerce") >= 2)
        & (pd.to_numeric(ranked["cluster_size_ratio_max"], errors="coerce") <= max_cluster_ratio)
        & (pd.to_numeric(ranked["n_small_clusters"], errors="coerce").fillna(999) <= 2)
    ].copy()
    if eligible.empty:
        eligible = ranked[ranked["ranking_score"].notna()].copy()

    chunks = []
    for (feature_set, algorithm), sdf in eligible.groupby(["feature_set", "algorithm"], sort=True):
        top = sdf.sort_values(["ranking_score", "silhouette_score"], ascending=[False, False], kind="stable").head(top_n)
        chunks.append(top)
    if not chunks:
        return pd.DataFrame(columns=ranked.columns.tolist())
    out = pd.concat(chunks, axis=0, ignore_index=True)
    return out.sort_values(["feature_set", "algorithm", "ranking_score"], ascending=[True, True, False], kind="stable").reset_index(drop=True)

