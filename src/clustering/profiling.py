from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from src.clustering.grid_search import fit_configuration
from src.clustering.preprocessing import METADATA_COLUMNS, split_metadata_and_features


def select_configurations(
    grid_df: pd.DataFrame,
    k_min: int,
    k_max: int,
    max_cluster_ratio: float,
    min_cluster_size: int,
) -> pd.DataFrame:
    """Select overall best and per-K best configs for base/with_chaos."""
    ranked = grid_df.copy()
    eligible = ranked[
        ranked["ranking_score"].notna()
        & (pd.to_numeric(ranked["cluster_size_ratio_max"], errors="coerce") <= float(max_cluster_ratio))
        & (pd.to_numeric(ranked["cluster_size_min"], errors="coerce") >= int(min_cluster_size))
    ].copy()

    selections: List[Dict[str, Any]] = []
    if not eligible.empty:
        best_overall = eligible.sort_values(["ranking_score", "silhouette_score"], ascending=[False, False], kind="stable").iloc[0]
        selections.append({"selection_scope": "overall", "k_bucket": "overall", **best_overall.to_dict()})

    for feature_set in ["base", "with_chaos"]:
        for k in range(int(k_min), int(k_max) + 1):
            sdf = eligible[
                (eligible["feature_set"] == feature_set)
                & (pd.to_numeric(eligible["n_clusters_requested"], errors="coerce") == int(k))
            ].copy()
            if sdf.empty:
                continue
            best = sdf.sort_values(["ranking_score", "silhouette_score"], ascending=[False, False], kind="stable").iloc[0]
            selections.append({"selection_scope": "best_k", "k_bucket": int(k), **best.to_dict()})

    if not selections:
        return pd.DataFrame(columns=["selection_scope", "k_bucket", *grid_df.columns.tolist()])
    out = pd.DataFrame(selections)
    return out.reset_index(drop=True)


def _build_assignments_from_fit(
    config_row: Mapping[str, Any],
    feature_table: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    fit = fit_configuration(
        feature_table=feature_table,
        scaler=str(config_row["scaler"]),
        space_type=str(config_row["space_type"]),
        pca_n_components=(
            None if pd.isna(config_row["pca_n_components"]) else int(config_row["pca_n_components"])
        ),
        algorithm=str(config_row["algorithm"]),
        n_clusters_requested=int(config_row["n_clusters_requested"]),
        random_state=int(random_state),
    )
    meta_cols = [c for c in METADATA_COLUMNS if c in feature_table.columns]
    rows = []
    for i in range(len(feature_table)):
        payload = {c: feature_table.iloc[i][c] for c in meta_cols}
        rows.append(
            {
                "config_id": str(config_row["config_id"]),
                "feature_set": str(config_row["feature_set"]),
                "scaler": str(config_row["scaler"]),
                "space_type": str(config_row["space_type"]),
                "algorithm": str(config_row["algorithm"]),
                "n_clusters": int(config_row["n_clusters_requested"]),
                **payload,
                "cluster_label": int(fit.labels[i]),
                "assignment_confidence": (
                    float(fit.assignment_confidence[i]) if fit.assignment_confidence is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_selected_assignments(
    selected_configs_df: pd.DataFrame,
    precomputed_assignments_df: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    random_state: int = 42,
) -> pd.DataFrame:
    """Get assignments for selected configs, recomputing missing ones if needed."""
    selected_unique = selected_configs_df.drop_duplicates(subset=["config_id"]).copy()
    rows: List[pd.DataFrame] = []
    for _, cfg in selected_unique.iterrows():
        cfg_id = str(cfg["config_id"])
        feature_set = str(cfg["feature_set"])
        expected_n = int(len(feature_tables[feature_set]))
        existing = precomputed_assignments_df[precomputed_assignments_df["config_id"] == cfg_id].copy()
        if not existing.empty and len(existing) == expected_n:
            rows.append(existing.reset_index(drop=True))
            continue
        rebuilt = _build_assignments_from_fit(cfg, feature_tables[feature_set].reset_index(drop=True), random_state=random_state)
        rows.append(rebuilt)
    if not rows:
        return pd.DataFrame(columns=precomputed_assignments_df.columns.tolist())
    return pd.concat(rows, axis=0, ignore_index=True)


def compute_cluster_balance(assignments_df: pd.DataFrame, selected_configs_df: pd.DataFrame) -> pd.DataFrame:
    cfg_cols = [
        "config_id",
        "feature_set",
        "scaler",
        "space_type",
        "pca_n_components",
        "algorithm",
        "n_clusters_requested",
        "ranking_score",
    ]
    cfg_ref = selected_configs_df[cfg_cols].drop_duplicates(subset=["config_id"]).copy()
    grouped = (
        assignments_df.groupby(["config_id", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "cluster_size"})
    )
    totals = grouped.groupby("config_id", as_index=False)["cluster_size"].sum().rename(columns={"cluster_size": "n_rows"})
    out = grouped.merge(totals, on="config_id", how="left")
    out["cluster_share"] = out["cluster_size"] / out["n_rows"]
    out["cluster_size_rank"] = out.groupby("config_id")["cluster_size"].rank(method="dense", ascending=False).astype(int)
    out = out.merge(cfg_ref, on="config_id", how="left")
    cols = [
        "config_id",
        "feature_set",
        "scaler",
        "space_type",
        "pca_n_components",
        "algorithm",
        "n_clusters_requested",
        "cluster_label",
        "cluster_size",
        "cluster_share",
        "cluster_size_rank",
        "ranking_score",
    ]
    return out[cols].sort_values(["feature_set", "config_id", "cluster_size_rank", "cluster_label"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class ProfileBuildResult:
    profiles: pd.DataFrame
    joined_table: pd.DataFrame


def build_cluster_profiles(
    selected_configs_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
) -> ProfileBuildResult:
    """Build cluster feature profiles with mean/median/std/z-score."""
    config_ref = selected_configs_df.drop_duplicates(subset=["config_id"]).copy()
    profile_rows: List[Dict[str, Any]] = []
    joined_rows: List[pd.DataFrame] = []

    for _, cfg in config_ref.iterrows():
        cfg_id = str(cfg["config_id"])
        feature_set = str(cfg["feature_set"])
        table = feature_tables[feature_set].copy()
        cfg_assign = assignments_df[assignments_df["config_id"] == cfg_id][["series_id", "cluster_label"]].copy()
        merged = table.merge(cfg_assign, on="series_id", how="inner")
        joined_rows.append(merged.assign(config_id=cfg_id))

        _, feature_df = split_metadata_and_features(table)
        feature_names = feature_df.columns.tolist()
        global_mean = feature_df.mean(axis=0, numeric_only=True)
        global_std = feature_df.std(axis=0, ddof=1, numeric_only=True)

        for cluster_label, cdf in merged.groupby("cluster_label", sort=True):
            cluster_features = cdf[feature_names].apply(pd.to_numeric, errors="coerce")
            c_mean = cluster_features.mean(axis=0, numeric_only=True)
            c_median = cluster_features.median(axis=0, numeric_only=True)
            c_std = cluster_features.std(axis=0, ddof=1, numeric_only=True)
            for feature_name in feature_names:
                g_mu = float(global_mean.get(feature_name, np.nan))
                g_sigma = float(global_std.get(feature_name, np.nan))
                mu = float(c_mean.get(feature_name, np.nan))
                med = float(c_median.get(feature_name, np.nan))
                sd = float(c_std.get(feature_name, np.nan))
                if np.isfinite(g_sigma) and g_sigma > 0 and np.isfinite(mu) and np.isfinite(g_mu):
                    z = (mu - g_mu) / g_sigma
                else:
                    z = 0.0
                profile_rows.append(
                    {
                        "config_id": cfg_id,
                        "feature_set": feature_set,
                        "scaler": str(cfg["scaler"]),
                        "space_type": str(cfg["space_type"]),
                        "pca_n_components": cfg["pca_n_components"],
                        "algorithm": str(cfg["algorithm"]),
                        "n_clusters_requested": int(cfg["n_clusters_requested"]),
                        "cluster_label": int(cluster_label),
                        "feature_name": feature_name,
                        "cluster_mean": mu,
                        "cluster_median": med,
                        "cluster_std": sd,
                        "global_mean": g_mu,
                        "global_std": g_sigma,
                        "z_score": float(z),
                        "abs_z_score": float(abs(z)),
                    }
                )

    profiles_df = pd.DataFrame(profile_rows)
    joined_df = pd.concat(joined_rows, axis=0, ignore_index=True) if joined_rows else pd.DataFrame()
    return ProfileBuildResult(profiles=profiles_df, joined_table=joined_df)


def select_key_features(cluster_profiles_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    sort_cols = ["abs_z_score", "feature_name"]
    for (_, _), sdf in cluster_profiles_df.groupby(["config_id", "cluster_label"], sort=True):
        top = sdf.sort_values(sort_cols, ascending=[False, True], kind="stable").head(int(top_n)).copy()
        top["rank_within_cluster"] = np.arange(1, len(top) + 1)
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=cluster_profiles_df.columns.tolist() + ["rank_within_cluster"])
    cols = [
        "config_id",
        "feature_set",
        "algorithm",
        "scaler",
        "space_type",
        "n_clusters_requested",
        "cluster_label",
        "rank_within_cluster",
        "feature_name",
        "z_score",
        "abs_z_score",
        "cluster_mean",
        "global_mean",
    ]
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out[cols].sort_values(["feature_set", "config_id", "cluster_label", "rank_within_cluster"], kind="stable").reset_index(drop=True)


def build_method_comparison(
    selected_configs_df: pd.DataFrame,
    cluster_balance_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare base vs with_chaos on selected configurations."""
    balance_agg = (
        cluster_balance_df.groupby("config_id", as_index=False)
        .agg(
            cluster_count=("cluster_label", "nunique"),
            cluster_share_min=("cluster_share", "min"),
            cluster_share_max=("cluster_share", "max"),
            cluster_share_std=("cluster_share", "std"),
        )
    )
    stability_ref = stability_df[["config_id", "ari_mean", "nmi_mean"]].copy() if not stability_df.empty else pd.DataFrame(columns=["config_id", "ari_mean", "nmi_mean"])

    ref = selected_configs_df.merge(balance_agg, on="config_id", how="left").merge(stability_ref, on="config_id", how="left")
    cmp_rows: List[Dict[str, Any]] = []

    scope_keys: Iterable[tuple[str, Any]]
    scope_keys = list(ref[["selection_scope", "k_bucket"]].drop_duplicates().itertuples(index=False, name=None))
    for selection_scope, k_bucket in scope_keys:
        base_row = ref[
            (ref["selection_scope"] == selection_scope)
            & (ref["k_bucket"] == k_bucket)
            & (ref["feature_set"] == "base")
        ]
        chaos_row = ref[
            (ref["selection_scope"] == selection_scope)
            & (ref["k_bucket"] == k_bucket)
            & (ref["feature_set"] == "with_chaos")
        ]
        b = base_row.iloc[0] if not base_row.empty else None
        c = chaos_row.iloc[0] if not chaos_row.empty else None
        cmp_rows.append(
            {
                "selection_scope": selection_scope,
                "k_bucket": k_bucket,
                "base_config_id": None if b is None else str(b["config_id"]),
                "with_chaos_config_id": None if c is None else str(c["config_id"]),
                "base_n_clusters": None if b is None else int(b["n_clusters_requested"]),
                "with_chaos_n_clusters": None if c is None else int(c["n_clusters_requested"]),
                "base_cluster_share_max": None if b is None else float(b["cluster_share_max"]),
                "with_chaos_cluster_share_max": None if c is None else float(c["cluster_share_max"]),
                "base_cluster_share_min": None if b is None else float(b["cluster_share_min"]),
                "with_chaos_cluster_share_min": None if c is None else float(c["cluster_share_min"]),
                "base_silhouette": None if b is None else float(b["silhouette_score"]),
                "with_chaos_silhouette": None if c is None else float(c["silhouette_score"]),
                "base_stability_ari_mean": None if b is None or pd.isna(b["ari_mean"]) else float(b["ari_mean"]),
                "with_chaos_stability_ari_mean": None if c is None or pd.isna(c["ari_mean"]) else float(c["ari_mean"]),
                "base_stability_nmi_mean": None if b is None or pd.isna(b["nmi_mean"]) else float(b["nmi_mean"]),
                "with_chaos_stability_nmi_mean": None if c is None or pd.isna(c["nmi_mean"]) else float(c["nmi_mean"]),
                "delta_silhouette_with_chaos_minus_base": (
                    None
                    if b is None or c is None
                    else float(c["silhouette_score"]) - float(b["silhouette_score"])
                ),
                "delta_share_max_with_chaos_minus_base": (
                    None
                    if b is None or c is None
                    else float(c["cluster_share_max"]) - float(b["cluster_share_max"])
                ),
                "delta_stability_ari_with_chaos_minus_base": (
                    None
                    if b is None or c is None or pd.isna(b["ari_mean"]) or pd.isna(c["ari_mean"])
                    else float(c["ari_mean"]) - float(b["ari_mean"])
                ),
            }
        )
    return pd.DataFrame(cmp_rows).sort_values(["selection_scope", "k_bucket"], kind="stable").reset_index(drop=True)
