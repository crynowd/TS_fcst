from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.clustering.preprocessing import preprocess_feature_table, split_metadata_and_features

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def _safe_slug(text: str) -> str:
    cleaned = []
    for ch in str(text):
        cleaned.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(cleaned)


def _config_slug(cfg: Mapping[str, Any]) -> str:
    pca_val = "none" if pd.isna(cfg["pca_n_components"]) else str(int(cfg["pca_n_components"]))
    return _safe_slug(
        f"{cfg['config_id']}_{cfg['feature_set']}_{cfg['algorithm']}_{cfg['scaler']}_{cfg['space_type']}_pca{pca_val}_k{int(cfg['n_clusters_requested'])}"
    )


def _resolve_labels_for_table(table: pd.DataFrame, cfg_assignments: pd.DataFrame) -> np.ndarray:
    ref = cfg_assignments.set_index("series_id")["cluster_label"]
    labels = table["series_id"].map(ref)
    if labels.isna().any():
        raise ValueError("Missing cluster labels for some series_id during visualization.")
    return labels.to_numpy(dtype=int, copy=True)


def _make_2d_coordinates(table: pd.DataFrame, cfg: Mapping[str, Any], random_state: int) -> np.ndarray:
    _, feature_df = split_metadata_and_features(table)
    prep = preprocess_feature_table(
        feature_df=feature_df,
        scaler=str(cfg["scaler"]),
        space_type=str(cfg["space_type"]),
        pca_n_components=None if pd.isna(cfg["pca_n_components"]) else int(cfg["pca_n_components"]),
        random_state=int(random_state),
    )
    if prep.matrix.shape[1] >= 2:
        return prep.matrix[:, :2]
    pca = PCA(n_components=2, random_state=int(random_state))
    return pca.fit_transform(prep.matrix)


def plot_pca_scatter_for_configs(
    selected_configs_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    dpi: int = 160,
    random_state: int = 42,
) -> List[Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for _, cfg in selected_configs_df.drop_duplicates(subset=["config_id"]).iterrows():
        cfg_id = str(cfg["config_id"])
        feature_set = str(cfg["feature_set"])
        table = feature_tables[feature_set].copy()
        cfg_assign = assignments_df[assignments_df["config_id"] == cfg_id].copy()
        labels = _resolve_labels_for_table(table, cfg_assign)
        coords = _make_2d_coordinates(table, cfg, random_state=int(random_state))

        fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.85,
                c=[cmap(i)],
                label=f"cluster {int(label)}",
                edgecolor="none",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"{feature_set} | {cfg['algorithm']} | {cfg['scaler']} | K={int(cfg['n_clusters_requested'])}"
        )
        ax.legend(loc="best", fontsize=8, frameon=False)

        slug = _config_slug(cfg)
        png_path = out_dir / f"cluster_scatter_{slug}.png"
        svg_path = out_dir / f"cluster_scatter_{slug}.svg"
        fig.savefig(png_path, dpi=int(dpi))
        fig.savefig(svg_path)
        plt.close(fig)
        saved_paths.extend([png_path, svg_path])

    return saved_paths


def plot_cluster_profile_heatmaps(
    cluster_profiles_df: pd.DataFrame,
    selected_configs_df: pd.DataFrame,
    output_dir: str | Path,
    cmap: str = "RdBu_r",
    dpi: int = 160,
) -> List[Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for _, cfg in selected_configs_df.drop_duplicates(subset=["config_id"]).iterrows():
        cfg_id = str(cfg["config_id"])
        sdf = cluster_profiles_df[cluster_profiles_df["config_id"] == cfg_id].copy()
        if sdf.empty:
            continue
        pivot = sdf.pivot(index="cluster_label", columns="feature_name", values="z_score")
        if pivot.empty:
            continue
        pivot = pivot.sort_index(axis=0)
        order = pivot.abs().max(axis=0).sort_values(ascending=False).index
        pivot = pivot[order]

        fig_w = max(10.0, min(24.0, 0.32 * len(pivot.columns)))
        fig_h = max(4.0, 1.0 + 0.75 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        if sns is not None:
            sns.heatmap(pivot, cmap=cmap, center=0.0, linewidths=0.25, linecolor="white", ax=ax, cbar=True)
        else:  # pragma: no cover
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=-3, vmax=3)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(str))

        ax.set_xlabel("feature")
        ax.set_ylabel("cluster_label")
        ax.set_title(
            f"Cluster profile z-score heatmap | {cfg['feature_set']} | {cfg['algorithm']} | K={int(cfg['n_clusters_requested'])}"
        )

        png_path = out_dir / f"cluster_profile_heatmap_{_config_slug(cfg)}.png"
        fig.savefig(png_path, dpi=int(dpi))
        plt.close(fig)
        saved_paths.append(png_path)
    return saved_paths
