from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.profiling import (
    build_cluster_profiles,
    compute_cluster_balance,
    select_configurations,
    select_key_features,
)
from src.clustering.visualization import plot_cluster_profile_heatmaps, plot_pca_scatter_for_configs


def _make_feature_tables() -> dict[str, pd.DataFrame]:
    n = 12
    meta = pd.DataFrame(
        {
            "series_id": [f"S{i:03d}" for i in range(n)],
            "ticker": [f"T{i:03d}" for i in range(n)],
            "market": ["RU"] * 6 + ["US"] * 6,
            "dataset_profile": ["core_balanced"] * n,
        }
    )
    rng = np.random.default_rng(42)
    base = meta.copy()
    base["f1"] = rng.normal(0, 1, n)
    base["f2"] = rng.normal(2, 1, n)
    with_chaos = base.copy()
    with_chaos["c1"] = rng.normal(-1, 0.5, n)
    return {"base": base, "with_chaos": with_chaos}


def _make_grid() -> pd.DataFrame:
    rows = [
        {
            "config_id": "cfg_base_k2",
            "feature_set": "base",
            "scaler": "identity",
            "space_type": "original",
            "pca_n_components": np.nan,
            "algorithm": "gmm",
            "n_clusters_requested": 2,
            "n_clusters_actual": 2,
            "silhouette_score": 0.22,
            "davies_bouldin_score": 1.1,
            "calinski_harabasz_score": 30.0,
            "cluster_size_min": 5,
            "cluster_size_max": 7,
            "cluster_size_ratio_max": 0.58,
            "n_singleton_clusters": 0,
            "n_small_clusters": 0,
            "n_rows": 12,
            "n_features_input": 2,
            "n_features_used": 2,
            "n_missing_imputed": 0,
            "explained_variance_ratio_sum": np.nan,
            "avg_assignment_confidence": 0.8,
            "min_assignment_confidence": 0.6,
            "entropy_of_assignment": 0.2,
            "ranking_score": 0.4,
            "notes": "",
        },
        {
            "config_id": "cfg_chaos_k2",
            "feature_set": "with_chaos",
            "scaler": "identity",
            "space_type": "original",
            "pca_n_components": np.nan,
            "algorithm": "gmm",
            "n_clusters_requested": 2,
            "n_clusters_actual": 2,
            "silhouette_score": 0.3,
            "davies_bouldin_score": 0.9,
            "calinski_harabasz_score": 33.0,
            "cluster_size_min": 5,
            "cluster_size_max": 7,
            "cluster_size_ratio_max": 0.58,
            "n_singleton_clusters": 0,
            "n_small_clusters": 0,
            "n_rows": 12,
            "n_features_input": 3,
            "n_features_used": 3,
            "n_missing_imputed": 0,
            "explained_variance_ratio_sum": np.nan,
            "avg_assignment_confidence": 0.83,
            "min_assignment_confidence": 0.61,
            "entropy_of_assignment": 0.19,
            "ranking_score": 0.5,
            "notes": "",
        },
    ]
    return pd.DataFrame(rows)


def _make_assignments(config_id: str, feature_set: str, n_clusters: int = 2, n_rows: int = 12) -> pd.DataFrame:
    labels = np.array([i % n_clusters for i in range(n_rows)], dtype=int)
    return pd.DataFrame(
        {
            "config_id": [config_id] * n_rows,
            "feature_set": [feature_set] * n_rows,
            "scaler": ["identity"] * n_rows,
            "space_type": ["original"] * n_rows,
            "algorithm": ["gmm"] * n_rows,
            "n_clusters": [n_clusters] * n_rows,
            "series_id": [f"S{i:03d}" for i in range(n_rows)],
            "ticker": [f"T{i:03d}" for i in range(n_rows)],
            "market": ["RU"] * 6 + ["US"] * 6,
            "dataset_profile": ["core_balanced"] * n_rows,
            "cluster_label": labels,
            "assignment_confidence": [0.8] * n_rows,
        }
    )


def test_cluster_balance_computation() -> None:
    selected = select_configurations(_make_grid(), k_min=2, k_max=2, max_cluster_ratio=0.9, min_cluster_size=5)
    assignments = pd.concat(
        [
            _make_assignments("cfg_base_k2", "base"),
            _make_assignments("cfg_chaos_k2", "with_chaos"),
        ],
        ignore_index=True,
    )
    balance = compute_cluster_balance(assignments, selected)
    assert not balance.empty
    assert {"cluster_size", "cluster_share", "cluster_size_rank"}.issubset(balance.columns)
    assert balance.groupby("config_id")["cluster_size"].sum().eq(12).all()


def test_profile_table_shape() -> None:
    tables = _make_feature_tables()
    selected = select_configurations(_make_grid(), k_min=2, k_max=2, max_cluster_ratio=0.9, min_cluster_size=5)
    assignments = pd.concat(
        [
            _make_assignments("cfg_base_k2", "base"),
            _make_assignments("cfg_chaos_k2", "with_chaos"),
        ],
        ignore_index=True,
    )
    profiles = build_cluster_profiles(selected, assignments, tables).profiles
    assert not profiles.empty
    expected_base = 2 * 2  # 2 clusters x 2 base features
    expected_chaos = 2 * 3  # 2 clusters x 3 with_chaos features
    assert len(profiles[profiles["config_id"] == "cfg_base_k2"]) == expected_base
    assert len(profiles[profiles["config_id"] == "cfg_chaos_k2"]) == expected_chaos


def test_key_feature_selection() -> None:
    tables = _make_feature_tables()
    selected = select_configurations(_make_grid(), k_min=2, k_max=2, max_cluster_ratio=0.9, min_cluster_size=5)
    assignments = pd.concat(
        [
            _make_assignments("cfg_base_k2", "base"),
            _make_assignments("cfg_chaos_k2", "with_chaos"),
        ],
        ignore_index=True,
    )
    profiles = build_cluster_profiles(selected, assignments, tables).profiles
    key_df = select_key_features(profiles, top_n=2)
    assert not key_df.empty
    assert key_df.groupby(["config_id", "cluster_label"]).size().eq(2).all()


def test_visualization_runs_without_error(tmp_path: Path) -> None:
    tables = _make_feature_tables()
    selected = select_configurations(_make_grid(), k_min=2, k_max=2, max_cluster_ratio=0.9, min_cluster_size=5)
    assignments = pd.concat(
        [
            _make_assignments("cfg_base_k2", "base"),
            _make_assignments("cfg_chaos_k2", "with_chaos"),
        ],
        ignore_index=True,
    )
    profiles = build_cluster_profiles(selected, assignments, tables).profiles

    out_dir = tmp_path / "figures"
    scatter = plot_pca_scatter_for_configs(selected, assignments, tables, output_dir=out_dir, dpi=120, random_state=42)
    heatmaps = plot_cluster_profile_heatmaps(profiles, selected, output_dir=out_dir, dpi=120)
    assert len(scatter) >= 2
    assert len(heatmaps) >= 1
    for p in [*scatter, *heatmaps]:
        assert p.exists()
