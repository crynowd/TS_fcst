from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_blobs

from src.cli.run_clustering_experiments import run_clustering_experiments_pipeline
from src.clustering.grid_search import run_grid_search
from src.clustering.preprocessing import apply_scaler
from src.clustering.selection import add_ranking_score, select_top_configurations
from src.clustering.stability import build_top_config_assignments, run_bootstrap_stability


def _make_feature_tables(n_samples: int = 90) -> dict[str, pd.DataFrame]:
    x, _ = make_blobs(n_samples=n_samples, centers=3, n_features=4, random_state=42, cluster_std=1.0)
    meta = pd.DataFrame(
        {
            "series_id": [f"S{i:04d}" for i in range(n_samples)],
            "ticker": [f"T{i:04d}" for i in range(n_samples)],
            "market": ["RU"] * (n_samples // 2) + ["US"] * (n_samples - n_samples // 2),
            "dataset_profile": ["core_balanced"] * n_samples,
        }
    )
    base = pd.concat([meta, pd.DataFrame(x, columns=["f1", "f2", "f3", "f4"])], axis=1)
    with_chaos = base.copy()
    with_chaos["d1"] = with_chaos["f1"] * 0.5 + with_chaos["f2"] * 0.1
    with_chaos.loc[::10, "d1"] = np.nan
    return {"base": base, "with_chaos": with_chaos}


def _cfg_small() -> dict:
    return {
        "feature_sets": ["base", "with_chaos"],
        "scalers": ["identity"],
        "spaces": ["original"],
        "pca_n_components": [2],
        "algorithms": ["gmm", "agglomerative"],
        "cluster_range": {"k_min": 2, "k_max": 3},
        "small_cluster_threshold": {"mode": "relative", "value": 0.05},
        "selection": {"top_n_per_algorithm_per_feature_set": 1, "max_cluster_ratio": 0.95},
        "stability": {"n_bootstrap": 5, "sample_fraction": 0.8, "random_state": 42},
    }


def test_identity_scaler_pass_through() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = apply_scaler(x, scaler="identity", random_state=42)
    assert np.allclose(out, x)


def test_grid_results_shape_nonempty() -> None:
    tables = _make_feature_tables()
    cfg = _cfg_small()
    cfg["feature_sets"] = ["base"]
    cfg["algorithms"] = ["gmm"]
    grid = run_grid_search(tables, cfg)
    assert len(grid) > 0
    assert {"config_id", "silhouette_score", "algorithm", "feature_set"}.issubset(grid.columns)


def test_top_config_selection_returns_expected_count() -> None:
    tables = _make_feature_tables()
    cfg = _cfg_small()
    grid = run_grid_search(tables, cfg)
    ranked = add_ranking_score(grid)
    top = select_top_configurations(ranked, cfg)
    # 2 feature sets x 2 algorithms x top_n(1)
    assert len(top) == 4


def test_cluster_assignments_written_for_top_configs() -> None:
    tables = _make_feature_tables()
    cfg = _cfg_small()
    top = select_top_configurations(add_ranking_score(run_grid_search(tables, cfg)), cfg)
    assignments, _ = build_top_config_assignments(top, tables, random_state=42)
    assert not assignments.empty
    assert {"config_id", "series_id", "cluster_label"}.issubset(assignments.columns)
    for _, row in top.iterrows():
        fs = str(row["feature_set"])
        expected = len(tables[fs])
        got = int((assignments["config_id"] == row["config_id"]).sum())
        assert got == expected


def test_stability_metrics_in_valid_range() -> None:
    tables = _make_feature_tables()
    cfg = _cfg_small()
    top = select_top_configurations(add_ranking_score(run_grid_search(tables, cfg)), cfg)
    _, ref = build_top_config_assignments(top, tables, random_state=42)
    st = run_bootstrap_stability(top, tables, ref, cfg)
    assert not st.empty
    assert st["ari_mean"].between(-1, 1).all()
    assert st["nmi_mean"].between(0, 1).all()


def test_clustering_experiment_stage_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    artifacts = root / "artifacts"
    features_dir = artifacts / "features"
    reports_dir = artifacts / "reports"
    logs_dir = artifacts / "logs"
    manifests_dir = artifacts / "manifests"
    clustering_dir = artifacts / "clustering"
    for d in [configs_dir, features_dir, reports_dir, logs_dir, manifests_dir, clustering_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tables = _make_feature_tables()
    tables["base"].to_parquet(features_dir / "final_clustering_features_base_v1.parquet", index=False)
    tables["with_chaos"].to_parquet(features_dir / "final_clustering_features_with_chaos_v1.parquet", index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "interim": str(artifacts / "interim"),
            "processed": str(artifacts / "processed"),
            "features": str(features_dir),
            "reports": str(reports_dir),
            "logs": str(logs_dir),
            "manifests": str(manifests_dir),
            "clustering": str(clustering_dir),
        },
    }
    (configs_dir / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    stage_cfg = {
        "run_name": "clustering_experiments_v1",
        "stage": "clustering_experiments",
        "input": {
            "final_base_parquet": "artifacts/features/final_clustering_features_base_v1.parquet",
            "final_with_chaos_parquet": "artifacts/features/final_clustering_features_with_chaos_v1.parquet",
        },
        "feature_sets": ["base", "with_chaos"],
        "scalers": ["identity"],
        "spaces": ["original"],
        "pca_n_components": [2],
        "algorithms": ["gmm", "agglomerative"],
        "cluster_range": {"k_min": 2, "k_max": 3},
        "imputation": {"strategy": "median"},
        "small_cluster_threshold": {"mode": "relative", "value": 0.05},
        "selection": {"top_n_per_algorithm_per_feature_set": 1, "max_cluster_ratio": 0.95},
        "stability": {"n_bootstrap": 4, "sample_fraction": 0.8, "random_state": 42},
        "output": {
            "grid_results_parquet_name": "clustering_grid_results_v1.parquet",
            "top_configs_parquet_name": "top_clustering_configs_v1.parquet",
            "stability_parquet_name": "clustering_stability_results_v1.parquet",
            "assignments_parquet_name": "cluster_assignments_v1.parquet",
            "excel_name": "clustering_experiments_v1.xlsx",
            "grid_results_csv_name": "clustering_grid_results_v1.csv",
            "top_configs_csv_name": "top_clustering_configs_v1.csv",
            "stability_csv_name": "clustering_stability_results_v1.csv",
        },
    }
    cfg_path = configs_dir / "clustering_experiments_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_clustering_experiments_pipeline(str(cfg_path))
    assert Path(result["grid_parquet_path"]).exists()
    assert Path(result["top_parquet_path"]).exists()
    assert Path(result["stability_parquet_path"]).exists()
    assert Path(result["assignments_parquet_path"]).exists()
    assert Path(result["excel_path"]).exists()
