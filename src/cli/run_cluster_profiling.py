from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.clustering.profiling import (
    build_cluster_profiles,
    build_method_comparison,
    build_selected_assignments,
    compute_cluster_balance,
    select_configurations,
    select_key_features,
)
from src.clustering.visualization import plot_cluster_profile_heatmaps, plot_pca_scatter_for_configs
from src.config.loader import load_cluster_profiling_config
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst cluster profiling and visualization stage")
    parser.add_argument(
        "--config",
        default="configs/cluster_profiling_v1.yaml",
        help="Path to cluster profiling config",
    )
    return parser.parse_args()


def _build_summary_sheet(
    selected_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    scatter_count: int,
    heatmap_count: int,
) -> pd.DataFrame:
    rows = [
        {"section": "overall", "label": "selected_configs_rows", "value": int(len(selected_df))},
        {"section": "overall", "label": "selected_configs_unique", "value": int(selected_df["config_id"].nunique())},
        {"section": "overall", "label": "scatter_plots", "value": int(scatter_count)},
        {"section": "overall", "label": "heatmaps", "value": int(heatmap_count)},
    ]
    for feature_set, n in selected_df["feature_set"].value_counts().sort_index().to_dict().items():
        rows.append({"section": "selected_by_feature_set", "label": str(feature_set), "value": int(n)})
    if not balance_df.empty:
        rows.append({"section": "cluster_balance", "label": "cluster_share_min", "value": float(balance_df["cluster_share"].min())})
        rows.append({"section": "cluster_balance", "label": "cluster_share_max", "value": float(balance_df["cluster_share"].max())})
    if not comparison_df.empty and "delta_silhouette_with_chaos_minus_base" in comparison_df.columns:
        valid_delta = pd.to_numeric(comparison_df["delta_silhouette_with_chaos_minus_base"], errors="coerce").dropna()
        rows.append(
            {
                "section": "method_comparison",
                "label": "delta_silhouette_with_chaos_minus_base_mean",
                "value": float(valid_delta.mean()) if not valid_delta.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _write_excel_report(
    excel_path: str | Path,
    summary_df: pd.DataFrame,
    selected_configs_df: pd.DataFrame,
    cluster_balance_df: pd.DataFrame,
    cluster_profiles_df: pd.DataFrame,
    cluster_key_features_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    readme_df: pd.DataFrame,
) -> Path:
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        selected_configs_df.to_excel(writer, sheet_name="selected_configs", index=False)
        cluster_balance_df.to_excel(writer, sheet_name="cluster_balance", index=False)
        cluster_profiles_df.to_excel(writer, sheet_name="cluster_profiles", index=False)
        cluster_key_features_df.to_excel(writer, sheet_name="cluster_key_features", index=False)
        comparison_df.to_excel(writer, sheet_name="method_comparison", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)
    return out_path


def run_cluster_profiling_pipeline(config_path: str = "configs/cluster_profiling_v1.yaml") -> Dict[str, Any]:
    cfg = load_cluster_profiling_config(config_path)

    run_name = str(cfg.get("run_name", "cluster_profiling_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts["logs"])
    logger.info("Cluster profiling started")
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    inputs = cfg["input"]
    grid_df = pd.read_parquet(inputs["grid_results_parquet"])
    precomputed_assignments_df = pd.read_parquet(inputs["assignments_parquet"])
    stability_df = pd.read_parquet(inputs["stability_parquet"])
    feature_tables = {
        "base": pd.read_parquet(inputs["final_base_parquet"]),
        "with_chaos": pd.read_parquet(inputs["final_with_chaos_parquet"]),
    }

    selection_cfg = cfg.get("selection", {})
    selected_df = select_configurations(
        grid_df=grid_df,
        k_min=int(selection_cfg.get("k_min", 2)),
        k_max=int(selection_cfg.get("k_max", 8)),
        max_cluster_ratio=float(selection_cfg.get("max_cluster_ratio", 0.90)),
        min_cluster_size=int(selection_cfg.get("min_cluster_size", 5)),
    )
    logger.info("Selected configurations: rows=%d unique=%d", len(selected_df), selected_df["config_id"].nunique())

    selected_assignments_df = build_selected_assignments(
        selected_configs_df=selected_df,
        precomputed_assignments_df=precomputed_assignments_df,
        feature_tables=feature_tables,
        random_state=int(selection_cfg.get("random_state", 42)),
    )
    balance_df = compute_cluster_balance(assignments_df=selected_assignments_df, selected_configs_df=selected_df)
    profile_result = build_cluster_profiles(
        selected_configs_df=selected_df,
        assignments_df=selected_assignments_df,
        feature_tables=feature_tables,
    )
    cluster_profiles_df = profile_result.profiles
    key_features_df = select_key_features(
        cluster_profiles_df=cluster_profiles_df,
        top_n=int(selection_cfg.get("key_features_top_n", 5)),
    )
    method_comparison_df = build_method_comparison(
        selected_configs_df=selected_df,
        cluster_balance_df=balance_df,
        stability_df=stability_df,
    )

    figures_dir = Path(artifacts.get("figures", Path(artifacts["reports"]).parent / "figures")).resolve() / "clusters"
    vis_cfg = cfg.get("visualization", {})
    scatter_paths = plot_pca_scatter_for_configs(
        selected_configs_df=selected_df,
        assignments_df=selected_assignments_df,
        feature_tables=feature_tables,
        output_dir=figures_dir,
        dpi=int(vis_cfg.get("dpi", 160)),
        random_state=int(selection_cfg.get("random_state", 42)),
    )
    heatmap_paths = plot_cluster_profile_heatmaps(
        cluster_profiles_df=cluster_profiles_df,
        selected_configs_df=selected_df,
        output_dir=figures_dir,
        cmap=str(vis_cfg.get("heatmap_cmap", "RdBu_r")),
        dpi=int(vis_cfg.get("dpi", 160)),
    )

    clustering_dir = Path(artifacts["clustering"]).resolve()
    reports_dir = Path(artifacts["reports"]).resolve()
    manifests_dir = Path(artifacts["manifests"]).resolve()
    for d in [clustering_dir, reports_dir, manifests_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    output = cfg["output"]
    selected_csv_path = clustering_dir / str(output["selected_configs_csv_name"])
    balance_csv_path = clustering_dir / str(output["cluster_balance_csv_name"])
    profiles_parquet_path = clustering_dir / str(output["cluster_profiles_parquet_name"])
    profiles_csv_path = clustering_dir / str(output["cluster_profiles_csv_name"])
    key_features_csv_path = clustering_dir / str(output["cluster_key_features_csv_name"])
    comparison_csv_path = reports_dir / str(output["method_comparison_csv_name"])
    excel_path = reports_dir / str(output["excel_name"])

    selected_df.to_csv(selected_csv_path, index=False)
    balance_df.to_csv(balance_csv_path, index=False)
    cluster_profiles_df.to_parquet(profiles_parquet_path, index=False)
    cluster_profiles_df.to_csv(profiles_csv_path, index=False)
    key_features_df.to_csv(key_features_csv_path, index=False)
    method_comparison_df.to_csv(comparison_csv_path, index=False)

    summary_df = _build_summary_sheet(
        selected_df=selected_df,
        balance_df=balance_df,
        comparison_df=method_comparison_df,
        scatter_count=len(scatter_paths),
        heatmap_count=len(heatmap_paths),
    )
    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": "cluster_profiling"},
            {"key": "config_path", "value": cfg["meta"]["config_path"]},
            {"key": "inputs", "value": "; ".join(f"{k}={v}" for k, v in inputs.items())},
            {"key": "selection_rules", "value": "best overall + best per K for base/with_chaos; drop degenerate by ratio>0.9 or min<5"},
            {"key": "cluster_profiles", "value": "mean/median/std and z-score against whole feature set"},
            {"key": "visualizations", "value": "PCA scatter and z-score heatmaps per selected config"},
        ]
    )
    excel_out = _write_excel_report(
        excel_path=excel_path,
        summary_df=summary_df,
        selected_configs_df=selected_df,
        cluster_balance_df=balance_df,
        cluster_profiles_df=cluster_profiles_df,
        cluster_key_features_df=key_features_df,
        comparison_df=method_comparison_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Analyzed configurations: %d", selected_df["config_id"].nunique())
    logger.info("Built plots: scatter=%d heatmaps=%d total=%d", len(scatter_paths), len(heatmap_paths), len(scatter_paths) + len(heatmap_paths))
    logger.info("selected_configs: %s", selected_csv_path)
    logger.info("cluster_balance: %s", balance_csv_path)
    logger.info("cluster_profiles_parquet: %s", profiles_parquet_path)
    logger.info("cluster_profiles_csv: %s", profiles_csv_path)
    logger.info("cluster_key_features: %s", key_features_csv_path)
    logger.info("method_comparison: %s", comparison_csv_path)
    logger.info("excel_report: %s", excel_out)
    logger.info("figures_dir: %s", figures_dir)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "cluster_profiling",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": inputs,
        "outputs": {
            "selected_configs_csv": str(selected_csv_path),
            "cluster_balance_csv": str(balance_csv_path),
            "cluster_profiles_parquet": str(profiles_parquet_path),
            "cluster_profiles_csv": str(profiles_csv_path),
            "cluster_key_features_csv": str(key_features_csv_path),
            "cluster_method_comparison_csv": str(comparison_csv_path),
            "excel_report": str(excel_out),
            "figures_dir": str(figures_dir),
            "scatter_plots": [str(p) for p in scatter_paths],
            "heatmaps": [str(p) for p in heatmap_paths],
            "log": str(log_path),
        },
        "summary": {
            "selected_rows": int(len(selected_df)),
            "selected_unique_configs": int(selected_df["config_id"].nunique()),
            "cluster_balance_rows": int(len(balance_df)),
            "cluster_profiles_rows": int(len(cluster_profiles_df)),
            "key_features_rows": int(len(key_features_df)),
            "plots_total": int(len(scatter_paths) + len(heatmap_paths)),
        },
    }
    manifest_path = write_manifest(manifest, manifests_dir=manifests_dir, run_id=run_id)

    return {
        "run_id": run_id,
        "selected_configs_rows": int(len(selected_df)),
        "selected_unique_configs": int(selected_df["config_id"].nunique()),
        "cluster_profiles_rows": int(len(cluster_profiles_df)),
        "plots_total": int(len(scatter_paths) + len(heatmap_paths)),
        "manifest_path": str(manifest_path),
        "selected_configs_csv_path": str(selected_csv_path),
        "cluster_balance_csv_path": str(balance_csv_path),
        "cluster_profiles_parquet_path": str(profiles_parquet_path),
        "cluster_key_features_csv_path": str(key_features_csv_path),
        "method_comparison_csv_path": str(comparison_csv_path),
        "excel_path": str(excel_out),
        "figures": [str(p) for p in [*scatter_paths, *heatmap_paths]],
    }


def main() -> None:
    args = parse_args()
    result = run_cluster_profiling_pipeline(args.config)
    print(
        "run_id={run_id} selected={selected} unique={unique} profiles={profiles} plots={plots} manifest={manifest}".format(
            run_id=result["run_id"],
            selected=result["selected_configs_rows"],
            unique=result["selected_unique_configs"],
            profiles=result["cluster_profiles_rows"],
            plots=result["plots_total"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
