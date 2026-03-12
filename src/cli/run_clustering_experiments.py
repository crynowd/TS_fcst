from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.clustering.grid_search import build_grid_configurations, run_grid_search
from src.clustering.selection import add_ranking_score, select_top_configurations
from src.clustering.stability import build_top_config_assignments, run_bootstrap_stability
from src.config.loader import load_clustering_experiments_config
from src.reporting.excel_export import export_clustering_experiments_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst clustering experiments framework")
    parser.add_argument(
        "--config",
        default="configs/clustering_experiments_v1.yaml",
        help="Path to clustering experiments config",
    )
    return parser.parse_args()


def _build_summary_sheet(grid_df: pd.DataFrame, top_df: pd.DataFrame, stability_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "section": "overall",
            "label": "total_grid_configurations",
            "value": int(len(grid_df)),
        }
    )
    for feature_set, n in grid_df["feature_set"].value_counts().sort_index().to_dict().items():
        rows.append({"section": "overall", "label": f"grid_configs_{feature_set}", "value": int(n)})

    for feature_set in sorted(grid_df["feature_set"].unique()):
        sdf = grid_df[grid_df["feature_set"] == feature_set].sort_values("ranking_score", ascending=False, kind="stable")
        if not sdf.empty and pd.notna(sdf.iloc[0]["ranking_score"]):
            best = sdf.iloc[0]
            rows.append(
                {
                    "section": "best_by_feature_set",
                    "label": str(feature_set),
                    "value": (
                        f"config_id={best['config_id']}; algo={best['algorithm']}; scaler={best['scaler']}; "
                        f"space={best['space_type']}; k={int(best['n_clusters_requested'])}; score={best['ranking_score']:.4f}"
                    ),
                }
            )

    for (feature_set, algorithm), sdf in top_df.groupby(["feature_set", "algorithm"], sort=True):
        top3 = sdf.sort_values("ranking_score", ascending=False, kind="stable").head(3)
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            rows.append(
                {
                    "section": "top3_by_feature_set_algorithm",
                    "label": f"{feature_set}_{algorithm}_rank{rank}",
                    "value": (
                        f"config_id={row['config_id']}; scaler={row['scaler']}; space={row['space_type']}; "
                        f"pca={row['pca_n_components']}; k={int(row['n_clusters_requested'])}; score={row['ranking_score']:.4f}"
                    ),
                }
            )

    if not stability_df.empty:
        rows.append(
            {
                "section": "stability_overall",
                "label": "ari_mean_over_top",
                "value": float(stability_df["ari_mean"].mean()),
            }
        )
        rows.append(
            {
                "section": "stability_overall",
                "label": "nmi_mean_over_top",
                "value": float(stability_df["nmi_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_clustering_experiments_pipeline(config_path: str = "configs/clustering_experiments_v1.yaml") -> Dict[str, Any]:
    cfg = load_clustering_experiments_config(config_path)

    run_name = str(cfg.get("run_name", "clustering_experiments_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts["logs"])

    logger.info("Clustering experiments started")
    logger.info("Config path: %s", cfg["meta"]["config_path"])
    logger.info("Feature sets: %s", cfg["feature_sets"])
    logger.info("Scalers: %s", cfg["scalers"])
    logger.info("Algorithms: %s", cfg["algorithms"])

    base_path = Path(cfg["input"]["final_base_parquet"]).resolve()
    chaos_path = Path(cfg["input"]["final_with_chaos_parquet"]).resolve()
    feature_tables: Dict[str, pd.DataFrame] = {
        "base": pd.read_parquet(base_path),
        "with_chaos": pd.read_parquet(chaos_path),
    }

    all_configs = build_grid_configurations(cfg)
    logger.info("Grid configurations total: %d", len(all_configs))

    grid_df = run_grid_search(feature_tables=feature_tables, cfg=cfg, logger=logger)
    grid_df = add_ranking_score(grid_df)
    top_df = select_top_configurations(grid_df=grid_df, cfg=cfg)
    logger.info("Top configurations selected: %d", len(top_df))

    assignments_df, reference_labels = build_top_config_assignments(
        top_configs_df=top_df,
        feature_tables=feature_tables,
        random_state=int(cfg["stability"]["random_state"]),
    )
    logger.info("Cluster assignments rows: %d", len(assignments_df))

    stability_df = run_bootstrap_stability(
        top_configs_df=top_df,
        feature_tables=feature_tables,
        reference_labels=reference_labels,
        cfg=cfg,
        logger=logger,
    )
    logger.info(
        "Bootstrap stability computed: configs=%d n_bootstrap=%d",
        len(stability_df),
        int(cfg["stability"]["n_bootstrap"]),
    )

    clustering_dir = Path(artifacts["clustering"]).resolve()
    reports_dir = Path(artifacts["reports"]).resolve()
    manifests_dir = Path(artifacts["manifests"]).resolve()
    for d in [clustering_dir, reports_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    output = cfg["output"]
    grid_parquet_path = clustering_dir / str(output["grid_results_parquet_name"])
    top_parquet_path = clustering_dir / str(output["top_configs_parquet_name"])
    stability_parquet_path = clustering_dir / str(output["stability_parquet_name"])
    assignments_parquet_path = clustering_dir / str(output["assignments_parquet_name"])
    grid_csv_path = reports_dir / str(output["grid_results_csv_name"])
    top_csv_path = reports_dir / str(output["top_configs_csv_name"])
    stability_csv_path = reports_dir / str(output["stability_csv_name"])
    excel_path = reports_dir / str(output["excel_name"])

    grid_df.to_parquet(grid_parquet_path, index=False)
    top_df.to_parquet(top_parquet_path, index=False)
    stability_df.to_parquet(stability_parquet_path, index=False)
    assignments_df.to_parquet(assignments_parquet_path, index=False)
    grid_df.to_csv(grid_csv_path, index=False)
    top_df.to_csv(top_csv_path, index=False)
    stability_df.to_csv(stability_csv_path, index=False)

    summary_df = _build_summary_sheet(grid_df=grid_df, top_df=top_df, stability_df=stability_df)
    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": "clustering_experiments"},
            {"key": "input_base_parquet", "value": str(base_path)},
            {"key": "input_with_chaos_parquet", "value": str(chaos_path)},
            {"key": "config_path", "value": cfg["meta"]["config_path"]},
            {"key": "selection_logic", "value": "ranking_score based on silhouette, DBI, CH, cluster balance, tiny cluster penalties"},
            {"key": "stability_logic", "value": "bootstrap subsampling with ARI/NMI versus reference assignment"},
        ]
    )
    excel_out = export_clustering_experiments_excel(
        excel_path=excel_path,
        summary_df=summary_df,
        grid_results_df=grid_df,
        top_configs_df=top_df,
        stability_df=stability_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Grid results path: %s", grid_parquet_path)
    logger.info("Top configs path: %s", top_parquet_path)
    logger.info("Stability path: %s", stability_parquet_path)
    logger.info("Assignments path: %s", assignments_parquet_path)
    logger.info("Excel report path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    summary_payload: Dict[str, Any] = {
        "n_grid_configurations": int(len(grid_df)),
        "n_top_configurations": int(len(top_df)),
        "n_assignments": int(len(assignments_df)),
        "stability_ari_mean": float(stability_df["ari_mean"].mean()) if not stability_df.empty else None,
        "stability_nmi_mean": float(stability_df["nmi_mean"].mean()) if not stability_df.empty else None,
    }

    manifest = {
        "run_id": run_id,
        "stage": "clustering_experiments",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": {
            "final_base_parquet": str(base_path),
            "final_with_chaos_parquet": str(chaos_path),
        },
        "outputs": {
            "grid_results_parquet": str(grid_parquet_path),
            "top_configs_parquet": str(top_parquet_path),
            "stability_parquet": str(stability_parquet_path),
            "assignments_parquet": str(assignments_parquet_path),
            "grid_results_csv": str(grid_csv_path),
            "top_configs_csv": str(top_csv_path),
            "stability_csv": str(stability_csv_path),
            "excel_report": str(excel_out),
            "log": str(log_path),
        },
        "summary": summary_payload,
    }
    manifest_path = write_manifest(manifest, manifests_dir=manifests_dir, run_id=run_id)

    return {
        "run_id": run_id,
        "grid_configurations": int(len(grid_df)),
        "top_configurations": int(len(top_df)),
        "stability_configurations": int(len(stability_df)),
        "manifest_path": str(manifest_path),
        "grid_parquet_path": str(grid_parquet_path),
        "top_parquet_path": str(top_parquet_path),
        "stability_parquet_path": str(stability_parquet_path),
        "assignments_parquet_path": str(assignments_parquet_path),
        "excel_path": str(excel_out),
    }


def main() -> None:
    args = parse_args()
    result = run_clustering_experiments_pipeline(args.config)
    print(
        "run_id={run_id} grid_configs={grid_n} top_configs={top_n} stability_configs={stab_n} manifest={manifest}".format(
            run_id=result["run_id"],
            grid_n=result["grid_configurations"],
            top_n=result["top_configurations"],
            stab_n=result["stability_configurations"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
