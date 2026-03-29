from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.analysis.cluster_forecasting import (
    append_relative_to_baseline,
    build_best_model_by_cluster,
    build_cluster_model_performance,
    build_tidy_cluster_model_metrics,
    build_tidy_series_cluster_metrics,
    join_forecasting_with_cluster_labels,
    validate_shortlist_configs,
)
from src.analysis.stat_tests import run_kruskal_tests
from src.analysis.utility_metrics import build_clustering_utility
from src.config.loader import load_cluster_forecasting_analysis_config
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cluster-conditioned forecasting analysis stage")
    parser.add_argument(
        "--config",
        default="configs/cluster_forecasting_analysis_smoke_v1.yaml",
        help="Path to Stage 11B config",
    )
    return parser.parse_args()


def _write_excel_report(
    excel_path: str | Path,
    summary_df: pd.DataFrame,
    best_model_by_cluster_df: pd.DataFrame,
    cluster_model_performance_df: pd.DataFrame,
    clustering_utility_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    readme_df: pd.DataFrame,
) -> Path:
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        best_model_by_cluster_df.to_excel(writer, sheet_name="best_model_by_cluster", index=False)
        cluster_model_performance_df.to_excel(writer, sheet_name="cluster_model_performance", index=False)
        clustering_utility_df.to_excel(writer, sheet_name="clustering_utility", index=False)
        tests_df.to_excel(writer, sheet_name="statistical_tests", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)
    return out_path


def _build_summary_sheet(
    cfg: dict[str, Any],
    used_configs: list[str],
    series_joined_df: pd.DataFrame,
    cluster_model_performance_df: pd.DataFrame,
    clustering_utility_df: pd.DataFrame,
) -> pd.DataFrame:
    horizons = sorted(pd.to_numeric(series_joined_df.get("horizon", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    models = sorted(series_joined_df.get("model_name", pd.Series(dtype=str)).astype(str).unique().tolist())

    rows = [
        {"section": "inputs", "label": "shortlist_requested", "value": ", ".join(cfg["shortlist_configs"])},
        {"section": "inputs", "label": "shortlist_used", "value": ", ".join(used_configs)},
        {"section": "inputs", "label": "horizons", "value": ", ".join(str(h) for h in horizons)},
        {"section": "inputs", "label": "models", "value": ", ".join(models)},
        {"section": "counts", "label": "joined_rows", "value": int(len(series_joined_df))},
        {"section": "counts", "label": "cluster_model_rows", "value": int(len(cluster_model_performance_df))},
    ]

    cfg_counts = (
        series_joined_df.groupby("clustering_config", as_index=False)["series_id"]
        .nunique()
        .rename(columns={"series_id": "n_series"})
        .sort_values("clustering_config", kind="stable")
    )
    for _, row in cfg_counts.iterrows():
        rows.append(
            {
                "section": "series_by_config",
                "label": str(row["clustering_config"]),
                "value": int(row["n_series"]),
            }
        )

    if not clustering_utility_df.empty:
        reg = pd.to_numeric(clustering_utility_df["regret_improvement"], errors="coerce")
        div = pd.to_numeric(clustering_utility_df["best_model_diversity_count"], errors="coerce")
        rows.append({"section": "utility", "label": "regret_improvement_mean", "value": float(reg.mean()) if reg.notna().any() else None})
        rows.append({"section": "utility", "label": "regret_improvement_median", "value": float(reg.median()) if reg.notna().any() else None})
        rows.append({"section": "utility", "label": "best_model_diversity_count_mean", "value": float(div.mean()) if div.notna().any() else None})
        rows.append({"section": "utility", "label": "best_model_diversity_count_max", "value": float(div.max()) if div.notna().any() else None})

    return pd.DataFrame(rows)


def run_cluster_forecasting_analysis_pipeline(config_path: str = "configs/cluster_forecasting_analysis_smoke_v1.yaml") -> Dict[str, Any]:
    cfg = load_cluster_forecasting_analysis_config(config_path)

    run_name = str(cfg.get("run_name", "cluster_forecasting_analysis_smoke_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    stage_name = str(cfg.get("stage", "cluster_forecasting_analysis_smoke"))
    start_ts = datetime.now(timezone.utc)

    logger, log_path = setup_logger(run_id=run_id, logs_dir=cfg["artifacts"]["logs"])
    logger.info("Cluster forecasting analysis started")
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    f_in = cfg["forecasting_inputs"]
    c_in = cfg["clustering_inputs"]
    series_metrics_df = pd.read_parquet(f_in["series_metrics_path"])
    fold_metrics_df = pd.read_parquet(f_in["fold_metrics_path"])
    raw_predictions_df = pd.read_parquet(f_in["raw_predictions_path"])
    cluster_assignments_df = pd.read_parquet(c_in["cluster_assignments_path"])
    selected_configs_df = pd.read_csv(c_in["selected_configs_path"])

    shortlist = [str(x) for x in cfg["shortlist_configs"]]
    validation = validate_shortlist_configs(
        shortlist_configs=shortlist,
        selected_configs_df=selected_configs_df,
        cluster_assignments_df=cluster_assignments_df,
    )
    used_configs = validation["available_configs"]
    logger.info("Shortlisted clusterings requested: %d", len(shortlist))
    logger.info("Shortlisted clusterings available in assignments: %d", len(used_configs))
    if validation["missing_in_selected"]:
        logger.error("Missing config_id in selected_configs: %s", ", ".join(validation["missing_in_selected"]))
    if validation["missing_in_assignments"]:
        logger.error("Missing config_id in cluster_assignments: %s", ", ".join(validation["missing_in_assignments"]))

    joined = join_forecasting_with_cluster_labels(
        series_metrics_df=series_metrics_df,
        fold_metrics_df=fold_metrics_df,
        raw_predictions_df=raw_predictions_df,
        cluster_assignments_df=cluster_assignments_df,
        available_configs=used_configs,
    )
    series_joined_df = joined["series_joined"]
    logger.info("Joined rows (series_metrics): %d", len(series_joined_df))
    logger.info("Joined rows (fold_metrics): %d", len(joined["fold_joined"]))
    logger.info("Joined rows (raw_predictions): %d", len(joined["raw_joined"]))

    primary_baseline = str(cfg.get("baselines", {}).get("primary", "naive_zero"))
    metrics_for_comparison = [str(m) for m in cfg.get("metrics_for_comparison", ["mae", "rmse", "mase", "directional_accuracy"])]

    cluster_model_perf = build_cluster_model_performance(series_joined_df)
    cluster_model_perf, missing_baseline_df = append_relative_to_baseline(
        cluster_model_performance_df=cluster_model_perf,
        baseline_model=primary_baseline,
    )
    if not missing_baseline_df.empty:
        logger.warning("Missing baseline rows for %s combinations: %d", primary_baseline, len(missing_baseline_df))
    else:
        logger.info("Baseline %s is available for all cluster/horizon combinations", primary_baseline)

    best_by_cluster_df = build_best_model_by_cluster(cluster_model_perf)
    utility_df = build_clustering_utility(cluster_model_perf, metrics=metrics_for_comparison)

    use_kruskal = bool(cfg.get("stat_tests", {}).get("use_kruskal", True))
    tests_df = run_kruskal_tests(series_joined_df, metrics=metrics_for_comparison) if use_kruskal else pd.DataFrame()

    tidy_cluster_model_df = build_tidy_cluster_model_metrics(cluster_model_perf)
    tidy_series_cluster_df = build_tidy_series_cluster_metrics(series_joined_df)

    outputs = cfg["outputs"]
    Path(outputs["cluster_model_performance_path"]).resolve().parent.mkdir(parents=True, exist_ok=True)

    cluster_model_perf.to_parquet(outputs["cluster_model_performance_path"], index=False)
    cluster_model_perf.to_csv(outputs["cluster_model_performance_csv_path"], index=False)
    best_by_cluster_df.to_csv(outputs["best_model_by_cluster_path"], index=False)
    utility_df.to_csv(outputs["clustering_utility_path"], index=False)
    tests_df.to_csv(outputs["cluster_metric_tests_path"], index=False)
    tidy_cluster_model_df.to_parquet(outputs["tidy_cluster_model_metrics_path"], index=False)
    tidy_series_cluster_df.to_parquet(outputs["tidy_series_cluster_metrics_path"], index=False)

    summary_df = _build_summary_sheet(
        cfg=cfg,
        used_configs=used_configs,
        series_joined_df=series_joined_df,
        cluster_model_performance_df=cluster_model_perf,
        clustering_utility_df=utility_df,
    )
    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": stage_name},
            {"key": "config_path", "value": cfg["meta"]["config_path"]},
            {"key": "inputs", "value": f"series={f_in['series_metrics_path']}; fold={f_in['fold_metrics_path']}; raw={f_in['raw_predictions_path']}; assignments={c_in['cluster_assignments_path']}; selected_configs={c_in['selected_configs_path']}"},
            {"key": "shortlist_requested", "value": ", ".join(shortlist)},
            {"key": "shortlist_used", "value": ", ".join(used_configs)},
            {"key": "baseline_primary", "value": primary_baseline},
            {"key": "notes", "value": "Stage 11B analysis only. No retraining. No clustering recomputation."},
        ]
    )
    excel_path = _write_excel_report(
        excel_path=outputs["excel_report_path"],
        summary_df=summary_df,
        best_model_by_cluster_df=best_by_cluster_df,
        cluster_model_performance_df=cluster_model_perf,
        clustering_utility_df=utility_df,
        tests_df=tests_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Number of models: %d", series_joined_df["model_name"].nunique())
    logger.info("Number of horizons: %d", series_joined_df["horizon"].nunique())
    logger.info("Number of statistical tests: %d", len(tests_df))
    logger.info("Generated tables: %d", 7)
    logger.info("Output cluster_model_performance: %s", outputs["cluster_model_performance_path"])
    logger.info("Output best_model_by_cluster: %s", outputs["best_model_by_cluster_path"])
    logger.info("Output clustering_utility: %s", outputs["clustering_utility_path"])
    logger.info("Output cluster_metric_tests: %s", outputs["cluster_metric_tests_path"])
    logger.info("Output tidy_cluster_model_metrics: %s", outputs["tidy_cluster_model_metrics_path"])
    logger.info("Output tidy_series_cluster_metrics: %s", outputs["tidy_series_cluster_metrics_path"])
    logger.info("Output excel_report: %s", excel_path)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg["meta"]["config_path"],
        "input_sources": {
            "series_metrics": f_in["series_metrics_path"],
            "fold_metrics": f_in["fold_metrics_path"],
            "raw_predictions": f_in["raw_predictions_path"],
            "cluster_assignments": c_in["cluster_assignments_path"],
            "selected_configs": c_in["selected_configs_path"],
        },
        "outputs": {
            "cluster_model_performance_parquet": outputs["cluster_model_performance_path"],
            "cluster_model_performance_csv": outputs["cluster_model_performance_csv_path"],
            "best_model_by_cluster_csv": outputs["best_model_by_cluster_path"],
            "clustering_utility_csv": outputs["clustering_utility_path"],
            "cluster_metric_tests_csv": outputs["cluster_metric_tests_path"],
            "tidy_cluster_model_metrics_parquet": outputs["tidy_cluster_model_metrics_path"],
            "tidy_series_cluster_metrics_parquet": outputs["tidy_series_cluster_metrics_path"],
            "excel_report": str(excel_path),
            "log": str(log_path),
        },
        "summary": {
            "shortlisted_configs_requested": len(shortlist),
            "shortlisted_configs_used": len(used_configs),
            "missing_in_selected": validation["missing_in_selected"],
            "missing_in_assignments": validation["missing_in_assignments"],
            "joined_rows_series_metrics": int(len(series_joined_df)),
            "models": int(series_joined_df["model_name"].nunique()),
            "horizons": int(series_joined_df["horizon"].nunique()),
            "baseline_missing_rows": int(len(missing_baseline_df)),
            "statistical_tests": int(len(tests_df)),
            "tables_generated": 7,
        },
    }
    manifest_path = write_manifest(manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)

    return {
        "run_id": run_id,
        "used_configs": used_configs,
        "validation": validation,
        "joined_rows": int(len(series_joined_df)),
        "cluster_model_performance_rows": int(len(cluster_model_perf)),
        "best_model_by_cluster_rows": int(len(best_by_cluster_df)),
        "clustering_utility_rows": int(len(utility_df)),
        "stat_tests_rows": int(len(tests_df)),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "excel_path": str(excel_path),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    result = run_cluster_forecasting_analysis_pipeline(args.config)
    print(
        "run_id={run_id} used_configs={used_cfg} joined_rows={joined} utility_rows={utility} tests={tests} manifest={manifest}".format(
            run_id=result["run_id"],
            used_cfg=len(result["used_configs"]),
            joined=result["joined_rows"],
            utility=result["clustering_utility_rows"],
            tests=result["stat_tests_rows"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
