from __future__ import annotations

import argparse

from src.features.fold_aware_rebuild import rebuild_fold_aware_features


def _str_to_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild final time-series features by forecasting fold using train-only data")
    parser.add_argument(
        "--log-returns",
        default="artifacts/processed/log_returns_v1.parquet",
        help="Path to log returns parquet",
    )
    parser.add_argument(
        "--split-metadata",
        default="artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet",
        help="Path to forecasting_benchmark_v2 split_metadata.parquet",
    )
    parser.add_argument(
        "--old-features",
        default="artifacts/features/final_clustering_features_with_chaos_v1.parquet",
        help="Path to old full-series final feature artifact for comparison",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/features/fold_aware_features_v2",
        help="Output directory for fold-aware feature artifacts",
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts/reports/forecasting_audit_v2",
        help="Output directory for audit report",
    )
    parser.add_argument("--max-series", type=int, default=None, help="Limit to first N series_id values after filters")
    parser.add_argument("--series-ids", nargs="+", default=None, help="Optional list of series_id values")
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="Optional list of horizons")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Optional list of fold IDs")
    parser.add_argument(
        "--overwrite",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Overwrite existing output parquet. Defaults to false.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        help="Resume from existing output parquet by skipping completed keys. Defaults to true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rebuild_fold_aware_features(
        log_returns_path=args.log_returns,
        split_metadata_path=args.split_metadata,
        old_features_path=args.old_features,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        max_series=args.max_series,
        series_ids=args.series_ids,
        horizons=args.horizons,
        folds=args.folds,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    coverage = result["coverage"]
    print(
        "features={features} rows={rows}/{expected} series={series} horizons={horizons} folds={folds} "
        "errors_or_warnings={errors} report={report}".format(
            features=result["outputs"]["features_parquet"],
            rows=coverage["output_rows"],
            expected=coverage["selected_split_rows"],
            series=coverage["series_count"],
            horizons=coverage["horizons"],
            folds=coverage["folds"],
            errors=result["errors"],
            report=result["outputs"]["audit_report_md"],
        )
    )


if __name__ == "__main__":
    main()
