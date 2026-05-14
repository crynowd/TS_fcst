from __future__ import annotations

import argparse

from src.meta_modeling.diagnostics import build_metamodeling_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight meta-modeling diagnostics without training")
    parser.add_argument(
        "--metrics",
        default="artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet",
        help="Path to forecasting metrics_long or aggregated series metrics artifact",
    )
    parser.add_argument(
        "--features",
        default="artifacts/features/final_clustering_features_with_chaos_v1.parquet",
        help="Path to series-level feature table",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/forecasting_audit_v2/metamodeling_diagnostic.json",
        help="Path for diagnostic JSON output",
    )
    parser.add_argument("--target-metrics", nargs="+", default=["rmse", "directional_accuracy"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 4, 5, 6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic = build_metamodeling_diagnostic(
        metrics_path=args.metrics,
        features_path=args.features,
        output_path=args.output,
        target_metrics=args.target_metrics,
        top_k_values=args.top_k,
    )
    agg = diagnostic["aggregation"]
    print(
        "diagnostic={output} series={series} models={models} horizons={horizons} n_folds_values={folds}".format(
            output=args.output,
            series=agg["n_series"],
            models=agg["n_models"],
            horizons=agg["horizons"],
            folds=agg["n_folds_values"],
        )
    )


if __name__ == "__main__":
    main()
