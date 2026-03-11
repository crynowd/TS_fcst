from __future__ import annotations

import argparse

from src.features.quality_screening import run_feature_screening_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst feature screening pipeline")
    parser.add_argument(
        "--config",
        default="configs/feature_screening_v1.yaml",
        help="Path to feature screening config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_screening_pipeline(args.config)
    print(
        "run_id={run_id} rows={rows} columns={columns} features={features} low_variance={low_variance} "
        "high_missing={high_missing} high_corr_pairs={high_corr_pairs} status_counts={status_counts}".format(
            run_id=result.run_id,
            rows=result.rows,
            columns=result.columns,
            features=result.n_features,
            low_variance=result.low_variance_count,
            high_missing=result.high_missing_count,
            high_corr_pairs=result.high_correlation_pairs,
            status_counts=result.screening_status_counts,
        )
    )


if __name__ == "__main__":
    main()
