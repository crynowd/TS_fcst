from __future__ import annotations

import argparse

from src.features.final_feature_sets import run_final_feature_sets_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst final manual feature set assembly")
    parser.add_argument(
        "--master-path",
        default="artifacts/features/features_master_v1.parquet",
        help="Path to features master parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_final_feature_sets_pipeline(master_path=args.master_path)
    print(
        "run_id={run_id} rows={rows} base_features={base_features} with_chaos_features={with_chaos_features} "
        "breakdown_base={breakdown_base} breakdown_with_chaos={breakdown_with_chaos}".format(
            run_id=result.run_id,
            rows=result.rows,
            base_features=result.base_feature_count,
            with_chaos_features=result.with_chaos_feature_count,
            breakdown_base=result.breakdown_base,
            breakdown_with_chaos=result.breakdown_with_chaos,
        )
    )


if __name__ == "__main__":
    main()
