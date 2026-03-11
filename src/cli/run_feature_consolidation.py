from __future__ import annotations

import argparse

from src.features.consolidation import run_feature_consolidation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst feature consolidation pipeline")
    parser.add_argument(
        "--config",
        default="configs/feature_consolidation_v1.yaml",
        help="Path to feature consolidation config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_consolidation_pipeline(args.config)
    print(
        "run_id={run_id} candidate_pool={candidate_pool} corr_groups={corr_groups} "
        "base_features={base_features} with_chaos_features={with_chaos_features} by_block={by_block}".format(
            run_id=result.run_id,
            candidate_pool=result.candidate_pool_size,
            corr_groups=result.correlation_groups_count,
            base_features=result.base_feature_count,
            with_chaos_features=result.with_chaos_feature_count,
            by_block=result.selected_by_block,
        )
    )


if __name__ == "__main__":
    main()
