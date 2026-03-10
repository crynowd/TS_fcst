from __future__ import annotations

import argparse

from src.features.registry import run_feature_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst feature block pipeline")
    parser.add_argument(
        "--block",
        required=True,
        help="Feature block code (currently supported: A)",
    )
    parser.add_argument(
        "--config",
        default="configs/features_block_A_v1.yaml",
        help="Path to feature block config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_block(args.block, args.config)
    print(
        "run_id={run_id} input_series={input_series} output_series={output_series} "
        "successful={successful} warnings={warnings}".format(
            run_id=result.run_id,
            input_series=result.input_series,
            output_series=result.output_series,
            successful=result.successful,
            warnings=result.warnings,
        )
    )


if __name__ == "__main__":
    main()

