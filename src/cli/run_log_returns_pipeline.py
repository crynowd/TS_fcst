from __future__ import annotations

import argparse

from src.data.log_returns import run_log_returns_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for log-returns pipeline stage."""
    parser = argparse.ArgumentParser(description="Run TS_fcst log-returns pipeline")
    parser.add_argument(
        "--config",
        default="configs/data_inventory_v1.yaml",
        help="Path to stage config YAML (default: configs/data_inventory_v1.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for log-returns pipeline run."""
    args = parse_args()
    result = run_log_returns_pipeline(args.config)
    print(
        "run_id={run_id} input={input_series} excluded={excluded} ge2000={ge2000} "
        "short={short} core={core} extended={extended}".format(
            run_id=result.run_id,
            input_series=result.input_series,
            excluded=result.excluded_too_short,
            ge2000=result.ge_target_count,
            short=result.short_series_count,
            core=result.core_size,
            extended=result.extended_size,
        )
    )


if __name__ == "__main__":
    main()
