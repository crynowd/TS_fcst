from __future__ import annotations

import argparse

from src.data.inventory import run_data_inventory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data inventory stage."""
    parser = argparse.ArgumentParser(description="Run TS_fcst data inventory pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to data inventory config YAML (e.g. configs/data_inventory_v1.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for data inventory run."""
    args = parse_args()
    result = run_data_inventory(args.config)
    print(
        "run_id={run_id} total_files={total} valid_target={vt} "
        "valid_min_only={vmo} invalid={invalid}".format(
            run_id=result.run_id,
            total=result.total_files,
            vt=result.valid_target,
            vmo=result.valid_min_only,
            invalid=result.invalid,
        )
    )


if __name__ == "__main__":
    main()
