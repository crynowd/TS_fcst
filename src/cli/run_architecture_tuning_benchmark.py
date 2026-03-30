from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.architecture_tuning.benchmark import run_architecture_tuning_benchmark
from src.config.loader import load_architecture_tuning_benchmark_config
from src.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst architecture tuning benchmark stage")
    parser.add_argument(
        "--config",
        default="configs/architecture_tuning_benchmark_v1.yaml",
        help="Path to architecture tuning benchmark config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_architecture_tuning_benchmark_config(args.config)

    run_name = str(cfg.get("run_name", "architecture_tuning_benchmark_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger, log_path = setup_logger(run_id=run_id, logs_dir=cfg["artifacts"]["logs"])
    cfg["meta"]["run_id"] = run_id
    cfg["meta"]["log_path"] = str(log_path)

    logger.info("Architecture tuning benchmark config loaded: %s", cfg["meta"]["config_path"])
    result = run_architecture_tuning_benchmark(cfg=cfg, logger=logger)
    print(
        "run_id={run_id} candidates={candidates} success={success} failed={failed} manifest={manifest}".format(
            run_id=result["run_id"],
            candidates=result["n_candidates"],
            success=result["success_count"],
            failed=result["failure_count"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
