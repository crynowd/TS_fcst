from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.runners import run_forecasting_benchmark
from src.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS_fcst forecasting benchmark (smoke)")
    parser.add_argument(
        "--config",
        default="configs/forecasting_benchmark_smoke_v1.yaml",
        help="Path to forecasting benchmark config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_forecasting_benchmark_config(args.config)

    run_name = str(cfg.get("outputs", {}).get("run_name", "forecasting_benchmark_smoke_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger, log_path = setup_logger(run_id=run_id, logs_dir=cfg["artifacts"]["logs"])
    cfg["meta"]["run_id"] = run_id
    cfg["meta"]["log_path"] = str(log_path)

    logger.info("Forecasting benchmark config loaded: %s", cfg["meta"]["config_path"])
    result = run_forecasting_benchmark(cfg, logger=logger)

    print(
        "run_id={run_id} tasks={tasks} success={success} timeout={timeout} error={error} manifest={manifest}".format(
            run_id=result["run_id"],
            tasks=result["n_tasks"],
            success=result["success_count"],
            timeout=result["timeout_count"],
            error=result["error_count"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
