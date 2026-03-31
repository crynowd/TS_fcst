from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config.loader import load_meta_modeling_experiments_config
from src.meta_modeling.experimental_pipeline import run_meta_modeling_experiments
from src.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental meta-modeling branches")
    parser.add_argument(
        "--config",
        default="configs/meta_modeling_experiments_v1.yaml",
        help="Path to experimental meta-modeling config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_meta_modeling_experiments_config(args.config)

    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger, log_path = setup_logger(run_id=run_id, logs_dir=cfg["artifacts"]["logs"])
    cfg["meta"]["run_id"] = run_id
    cfg["meta"]["log_path"] = str(log_path)

    logger.info("Meta-modeling experiments config loaded: %s", cfg["meta"]["config_path"])
    result = run_meta_modeling_experiments(cfg=cfg, logger=logger)
    print(
        "run_id={run_id} tasks_total={total} tasks_success={success} manifest={manifest}".format(
            run_id=result["run_id"],
            total=result["tasks_total"],
            success=result["tasks_success"],
            manifest=result["manifest_path"],
        )
    )


if __name__ == "__main__":
    main()
