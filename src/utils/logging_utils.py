from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(run_id: str, logs_dir: str | Path) -> tuple[logging.Logger, Path]:
    """Create stage logger writing both to console and a dedicated file.

    Args:
        run_id: Run identifier used as log file name.
        logs_dir: Directory where logs are written.

    Returns:
        Tuple of configured logger and absolute log file path.
    """
    logs_path = Path(logs_dir).resolve()
    logs_path.mkdir(parents=True, exist_ok=True)

    log_file = logs_path / f"{run_id}.log"
    logger = logging.getLogger(f"data_inventory.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_file
