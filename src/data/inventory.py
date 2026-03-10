from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.config.loader import load_inventory_config
from src.data.readers import read_ru_daily_file, read_us_daily_file
from src.data.schema import CANONICAL_COLUMNS, get_market_label, get_source_schema
from src.reporting.excel_export import export_series_catalog_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest
from src.utils.validators import SchemaError, ensure_columns_exist, evaluate_series_quality


@dataclass(frozen=True)
class InventoryOutputs:
    """Output paths and summary stats of one inventory run."""

    run_id: str
    parquet_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    total_files: int
    valid_target: int
    valid_min_only: int
    invalid: int


def _extract_ticker(path: Path, market: str) -> str:
    """Extract ticker symbol from source filename based on market naming rules."""
    if market.lower() == "ru":
        name = path.name
        suffix = "_D1.csv"
        if not name.endswith(suffix):
            return path.stem
        return name[: -len(suffix)]
    return path.stem


def _safe_date(value: Any) -> str | None:
    """Convert date-like value to ISO date string."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def standardize_series_dataframe(
    df: pd.DataFrame,
    ticker: str,
    market: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Standardize one source dataframe to canonical TS_fcst schema.

    Args:
        df: Raw dataframe from RU or US reader.
        ticker: Series ticker.
        market: Market identifier (`ru`/`us` or canonical labels).
        config: Loaded inventory configuration.

    Returns:
        Standardized dataframe with canonical columns.

    Raises:
        SchemaError: If source schema is incompatible.
    """
    market_key = market.lower()
    source_schema = get_source_schema(config, market_key)

    ensure_columns_exist(
        df,
        [
            source_schema.date_col,
            source_schema.open_col,
            source_schema.high_col,
            source_schema.low_col,
            source_schema.close_col,
            source_schema.volume_col,
        ],
    )

    standardized = pd.DataFrame(
        {
            "series_id": f"{get_market_label(config, market_key)}:{ticker}",
            "ticker": ticker,
            "market": get_market_label(config, market_key),
            "date": pd.to_datetime(df[source_schema.date_col], errors="coerce"),
            "open": pd.to_numeric(df[source_schema.open_col], errors="coerce"),
            "high": pd.to_numeric(df[source_schema.high_col], errors="coerce"),
            "low": pd.to_numeric(df[source_schema.low_col], errors="coerce"),
            "close": pd.to_numeric(df[source_schema.close_col], errors="coerce"),
            "volume": pd.to_numeric(df[source_schema.volume_col], errors="coerce"),
        }
    )

    ensure_columns_exist(standardized, CANONICAL_COLUMNS)

    if bool(config.get("quality_checks", {}).get("sort_by_date", True)):
        standardized = standardized.sort_values("date", kind="stable").reset_index(drop=True)

    out_columns = config.get("standardization", {}).get("output_columns", CANONICAL_COLUMNS)
    ensure_columns_exist(standardized, out_columns)

    standardized = standardized[out_columns]
    return standardized


def build_series_summary(
    df: pd.DataFrame,
    source_path: str | Path,
    ticker: str,
    market: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build catalog summary for one standardized series dataframe.

    Args:
        df: Standardized dataframe.
        source_path: Source file path.
        ticker: Ticker symbol.
        market: Canonical market label.
        config: Loaded inventory configuration.

    Returns:
        Catalog row as dictionary.
    """
    policy = config.get("series_policy", {})
    quality = evaluate_series_quality(df, policy)
    path_obj = Path(source_path)

    return {
        "series_id": f"{market}:{ticker}",
        "ticker": ticker,
        "market": market,
        "source_path": str(path_obj),
        "file_name": path_obj.name,
        "n_rows_raw": int(df.attrs.get("n_rows_raw", len(df))),
        "n_rows_after_standardization": int(quality["n_rows_after_standardization"]),
        "min_date": _safe_date(quality["min_date"]),
        "max_date": _safe_date(quality["max_date"]),
        "is_sorted": bool(quality["is_sorted"]),
        "n_duplicate_dates": int(quality["n_duplicate_dates"]),
        "n_missing_open": int(quality["n_missing_open"]),
        "n_missing_high": int(quality["n_missing_high"]),
        "n_missing_low": int(quality["n_missing_low"]),
        "n_missing_close": int(quality["n_missing_close"]),
        "n_missing_volume": int(quality["n_missing_volume"]),
        "has_nonpositive_close": bool(quality["has_nonpositive_close"]),
        "can_compute_log_returns": bool(quality["can_compute_log_returns"]),
        "eligible_target_2000": bool(quality["eligible_target_2000"]),
        "eligible_min_1500": bool(quality["eligible_min_1500"]),
        "proposed_slice_mode": str(quality["proposed_slice_mode"]),
        "status": str(quality["status"]),
        "status_reason": str(quality["status_reason"]),
    }


def _build_error_row(
    source_path: Path,
    ticker: str,
    market: str,
    status_reason: str,
    n_rows_raw: int = 0,
) -> Dict[str, Any]:
    """Build fallback catalog row for read/schema errors."""
    return {
        "series_id": f"{market}:{ticker}",
        "ticker": ticker,
        "market": market,
        "source_path": str(source_path),
        "file_name": source_path.name,
        "n_rows_raw": int(n_rows_raw),
        "n_rows_after_standardization": 0,
        "min_date": None,
        "max_date": None,
        "is_sorted": False,
        "n_duplicate_dates": 0,
        "n_missing_open": 0,
        "n_missing_high": 0,
        "n_missing_low": 0,
        "n_missing_close": 0,
        "n_missing_volume": 0,
        "has_nonpositive_close": False,
        "can_compute_log_returns": False,
        "eligible_target_2000": False,
        "eligible_min_1500": False,
        "proposed_slice_mode": "last_n",
        "status": "invalid",
        "status_reason": status_reason,
    }


def _scan_files(source_dir: str | Path, pattern: str) -> List[Path]:
    """Scan source directory by glob pattern and return sorted file list."""
    base = Path(source_dir)
    if not base.exists():
        return []
    return sorted(base.glob(pattern))


def _ensure_catalog_columns(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Create catalog dataframe with stable column order."""
    columns = [
        "series_id",
        "ticker",
        "market",
        "source_path",
        "file_name",
        "n_rows_raw",
        "n_rows_after_standardization",
        "min_date",
        "max_date",
        "is_sorted",
        "n_duplicate_dates",
        "n_missing_open",
        "n_missing_high",
        "n_missing_low",
        "n_missing_close",
        "n_missing_volume",
        "has_nonpositive_close",
        "can_compute_log_returns",
        "eligible_target_2000",
        "eligible_min_1500",
        "proposed_slice_mode",
        "status",
        "status_reason",
    ]
    df = pd.DataFrame(list(rows))
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def run_data_inventory(config_path: str) -> InventoryOutputs:
    """Execute stage-1 data inventory pipeline.

    The run scans external RU/US sources, reads and standardizes each series,
    computes quality catalog, and writes parquet, Excel report, logs, and
    manifest.

    Args:
        config_path: Path to inventory stage config YAML.

    Returns:
        InventoryOutputs with artifact paths and summary counts.
    """
    cfg = load_inventory_config(config_path)

    run_name = str(cfg.get("run_name", "data_inventory_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg.get("artifacts", {})
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    logger.info("Data inventory run started")
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    ru_files = _scan_files(cfg["data_sources"]["ru_daily"], cfg["input"].get("ru_pattern", "*_D1.csv"))
    us_files = _scan_files(cfg["data_sources"]["us_daily"], cfg["input"].get("us_pattern", "*.txt"))

    logger.info("RU files found: %d", len(ru_files))
    logger.info("US files found: %d", len(us_files))

    catalog_rows: List[Dict[str, Any]] = []

    for file_path in ru_files:
        ticker = _extract_ticker(file_path, "ru")
        market_label = get_market_label(cfg, "ru")
        try:
            raw_df = read_ru_daily_file(file_path)
        except Exception as exc:
            logger.error("Read error [%s]: %s", file_path, exc)
            catalog_rows.append(_build_error_row(file_path, ticker, market_label, "read_error"))
            continue

        try:
            std_df = standardize_series_dataframe(raw_df, ticker=ticker, market="ru", config=cfg)
            std_df.attrs["n_rows_raw"] = int(len(raw_df))
            row = build_series_summary(std_df, file_path, ticker, market_label, cfg)
            catalog_rows.append(row)
            if row["status"] == "invalid":
                logger.warning("Invalid series [%s]: %s", file_path, row["status_reason"])
        except SchemaError as exc:
            logger.error("Schema error [%s]: %s", file_path, exc)
            catalog_rows.append(
                _build_error_row(file_path, ticker, market_label, "schema_error", n_rows_raw=len(raw_df))
            )
        except Exception as exc:
            logger.error("Unexpected processing error [%s]: %s", file_path, exc)
            catalog_rows.append(
                _build_error_row(file_path, ticker, market_label, "processing_error", n_rows_raw=len(raw_df))
            )

    for file_path in us_files:
        ticker = _extract_ticker(file_path, "us")
        market_label = get_market_label(cfg, "us")
        try:
            raw_df = read_us_daily_file(file_path)
        except Exception as exc:
            logger.error("Read error [%s]: %s", file_path, exc)
            catalog_rows.append(_build_error_row(file_path, ticker, market_label, "read_error"))
            continue

        try:
            std_df = standardize_series_dataframe(raw_df, ticker=ticker, market="us", config=cfg)
            std_df.attrs["n_rows_raw"] = int(len(raw_df))
            row = build_series_summary(std_df, file_path, ticker, market_label, cfg)
            catalog_rows.append(row)
            if row["status"] == "invalid":
                logger.warning("Invalid series [%s]: %s", file_path, row["status_reason"])
        except SchemaError as exc:
            logger.error("Schema error [%s]: %s", file_path, exc)
            catalog_rows.append(
                _build_error_row(file_path, ticker, market_label, "schema_error", n_rows_raw=len(raw_df))
            )
        except Exception as exc:
            logger.error("Unexpected processing error [%s]: %s", file_path, exc)
            catalog_rows.append(
                _build_error_row(file_path, ticker, market_label, "processing_error", n_rows_raw=len(raw_df))
            )

    catalog_df = _ensure_catalog_columns(catalog_rows)

    version_suffix = run_name.replace("data_inventory_", "")
    parquet_path = Path(artifacts_cfg["processed"]).resolve() / f"series_catalog_{version_suffix}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_df.to_parquet(parquet_path, index=False)

    excel_path = Path(artifacts_cfg["reports"]).resolve() / f"series_catalog_{version_suffix}.xlsx"
    timestamp_now = datetime.now(timezone.utc).isoformat()
    excel_out = export_series_catalog_excel(
        catalog_df,
        excel_path=excel_path,
        run_metadata={
            "run_id": run_id,
            "stage": "data_inventory",
            "timestamp": timestamp_now,
            "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
            "config_path": cfg["meta"]["config_path"],
            "parquet_path": str(parquet_path),
        },
        source_paths=[cfg["data_sources"]["ru_daily"], cfg["data_sources"]["us_daily"]],
    )

    end_ts = datetime.now(timezone.utc)
    valid_target = int((catalog_df["status"] == "valid_target").sum())
    valid_min_only = int((catalog_df["status"] == "valid_min_only").sum())
    invalid = int((catalog_df["status"] == "invalid").sum())

    manifest = {
        "run_id": run_id,
        "stage": "data_inventory",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": {
            "ru_daily": cfg["data_sources"]["ru_daily"],
            "us_daily": cfg["data_sources"]["us_daily"],
            "ru_files": len(ru_files),
            "us_files": len(us_files),
        },
        "outputs": {
            "parquet": str(parquet_path),
            "excel": str(excel_out),
            "log": str(log_path),
        },
        "summary": {
            "total_files": int(len(catalog_df)),
            "valid_target": valid_target,
            "valid_min_only": valid_min_only,
            "invalid": invalid,
        },
    }

    manifest_path = write_manifest(manifest, artifacts_cfg["manifests"], run_id)

    logger.info("Successfully processed series: %d", valid_target + valid_min_only)
    logger.info("Invalid series: %d", invalid)
    logger.info("Parquet path: %s", parquet_path)
    logger.info("Excel path: %s", excel_out)
    logger.info("Manifest path: %s", manifest_path)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    return InventoryOutputs(
        run_id=run_id,
        parquet_path=str(parquet_path),
        excel_path=str(excel_out),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        total_files=int(len(catalog_df)),
        valid_target=valid_target,
        valid_min_only=valid_min_only,
        invalid=invalid,
    )
