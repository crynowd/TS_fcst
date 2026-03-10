from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.loader import load_inventory_config
from src.data.dataset_profiles import build_dataset_profiles
from src.data.inventory import standardize_series_dataframe
from src.data.readers import read_ru_daily_file, read_us_daily_file
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class LogReturnsPipelineOutputs:
    """Output paths and summary stats for log-returns stage run."""

    run_id: str
    log_returns_parquet_path: str
    dataset_profiles_parquet_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    input_series: int
    excluded_too_short: int
    ge_target_count: int
    short_series_count: int
    valid_ru: int
    valid_us: int
    core_size: int
    extended_size: int


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns from price series `close` as log(P_t)-log(P_{t-1})."""
    if "close" not in df.columns:
        raise ValueError("Input dataframe must contain 'close' column")

    close = pd.to_numeric(df["close"], errors="coerce")
    if close.isna().any():
        raise ValueError("Close series contains NaN values")
    if (close <= 0).any():
        raise ValueError("Close series must be strictly positive")

    out = df.copy()
    out["log_return"] = np.log(close).diff()
    out = out.iloc[1:].copy()

    if out.empty:
        raise ValueError("Returns series is empty after differencing")
    if out["log_return"].isna().any():
        raise ValueError("Log-returns contain NaN")
    if not np.isfinite(out["log_return"]).all():
        raise ValueError("Log-returns contain non-finite values")

    return out


def apply_series_length_policy(
    returns_df: pd.DataFrame,
    target_length: int = 2000,
    min_length: int = 1500,
    slice_mode: str = "last_n",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply length policy to log-returns series."""
    if slice_mode != "last_n":
        raise ValueError(f"Unsupported slice_mode: {slice_mode}")

    returns_length = int(len(returns_df))
    if returns_length < min_length:
        return (
            returns_df.iloc[0:0].copy(),
            {
                "returns_length": returns_length,
                "selected_length": 0,
                "short_series": False,
                "status": "excluded_too_short",
            },
        )

    if returns_length >= target_length:
        selected = returns_df.tail(target_length).copy()
        return (
            selected,
            {
                "returns_length": returns_length,
                "selected_length": int(len(selected)),
                "short_series": False,
                "status": "valid_target",
            },
        )

    return (
        returns_df.copy(),
        {
            "returns_length": returns_length,
            "selected_length": returns_length,
            "short_series": True,
            "status": "valid_min_only",
        },
    )


def _read_standardized_series(row: pd.Series, cfg: Dict[str, Any]) -> pd.DataFrame:
    market = str(row["market"]).upper()
    source_path = row["source_path"]
    ticker = row["ticker"]

    if market == "RU":
        raw = read_ru_daily_file(source_path)
        return standardize_series_dataframe(raw, ticker=ticker, market="ru", config=cfg)

    if market == "US":
        raw = read_us_daily_file(source_path)
        return standardize_series_dataframe(raw, ticker=ticker, market="us", config=cfg)

    raise ValueError(f"Unsupported market: {market}")


def _build_length_distribution(series_meta_df: pd.DataFrame) -> pd.DataFrame:
    if series_meta_df.empty:
        return pd.DataFrame(columns=["length_bucket", "series_count"])

    returns_len = series_meta_df["returns_length"]
    bucket = pd.Series(index=returns_len.index, dtype="object")
    bucket.loc[returns_len < 1500] = "<1500"
    bucket.loc[(returns_len >= 1500) & (returns_len < 2000)] = "1500-1999"
    bucket.loc[returns_len >= 2000] = ">=2000"

    dist = (
        bucket.value_counts()
        .rename_axis("length_bucket")
        .reset_index(name="series_count")
        .sort_values("length_bucket", kind="stable")
        .reset_index(drop=True)
    )
    return dist


def _export_log_returns_report(
    excel_path: str | Path,
    run_id: str,
    config_path: str,
    log_returns_parquet_path: str,
    dataset_profiles_parquet_path: str,
    summary_df: pd.DataFrame,
    core_profile_df: pd.DataFrame,
    extended_profile_df: pd.DataFrame,
    length_distribution_df: pd.DataFrame,
) -> Path:
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "config_path", "value": config_path},
            {"key": "log_returns_parquet", "value": log_returns_parquet_path},
            {"key": "dataset_profiles_parquet", "value": dataset_profiles_parquet_path},
            {"key": "rules", "value": "target=2000, min=1500, slice_mode=last_n"},
            {
                "key": "dataset_profiles",
                "value": "core_balanced=all RU + random US of equal size; extended_us_heavy=all RU + random US up to 800",
            },
        ]
    )

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        core_profile_df[["series_id"]].drop_duplicates().to_excel(
            writer, sheet_name="dataset_core_balanced", index=False
        )
        extended_profile_df[["series_id"]].drop_duplicates().to_excel(
            writer, sheet_name="dataset_extended", index=False
        )
        length_distribution_df.to_excel(writer, sheet_name="length_distribution", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path


def run_log_returns_pipeline(config_path: str = "configs/data_inventory_v1.yaml") -> LogReturnsPipelineOutputs:
    """Execute stage-2 pipeline: log-returns computation and dataset profiles."""
    cfg = load_inventory_config(config_path)

    run_name = "log_returns_v1"
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    catalog_path = Path(artifacts_cfg["processed"]).resolve() / "series_catalog_v1.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog parquet not found: {catalog_path}")

    logger.info("Log-returns pipeline run started")
    logger.info("Config path: %s", cfg["meta"]["config_path"])
    logger.info("Input catalog path: %s", catalog_path)

    catalog_df = pd.read_parquet(catalog_path)
    input_series = int(len(catalog_df))

    policy = cfg.get("series_policy", {})
    target_length = int(policy.get("target_length", 2000))
    min_length = int(policy.get("min_length", 1500))
    slice_mode = str(policy.get("slice_mode", "last_n"))

    logger.info("Input series count: %d", input_series)
    logger.info(
        "Length policy: target=%d min=%d slice_mode=%s",
        target_length,
        min_length,
        slice_mode,
    )

    series_returns_rows: List[pd.DataFrame] = []
    series_meta_rows: List[Dict[str, Any]] = []
    errors = 0

    for _, row in catalog_df.iterrows():
        try:
            std_df = _read_standardized_series(row, cfg)
            original_length = int(len(std_df))

            returns_df = compute_log_returns(std_df)
            selected_df, policy_meta = apply_series_length_policy(
                returns_df,
                target_length=target_length,
                min_length=min_length,
                slice_mode=slice_mode,
            )

            series_id = str(row["series_id"])
            ticker = str(row["ticker"])
            market = str(row["market"]).upper()

            series_meta = {
                "series_id": series_id,
                "ticker": ticker,
                "market": market,
                "original_length": original_length,
                "returns_length": int(policy_meta["returns_length"]),
                "selected_length": int(policy_meta["selected_length"]),
                "short_series": bool(policy_meta["short_series"]),
                "status": str(policy_meta["status"]),
            }
            series_meta_rows.append(series_meta)

            if policy_meta["status"] == "excluded_too_short":
                continue

            use_cols = ["date", "log_return"]
            out_df = selected_df[use_cols].copy()
            out_df["series_id"] = series_id
            out_df["ticker"] = ticker
            out_df["market"] = market
            series_returns_rows.append(out_df)

        except Exception as exc:
            errors += 1
            logger.warning("Series processing failed [%s]: %s", row.get("series_id", "unknown"), exc)

    series_meta_df = pd.DataFrame(series_meta_rows)
    if series_meta_df.empty:
        series_meta_df = pd.DataFrame(
            columns=[
                "series_id",
                "ticker",
                "market",
                "original_length",
                "returns_length",
                "selected_length",
                "short_series",
                "status",
            ]
        )

    returns_df = (
        pd.concat(series_returns_rows, ignore_index=True)
        if series_returns_rows
        else pd.DataFrame(columns=["date", "log_return", "series_id", "ticker", "market"])
    )

    valid_series_df = series_meta_df[series_meta_df["status"].isin(["valid_target", "valid_min_only"])].copy()
    core_profile_df, extended_profile_df = build_dataset_profiles(valid_series_df, random_seed=42, extended_us_target=800)

    dataset_profiles_df = pd.concat([core_profile_df, extended_profile_df], ignore_index=True)
    if not dataset_profiles_df.empty:
        dataset_profiles_df = dataset_profiles_df[
            [
                "series_id",
                "ticker",
                "market",
                "dataset_profile",
                "original_length",
                "returns_length",
                "selected_length",
                "short_series",
                "status",
            ]
        ].sort_values(["dataset_profile", "market", "series_id"], kind="stable")

    core_ids = set(core_profile_df["series_id"].tolist())
    ext_ids = set(extended_profile_df["series_id"].tolist())

    core_returns = returns_df[returns_df["series_id"].isin(core_ids)].copy()
    core_returns["dataset_profile"] = "core_balanced"

    ext_returns = returns_df[returns_df["series_id"].isin(ext_ids)].copy()
    ext_returns["dataset_profile"] = "extended_us_heavy"

    log_returns_out_df = pd.concat([core_returns, ext_returns], ignore_index=True)
    if not log_returns_out_df.empty:
        log_returns_out_df = log_returns_out_df[
            ["series_id", "ticker", "market", "date", "log_return", "dataset_profile"]
        ].sort_values(["dataset_profile", "series_id", "date"], kind="stable")

    processed_dir = Path(artifacts_cfg["processed"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_returns_parquet_path = processed_dir / "log_returns_v1.parquet"
    dataset_profiles_parquet_path = processed_dir / "dataset_profiles_v1.parquet"
    excel_path = reports_dir / "log_returns_summary_v1.xlsx"

    log_returns_out_df.to_parquet(log_returns_parquet_path, index=False)
    dataset_profiles_df.to_parquet(dataset_profiles_parquet_path, index=False)

    excluded_too_short = int((series_meta_df["status"] == "excluded_too_short").sum())
    ge_target_count = int((series_meta_df["status"] == "valid_target").sum())
    short_series_count = int((series_meta_df["status"] == "valid_min_only").sum())
    valid_ru = int(((valid_series_df["market"] == "RU")).sum())
    valid_us = int(((valid_series_df["market"] == "US")).sum())

    core_size = int(len(core_profile_df))
    extended_size = int(len(extended_profile_df))

    summary_df = pd.DataFrame(
        [
            {
                "total_input_series": input_series,
                "ru_series_valid": valid_ru,
                "us_series_valid": valid_us,
                "ge_2000": ge_target_count,
                "between_1500_2000": short_series_count,
                "excluded_lt_1500": excluded_too_short,
                "processing_errors": errors,
            }
        ]
    )
    length_distribution_df = _build_length_distribution(series_meta_df)

    excel_out = _export_log_returns_report(
        excel_path=excel_path,
        run_id=run_id,
        config_path=cfg["meta"]["config_path"],
        log_returns_parquet_path=str(log_returns_parquet_path),
        dataset_profiles_parquet_path=str(dataset_profiles_parquet_path),
        summary_df=summary_df,
        core_profile_df=core_profile_df,
        extended_profile_df=extended_profile_df,
        length_distribution_df=length_distribution_df,
    )

    end_ts = datetime.now(timezone.utc)

    logger.info("Excluded (<%d): %d", min_length, excluded_too_short)
    logger.info("Series with returns >= %d: %d", target_length, ge_target_count)
    logger.info("Short series (%d..%d): %d", min_length, target_length - 1, short_series_count)
    logger.info("Valid RU series: %d", valid_ru)
    logger.info("Valid US series: %d", valid_us)
    logger.info("Core dataset size: %d", core_size)
    logger.info("Extended dataset size: %d", extended_size)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "log_returns_pipeline",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "inputs": {
            "series_catalog_parquet": str(catalog_path),
            "ru_daily": cfg["data_sources"]["ru_daily"],
            "us_daily": cfg["data_sources"]["us_daily"],
        },
        "outputs": {
            "log_returns_parquet": str(log_returns_parquet_path),
            "dataset_profiles_parquet": str(dataset_profiles_parquet_path),
            "excel_report": str(excel_out),
            "log": str(log_path),
        },
        "summary": {
            "input_series": input_series,
            "excluded_too_short": excluded_too_short,
            "ge_target_count": ge_target_count,
            "short_series_count": short_series_count,
            "valid_ru": valid_ru,
            "valid_us": valid_us,
            "core_size": core_size,
            "extended_size": extended_size,
            "processing_errors": errors,
        },
    }

    manifest_path = write_manifest(manifest, artifacts_cfg["manifests"], run_id)

    return LogReturnsPipelineOutputs(
        run_id=run_id,
        log_returns_parquet_path=str(log_returns_parquet_path),
        dataset_profiles_parquet_path=str(dataset_profiles_parquet_path),
        excel_path=str(excel_out),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        input_series=input_series,
        excluded_too_short=excluded_too_short,
        ge_target_count=ge_target_count,
        short_series_count=short_series_count,
        valid_ru=valid_ru,
        valid_us=valid_us,
        core_size=core_size,
        extended_size=extended_size,
    )
