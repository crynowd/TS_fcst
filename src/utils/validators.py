from __future__ import annotations

from typing import Dict

import pandas as pd


class SchemaError(Exception):
    """Raised when source dataframe does not match required schema."""


REQUIRED_CANONICAL_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def ensure_columns_exist(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that required columns are present in dataframe.

    Args:
        df: Input dataframe.
        required_columns: Column names that must exist.

    Raises:
        SchemaError: If at least one required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def evaluate_series_quality(df: pd.DataFrame, policy: Dict[str, int | str]) -> Dict[str, object]:
    """Compute quality metrics and status flags for one standardized series.

    Args:
        df: Standardized dataframe with canonical columns.
        policy: Series policy from config.

    Returns:
        Dictionary with quality stats and status fields.
    """
    n_rows = int(len(df))
    date_series = df["date"]
    sorted_dates = date_series.is_monotonic_increasing
    duplicate_dates = int(date_series.duplicated().sum())

    missing_counts = {
        "n_missing_open": int(df["open"].isna().sum()),
        "n_missing_high": int(df["high"].isna().sum()),
        "n_missing_low": int(df["low"].isna().sum()),
        "n_missing_close": int(df["close"].isna().sum()),
        "n_missing_volume": int(df["volume"].isna().sum()),
    }

    has_nonpositive_close = bool((df["close"] <= 0).fillna(False).any())
    can_compute_log_returns = bool(n_rows >= 2 and (df["close"] > 0).all())

    target_length = int(policy.get("target_length", 2000))
    min_length = int(policy.get("min_length", 1500))
    eligible_target = n_rows >= target_length
    eligible_min = n_rows >= min_length

    if has_nonpositive_close:
        status = "invalid"
        status_reason = "nonpositive_close"
    elif eligible_target and can_compute_log_returns:
        status = "valid_target"
        status_reason = "ok"
    elif eligible_min and can_compute_log_returns:
        status = "valid_min_only"
        status_reason = "ok"
    else:
        status = "invalid"
        status_reason = "too_short" if not eligible_min else "validation_failed"

    return {
        "n_rows_after_standardization": n_rows,
        "min_date": None if n_rows == 0 else date_series.min(),
        "max_date": None if n_rows == 0 else date_series.max(),
        "is_sorted": bool(sorted_dates),
        "n_duplicate_dates": duplicate_dates,
        "has_nonpositive_close": has_nonpositive_close,
        "can_compute_log_returns": can_compute_log_returns,
        "eligible_target_2000": bool(eligible_target),
        "eligible_min_1500": bool(eligible_min),
        "proposed_slice_mode": str(policy.get("slice_mode", "last_n")),
        "status": status,
        "status_reason": status_reason,
        **missing_counts,
    }
