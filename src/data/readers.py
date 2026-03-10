from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_ru_daily_file(path: str | Path) -> pd.DataFrame:
    """Read RU daily OHLCV file.

    Expected CSV columns:
    `datetime,open,high,low,close,volume`.

    Args:
        path: File path.

    Returns:
        Raw dataframe as read from CSV.
    """
    return pd.read_csv(path)


def read_us_daily_file(path: str | Path) -> pd.DataFrame:
    """Read US daily OHLCV file stored as .txt with CSV content.

    Expected columns:
    `Date,Open,High,Low,Close,Volume,OpenInt`.

    Args:
        path: File path.

    Returns:
        Raw dataframe as read from CSV.
    """
    return pd.read_csv(path)
