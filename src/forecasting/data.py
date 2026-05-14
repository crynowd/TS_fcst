from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SelectedSeries:
    series_id: str
    ticker: str
    market: str


REQUIRED_COLUMNS = {"series_id", "ticker", "market", "date", "log_return", "dataset_profile"}


def load_log_returns_table(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path).resolve()
    df = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"log-returns parquet missing columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out = out.sort_values(["series_id", "date"], kind="stable")
    return out


def select_series(
    log_returns_df: pd.DataFrame,
    dataset_profile: str,
    max_series: int | None = None,
    series_selection_mode: str = "first_n",
    series_ids: list[str] | None = None,
) -> list[SelectedSeries]:
    scoped = log_returns_df[log_returns_df["dataset_profile"] == dataset_profile].copy()
    if series_ids:
        scope_set = {str(s) for s in series_ids}
        scoped = scoped[scoped["series_id"].isin(scope_set)]

    uniq = (
        scoped[["series_id", "ticker", "market"]]
        .drop_duplicates(subset=["series_id"], keep="first")
        .reset_index(drop=True)
    )

    if series_selection_mode == "first_n":
        pass
    elif series_selection_mode == "sorted":
        uniq = uniq.sort_values(["series_id"], kind="stable").reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported series_selection_mode: {series_selection_mode}")

    if max_series is not None and max_series > 0:
        uniq = uniq.head(int(max_series))

    return [
        SelectedSeries(series_id=str(r.series_id), ticker=str(r.ticker), market=str(r.market))
        for r in uniq.itertuples(index=False)
    ]


def build_series_lookup(
    log_returns_df: pd.DataFrame,
    selected: list[SelectedSeries],
    dataset_profile: str | None = None,
) -> dict[str, pd.DataFrame]:
    selected_ids = {s.series_id for s in selected}
    subset = log_returns_df[log_returns_df["series_id"].isin(selected_ids)].copy()
    if dataset_profile:
        subset = subset[subset["dataset_profile"] == dataset_profile].copy()
    grouped: dict[str, pd.DataFrame] = {}
    for sid, sdf in subset.groupby("series_id", sort=False):
        grouped[str(sid)] = sdf.sort_values("date", kind="stable").reset_index(drop=True)
    return grouped
