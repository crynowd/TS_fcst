from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SELECTED_SERIES_COLUMNS = ["series_id", "source_path", "file_name"]


@dataclass
class ExternalSeries:
    series_id: str
    source_path: str
    file_name: str


def list_external_csv_files(external_data_dir: str | Path, file_pattern: str = "*.csv") -> list[Path]:
    root = Path(external_data_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"External data directory not found: {root}")
    files = [p.resolve() for p in root.glob(file_pattern) if p.is_file()]
    return sorted(files, key=lambda p: p.name.lower())


def sample_external_series(
    files: list[Path],
    sample_size: int,
    random_seed: int,
) -> list[ExternalSeries]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(files) < sample_size:
        raise ValueError(f"Not enough files for sampling: requested={sample_size}, available={len(files)}")

    rng = np.random.default_rng(int(random_seed))
    idx = rng.choice(len(files), size=sample_size, replace=False)
    selected_paths = [files[int(i)] for i in sorted(idx.tolist())]

    return [
        ExternalSeries(
            series_id=p.stem,
            source_path=str(p),
            file_name=p.name,
        )
        for p in selected_paths
    ]


def selected_series_to_frame(selected: list[ExternalSeries]) -> pd.DataFrame:
    rows = [
        {
            "series_id": s.series_id,
            "source_path": s.source_path,
            "file_name": s.file_name,
        }
        for s in selected
    ]
    out = pd.DataFrame(rows)
    for col in SELECTED_SERIES_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[SELECTED_SERIES_COLUMNS]


def save_selected_series_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def load_external_log_returns_series(
    source_path: str | Path,
    date_column: str = "Date",
    close_column: str = "Close",
    min_history: int = 120,
) -> pd.DataFrame:
    path = Path(source_path).resolve()
    df = pd.read_csv(path)
    if date_column not in df.columns:
        raise ValueError(f"{path.name}: missing date column '{date_column}'")
    if close_column not in df.columns:
        raise ValueError(f"{path.name}: missing close column '{close_column}'")

    out = df[[date_column, close_column]].copy()
    out = out.rename(columns={date_column: "date", close_column: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    out = out[out["close"] > 0.0]

    if len(out) < int(min_history):
        raise ValueError(f"{path.name}: history too short after cleaning ({len(out)} rows)")

    out["log_return"] = np.log(out["close"]).diff()
    out = out.dropna(subset=["log_return"]).reset_index(drop=True)
    out = out[["date", "log_return"]]
    if out.empty:
        raise ValueError(f"{path.name}: no usable log returns after transformation")
    return out
