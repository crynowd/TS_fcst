from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _build_summary_table(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row summary table for Excel `summary` sheet."""
    n_total = int(len(catalog_df))
    n_ru = int((catalog_df["market"] == "RU").sum())
    n_us = int((catalog_df["market"] == "US").sum())

    lengths = catalog_df["n_rows_after_standardization"].fillna(0)
    length_range = (
        f"{int(lengths.min())}..{int(lengths.max())}" if n_total > 0 else "0..0"
    )

    return pd.DataFrame(
        [
            {
                "total_files": n_total,
                "ru_files": n_ru,
                "us_files": n_us,
                "valid_target": int((catalog_df["status"] == "valid_target").sum()),
                "valid_min_only": int((catalog_df["status"] == "valid_min_only").sum()),
                "invalid": int((catalog_df["status"] == "invalid").sum()),
                "series_with_nonpositive_close": int(catalog_df["has_nonpositive_close"].fillna(False).sum()),
                "series_with_duplicate_dates": int((catalog_df["n_duplicate_dates"].fillna(0) > 0).sum()),
                "length_range": length_range,
            }
        ]
    )


def export_series_catalog_excel(
    catalog_df: pd.DataFrame,
    excel_path: str | Path,
    run_metadata: Dict[str, str],
    source_paths: Iterable[str],
) -> Path:
    """Export catalog and run metadata to a multi-sheet Excel report.

    Args:
        catalog_df: Full series catalog dataframe.
        excel_path: Target `.xlsx` path.
        run_metadata: Metadata for readme sheet.
        source_paths: Input source directories.

    Returns:
        Absolute output path.
    """
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = _build_summary_table(catalog_df)
    invalid_df = catalog_df[catalog_df["status"] == "invalid"].copy()

    readme_rows = [
        {"key": "run_id", "value": run_metadata.get("run_id", "")},
        {"key": "stage", "value": run_metadata.get("stage", "")},
        {"key": "timestamp", "value": run_metadata.get("timestamp", "")},
        {"key": "git_commit", "value": run_metadata.get("git_commit", "")},
        {"key": "config_path", "value": run_metadata.get("config_path", "")},
        {"key": "parquet_path", "value": run_metadata.get("parquet_path", "")},
        {"key": "source_paths", "value": "; ".join(source_paths)},
        {
            "key": "sheets_description",
            "value": "summary=run summary; series_catalog=full catalog; invalid_series=invalid only",
        },
    ]
    readme_df = pd.DataFrame(readme_rows)

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        catalog_df.to_excel(writer, sheet_name="series_catalog", index=False)
        invalid_df.to_excel(writer, sheet_name="invalid_series", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path
