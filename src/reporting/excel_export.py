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


def export_features_block_a_excel(
    features_df: pd.DataFrame,
    summary: Dict[str, object],
    warnings_df: pd.DataFrame,
    excel_path: str | Path,
    run_id: str,
    dataset_profile: str,
    input_paths: Dict[str, str],
    output_parquet: str,
) -> Path:
    """Export feature block A artifacts into multi-sheet Excel report."""
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "dataset_profile": dataset_profile,
                "series_total": int(summary.get("series_total", 0)),
                "series_successful": int(summary.get("series_successful", 0)),
                "series_with_warnings": int(summary.get("series_with_warnings", 0)),
                "nan_hurst_rs": int(summary.get("nan_hurst_rs", 0)),
                "nan_hurst_dfa": int(summary.get("nan_hurst_dfa", 0)),
                "pvalues_outside_0_1": int(summary.get("pvalues_outside_0_1", 0)),
            }
        ]
    )

    range_metrics = [
        "hurst_rs",
        "hurst_dfa",
        "iacf_abs_1_20",
        "iacf_absret_1_20",
        "vr_q2",
        "vr_q5",
        "vr_q10",
    ]
    rows = []
    for col in range_metrics:
        if col not in features_df.columns:
            continue
        vals = pd.to_numeric(features_df[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "min": float(vals.min()) if vals.notna().any() else None,
                "max": float(vals.max()) if vals.notna().any() else None,
                "mean": float(vals.mean()) if vals.notna().any() else None,
                "median": float(vals.median()) if vals.notna().any() else None,
            }
        )
    ranges_df = pd.DataFrame(rows)

    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": "feature_block_A"},
            {"key": "dataset_profile", "value": dataset_profile},
            {"key": "input_log_returns_parquet", "value": input_paths.get("log_returns_parquet", "")},
            {"key": "input_dataset_profiles_parquet", "value": input_paths.get("dataset_profiles_parquet", "")},
            {"key": "output_features_parquet", "value": output_parquet},
            {
                "key": "metrics",
                "value": "Hurst RS/DFA, ACF and integrated ACF, Ljung-Box, Variance Ratio",
            },
        ]
    )

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        features_df.to_excel(writer, sheet_name="features_block_A", index=False)
        warnings_df.to_excel(writer, sheet_name="warnings", index=False)
        ranges_df.to_excel(writer, sheet_name="ranges", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path
