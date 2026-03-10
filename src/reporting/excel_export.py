from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

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


def export_features_block_b_excel(
    features_df: pd.DataFrame,
    summary: Dict[str, object],
    warnings_df: pd.DataFrame,
    excel_path: str | Path,
    run_id: str,
    dataset_profile: str,
    input_paths: Dict[str, str],
    output_parquet: str,
    config_params: Dict[str, Any],
) -> Path:
    """Export feature block B artifacts into a multi-sheet Excel report."""
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "spectral_slope_beta",
        "spectral_entropy",
        "spectral_flatness",
        "noise_fn",
        "permutation_entropy",
        "lz_complexity",
        "sample_entropy",
    ]

    summary_row: Dict[str, object] = {
        "dataset_profile": dataset_profile,
        "series_total": int(summary.get("series_total", 0)),
        "series_successful": int(summary.get("series_successful", 0)),
        "series_with_warnings": int(summary.get("series_with_warnings", 0)),
        "spectral_entropy_lt_0": int(summary.get("spectral_entropy_lt_0", 0)),
        "spectral_flatness_lt_0": int(summary.get("spectral_flatness_lt_0", 0)),
    }
    for metric in metric_columns:
        summary_row[f"nan_{metric}"] = int(summary.get(f"nan_{metric}", 0))
    if "spectral_entropy_outside_0_1" in summary:
        summary_row["spectral_entropy_outside_0_1"] = int(summary.get("spectral_entropy_outside_0_1", 0))
    if "permutation_entropy_outside_0_1" in summary:
        summary_row["permutation_entropy_outside_0_1"] = int(summary.get("permutation_entropy_outside_0_1", 0))

    summary_df = pd.DataFrame([summary_row])

    rows = []
    for col in metric_columns:
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
            {"key": "stage", "value": "feature_block_B"},
            {"key": "dataset_profile", "value": dataset_profile},
            {"key": "input_log_returns_parquet", "value": input_paths.get("log_returns_parquet", "")},
            {"key": "input_dataset_profiles_parquet", "value": input_paths.get("dataset_profiles_parquet", "")},
            {"key": "output_features_parquet", "value": output_parquet},
            {
                "key": "metrics",
                "value": "Spectral slope/entropy/flatness, NoiseFN, permutation entropy, Lempel-Ziv complexity, sample entropy",
            },
            {"key": "spectral_params", "value": str(config_params.get("spectral", {}))},
            {"key": "permutation_entropy_params", "value": str(config_params.get("permutation_entropy", {}))},
            {"key": "lz_complexity_params", "value": str(config_params.get("lz_complexity", {}))},
            {"key": "sample_entropy_params", "value": str(config_params.get("sample_entropy", {}))},
        ]
    )

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        features_df.to_excel(writer, sheet_name="features_block_B", index=False)
        warnings_df.to_excel(writer, sheet_name="warnings", index=False)
        ranges_df.to_excel(writer, sheet_name="ranges", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path


def export_features_block_c_excel(
    features_df: pd.DataFrame,
    summary: Dict[str, object],
    warnings_df: pd.DataFrame,
    excel_path: str | Path,
    run_id: str,
    dataset_profile: str,
    input_paths: Dict[str, str],
    output_parquet: str,
    config_params: Dict[str, Any],
) -> Path:
    """Export feature block C artifacts into a multi-sheet Excel report."""
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "kurtosis",
        "robust_kurtosis",
        "tail_ratio_symmetric",
        "tail_ratio_upper",
        "tail_ratio_lower",
        "hill_tail_index",
    ]

    summary_row: Dict[str, object] = {
        "dataset_profile": dataset_profile,
        "series_total": int(summary.get("series_total", 0)),
        "series_successful": int(summary.get("series_successful", 0)),
        "series_with_warnings": int(summary.get("series_with_warnings", 0)),
        "kurtosis_lt_0": int(summary.get("kurtosis_lt_0", 0)),
        "tail_ratio_symmetric_negative": int(summary.get("tail_ratio_symmetric_negative", 0)),
        "tail_ratio_upper_negative": int(summary.get("tail_ratio_upper_negative", 0)),
        "tail_ratio_lower_negative": int(summary.get("tail_ratio_lower_negative", 0)),
        "hill_tail_index_gt_10": int(summary.get("hill_tail_index_gt_10", 0)),
    }
    for metric in metric_columns:
        summary_row[f"nan_{metric}"] = int(summary.get(f"nan_{metric}", 0))

    summary_df = pd.DataFrame([summary_row])

    rows = []
    for col in metric_columns:
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
            {"key": "stage", "value": "feature_block_C"},
            {"key": "dataset_profile", "value": dataset_profile},
            {"key": "input_log_returns_parquet", "value": input_paths.get("log_returns_parquet", "")},
            {"key": "input_dataset_profiles_parquet", "value": input_paths.get("dataset_profiles_parquet", "")},
            {"key": "output_features_parquet", "value": output_parquet},
            {
                "key": "metrics",
                "value": "Kurtosis (Fisher), Moors robust kurtosis, quantile tail ratios, reserve Hill tail index",
            },
            {"key": "quantiles", "value": str(config_params.get("quantiles", []))},
            {"key": "hill_k_fraction", "value": str(config_params.get("hill_k_fraction", 0.05))},
            {"key": "use_hill", "value": str(config_params.get("use_hill", True))},
        ]
    )

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        features_df.to_excel(writer, sheet_name="features_block_C", index=False)
        warnings_df.to_excel(writer, sheet_name="warnings", index=False)
        ranges_df.to_excel(writer, sheet_name="ranges", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path


def export_features_block_d_excel(
    features_df: pd.DataFrame,
    summary: Dict[str, object],
    warnings_df: pd.DataFrame,
    excel_path: str | Path,
    run_id: str,
    dataset_profile: str,
    input_paths: Dict[str, str],
    output_parquet: str,
    config_params: Dict[str, Any],
) -> Path:
    """Export feature block D artifacts into a multi-sheet Excel report."""
    out_path = Path(excel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tau_breakdown = summary.get("tau_selection_breakdown", {})
    summary_row: Dict[str, object] = {
        "dataset_profile": dataset_profile,
        "series_total": int(summary.get("series_total", 0)),
        "series_successful": int(summary.get("series_successful", 0)),
        "series_with_warnings": int(summary.get("series_with_warnings", 0)),
        "nan_embedding_dimension": int(summary.get("nan_embedding_dimension", 0)),
        "nan_correlation_dimension": int(summary.get("nan_correlation_dimension", 0)),
        "nan_largest_lyapunov_exponent": int(summary.get("nan_largest_lyapunov_exponent", 0)),
        "nan_lyapunov_time": int(summary.get("nan_lyapunov_time", 0)),
        "tau_ami": int(tau_breakdown.get("ami", 0)),
        "tau_acf_zero": int(tau_breakdown.get("acf_zero", 0)),
        "tau_acf_einv": int(tau_breakdown.get("acf_einv", 0)),
        "tau_fallback_default": int(tau_breakdown.get("fallback_default", 0)),
        "exponent_nonpositive": int(summary.get("exponent_nonpositive", 0)),
        "correlation_dimension_negative": int(summary.get("correlation_dimension_negative", 0)),
        "embedding_dimension_out_of_bounds": int(summary.get("embedding_dimension_out_of_bounds", 0)),
    }
    summary_df = pd.DataFrame([summary_row])

    range_metrics = [
        "selected_delay_tau",
        "embedding_dimension",
        "correlation_dimension",
        "largest_lyapunov_exponent",
        "lyapunov_time",
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
            {"key": "stage", "value": "feature_block_D"},
            {"key": "dataset_profile", "value": dataset_profile},
            {"key": "input_log_returns_parquet", "value": input_paths.get("log_returns_parquet", "")},
            {"key": "input_dataset_profiles_parquet", "value": input_paths.get("dataset_profiles_parquet", "")},
            {"key": "output_features_parquet", "value": output_parquet},
            {
                "key": "metrics",
                "value": "Embedding dimension (FNN), correlation dimension, largest Lyapunov exponent (Rosenstein), Lyapunov time",
            },
            {"key": "delay_selection", "value": str(config_params.get("delay_selection", {}))},
            {"key": "embedding", "value": str(config_params.get("embedding", {}))},
            {"key": "correlation_dimension", "value": str(config_params.get("correlation_dimension", {}))},
            {"key": "lyapunov", "value": str(config_params.get("lyapunov", {}))},
            {"key": "minimum_series_length", "value": str(config_params.get("minimum_series_length", ""))},
        ]
    )

    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        features_df.to_excel(writer, sheet_name="features_block_D", index=False)
        warnings_df.to_excel(writer, sheet_name="warnings", index=False)
        ranges_df.to_excel(writer, sheet_name="ranges", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    return out_path
