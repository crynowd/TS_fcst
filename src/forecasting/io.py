from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RAW_PREDICTION_COLUMNS = [
    "run_id",
    "model_name",
    "series_id",
    "ticker",
    "market",
    "horizon",
    "fold_id",
    "timestamp",
    "y_true",
    "y_pred",
    "status",
    "y_train_mean",
    "y_naive_baseline",
]

FOLD_METRICS_COLUMNS = [
    "run_id",
    "model_name",
    "series_id",
    "ticker",
    "market",
    "horizon",
    "fold_id",
    "n_train",
    "n_test",
    "fit_seconds",
    "predict_seconds",
    "status",
    "mae",
    "mse",
    "rmse",
    "mase",
    "mape",
    "smape",
    "huber",
    "directional_accuracy",
    "medae",
    "bias",
    "r2",
    "r2_oos",
]

TASK_AUDIT_COLUMNS = [
    "task_id",
    "run_id",
    "model_name",
    "series_id",
    "ticker",
    "market",
    "horizon",
    "fold_id",
    "status",
    "error_type",
    "fit_seconds",
    "predict_seconds",
    "notes",
]


def ensure_output_dirs(paths: dict[str, str]) -> None:
    for p in paths.values():
        Path(p).resolve().parent.mkdir(parents=True, exist_ok=True)


def ensure_table_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out[columns]


def save_tables(
    raw_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    series_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    outputs: dict[str, str],
) -> dict[str, str]:
    ensure_output_dirs(outputs)

    raw = ensure_table_schema(raw_df, RAW_PREDICTION_COLUMNS)
    fold = ensure_table_schema(fold_df, FOLD_METRICS_COLUMNS)
    audit = ensure_table_schema(audit_df, TASK_AUDIT_COLUMNS)

    raw_path = Path(outputs["raw_predictions"]).resolve()
    fold_path = Path(outputs["fold_metrics"]).resolve()
    series_path = Path(outputs["series_metrics"]).resolve()
    audit_path = Path(outputs["task_audit"]).resolve()

    raw.to_parquet(raw_path, index=False)
    fold.to_parquet(fold_path, index=False)
    series_df.to_parquet(series_path, index=False)
    audit.to_parquet(audit_path, index=False)

    return {
        "raw_predictions": str(raw_path),
        "fold_metrics": str(fold_path),
        "series_metrics": str(series_path),
        "task_audit": str(audit_path),
    }


def export_forecasting_summary_excel(
    excel_path: str | Path,
    summary_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    task_audit_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    series_metrics_df: pd.DataFrame,
    readme_df: pd.DataFrame,
) -> Path:
    out = Path(excel_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        model_registry_df.to_excel(writer, sheet_name="model_registry", index=False)
        task_audit_df.head(500).to_excel(writer, sheet_name="task_audit_preview", index=False)
        fold_metrics_df.head(1000).to_excel(writer, sheet_name="fold_metrics_preview", index=False)
        series_metrics_df.head(1000).to_excel(writer, sheet_name="series_metrics_preview", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)
    return out
