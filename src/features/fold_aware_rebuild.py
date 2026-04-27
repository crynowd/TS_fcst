from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from src.config.loader import load_feature_block_config
from src.features import block_a_dependence, block_b_spectrum, block_c_tails, block_d_chaos
from src.features.final_feature_sets import BASE_FEATURES_BY_BLOCK, CHAOS_FEATURES_BY_BLOCK


DEFAULT_OUTPUT_NAME = "final_train_only_features_by_fold.parquet"
MANIFEST_NAME = "feature_manifest.json"
SUMMARY_NAME = "feature_summary.csv"
ERRORS_NAME = "feature_errors.csv"
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"
REPORT_NAME = "fold_aware_feature_rebuild.md"

KEY_COLUMNS = ["series_id", "horizon", "fold_id"]
SEGMENT_COLUMNS = ["train_start", "train_end", "n_train"]
METADATA_COLUMNS = ["series_id", "horizon", "fold_id", "train_start", "train_end", "n_train"]

BLOCK_CONFIGS = {
    "A": "configs/features_block_A_v1.yaml",
    "B": "configs/features_block_B_v1.yaml",
    "C": "configs/features_block_C_v1.yaml",
    "D": "configs/features_block_D_v1.yaml",
}

BLOCK_MODULES = {
    "A": block_a_dependence,
    "B": block_b_spectrum,
    "C": block_c_tails,
    "D": block_d_chaos,
}

FEATURE_ENGINEERING_FILES = [
    "src/features/block_a_dependence.py",
    "src/features/block_b_spectrum.py",
    "src/features/block_c_tails.py",
    "src/features/block_d_chaos.py",
    "src/features/consolidation.py",
    "src/features/final_feature_sets.py",
    "src/cli/run_feature_block.py",
    "src/cli/run_feature_consolidation.py",
    "src/cli/run_final_feature_sets.py",
]


def final_feature_columns() -> list[str]:
    out: list[str] = []
    for block in ["A", "B", "C"]:
        out.extend(BASE_FEATURES_BY_BLOCK[block])
    out.extend(CHAOS_FEATURES_BY_BLOCK["D"])
    return out


def _parse_optional_ints(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(v) for v in values}


def _parse_optional_strings(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(v) for v in values}


def _git_status(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"git status unavailable: {exc}"
    return result.stdout.strip() or "(clean)"


def _gitignore_findings(project_root: Path) -> dict[str, Any]:
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return {"exists": False, "matched_rules": []}
    lines = [line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()]
    relevant = [
        line
        for line in lines
        if line
        and not line.startswith("#")
        and (
            line.startswith("artifacts/features")
            or line.startswith("artifacts/reports")
            or line.startswith("*.parquet")
            or line.startswith("*.csv")
            or line.startswith("*.xlsx")
            or line.startswith("*.xls")
            or line.startswith("*.log")
        )
    ]
    return {"exists": True, "matched_rules": relevant}


def _load_block_configs() -> dict[str, dict[str, Any]]:
    return {block: load_feature_block_config(path) for block, path in BLOCK_CONFIGS.items()}


def _metric_columns_by_block(configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {
        block: [col for col in module._feature_metric_columns(configs[block])]  # type: ignore[attr-defined]
        for block, module in BLOCK_MODULES.items()
    }


def _selected_features_by_block() -> dict[str, list[str]]:
    selected = {block: list(cols) for block, cols in BASE_FEATURES_BY_BLOCK.items()}
    selected.update({block: list(cols) for block, cols in CHAOS_FEATURES_BY_BLOCK.items()})
    return selected


def _compute_block_row(
    block: str,
    segment_df: pd.DataFrame,
    dataset_profile: str,
    cfg: dict[str, Any],
    metric_columns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module = BLOCK_MODULES[block]
    result = module._compute_single_series_features(segment_df, dataset_profile=dataset_profile, cfg=cfg)  # type: ignore[attr-defined]
    row = result.as_row(metric_columns)
    records: list[dict[str, Any]] = []
    flags = str(row.get("feature_warning_flags", "")).strip()
    status = str(row.get("feature_status", "success"))
    if status != "success" or flags:
        records.append(
            {
                "severity": "warning" if status != "failed" else "error",
                "block": block,
                "feature_status": status,
                "feature_warning_flags": flags,
            }
        )
    return row, records


def _select_splits(
    split_df: pd.DataFrame,
    series_ids: set[str] | None,
    horizons: set[int] | None,
    folds: set[int] | None,
    max_series: int | None,
) -> pd.DataFrame:
    scoped = split_df.copy()
    scoped["series_id"] = scoped["series_id"].astype(str)
    scoped["horizon"] = scoped["horizon"].astype(int)
    scoped["fold_id"] = scoped["fold_id"].astype(int)
    if series_ids is not None:
        scoped = scoped[scoped["series_id"].isin(series_ids)].copy()
    if horizons is not None:
        scoped = scoped[scoped["horizon"].isin(horizons)].copy()
    if folds is not None:
        scoped = scoped[scoped["fold_id"].isin(folds)].copy()
    if max_series is not None:
        keep = sorted(scoped["series_id"].drop_duplicates().tolist())[: int(max_series)]
        scoped = scoped[scoped["series_id"].isin(keep)].copy()
    return scoped.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _existing_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=KEY_COLUMNS)
    return {
        (str(r.series_id), int(r.horizon), int(r.fold_id))
        for r in df.itertuples(index=False)
    }


def _sanitize_nonfinite(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    out = df.copy()
    for col in feature_cols:
        numeric = pd.to_numeric(out[col], errors="coerce")
        inf_mask = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
        if inf_mask.any():
            for r in out.loc[inf_mask, KEY_COLUMNS].itertuples(index=False):
                rows.append(
                    {
                        "severity": "warning",
                        "series_id": str(r.series_id),
                        "horizon": int(r.horizon),
                        "fold_id": int(r.fold_id),
                        "block": "",
                        "feature": col,
                        "message": "nonfinite_inf_replaced_with_nan",
                    }
                )
            out.loc[inf_mask, col] = np.nan
    return out, pd.DataFrame(rows)


def _feature_summary(df: pd.DataFrame, feature_cols: list[str], split_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = int(len(df))
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        finite = np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))
        rows.append(
            {
                "feature": col,
                "rows": total_rows,
                "nan_count": int(vals.isna().sum()),
                "nan_rate": float(vals.isna().mean()) if total_rows else np.nan,
                "inf_count": int(np.isinf(vals.to_numpy(dtype=float, na_value=np.nan)).sum()),
                "finite_count": int(finite.sum()),
                "mean": float(vals[finite].mean()) if finite.any() else np.nan,
                "std": float(vals[finite].std()) if finite.sum() > 1 else np.nan,
                "min": float(vals[finite].min()) if finite.any() else np.nan,
                "median": float(vals[finite].median()) if finite.any() else np.nan,
                "max": float(vals[finite].max()) if finite.any() else np.nan,
            }
        )
    coverage = pd.DataFrame(
        [
            {"feature": "__coverage_rows__", "rows": total_rows, "nan_count": np.nan, "nan_rate": np.nan, "inf_count": np.nan},
            {"feature": "__coverage_series__", "rows": int(df["series_id"].nunique()) if total_rows else 0},
            {"feature": "__coverage_horizons__", "rows": int(df["horizon"].nunique()) if total_rows else 0},
            {"feature": "__coverage_folds__", "rows": int(df["fold_id"].nunique()) if total_rows else 0},
            {"feature": "__expected_rows_from_selected_splits__", "rows": int(len(split_df))},
        ]
    )
    return pd.concat([coverage, pd.DataFrame(rows)], ignore_index=True)


def _comparison_payload(new_df: pd.DataFrame, old_features_path: Path, feature_cols: list[str]) -> dict[str, Any]:
    if not old_features_path.exists():
        return {"old_features_exists": False}
    old_df = pd.read_parquet(old_features_path)
    old_feature_cols = [c for c in old_df.columns if c not in {"series_id", "ticker", "market", "dataset_profile"}]
    common = sorted(set(old_feature_cols) & set(feature_cols))
    old_only = sorted(set(old_feature_cols) - set(feature_cols))
    new_only = sorted(set(feature_cols) - set(old_feature_cols))

    nan_rates = {
        "old": {c: float(pd.to_numeric(old_df[c], errors="coerce").isna().mean()) for c in old_feature_cols},
        "new": {c: float(pd.to_numeric(new_df[c], errors="coerce").isna().mean()) for c in feature_cols if c in new_df.columns},
    }

    describe = {
        "old": old_df[common].describe().to_dict() if common else {},
        "new": new_df[common].describe().to_dict() if common else {},
    }

    fold_compare: dict[str, Any] = {}
    subset = new_df[(new_df["fold_id"] == 1) & (new_df["horizon"] == 1)].copy()
    if subset.empty:
        subset = new_df[(new_df["fold_id"] == 0) & (new_df["horizon"] == 1)].copy()
    if not subset.empty and common:
        merged = old_df[["series_id"] + common].merge(
            subset[["series_id"] + common],
            on="series_id",
            suffixes=("_full_series", "_train_only"),
        )
        deltas: dict[str, dict[str, float]] = {}
        for col in common:
            old_vals = pd.to_numeric(merged[f"{col}_full_series"], errors="coerce")
            new_vals = pd.to_numeric(merged[f"{col}_train_only"], errors="coerce")
            diff = new_vals - old_vals
            deltas[col] = {
                "paired_rows": int(diff.notna().sum()),
                "mean_abs_delta": float(diff.abs().mean()) if diff.notna().any() else np.nan,
                "median_abs_delta": float(diff.abs().median()) if diff.notna().any() else np.nan,
            }
        fold_compare = {"fold_filter": "fold_id=1,horizon=1", "rows": int(len(merged)), "deltas": deltas}

    return {
        "old_features_exists": True,
        "old_shape": list(old_df.shape),
        "new_shape": list(new_df.shape),
        "old_feature_columns": old_feature_cols,
        "new_feature_columns": feature_cols,
        "common_features": common,
        "old_only_features": old_only,
        "new_only_features": new_only,
        "nan_rates": nan_rates,
        "describe": describe,
        "full_vs_train_only_comparison": fold_compare,
    }


def _write_report(
    path: Path,
    *,
    selected_splits: pd.DataFrame,
    output_paths: dict[str, str],
    feature_cols: list[str],
    selected_by_block: dict[str, list[str]],
    errors_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison: dict[str, Any],
    project_root: Path,
    command_example: str,
) -> None:
    def _markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = [str(c) for c in df.columns]
        lines_local = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for record in df.to_dict(orient="records"):
            vals = []
            for col in cols:
                value = record.get(col, "")
                if isinstance(value, float):
                    vals.append("" if pd.isna(value) else f"{value:.6g}")
                else:
                    vals.append(str(value))
            lines_local.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines_local)

    expected_rows = int(len(selected_splits))
    success_rows = int(pd.read_parquet(output_paths["features_parquet"]).shape[0]) if Path(output_paths["features_parquet"]).exists() else 0
    short_threshold = 300
    short_rows = int((selected_splits["n_train"] < short_threshold).sum()) if "n_train" in selected_splits else 0
    lines: list[str] = []
    lines.append("# Fold-aware Train-only Feature Rebuild\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("## Feature engineering files found\n")
    for file in FEATURE_ENGINEERING_FILES:
        lines.append(f"- `{file}`")
    lines.append("\n## Recomputed features\n")
    for block, cols in selected_by_block.items():
        lines.append(f"- Block {block}: {', '.join(cols)}")
    lines.append("\n## Coverage\n")
    lines.append(f"- unique series_id: {selected_splits['series_id'].nunique()}")
    horizons = sorted(int(x) for x in selected_splits["horizon"].unique())
    folds = sorted(int(x) for x in selected_splits["fold_id"].unique())
    lines.append(f"- horizons: {horizons}")
    lines.append(f"- folds: {folds}")
    lines.append(f"- expected product: {selected_splits['series_id'].nunique()} x {len(horizons)} x {len(folds)} = {selected_splits['series_id'].nunique() * len(horizons) * len(folds)}")
    lines.append(f"- expected rows: {expected_rows}")
    lines.append(f"- output rows: {success_rows}")
    lines.append(f"- train segments below {short_threshold}: {short_rows}")
    lines.append("\n## NaN/Inf summary\n")
    display_summary = summary_df[summary_df["feature"].isin(feature_cols)].copy()
    if display_summary.empty:
        lines.append("- no feature summary available")
    else:
        top_nan = display_summary.sort_values(["nan_rate", "feature"], ascending=[False, True]).head(25)
        lines.append(_markdown_table(top_nan[["feature", "nan_count", "nan_rate", "inf_count"]]))
    lines.append("\n## Errors and warnings\n")
    if errors_df.empty:
        lines.append("- no feature computation warnings/errors recorded")
    else:
        lines.append(f"- warning/error rows: {len(errors_df)}")
        grouped = errors_df.groupby(["severity", "block"], dropna=False).size().reset_index(name="rows")
        lines.append(_markdown_table(grouped))
        flag_grouped = (
            errors_df.groupby(["severity", "block", "feature_warning_flags"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["rows", "severity", "block"], ascending=[False, True, True])
            .head(30)
        )
        lines.append("\nWarning/error flag breakdown:")
        lines.append(_markdown_table(flag_grouped))
    lines.append("\n## Old vs new artifact comparison\n")
    lines.append(f"- old artifact exists: {comparison.get('old_features_exists', False)}")
    if comparison.get("old_features_exists"):
        lines.append(f"- old shape: {comparison.get('old_shape')}")
        lines.append(f"- new shape: {comparison.get('new_shape')}")
        lines.append(f"- common features: {len(comparison.get('common_features', []))}")
        lines.append(f"- old-only features: {comparison.get('old_only_features', [])}")
        lines.append(f"- new-only features: {comparison.get('new_only_features', [])}")
        fold_cmp = comparison.get("full_vs_train_only_comparison", {})
        lines.append(f"- full vs train-only comparison rows: {fold_cmp.get('rows', 0)} ({fold_cmp.get('fold_filter', '')})")
        if fold_cmp.get("deltas"):
            delta_rows = []
            for feature, stats in fold_cmp["deltas"].items():
                delta_rows.append({"feature": feature, **stats})
            delta_df = pd.DataFrame(delta_rows).sort_values("mean_abs_delta", ascending=False).head(15)
            lines.append("\nLargest full-series vs train-only mean absolute deltas:")
            lines.append(_markdown_table(delta_df))
    lines.append("\n## Outputs\n")
    for key, value in output_paths.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("\n## Full rebuild command\n")
    lines.append(f"```bash\n{command_example}\n```")
    lines.append("\n## Git status\n")
    lines.append("```text")
    lines.append(_git_status(project_root))
    lines.append("```")
    lines.append("\n## .gitignore check\n")
    findings = _gitignore_findings(project_root)
    if findings["exists"]:
        lines.append("Relevant ignore rules:")
        for rule in findings["matched_rules"]:
            lines.append(f"- `{rule}`")
        lines.append("- New parquet/csv/json/yaml outputs under `artifacts/features/` or report heavy formats are ignored.")
        lines.append("- The markdown audit report is not ignored by the current `.gitignore` rules.")
    else:
        lines.append("- `.gitignore` not found")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_fold_aware_features(
    *,
    log_returns_path: str,
    split_metadata_path: str,
    old_features_path: str,
    output_dir: str,
    report_dir: str,
    max_series: int | None = None,
    series_ids: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
    overwrite: bool = False,
    resume: bool = True,
    project_root: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    log_returns = Path(log_returns_path).resolve()
    split_path = Path(split_metadata_path).resolve()
    old_path = Path(old_features_path).resolve()
    out_dir = Path(output_dir).resolve()
    rep_dir = Path(report_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    feature_path = out_dir / DEFAULT_OUTPUT_NAME
    manifest_path = out_dir / MANIFEST_NAME
    summary_path = out_dir / SUMMARY_NAME
    errors_path = out_dir / ERRORS_NAME
    config_snapshot_path = out_dir / CONFIG_SNAPSHOT_NAME
    report_path = rep_dir / REPORT_NAME

    if feature_path.exists() and overwrite:
        feature_path.unlink()
    elif feature_path.exists() and not resume:
        raise FileExistsError(f"Output exists and overwrite/resume disabled: {feature_path}")

    configs = _load_block_configs()
    dataset_profile_filter = str(configs["A"].get("dataset_profile", "")).strip()
    metric_cols_by_block = _metric_columns_by_block(configs)
    selected_by_block = _selected_features_by_block()
    feature_cols = final_feature_columns()

    returns_df = pd.read_parquet(log_returns)
    returns_df["series_id"] = returns_df["series_id"].astype(str)
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    if dataset_profile_filter and "dataset_profile" in returns_df.columns:
        returns_df = returns_df[returns_df["dataset_profile"].astype(str) == dataset_profile_filter].copy()
    split_df = pd.read_parquet(split_path)
    split_df["train_start"] = pd.to_datetime(split_df["train_start"])
    split_df["train_end"] = pd.to_datetime(split_df["train_end"])

    selected_splits = _select_splits(
        split_df,
        series_ids=_parse_optional_strings(series_ids),
        horizons=_parse_optional_ints(horizons),
        folds=_parse_optional_ints(folds),
        max_series=max_series,
    )
    selected_series = set(selected_splits["series_id"].astype(str).unique())
    returns_df = returns_df[returns_df["series_id"].isin(selected_series)].copy()
    by_series = {str(sid): sdf.sort_values("date", kind="stable") for sid, sdf in returns_df.groupby("series_id", sort=False)}

    existing = _existing_keys(feature_path) if resume and feature_path.exists() else set()
    rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    start_ts = datetime.now(timezone.utc)

    for split in selected_splits.itertuples(index=False):
        key = (str(split.series_id), int(split.horizon), int(split.fold_id))
        if key in existing:
            continue
        sdf = by_series.get(key[0], pd.DataFrame())
        train_start = pd.Timestamp(split.train_start)
        train_end = pd.Timestamp(split.train_end)
        segment = sdf[(sdf["date"] >= train_start) & (sdf["date"] <= train_end)].copy() if not sdf.empty else sdf.copy()
        base_row: dict[str, Any] = {
            "series_id": key[0],
            "horizon": key[1],
            "fold_id": key[2],
            "train_start": train_start,
            "train_end": train_end,
            "n_train": int(len(segment)),
        }
        feature_values: dict[str, Any] = {col: np.nan for col in feature_cols}
        status_parts: list[str] = []
        warning_parts: list[str] = []

        if segment.empty:
            error_rows.append(
                {
                    **{k: base_row[k] for k in METADATA_COLUMNS},
                    "severity": "error",
                    "block": "",
                    "feature": "",
                    "feature_status": "failed",
                    "feature_warning_flags": "empty_train_segment",
                    "message": "No rows found for series_id within train_start/train_end",
                }
            )
            rows.append({**base_row, **feature_values, "feature_status": "failed", "feature_warning_flags": "empty_train_segment"})
            continue

        dataset_profile = str(segment["dataset_profile"].iloc[0]) if "dataset_profile" in segment else ""
        for block in ["A", "B", "C", "D"]:
            try:
                block_row, block_records = _compute_block_row(
                    block=block,
                    segment_df=segment,
                    dataset_profile=dataset_profile,
                    cfg=configs[block],
                    metric_columns=metric_cols_by_block[block],
                )
            except Exception as exc:
                status_parts.append(f"{block}:failed")
                error_rows.append(
                    {
                        **{k: base_row[k] for k in METADATA_COLUMNS},
                        "severity": "error",
                        "block": block,
                        "feature": "",
                        "feature_status": "failed",
                        "feature_warning_flags": "series_processing_error",
                        "message": repr(exc),
                    }
                )
                continue
            status_parts.append(f"{block}:{block_row.get('feature_status', 'success')}")
            flags = str(block_row.get("feature_warning_flags", "")).strip()
            if flags:
                warning_parts.append(f"{block}:{flags}")
            for record in block_records:
                error_rows.append(
                    {
                        **{k: base_row[k] for k in METADATA_COLUMNS},
                        "feature": "",
                        "message": "",
                        **record,
                    }
                )
            for col in selected_by_block[block]:
                feature_values[col] = block_row.get(col, np.nan)

        rows.append(
            {
                **base_row,
                **feature_values,
                "feature_status": ";".join(status_parts),
                "feature_warning_flags": "|".join(warning_parts),
            }
        )

    new_rows_df = pd.DataFrame(rows)
    if feature_path.exists() and resume and not overwrite:
        existing_df = pd.read_parquet(feature_path)
        combined = pd.concat([existing_df, new_rows_df], ignore_index=True)
        combined = combined.drop_duplicates(KEY_COLUMNS, keep="last")
    else:
        combined = new_rows_df
    if combined.empty:
        combined = pd.DataFrame(columns=METADATA_COLUMNS + feature_cols + ["feature_status", "feature_warning_flags"])
    combined = combined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    combined, inf_errors = _sanitize_nonfinite(combined, feature_cols)
    combined.to_parquet(feature_path, index=False)

    errors_df = pd.DataFrame(error_rows)
    if not inf_errors.empty:
        errors_df = pd.concat([errors_df, inf_errors], ignore_index=True)
    if errors_path.exists() and resume and not overwrite:
        existing_errors = pd.read_csv(errors_path)
        if not existing_errors.empty:
            errors_df = pd.concat([existing_errors, errors_df], ignore_index=True)
            dedupe_cols = [c for c in errors_df.columns if c in METADATA_COLUMNS + ["severity", "block", "feature", "feature_status", "feature_warning_flags", "message"]]
            errors_df = errors_df.drop_duplicates(dedupe_cols, keep="last")
    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=METADATA_COLUMNS + ["severity", "block", "feature", "feature_status", "feature_warning_flags", "message"]
        )
    errors_df.to_csv(errors_path, index=False)

    summary_df = _feature_summary(combined, feature_cols, selected_splits)
    summary_df.to_csv(summary_path, index=False)

    comparison = _comparison_payload(combined, old_path, feature_cols)
    end_ts = datetime.now(timezone.utc)
    output_paths = {
        "features_parquet": str(feature_path),
        "feature_manifest_json": str(manifest_path),
        "feature_summary_csv": str(summary_path),
        "feature_errors_csv": str(errors_path),
        "config_snapshot_yaml": str(config_snapshot_path),
        "audit_report_md": str(report_path),
    }

    config_snapshot = {
        "stage": "fold_aware_feature_rebuild_v2",
        "inputs": {
            "log_returns": str(log_returns),
            "split_metadata": str(split_path),
            "old_features": str(old_path),
            "dataset_profile_filter": dataset_profile_filter,
        },
        "outputs": output_paths,
        "filters": {
            "max_series": max_series,
            "series_ids": sorted(_parse_optional_strings(series_ids) or []),
            "horizons": sorted(_parse_optional_ints(horizons) or []),
            "folds": sorted(_parse_optional_ints(folds) or []),
            "overwrite": bool(overwrite),
            "resume": bool(resume),
        },
        "block_configs": BLOCK_CONFIGS,
        "dataset_profile_filter": dataset_profile_filter,
        "selected_features_by_block": selected_by_block,
    }
    config_snapshot_path.write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")

    manifest = {
        "run_id": f"fold_aware_feature_rebuild_v2_{start_ts.strftime('%Y%m%dT%H%M%SZ')}",
        "stage": "fold_aware_feature_rebuild_v2",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "feature_engineering_files": FEATURE_ENGINEERING_FILES,
        "input_sources": config_snapshot["inputs"],
        "outputs": output_paths,
        "selected_features": feature_cols,
        "selected_features_by_block": selected_by_block,
        "coverage": {
            "selected_split_rows": int(len(selected_splits)),
            "output_rows": int(len(combined)),
            "series_count": int(selected_splits["series_id"].nunique()),
            "horizons": sorted(int(x) for x in selected_splits["horizon"].unique()),
            "folds": sorted(int(x) for x in selected_splits["fold_id"].unique()),
            "new_rows_computed": int(len(new_rows_df)),
            "resume_existing_keys": int(len(existing)),
        },
        "comparison": {
            "old_features_exists": comparison.get("old_features_exists", False),
            "old_shape": comparison.get("old_shape"),
            "new_shape": comparison.get("new_shape"),
            "common_feature_count": len(comparison.get("common_features", [])),
            "old_only_features": comparison.get("old_only_features", []),
            "new_only_features": comparison.get("new_only_features", []),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    command_example = (
        "python -m src.cli.run_fold_aware_feature_rebuild "
        "--log-returns artifacts/processed/log_returns_v1.parquet "
        "--split-metadata artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet "
        "--old-features artifacts/features/final_clustering_features_with_chaos_v1.parquet "
        "--output-dir artifacts/features/fold_aware_features_v2 "
        "--report-dir artifacts/reports/forecasting_audit_v2 "
        "--overwrite"
    )
    _write_report(
        report_path,
        selected_splits=selected_splits,
        output_paths=output_paths,
        feature_cols=feature_cols,
        selected_by_block=selected_by_block,
        errors_df=errors_df,
        summary_df=summary_df,
        comparison=comparison,
        project_root=root,
        command_example=command_example,
    )

    return {
        "outputs": output_paths,
        "coverage": manifest["coverage"],
        "errors": int(len(errors_df)),
        "feature_count": int(len(feature_cols)),
        "gitignore": _gitignore_findings(root),
    }
