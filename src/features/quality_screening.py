from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from src.config.loader import load_feature_screening_config
from src.features.master_table import build_master_feature_table, load_feature_block_tables
from src.reporting.excel_export import export_feature_screening_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class FeatureScreeningOutputs:
    run_id: str
    master_parquet_path: str
    excel_path: str
    correlation_pairs_csv_path: str
    shortlist_csv_path: str
    log_path: str
    manifest_path: str
    dataset_profile: str
    rows: int
    columns: int
    n_features: int
    low_variance_count: int
    high_missing_count: int
    high_correlation_pairs: int
    screening_status_counts: Dict[str, int]


def _is_warning_heavy(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value >= threshold)


def _feature_dtype(series: pd.Series) -> str:
    return str(series.dtype)


def _compute_iqr(values: pd.Series) -> float:
    q75 = values.quantile(0.75)
    q25 = values.quantile(0.25)
    if pd.isna(q75) or pd.isna(q25):
        return np.nan
    return float(q75 - q25)


def compute_feature_quality_metrics(
    master_df: pd.DataFrame,
    feature_to_block: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    n_rows = int(len(master_df))
    rows: List[Dict[str, Any]] = []

    missing_thr = float(cfg["high_missing_threshold"])
    low_variance_thr = float(cfg["low_variance_threshold"])
    near_const_unique_thr = int(cfg["near_constant_unique_threshold"])

    for feature in feature_to_block:
        series = master_df[feature]
        nonnull = series.dropna()
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_nonnull = numeric.dropna().astype(float)

        missing_count = int(series.isna().sum())
        missing_rate = float(missing_count / n_rows) if n_rows else np.nan
        n_unique = int(nonnull.nunique(dropna=True))

        block = feature_to_block[feature]
        warning_col = f"block_{block}_warning_flags"
        warning_coverage = (
            float(master_df[warning_col].astype(str).str.strip().ne("").mean()) if warning_col in master_df.columns else np.nan
        )

        if numeric_nonnull.empty:
            mean = std = min_v = max_v = median = iqr = cv = np.nan
        else:
            mean = float(numeric_nonnull.mean())
            std = float(numeric_nonnull.std(ddof=1)) if len(numeric_nonnull) > 1 else 0.0
            min_v = float(numeric_nonnull.min())
            max_v = float(numeric_nonnull.max())
            median = float(numeric_nonnull.median())
            iqr = _compute_iqr(numeric_nonnull)
            cv = float(std / abs(mean)) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan

        near_constant = bool(n_unique <= near_const_unique_thr)
        low_variance = bool(np.isfinite(std) and std <= low_variance_thr)
        high_missing = bool(np.isfinite(missing_rate) and missing_rate >= missing_thr)

        rows.append(
            {
                "feature_name": feature,
                "source_block": block,
                "dtype": _feature_dtype(series),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "median": median,
                "iqr": iqr,
                "coefficient_of_variation": cv,
                "warning_coverage": warning_coverage,
                "near_constant_flag": near_constant,
                "low_variance_flag": low_variance,
                "high_missing_flag": high_missing,
            }
        )

    return pd.DataFrame(rows).sort_values(["source_block", "feature_name"], kind="stable").reset_index(drop=True)


def compute_high_correlation_pairs(
    master_df: pd.DataFrame,
    feature_to_block: Mapping[str, str],
    threshold: float,
) -> pd.DataFrame:
    numeric_features = [f for f in feature_to_block if pd.api.types.is_numeric_dtype(master_df[f])]
    if len(numeric_features) < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "pearson_corr",
                "spearman_corr",
                "suggested_relation",
            ]
        )

    numeric_df = master_df[numeric_features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric_df.corr(method="pearson", min_periods=3)
    spearman = numeric_df.corr(method="spearman", min_periods=3)

    rows: List[Dict[str, Any]] = []
    for i, feature_1 in enumerate(numeric_features):
        for feature_2 in numeric_features[i + 1 :]:
            p = float(pearson.loc[feature_1, feature_2]) if pd.notna(pearson.loc[feature_1, feature_2]) else np.nan
            s = float(spearman.loc[feature_1, feature_2]) if pd.notna(spearman.loc[feature_1, feature_2]) else np.nan
            if not (np.isfinite(p) or np.isfinite(s)):
                continue
            if max(abs(p) if np.isfinite(p) else 0.0, abs(s) if np.isfinite(s) else 0.0) < threshold:
                continue
            relation = "potential_duplicate" if max(abs(p), abs(s)) >= 0.98 else "strongly_related"
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "pearson_corr": p,
                    "spearman_corr": s,
                    "suggested_relation": relation,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out.reindex(
            columns=["feature_1", "feature_2", "pearson_corr", "spearman_corr", "suggested_relation"],
        )
    return out.sort_values(["suggested_relation", "feature_1", "feature_2"], kind="stable").reset_index(drop=True)


def build_screening_shortlist(
    quality_df: pd.DataFrame,
    corr_pairs_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    warning_heavy_thr = float(cfg["warning_heavy_threshold"])
    corr_map: Dict[str, str] = {}
    corr_features: set[str] = set()
    if not corr_pairs_df.empty:
        for _, row in corr_pairs_df.iterrows():
            f1, f2 = str(row["feature_1"]), str(row["feature_2"])
            corr_features.add(f1)
            corr_features.add(f2)
            corr_map.setdefault(f1, f2)
            corr_map.setdefault(f2, f1)

    shortlist = quality_df.copy()
    shortlist["correlation_flag"] = shortlist["feature_name"].isin(corr_features)

    statuses: List[str] = []
    notes: List[str] = []
    for _, row in shortlist.iterrows():
        feature = str(row["feature_name"])
        block = str(row["source_block"])
        high_missing = bool(row["high_missing_flag"])
        low_variance = bool(row["low_variance_flag"])
        near_constant = bool(row["near_constant_flag"])
        warning_heavy = _is_warning_heavy(float(row["warning_coverage"]), warning_heavy_thr)
        correlated = bool(row["correlation_flag"])
        missing_rate = float(row["missing_rate"]) if np.isfinite(row["missing_rate"]) else np.nan

        note_tokens: List[str] = []
        if np.isfinite(missing_rate) and missing_rate >= 0.99:
            status = "drop_candidate"
            note_tokens.append("all values missing")
        elif high_missing:
            status = "review_high_missing"
            note_tokens.append("high missingness")
        elif near_constant or low_variance:
            status = "review_low_variance"
            note_tokens.append("very low variance")
        elif warning_heavy and block == "D":
            status = "reserve_candidate"
            note_tokens.append("heavy warning coverage in block D")
        elif warning_heavy:
            status = "review_warning_heavy"
            note_tokens.append("heavy warning coverage")
        elif correlated:
            status = "review_highly_correlated"
            note_tokens.append(f"highly correlated with {corr_map.get(feature, 'another feature')}")
        else:
            status = "keep_candidate"
            note_tokens.append("no major issues detected")

        if correlated and status not in {"review_highly_correlated", "drop_candidate"}:
            note_tokens.append(f"highly correlated with {corr_map.get(feature, 'another feature')}")

        statuses.append(status)
        notes.append("; ".join(dict.fromkeys(note_tokens)))

    shortlist["screening_status"] = statuses
    shortlist["notes"] = notes

    columns = [
        "feature_name",
        "source_block",
        "dtype",
        "missing_rate",
        "std",
        "near_constant_flag",
        "low_variance_flag",
        "high_missing_flag",
        "warning_coverage",
        "correlation_flag",
        "screening_status",
        "notes",
    ]
    return shortlist[columns].sort_values(["source_block", "feature_name"], kind="stable").reset_index(drop=True)


def run_feature_screening_pipeline(config_path: str = "configs/feature_screening_v1.yaml") -> FeatureScreeningOutputs:
    cfg = load_feature_screening_config(config_path)
    run_name = str(cfg.get("run_name", "feature_screening_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    screening_cfg = cfg["screening"]
    logger.info("Feature screening started")
    logger.info("Dataset profile: %s", dataset_profile)
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    block_tables = load_feature_block_tables(cfg["input"], dataset_profile=dataset_profile)
    if not block_tables:
        raise ValueError("No feature block inputs were loaded from config input section.")
    logger.info("Feature files loaded: %d", len(block_tables))

    master_build = build_master_feature_table(block_tables, dataset_profile=dataset_profile)
    master_df = master_build.master_df
    feature_to_block = master_build.feature_to_block
    logger.info("Master rows: %d", len(master_df))
    logger.info("Master columns: %d", len(master_df.columns))
    logger.info("Feature columns: %d", len(feature_to_block))

    quality_df = compute_feature_quality_metrics(master_df, feature_to_block, screening_cfg)
    corr_pairs_df = compute_high_correlation_pairs(
        master_df,
        feature_to_block,
        threshold=float(screening_cfg["high_correlation_threshold"]),
    )
    shortlist_df = build_screening_shortlist(quality_df, corr_pairs_df, screening_cfg)

    high_missing_count = int(shortlist_df["high_missing_flag"].sum())
    low_variance_count = int(shortlist_df["low_variance_flag"].sum())
    status_counts = {str(k): int(v) for k, v in shortlist_df["screening_status"].value_counts().to_dict().items()}
    high_corr_pairs_count = int(len(corr_pairs_df))

    logger.info("Features with high missingness: %d", high_missing_count)
    logger.info("Features with low variance: %d", low_variance_count)
    logger.info("High-correlation pairs: %d", high_corr_pairs_count)
    logger.info("Screening status counts: %s", status_counts)

    features_dir = Path(artifacts_cfg["features"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    master_parquet_path = features_dir / str(cfg["output"]["master_parquet_name"])
    excel_path = reports_dir / str(cfg["output"]["excel_name"])
    corr_csv_path = reports_dir / str(cfg["output"]["high_correlation_csv_name"])
    shortlist_csv_path = reports_dir / str(cfg["output"]["shortlist_csv_name"])

    master_df.to_parquet(master_parquet_path, index=False)
    corr_pairs_df.to_csv(corr_csv_path, index=False)
    shortlist_df.to_csv(shortlist_csv_path, index=False)

    summary_payload = {
        "rows": int(len(master_df)),
        "columns": int(len(master_df.columns)),
        "features_total": int(len(feature_to_block)),
        "numeric_features": int(
            sum(1 for f in feature_to_block if pd.api.types.is_numeric_dtype(master_df[f])),
        ),
        "features_with_missingness": int((quality_df["missing_rate"] > 0).sum()),
        "features_low_variance": low_variance_count,
        "high_correlation_pairs": high_corr_pairs_count,
        "screening_status_counts": status_counts,
    }
    excel_out = export_feature_screening_excel(
        excel_path=excel_path,
        run_id=run_id,
        dataset_profile=dataset_profile,
        master_df=master_df,
        quality_df=quality_df,
        corr_pairs_df=corr_pairs_df,
        shortlist_df=shortlist_df,
        summary=summary_payload,
        input_paths={k: str(v) for k, v in cfg["input"].items()},
        output_parquet=str(master_parquet_path),
        screening_cfg=screening_cfg,
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Master parquet path: %s", master_parquet_path)
    logger.info("Excel report path: %s", excel_out)
    logger.info("High correlation CSV path: %s", corr_csv_path)
    logger.info("Shortlist CSV path: %s", shortlist_csv_path)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_screening",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": {
            **{k: str(v) for k, v in cfg["input"].items()},
            "dataset_profile": dataset_profile,
            "rows_by_block": master_build.input_rows_by_block,
        },
        "outputs": {
            "master_parquet": str(master_parquet_path),
            "excel_report": str(excel_out),
            "high_correlation_pairs_csv": str(corr_csv_path),
            "screening_shortlist_csv": str(shortlist_csv_path),
            "log": str(log_path),
        },
        "summary": summary_payload,
    }
    manifest_path = write_manifest(manifest, artifacts_cfg["manifests"], run_id)

    return FeatureScreeningOutputs(
        run_id=run_id,
        master_parquet_path=str(master_parquet_path),
        excel_path=str(excel_out),
        correlation_pairs_csv_path=str(corr_csv_path),
        shortlist_csv_path=str(shortlist_csv_path),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        dataset_profile=dataset_profile,
        rows=int(len(master_df)),
        columns=int(len(master_df.columns)),
        n_features=int(len(feature_to_block)),
        low_variance_count=low_variance_count,
        high_missing_count=high_missing_count,
        high_correlation_pairs=high_corr_pairs_count,
        screening_status_counts=status_counts,
    )
