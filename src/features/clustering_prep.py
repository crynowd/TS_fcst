from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


METADATA_COLUMNS = ["series_id", "ticker", "market", "dataset_profile"]


def build_clustering_table(master_df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    selected_features = [c for c in feature_columns if c in master_df.columns]
    missing = [c for c in feature_columns if c not in master_df.columns]
    if missing:
        raise ValueError(f"Selected features are missing in master table: {missing}")
    cols = [c for c in METADATA_COLUMNS if c in master_df.columns] + selected_features
    return master_df[cols].copy()


def _outlier_share(series: pd.Series) -> float:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = float(q3 - q1)
    if not np.isfinite(iqr):
        return np.nan
    if iqr <= 0:
        return 0.0
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return float(((series < low) | (series > high)).mean())


def build_scaling_plan(
    table_df: pd.DataFrame,
    feature_source_map: Mapping[str, str],
    cfg: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    robust_outlier_threshold = float(cfg.get("robust_outlier_threshold", 0.05))
    rank_outlier_threshold = float(cfg.get("rank_outlier_threshold", 0.15))
    robust_abs_skew_threshold = float(cfg.get("robust_abs_skew_threshold", 1.0))
    rank_abs_skew_threshold = float(cfg.get("rank_abs_skew_threshold", 2.0))

    rows: List[Dict[str, object]] = []
    for feature, source_block in feature_source_map.items():
        if feature not in table_df.columns:
            continue
        s = pd.to_numeric(table_df[feature], errors="coerce").dropna().astype(float)
        if s.empty:
            mean = std = median = iqr = min_v = max_v = skew = outlier = np.nan
        else:
            mean = float(s.mean())
            std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            median = float(s.median())
            iqr = float(s.quantile(0.75) - s.quantile(0.25))
            min_v = float(s.min())
            max_v = float(s.max())
            skew = float(s.skew())
            outlier = _outlier_share(s)

        if (np.isfinite(outlier) and outlier >= rank_outlier_threshold) or (
            np.isfinite(skew) and abs(skew) >= rank_abs_skew_threshold
        ):
            scaling = "rank_transform_candidate"
            reason = "high skew/outlier profile"
        elif (np.isfinite(outlier) and outlier >= robust_outlier_threshold) or (
            np.isfinite(skew) and abs(skew) >= robust_abs_skew_threshold
        ):
            scaling = "robust_scale_candidate"
            reason = "moderate skew/outlier profile"
        else:
            scaling = "standard_scale_candidate"
            reason = "stable distribution profile"

        rows.append(
            {
                "feature_name": feature,
                "source_block": source_block,
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
                "min": min_v,
                "max": max_v,
                "skewness": skew,
                "outlier_proxy_rate": outlier,
                "recommended_scaling": scaling,
                "scaling_reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values(["source_block", "feature_name"], kind="stable").reset_index(drop=True)
