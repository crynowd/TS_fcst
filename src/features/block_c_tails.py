from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from src.config.loader import load_feature_block_config
from src.features.base import SeriesFeatureResult
from src.reporting.excel_export import export_features_block_c_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class FeatureBlockCOutputs:
    run_id: str
    parquet_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    input_series: int
    output_series: int
    successful: int
    warnings: int


def _feature_metric_columns(cfg: Dict[str, Any]) -> List[str]:
    configured = cfg.get("output", {}).get("metric_columns", [])
    if configured:
        return [str(c) for c in configured]
    return [
        "kurtosis",
        "robust_kurtosis",
        "tail_ratio_symmetric",
        "tail_ratio_upper",
        "tail_ratio_lower",
        "hill_tail_index",
    ]


def _clean_and_demean(values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    flags: List[str] = []
    if not np.all(mask):
        flags.append("nonfinite_removed")
    x = x[mask]
    if x.size == 0:
        flags.append("series_empty_after_clean")
        return x, flags
    x = x - float(np.mean(x))
    return x, flags


def _safe_ratio(numerator: float, denominator: float, den_eps: float, near_zero_flag: str) -> Tuple[float, List[str]]:
    flags: List[str] = []
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan, ["ratio_nonfinite"]
    if abs(denominator) <= den_eps:
        return np.nan, [near_zero_flag]
    value = numerator / denominator
    if not np.isfinite(value):
        return np.nan, ["ratio_nonfinite"]
    return float(value), flags


def fisher_kurtosis(series: np.ndarray) -> Tuple[float, List[str]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan, ["kurtosis_too_short"]
    value = scipy_kurtosis(x, fisher=True, bias=False, nan_policy="raise")
    if not np.isfinite(value):
        return np.nan, ["kurtosis_nonfinite"]
    return float(value), []


def _moors_kurtosis(x: np.ndarray) -> Tuple[float, List[str]]:
    """Moors kurtosis based on octiles.

    Formula:
    (Q0.875 - Q0.625 + Q0.375 - Q0.125) / (Q0.75 - Q0.25)
    """
    if x.size < 8:
        return np.nan, ["robust_kurtosis_too_short"]

    q = np.quantile(x, [0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
    den = float(q[4] - q[1])
    if abs(den) <= 1e-12:
        return np.nan, ["robust_kurtosis_denominator_near_zero"]

    num = float((q[5] - q[3]) + (q[2] - q[0]))
    value = num / den
    if not np.isfinite(value):
        return np.nan, ["robust_kurtosis_nonfinite"]
    return float(value), []


def robust_kurtosis_moors(series: np.ndarray) -> Tuple[float, List[str]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    return _moors_kurtosis(x)


def quantile_tail_ratios(series: np.ndarray, quantiles: Sequence[float] = (0.01, 0.25, 0.75, 0.99)) -> Tuple[Dict[str, float], List[str]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    flags: List[str] = []
    if x.size < 8:
        return {
            "tail_ratio_symmetric": np.nan,
            "tail_ratio_upper": np.nan,
            "tail_ratio_lower": np.nan,
        }, ["tail_ratios_too_short"]

    q_levels = np.asarray([float(v) for v in quantiles], dtype=float)
    q_values = np.quantile(x, q_levels)

    def _get_q(target: float) -> float:
        idx = int(np.argmin(np.abs(q_levels - target)))
        if abs(float(q_levels[idx]) - target) > 1e-9:
            flags.append(f"quantile_{target:.2f}_missing")
        return float(q_values[idx])

    q01 = _get_q(0.01)
    q25 = _get_q(0.25)
    q75 = _get_q(0.75)
    q99 = _get_q(0.99)

    den_eps = 1e-12
    sym, f = _safe_ratio(abs(q99) + abs(q01), abs(q75) + abs(q25), den_eps, "tail_ratio_symmetric_denominator_near_zero")
    flags.extend(f)
    up, f = _safe_ratio(q99, q75, den_eps, "tail_ratio_upper_denominator_near_zero")
    flags.extend(f)
    lo, f = _safe_ratio(abs(q01), abs(q25), den_eps, "tail_ratio_lower_denominator_near_zero")
    flags.extend(f)

    return {
        "tail_ratio_symmetric": sym,
        "tail_ratio_upper": up,
        "tail_ratio_lower": lo,
    }, flags


def hill_tail_index(abs_tail_values: np.ndarray, k_fraction: float) -> Tuple[float, List[str]]:
    flags: List[str] = []
    abs_x = np.asarray(abs_tail_values, dtype=float)
    abs_x = abs_x[np.isfinite(abs_x) & (abs_x > 0)]
    n = int(abs_x.size)
    if n < 50:
        return np.nan, ["hill_too_short"]

    k_fraction = float(k_fraction)
    if not np.isfinite(k_fraction) or k_fraction <= 0.0 or k_fraction >= 1.0:
        return np.nan, ["hill_invalid_k_fraction"]

    k = int(np.floor(n * k_fraction))
    k = max(5, k)
    k = min(k, n - 1)
    if k < 5:
        return np.nan, ["hill_invalid_k"]

    xs = np.sort(abs_x)
    top = xs[-k:]
    x_k = float(xs[-k - 1])
    if x_k <= 0 or not np.isfinite(x_k):
        return np.nan, ["hill_threshold_nonpositive"]

    logs = np.log(top / x_k)
    if np.any(~np.isfinite(logs)):
        return np.nan, ["hill_nonfinite_logs"]

    h = float(np.mean(logs))
    if h <= 0 or not np.isfinite(h):
        return np.nan, ["hill_nonpositive_h"]

    # Returns alpha (tail index) where smaller alpha corresponds to heavier tails.
    alpha = 1.0 / h
    if not np.isfinite(alpha):
        return np.nan, ["hill_nonfinite"]

    return float(alpha), flags


def _compute_single_series_features(series_df: pd.DataFrame, dataset_profile: str, cfg: Dict[str, Any]) -> SeriesFeatureResult:
    sid = str(series_df["series_id"].iloc[0])
    ticker = str(series_df["ticker"].iloc[0])
    market = str(series_df["market"].iloc[0])

    raw = series_df.sort_values("date", kind="stable")["log_return"].to_numpy(dtype=float)
    x, clean_flags = _clean_and_demean(raw)

    result = SeriesFeatureResult(
        series_id=sid,
        ticker=ticker,
        market=market,
        dataset_profile=dataset_profile,
        feature_block="C",
        n_obs_used=int(x.size),
    )
    for flag in clean_flags:
        result.add_warning(flag)

    metric_columns = _feature_metric_columns(cfg)
    if x.size < 8:
        for metric in metric_columns:
            result.metrics[metric] = np.nan
        result.add_warning("series_too_short_for_block_c")
        return result

    # C1: classical Fisher kurtosis (excess kurtosis).
    k, flags = fisher_kurtosis(x)
    result.metrics["kurtosis"] = k
    for flag in flags:
        result.add_warning(flag)
    if np.isfinite(k) and k < 0:
        result.add_warning("kurtosis_negative")

    # C2: robust kurtosis via Moors octile-based measure.
    rk, flags = _moors_kurtosis(x)
    result.metrics["robust_kurtosis"] = rk
    for flag in flags:
        result.add_warning(flag)

    ratios, flags = quantile_tail_ratios(
        x,
        quantiles=[float(v) for v in cfg.get("metrics", {}).get("quantiles", [0.01, 0.25, 0.75, 0.99])],
    )
    result.metrics["tail_ratio_symmetric"] = ratios["tail_ratio_symmetric"]
    result.metrics["tail_ratio_upper"] = ratios["tail_ratio_upper"]
    result.metrics["tail_ratio_lower"] = ratios["tail_ratio_lower"]
    for flag in flags:
        result.add_warning(flag)

    for col in ["tail_ratio_symmetric", "tail_ratio_upper", "tail_ratio_lower"]:
        val = result.metrics.get(col, np.nan)
        if np.isfinite(val) and val < 0:
            result.add_warning(f"{col}_negative")

    # C6 (reserve): Hill tail index.
    use_hill = bool(cfg.get("metrics", {}).get("use_hill", True))
    if use_hill:
        hill, flags = hill_tail_index(np.abs(x), k_fraction=float(cfg.get("metrics", {}).get("hill_k_fraction", 0.05)))
        result.metrics["hill_tail_index"] = hill
        for flag in flags:
            result.add_warning(flag)
        if np.isfinite(hill) and hill > 10:
            result.add_warning("hill_tail_index_gt_10")
    else:
        result.metrics["hill_tail_index"] = np.nan
        result.add_warning("hill_disabled")

    return result


def _metric_ranges(features_df: pd.DataFrame, metric_columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in metric_columns:
        vals = pd.to_numeric(features_df[col], errors="coerce")
        out[col] = {
            "min": float(vals.min()) if vals.notna().any() else np.nan,
            "max": float(vals.max()) if vals.notna().any() else np.nan,
            "mean": float(vals.mean()) if vals.notna().any() else np.nan,
            "median": float(vals.median()) if vals.notna().any() else np.nan,
        }
    return out


def _summarize_feature_table(features_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    metric_columns = _feature_metric_columns(cfg)
    nan_counts = {f"nan_{col}": int(features_df[col].isna().sum()) for col in metric_columns if col in features_df.columns}

    kurt = pd.to_numeric(features_df.get("kurtosis"), errors="coerce")
    tr_sym = pd.to_numeric(features_df.get("tail_ratio_symmetric"), errors="coerce")
    tr_up = pd.to_numeric(features_df.get("tail_ratio_upper"), errors="coerce")
    tr_lo = pd.to_numeric(features_df.get("tail_ratio_lower"), errors="coerce")
    hill = pd.to_numeric(features_df.get("hill_tail_index"), errors="coerce")

    summary: Dict[str, Any] = {
        "series_total": int(len(features_df)),
        "series_successful": int((features_df["feature_status"] == "success").sum()),
        "series_with_warnings": int((features_df["feature_status"] != "success").sum()),
        "kurtosis_lt_0": int((kurt < 0).sum()),
        "tail_ratio_symmetric_negative": int((tr_sym < 0).sum()),
        "tail_ratio_upper_negative": int((tr_up < 0).sum()),
        "tail_ratio_lower_negative": int((tr_lo < 0).sum()),
        "hill_tail_index_gt_10": int((hill > 10).sum()),
        "ranges": _metric_ranges(features_df, metric_columns),
        **nan_counts,
    }
    return summary


def run_feature_block_c_pipeline(config_path: str = "configs/features_block_C_v1.yaml") -> FeatureBlockCOutputs:
    cfg = load_feature_block_config(config_path)
    run_name = str(cfg.get("run_name", "feature_block_C_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    logger.info("Feature block C started")
    logger.info("Dataset profile: %s", dataset_profile)
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    log_returns_path = Path(cfg["input"]["log_returns_parquet"]).resolve()
    profiles_path = Path(cfg["input"]["dataset_profiles_parquet"]).resolve()
    if not log_returns_path.exists():
        raise FileNotFoundError(f"Input not found: {log_returns_path}")
    if not profiles_path.exists():
        raise FileNotFoundError(f"Input not found: {profiles_path}")

    returns_df = pd.read_parquet(log_returns_path)
    profiles_df = pd.read_parquet(profiles_path)
    profile_series = profiles_df[profiles_df["dataset_profile"] == dataset_profile]["series_id"].drop_duplicates()
    scoped_returns = returns_df[returns_df["series_id"].isin(set(profile_series))].copy()
    scoped_returns = scoped_returns[scoped_returns["dataset_profile"] == dataset_profile].copy()

    input_series = int(scoped_returns["series_id"].nunique())
    logger.info("Input rows: %d", len(scoped_returns))
    logger.info("Input series: %d", input_series)

    metric_columns = _feature_metric_columns(cfg)
    results: List[SeriesFeatureResult] = []
    errors = 0

    for series_id, sdf in scoped_returns.groupby("series_id", sort=True):
        try:
            row = _compute_single_series_features(sdf, dataset_profile=dataset_profile, cfg=cfg)
            results.append(row)
        except Exception as exc:
            errors += 1
            logger.exception("Feature computation failed for %s: %s", series_id, exc)
            fallback = SeriesFeatureResult(
                series_id=str(series_id),
                ticker=str(sdf["ticker"].iloc[0]) if not sdf.empty else "",
                market=str(sdf["market"].iloc[0]) if not sdf.empty else "",
                dataset_profile=dataset_profile,
                feature_block="C",
                n_obs_used=int(len(sdf)),
                status="failed",
            )
            fallback.add_warning("series_processing_error")
            results.append(fallback)

    features_df = pd.DataFrame([r.as_row(metric_columns) for r in results])
    if not features_df.empty:
        features_df = features_df.sort_values(["series_id"], kind="stable").reset_index(drop=True)

    summary = _summarize_feature_table(features_df, cfg=cfg) if not features_df.empty else {}
    warnings_df = features_df[features_df["feature_warning_flags"].astype(str) != ""] if not features_df.empty else pd.DataFrame()

    features_dir = Path(artifacts_cfg["features"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = features_dir / str(cfg["output"]["parquet_name"])
    excel_path = reports_dir / str(cfg["output"]["excel_name"])
    features_df.to_parquet(parquet_path, index=False)

    excel_out = export_features_block_c_excel(
        features_df=features_df,
        summary=summary,
        warnings_df=warnings_df,
        excel_path=excel_path,
        run_id=run_id,
        dataset_profile=dataset_profile,
        input_paths={
            "log_returns_parquet": str(log_returns_path),
            "dataset_profiles_parquet": str(profiles_path),
        },
        output_parquet=str(parquet_path),
        config_params={
            "quantiles": cfg.get("metrics", {}).get("quantiles", []),
            "hill_k_fraction": cfg.get("metrics", {}).get("hill_k_fraction", 0.05),
            "use_hill": cfg.get("metrics", {}).get("use_hill", True),
        },
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Output series: %d", int(len(features_df)))
    logger.info("Series successful: %d", summary.get("series_successful", 0))
    logger.info("Series with warnings: %d", summary.get("series_with_warnings", 0))
    for col in metric_columns:
        logger.info("NaN %s: %d", col, int(summary.get(f"nan_{col}", 0)))

    logger.info("kurtosis < 0: %d", int(summary.get("kurtosis_lt_0", 0)))
    logger.info("tail_ratio_symmetric < 0: %d", int(summary.get("tail_ratio_symmetric_negative", 0)))
    logger.info("tail_ratio_upper < 0: %d", int(summary.get("tail_ratio_upper_negative", 0)))
    logger.info("tail_ratio_lower < 0: %d", int(summary.get("tail_ratio_lower_negative", 0)))
    logger.info("hill_tail_index > 10: %d", int(summary.get("hill_tail_index_gt_10", 0)))

    logger.info("Processing errors: %d", errors)
    logger.info("Parquet path: %s", parquet_path)
    logger.info("Excel path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_block_C",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": {
            "log_returns_parquet": str(log_returns_path),
            "dataset_profiles_parquet": str(profiles_path),
            "dataset_profile": dataset_profile,
            "input_series": input_series,
        },
        "outputs": {
            "features_parquet": str(parquet_path),
            "excel_report": str(excel_out),
            "log": str(log_path),
        },
        "summary": {
            **summary,
            "processing_errors": errors,
        },
    }
    manifest_path = write_manifest(manifest, artifacts_cfg["manifests"], run_id)

    return FeatureBlockCOutputs(
        run_id=run_id,
        parquet_path=str(parquet_path),
        excel_path=str(excel_out),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        input_series=input_series,
        output_series=int(len(features_df)),
        successful=int(summary.get("series_successful", 0)),
        warnings=int(summary.get("series_with_warnings", 0)),
    )
