from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from src.config.loader import load_feature_block_config
from src.features.base import SeriesFeatureResult
from src.reporting.excel_export import export_features_block_a_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class FeatureBlockAOutputs:
    run_id: str
    parquet_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    input_series: int
    output_series: int
    successful: int
    warnings: int


def _build_log_windows(min_window: int, max_window: int, num_windows: int) -> np.ndarray:
    raw = np.geomspace(min_window, max_window, num=num_windows)
    windows = np.unique(np.maximum(raw.astype(int), 2))
    return windows


def hurst_rs(
    series: np.ndarray,
    min_window: int = 16,
    max_window: int = 256,
    num_windows: int = 12,
    min_scales: int = 4,
) -> Tuple[float, List[str]]:
    """Estimate Hurst exponent with rescaled-range regression over multiple scales."""
    flags: List[str] = []
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max(min_window * 2, 64):
        return np.nan, ["hurst_rs_too_short"]

    windows = _build_log_windows(min_window=min_window, max_window=min(max_window, len(x) // 2), num_windows=num_windows)
    if len(windows) < min_scales:
        return np.nan, ["hurst_rs_insufficient_scales"]

    log_w: List[float] = []
    log_rs: List[float] = []
    for w in windows:
        n_chunks = len(x) // w
        if n_chunks < 2:
            continue
        rs_vals: List[float] = []
        for i in range(n_chunks):
            chunk = x[i * w : (i + 1) * w]
            centered = chunk - np.mean(chunk)
            y = np.cumsum(centered)
            r = np.max(y) - np.min(y)
            s = np.std(chunk, ddof=1)
            if s <= 0:
                continue
            rs = r / s
            if np.isfinite(rs) and rs > 0:
                rs_vals.append(float(rs))
        if rs_vals:
            log_w.append(float(np.log(w)))
            log_rs.append(float(np.log(np.mean(rs_vals))))

    if len(log_w) < min_scales:
        return np.nan, ["hurst_rs_unstable"]

    slope = np.polyfit(log_w, log_rs, deg=1)[0]
    if not np.isfinite(slope):
        return np.nan, ["hurst_rs_nonfinite"]
    if slope < 0 or slope > 1.5:
        flags.append("hurst_rs_out_of_range")
    return float(slope), flags


def hurst_dfa(
    series: np.ndarray,
    min_window: int = 16,
    max_window: int = 256,
    num_windows: int = 12,
    min_scales: int = 4,
) -> Tuple[float, List[str]]:
    """Estimate Hurst exponent via first-order DFA (linear detrending per segment)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(min_window * 2, 64):
        return np.nan, ["hurst_dfa_too_short"]

    y = np.cumsum(x - np.mean(x))
    windows = _build_log_windows(min_window=min_window, max_window=min(max_window, len(x) // 2), num_windows=num_windows)
    if len(windows) < min_scales:
        return np.nan, ["hurst_dfa_insufficient_scales"]

    log_w: List[float] = []
    log_f: List[float] = []
    for w in windows:
        n_chunks = len(y) // w
        if n_chunks < 2:
            continue

        fluctuations: List[float] = []
        t = np.arange(w, dtype=float)
        for i in range(n_chunks):
            seg = y[i * w : (i + 1) * w]
            coeff = np.polyfit(t, seg, deg=1)
            trend = coeff[0] * t + coeff[1]
            resid = seg - trend
            f = np.sqrt(np.mean(resid**2))
            if np.isfinite(f) and f > 0:
                fluctuations.append(float(f))
        if fluctuations:
            log_w.append(float(np.log(w)))
            log_f.append(float(np.log(np.mean(fluctuations))))

    if len(log_w) < min_scales:
        return np.nan, ["hurst_dfa_unstable"]

    slope = np.polyfit(log_w, log_f, deg=1)[0]
    flags: List[str] = []
    if not np.isfinite(slope):
        return np.nan, ["hurst_dfa_nonfinite"]
    if slope < 0 or slope > 1.5:
        flags.append("hurst_dfa_out_of_range")
    return float(slope), flags


def compute_acf_grid(series: np.ndarray, lags: Sequence[int]) -> Dict[int, float]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {int(k): np.nan for k in lags}

    max_lag = int(max(lags))
    if len(x) <= max_lag:
        max_lag = len(x) - 1
    if max_lag < 1:
        return {int(k): np.nan for k in lags}

    acf_vals = acf(x, nlags=max_lag, fft=True, missing="raise")
    out: Dict[int, float] = {}
    for lag in lags:
        k = int(lag)
        out[k] = float(acf_vals[k]) if k <= max_lag else np.nan
    return out


def integrated_acf(acf_values: Dict[int, float], k: int) -> float:
    vals = [abs(v) for lag, v in acf_values.items() if lag <= k and np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.sum(vals))


def compute_ljung_box(series: np.ndarray, lags: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[int, Tuple[float, float]] = {}
    if len(x) < 5:
        return {int(lag): (np.nan, np.nan) for lag in lags}

    max_lag = int(max(lags))
    safe_lag = min(max_lag, len(x) - 1)
    if safe_lag < 1:
        return {int(lag): (np.nan, np.nan) for lag in lags}

    lb = acorr_ljungbox(x, lags=list(range(1, safe_lag + 1)), return_df=True)
    for lag in lags:
        k = int(lag)
        if k <= safe_lag:
            row = lb.iloc[k - 1]
            out[k] = (float(row["lb_stat"]), float(row["lb_pvalue"]))
        else:
            out[k] = (np.nan, np.nan)
    return out


def variance_ratio(series: np.ndarray, q: int) -> Tuple[float, List[str]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= q + 2:
        return np.nan, [f"vr_q{q}_too_short"]
    var_1 = np.var(x, ddof=1)
    if var_1 <= 0:
        return np.nan, [f"vr_q{q}_zero_var"]
    agg = pd.Series(x).rolling(window=q).sum().dropna().to_numpy()
    if len(agg) < 3:
        return np.nan, [f"vr_q{q}_unstable"]
    var_q = np.var(agg, ddof=1)
    vr = var_q / (q * var_1)
    if not np.isfinite(vr):
        return np.nan, [f"vr_q{q}_nonfinite"]
    return float(vr), []


def _feature_metric_columns(cfg: Dict[str, Any]) -> List[str]:
    acf_lags = [int(x) for x in cfg["metrics"]["acf_lags"]]
    metric_columns: List[str] = [
        "hurst_rs",
        "hurst_dfa",
    ]
    metric_columns += [f"acf_lag_{k}" for k in acf_lags]
    metric_columns += [f"abs_acf_lag_{k}" for k in acf_lags]
    metric_columns += [f"iacf_abs_1_{k}" for k in cfg["metrics"]["integrated_acf_windows"]]
    metric_columns += [f"iacf_absret_1_{k}" for k in cfg["metrics"]["integrated_acf_windows"]]
    for lag in cfg["metrics"]["ljung_box_lags"]:
        metric_columns.append(f"lb_ret_stat_{lag}")
        metric_columns.append(f"lb_ret_p_{lag}")
    for lag in cfg["metrics"]["ljung_box_lags"]:
        metric_columns.append(f"lb_absret_stat_{lag}")
        metric_columns.append(f"lb_absret_p_{lag}")
    metric_columns += [f"vr_q{q}" for q in cfg["metrics"]["variance_ratio_q"]]
    return metric_columns


def _compute_single_series_features(
    series_df: pd.DataFrame,
    dataset_profile: str,
    cfg: Dict[str, Any],
) -> SeriesFeatureResult:
    sid = str(series_df["series_id"].iloc[0])
    ticker = str(series_df["ticker"].iloc[0])
    market = str(series_df["market"].iloc[0])
    x = series_df.sort_values("date", kind="stable")["log_return"].to_numpy(dtype=float)
    result = SeriesFeatureResult(
        series_id=sid,
        ticker=ticker,
        market=market,
        dataset_profile=dataset_profile,
        feature_block="A",
        n_obs_used=int(len(x)),
    )

    rs_params = cfg["metrics"]["hurst_rs"]
    dfa_params = cfg["metrics"]["hurst_dfa"]
    acf_lags = [int(v) for v in cfg["metrics"]["acf_lags"]]
    iacf_windows = [int(v) for v in cfg["metrics"]["integrated_acf_windows"]]
    lb_lags = [int(v) for v in cfg["metrics"]["ljung_box_lags"]]
    vr_qs = [int(v) for v in cfg["metrics"]["variance_ratio_q"]]

    h_rs, rs_flags = hurst_rs(
        x,
        min_window=int(rs_params["min_window"]),
        max_window=int(rs_params["max_window"]),
        num_windows=int(rs_params["num_windows"]),
        min_scales=int(rs_params["min_scales"]),
    )
    result.metrics["hurst_rs"] = h_rs
    for flag in rs_flags:
        result.add_warning(flag)

    h_dfa, dfa_flags = hurst_dfa(
        x,
        min_window=int(dfa_params["min_window"]),
        max_window=int(dfa_params["max_window"]),
        num_windows=int(dfa_params["num_windows"]),
        min_scales=int(dfa_params["min_scales"]),
    )
    result.metrics["hurst_dfa"] = h_dfa
    for flag in dfa_flags:
        result.add_warning(flag)

    acf_ret = compute_acf_grid(x, acf_lags)
    acf_abs = compute_acf_grid(np.abs(x), acf_lags)
    for k in acf_lags:
        result.metrics[f"acf_lag_{k}"] = acf_ret.get(k, np.nan)
        result.metrics[f"abs_acf_lag_{k}"] = acf_abs.get(k, np.nan)

    for k in iacf_windows:
        result.metrics[f"iacf_abs_1_{k}"] = integrated_acf(acf_ret, k)
        result.metrics[f"iacf_absret_1_{k}"] = integrated_acf(acf_abs, k)

    lb_ret = compute_ljung_box(x, lb_lags)
    lb_abs = compute_ljung_box(np.abs(x), lb_lags)
    for lag in lb_lags:
        stat, p = lb_ret[lag]
        result.metrics[f"lb_ret_stat_{lag}"] = stat
        result.metrics[f"lb_ret_p_{lag}"] = p
        if np.isfinite(p) and (p < 0 or p > 1):
            result.add_warning("lb_ret_p_out_of_range")

    for lag in lb_lags:
        stat, p = lb_abs[lag]
        result.metrics[f"lb_absret_stat_{lag}"] = stat
        result.metrics[f"lb_absret_p_{lag}"] = p
        if np.isfinite(p) and (p < 0 or p > 1):
            result.add_warning("lb_absret_p_out_of_range")

    for q in vr_qs:
        vr, flags = variance_ratio(x, q)
        result.metrics[f"vr_q{q}"] = vr
        for flag in flags:
            result.add_warning(flag)

    return result


def _summarize_feature_table(features_df: pd.DataFrame) -> Dict[str, Any]:
    def _min_max(col: str) -> Dict[str, float]:
        vals = pd.to_numeric(features_df[col], errors="coerce")
        return {
            "min": float(vals.min()) if vals.notna().any() else np.nan,
            "max": float(vals.max()) if vals.notna().any() else np.nan,
        }

    pvalue_cols = [c for c in features_df.columns if c.endswith("_p_5") or c.endswith("_p_10") or c.endswith("_p_20") or c.endswith("_p_50")]
    pvals = pd.concat([pd.to_numeric(features_df[c], errors="coerce") for c in pvalue_cols], axis=0)
    pvals = pvals[np.isfinite(pvals)]

    summary = {
        "series_total": int(len(features_df)),
        "series_successful": int((features_df["feature_status"] == "success").sum()),
        "series_with_warnings": int((features_df["feature_status"] != "success").sum()),
        "nan_hurst_rs": int(features_df["hurst_rs"].isna().sum()),
        "nan_hurst_dfa": int(features_df["hurst_dfa"].isna().sum()),
        "hurst_rs_range": _min_max("hurst_rs"),
        "hurst_dfa_range": _min_max("hurst_dfa"),
        "iacf_abs_1_20_range": _min_max("iacf_abs_1_20"),
        "iacf_absret_1_20_range": _min_max("iacf_absret_1_20"),
        "vr_q2_range": _min_max("vr_q2"),
        "vr_q5_range": _min_max("vr_q5"),
        "vr_q10_range": _min_max("vr_q10"),
        "pvalues_outside_0_1": int(((pvals < 0) | (pvals > 1)).sum()),
    }
    return summary


def run_feature_block_a_pipeline(config_path: str = "configs/features_block_A_v1.yaml") -> FeatureBlockAOutputs:
    cfg = load_feature_block_config(config_path)
    run_name = str(cfg.get("run_name", "features_block_A_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    logger.info("Feature block A started")
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
                feature_block="A",
                n_obs_used=int(len(sdf)),
                status="failed",
            )
            fallback.add_warning("series_processing_error")
            results.append(fallback)

    features_df = pd.DataFrame([r.as_row(metric_columns) for r in results])
    if not features_df.empty:
        features_df = features_df.sort_values(["series_id"], kind="stable").reset_index(drop=True)

    summary = _summarize_feature_table(features_df) if not features_df.empty else {}
    warning_df = features_df[features_df["feature_warning_flags"].astype(str) != ""] if not features_df.empty else pd.DataFrame()

    features_dir = Path(artifacts_cfg["features"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = features_dir / str(cfg["output"]["parquet_name"])
    excel_path = reports_dir / str(cfg["output"]["excel_name"])
    features_df.to_parquet(parquet_path, index=False)

    excel_out = export_features_block_a_excel(
        features_df=features_df,
        summary=summary,
        warnings_df=warning_df,
        excel_path=excel_path,
        run_id=run_id,
        dataset_profile=dataset_profile,
        input_paths={
            "log_returns_parquet": str(log_returns_path),
            "dataset_profiles_parquet": str(profiles_path),
        },
        output_parquet=str(parquet_path),
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Output series: %d", int(len(features_df)))
    logger.info("Series successful: %d", summary.get("series_successful", 0))
    logger.info("Series with warnings: %d", summary.get("series_with_warnings", 0))
    logger.info("NaN hurst_rs: %d", summary.get("nan_hurst_rs", 0))
    logger.info("NaN hurst_dfa: %d", summary.get("nan_hurst_dfa", 0))
    logger.info("p-values outside [0,1]: %d", summary.get("pvalues_outside_0_1", 0))
    logger.info("Processing errors: %d", errors)
    logger.info("Parquet path: %s", parquet_path)
    logger.info("Excel path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_block_A",
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

    return FeatureBlockAOutputs(
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

