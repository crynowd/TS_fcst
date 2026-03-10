from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import factorial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import periodogram, welch
from scipy.spatial.distance import pdist

from src.config.loader import load_feature_block_config
from src.features.base import SeriesFeatureResult
from src.reporting.excel_export import export_features_block_b_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class FeatureBlockBOutputs:
    run_id: str
    parquet_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    input_series: int
    output_series: int
    successful: int
    warnings: int


def _clean_series(values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    flags: List[str] = []
    if not np.all(mask):
        flags.append("nonfinite_removed")
    x = x[mask]
    if x.size < 3:
        flags.append("series_too_short")
    return x, flags


def _compute_psd(x: np.ndarray, spectral_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    method = str(spectral_cfg.get("method", "welch")).strip().lower()
    flags: List[str] = []

    if x.size < 8:
        return np.array([]), np.array([]), ["spectral_too_short"]

    if method == "welch":
        nperseg_cfg = int(spectral_cfg.get("welch_nperseg", 256))
        nperseg = int(min(max(8, nperseg_cfg), x.size))
        freqs, power = welch(x, fs=1.0, nperseg=nperseg, detrend=False, scaling="density")
    elif method == "periodogram":
        freqs, power = periodogram(x, fs=1.0, detrend=False, scaling="density")
    else:
        return np.array([]), np.array([]), ["spectral_method_unsupported"]

    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    finite_mask = np.isfinite(freqs) & np.isfinite(power)
    freqs = freqs[finite_mask]
    power = power[finite_mask]

    if freqs.size == 0:
        flags.append("spectral_psd_empty")
    return freqs, power, flags


def spectral_slope_beta(x_demeaned: np.ndarray, spectral_cfg: Dict[str, Any]) -> Tuple[float, List[str]]:
    freqs, power, flags = _compute_psd(x_demeaned, spectral_cfg)
    if freqs.size == 0:
        return np.nan, flags or ["spectral_slope_unavailable"]

    nyquist = 0.5
    min_ratio = float(spectral_cfg.get("regression_freq_min_ratio", 0.02))
    max_ratio = float(spectral_cfg.get("regression_freq_max_ratio", 1.0))
    min_ratio = max(0.0, min(min_ratio, 1.0))
    max_ratio = max(min_ratio, min(max_ratio, 1.0))

    f_min = nyquist * min_ratio
    f_max = nyquist * max_ratio

    mask = (freqs > 0.0) & (power > 0.0) & (freqs >= f_min) & (freqs <= f_max)
    log_f = np.log(freqs[mask])
    log_p = np.log(power[mask])

    min_points = int(spectral_cfg.get("min_regression_points", 8))
    if log_f.size < min_points:
        return np.nan, flags + ["spectral_slope_insufficient_points"]

    slope = np.polyfit(log_f, log_p, deg=1)[0]
    if not np.isfinite(slope):
        return np.nan, flags + ["spectral_slope_nonfinite"]
    return float(slope), flags


def spectral_entropy(x_demeaned: np.ndarray, spectral_cfg: Dict[str, Any]) -> Tuple[float, List[str]]:
    freqs, power, flags = _compute_psd(x_demeaned, spectral_cfg)
    if freqs.size == 0:
        return np.nan, flags or ["spectral_entropy_unavailable"]

    mask = (freqs > 0.0) & (power >= 0.0)
    p = power[mask]
    if p.size < 2:
        return np.nan, flags + ["spectral_entropy_insufficient_bins"]

    p = np.maximum(p, 0.0)
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= 0.0:
        return np.nan, flags + ["spectral_entropy_zero_power"]

    prob = p / total
    prob = np.clip(prob, 1e-18, 1.0)
    ent = -np.sum(prob * np.log(prob))

    normalized = bool(spectral_cfg.get("entropy_normalized", True))
    if normalized:
        denom = np.log(float(prob.size))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, flags + ["spectral_entropy_norm_invalid"]
        ent = ent / denom

    if not np.isfinite(ent):
        return np.nan, flags + ["spectral_entropy_nonfinite"]
    return float(ent), flags


def spectral_flatness(x_demeaned: np.ndarray, spectral_cfg: Dict[str, Any]) -> Tuple[float, List[str]]:
    freqs, power, flags = _compute_psd(x_demeaned, spectral_cfg)
    if freqs.size == 0:
        return np.nan, flags or ["spectral_flatness_unavailable"]

    mask = (freqs > 0.0) & (power >= 0.0)
    p = power[mask]
    if p.size < 2:
        return np.nan, flags + ["spectral_flatness_insufficient_bins"]

    p = np.maximum(p, 1e-18)
    gmean = float(np.exp(np.mean(np.log(p))))
    amean = float(np.mean(p))
    if amean <= 0 or not np.isfinite(amean):
        return np.nan, flags + ["spectral_flatness_zero_mean"]

    flatness = gmean / amean
    if not np.isfinite(flatness):
        return np.nan, flags + ["spectral_flatness_nonfinite"]
    return float(flatness), flags


def noise_fn(x: np.ndarray) -> Tuple[float, List[str]]:
    """Compute NoiseFN = std(diff(x)) / std(x), where x is log-return series."""
    if x.size < 3:
        return np.nan, ["noise_fn_too_short"]

    sigma_x = float(np.std(x, ddof=1))
    if sigma_x <= 0 or not np.isfinite(sigma_x):
        return np.nan, ["noise_fn_zero_std"]

    dx = np.diff(x)
    sigma_dx = float(np.std(dx, ddof=1))
    if not np.isfinite(sigma_dx):
        return np.nan, ["noise_fn_nonfinite"]

    value = sigma_dx / sigma_x
    if not np.isfinite(value):
        return np.nan, ["noise_fn_nonfinite"]
    return float(value), []


def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalized: bool = True,
) -> Tuple[float, List[str]]:
    if order < 2:
        return np.nan, ["perm_entropy_invalid_order"]
    if delay < 1:
        return np.nan, ["perm_entropy_invalid_delay"]

    n = int(x.size)
    n_vectors = n - (order - 1) * delay
    if n_vectors <= 1:
        return np.nan, ["perm_entropy_too_short"]

    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(n_vectors):
        window = x[i : i + order * delay : delay]
        perm = tuple(np.argsort(window, kind="mergesort"))
        counts[perm] = counts.get(perm, 0) + 1

    probs = np.asarray(list(counts.values()), dtype=float)
    probs /= probs.sum()
    probs = np.clip(probs, 1e-18, 1.0)

    ent = -np.sum(probs * np.log(probs))
    if normalized:
        denom = np.log(float(factorial(order)))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, ["perm_entropy_norm_invalid"]
        ent = ent / denom

    if not np.isfinite(ent):
        return np.nan, ["perm_entropy_nonfinite"]
    return float(ent), []


def _lz76_complexity(binary_sequence: np.ndarray) -> int:
    s = "".join(binary_sequence.astype(str).tolist())
    n = len(s)
    if n == 0:
        return 0

    i = 0
    c = 1
    l = 1
    k = 1
    k_max = 1
    stop = False

    while True:
        if i + k > n or l + k > n:
            c += 1
            break

        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l >= n:
                    stop = True
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    if stop:
        return c
    return c


def lz_complexity(
    x: np.ndarray,
    binarization: str = "median",
    normalized: bool = True,
) -> Tuple[float, List[str]]:
    if x.size < 8:
        return np.nan, ["lz_too_short"]

    method = str(binarization).strip().lower()
    if method == "median":
        threshold = float(np.median(x))
        bits = (x > threshold).astype(int)
    elif method == "mean":
        threshold = float(np.mean(x))
        bits = (x > threshold).astype(int)
    elif method == "sign":
        bits = (x > 0.0).astype(int)
    else:
        return np.nan, ["lz_binarization_unsupported"]

    if np.unique(bits).size < 2:
        return np.nan, ["lz_single_symbol"]

    c = float(_lz76_complexity(bits))
    if not np.isfinite(c):
        return np.nan, ["lz_nonfinite"]

    if normalized:
        n = float(bits.size)
        c = c * np.log2(n) / n

    if not np.isfinite(c):
        return np.nan, ["lz_nonfinite"]
    return float(c), []


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> Tuple[float, List[str]]:
    if m < 1:
        return np.nan, ["sampen_invalid_m"]
    if r_ratio <= 0:
        return np.nan, ["sampen_invalid_r_ratio"]
    if x.size <= (m + 2):
        return np.nan, ["sampen_too_short"]

    sigma = float(np.std(x, ddof=1))
    if sigma <= 0 or not np.isfinite(sigma):
        return np.nan, ["sampen_zero_std"]

    r = float(r_ratio * sigma)

    t_m = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)
    t_m1 = np.lib.stride_tricks.sliding_window_view(x, window_shape=m + 1)

    if t_m.shape[0] < 2 or t_m1.shape[0] < 2:
        return np.nan, ["sampen_insufficient_templates"]

    d_m = pdist(t_m, metric="chebyshev")
    d_m1 = pdist(t_m1, metric="chebyshev")

    b = int(np.count_nonzero(d_m <= r))
    a = int(np.count_nonzero(d_m1 <= r))

    if b == 0:
        return np.nan, ["sampen_no_matches_m"]
    if a == 0:
        return np.nan, ["sampen_no_matches_m1"]

    value = -np.log(a / b)
    if not np.isfinite(value):
        return np.nan, ["sampen_nonfinite"]
    return float(value), []


def _feature_metric_columns(cfg: Dict[str, Any]) -> List[str]:
    configured = cfg.get("output", {}).get("metric_columns", [])
    if configured:
        return [str(c) for c in configured]
    return [
        "spectral_slope_beta",
        "spectral_entropy",
        "spectral_flatness",
        "noise_fn",
        "permutation_entropy",
        "lz_complexity",
        "sample_entropy",
    ]


def _compute_single_series_features(series_df: pd.DataFrame, dataset_profile: str, cfg: Dict[str, Any]) -> SeriesFeatureResult:
    sid = str(series_df["series_id"].iloc[0])
    ticker = str(series_df["ticker"].iloc[0])
    market = str(series_df["market"].iloc[0])

    raw = series_df.sort_values("date", kind="stable")["log_return"].to_numpy(dtype=float)
    x, clean_flags = _clean_series(raw)

    result = SeriesFeatureResult(
        series_id=sid,
        ticker=ticker,
        market=market,
        dataset_profile=dataset_profile,
        feature_block="B",
        n_obs_used=int(x.size),
    )
    for flag in clean_flags:
        result.add_warning(flag)

    if x.size < 8:
        for metric in _feature_metric_columns(cfg):
            result.metrics[metric] = np.nan
        result.add_warning("series_too_short_for_block_b")
        return result

    spectral_cfg = cfg.get("metrics", {}).get("spectral", {})
    x_demeaned = x - float(np.mean(x))

    slope, flags = spectral_slope_beta(x_demeaned, spectral_cfg=spectral_cfg)
    result.metrics["spectral_slope_beta"] = slope
    for flag in flags:
        result.add_warning(flag)

    ent, flags = spectral_entropy(x_demeaned, spectral_cfg=spectral_cfg)
    result.metrics["spectral_entropy"] = ent
    for flag in flags:
        result.add_warning(flag)

    flat, flags = spectral_flatness(x_demeaned, spectral_cfg=spectral_cfg)
    result.metrics["spectral_flatness"] = flat
    for flag in flags:
        result.add_warning(flag)

    nfn, flags = noise_fn(x)
    result.metrics["noise_fn"] = nfn
    for flag in flags:
        result.add_warning(flag)

    pe_cfg = cfg.get("metrics", {}).get("permutation_entropy", {})
    pe, flags = permutation_entropy(
        x,
        order=int(pe_cfg.get("order", 3)),
        delay=int(pe_cfg.get("delay", 1)),
        normalized=bool(pe_cfg.get("normalized", True)),
    )
    result.metrics["permutation_entropy"] = pe
    for flag in flags:
        result.add_warning(flag)

    lz_cfg = cfg.get("metrics", {}).get("lz_complexity", {})
    lz, flags = lz_complexity(
        x,
        binarization=str(lz_cfg.get("binarization", "median")),
        normalized=bool(lz_cfg.get("normalized", True)),
    )
    result.metrics["lz_complexity"] = lz
    for flag in flags:
        result.add_warning(flag)

    se_cfg = cfg.get("metrics", {}).get("sample_entropy", {})
    se, flags = sample_entropy(
        x,
        m=int(se_cfg.get("m", 2)),
        r_ratio=float(se_cfg.get("r_ratio", 0.2)),
    )
    result.metrics["sample_entropy"] = se
    for flag in flags:
        result.add_warning(flag)

    return result


def _metric_ranges(features_df: pd.DataFrame, metric_columns: List[str]) -> Dict[str, Dict[str, float]]:
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
    ranges = _metric_ranges(features_df, metric_columns)

    spectral_norm = bool(cfg.get("metrics", {}).get("spectral", {}).get("entropy_normalized", True))
    perm_norm = bool(cfg.get("metrics", {}).get("permutation_entropy", {}).get("normalized", True))
    lz_norm = bool(cfg.get("metrics", {}).get("lz_complexity", {}).get("normalized", True))

    spectral_entropy_vals = pd.to_numeric(features_df.get("spectral_entropy"), errors="coerce")
    flatness_vals = pd.to_numeric(features_df.get("spectral_flatness"), errors="coerce")
    perm_vals = pd.to_numeric(features_df.get("permutation_entropy"), errors="coerce")
    lz_vals = pd.to_numeric(features_df.get("lz_complexity"), errors="coerce")

    summary: Dict[str, Any] = {
        "series_total": int(len(features_df)),
        "series_successful": int((features_df["feature_status"] == "success").sum()),
        "series_with_warnings": int((features_df["feature_status"] != "success").sum()),
        "spectral_entropy_lt_0": int((spectral_entropy_vals < 0).sum()),
        "spectral_flatness_lt_0": int((flatness_vals < 0).sum()),
        "ranges": ranges,
        **nan_counts,
    }

    if spectral_norm:
        summary["spectral_entropy_outside_0_1"] = int(((spectral_entropy_vals < 0) | (spectral_entropy_vals > 1)).sum())
    if perm_norm:
        summary["permutation_entropy_outside_0_1"] = int(((perm_vals < 0) | (perm_vals > 1)).sum())
    if lz_norm:
        summary["lz_complexity_negative"] = int((lz_vals < 0).sum())

    return summary


def run_feature_block_b_pipeline(config_path: str = "configs/features_block_B_v1.yaml") -> FeatureBlockBOutputs:
    cfg = load_feature_block_config(config_path)
    run_name = str(cfg.get("run_name", "feature_block_B_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    logger.info("Feature block B started")
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
                feature_block="B",
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

    excel_out = export_features_block_b_excel(
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
            "spectral": cfg.get("metrics", {}).get("spectral", {}),
            "permutation_entropy": cfg.get("metrics", {}).get("permutation_entropy", {}),
            "lz_complexity": cfg.get("metrics", {}).get("lz_complexity", {}),
            "sample_entropy": cfg.get("metrics", {}).get("sample_entropy", {}),
        },
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Output series: %d", int(len(features_df)))
    logger.info("Series successful: %d", summary.get("series_successful", 0))
    logger.info("Series with warnings: %d", summary.get("series_with_warnings", 0))
    for col in metric_columns:
        logger.info("NaN %s: %d", col, int(summary.get(f"nan_{col}", 0)))

    logger.info("spectral_entropy < 0: %d", int(summary.get("spectral_entropy_lt_0", 0)))
    logger.info("spectral_flatness < 0: %d", int(summary.get("spectral_flatness_lt_0", 0)))
    if "spectral_entropy_outside_0_1" in summary:
        logger.info("spectral_entropy outside [0,1]: %d", int(summary["spectral_entropy_outside_0_1"]))
    if "permutation_entropy_outside_0_1" in summary:
        logger.info("permutation_entropy outside [0,1]: %d", int(summary["permutation_entropy_outside_0_1"]))

    logger.info("Processing errors: %d", errors)
    logger.info("Parquet path: %s", parquet_path)
    logger.info("Excel path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_block_B",
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

    return FeatureBlockBOutputs(
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
