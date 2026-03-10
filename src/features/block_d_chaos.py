from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf

from src.config.loader import load_feature_block_config
from src.features.base import SeriesFeatureResult
from src.reporting.excel_export import export_features_block_d_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


@dataclass(frozen=True)
class FeatureBlockDOutputs:
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
    configured = cfg.get("metrics", {}).get("metric_columns", [])
    if configured:
        return [str(c) for c in configured]
    configured_out = cfg.get("output", {}).get("metric_columns", [])
    if configured_out:
        return [str(c) for c in configured_out]
    return [
        "selected_delay_tau",
        "tau_selection_method",
        "fnn_success_flag",
        "fnn_selection_mode",
        "fnn_min_fraction",
        "embedding_dimension",
        "correlation_dimension",
        "largest_lyapunov_exponent",
        "lyapunov_time",
    ]


def _clean_and_demean(values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(values, dtype=float)
    flags: List[str] = []
    finite_mask = np.isfinite(x)
    if not np.all(finite_mask):
        flags.append("nonfinite_removed")
    x = x[finite_mask]
    if x.size == 0:
        flags.append("series_empty_after_clean")
        return x, flags
    x = x - float(np.mean(x))
    return x, flags


def _mutual_information_lag(x: np.ndarray, tau: int, n_bins: int) -> float:
    if tau < 1 or x.size <= tau + 4:
        return np.nan
    x1 = x[:-tau]
    x2 = x[tau:]
    if x1.size < 5:
        return np.nan

    hist2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
    pxy = hist2d / np.sum(hist2d)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)

    nz = pxy > 0
    if not np.any(nz):
        return np.nan
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz]))
    return float(mi) if np.isfinite(mi) else np.nan


def select_delay_tau(x: np.ndarray, delay_cfg: Dict[str, Any]) -> Tuple[int, str, Dict[str, float], List[str]]:
    tau_min = max(1, int(delay_cfg.get("tau_min", 1)))
    tau_max = max(tau_min, int(delay_cfg.get("tau_max", 50)))
    ami_bins = max(6, int(delay_cfg.get("ami_bins", 24)))

    flags: List[str] = []
    diagnostics: Dict[str, float] = {}

    if x.size <= tau_min + 4:
        flags.append("tau_selection_series_too_short")
        return tau_min, "fallback_default", diagnostics, flags

    tau_hi = min(tau_max, int(x.size // 3))
    if tau_hi < tau_min:
        flags.append("tau_selection_tau_range_empty")
        return tau_min, "fallback_default", diagnostics, flags

    taus = list(range(tau_min, tau_hi + 1))
    ami_vals = np.asarray([_mutual_information_lag(x, t, n_bins=ami_bins) for t in taus], dtype=float)
    diagnostics["ami_candidates"] = float(np.isfinite(ami_vals).sum())
    if np.isfinite(ami_vals).sum() >= 3:
        for i in range(1, len(ami_vals) - 1):
            if np.isfinite(ami_vals[i - 1]) and np.isfinite(ami_vals[i]) and np.isfinite(ami_vals[i + 1]):
                if ami_vals[i] < ami_vals[i - 1] and ami_vals[i] <= ami_vals[i + 1]:
                    return int(taus[i]), "ami", diagnostics, flags

    try:
        acf_vals = acf(x, nlags=tau_hi, fft=True, missing="raise")
    except Exception:
        acf_vals = np.asarray([], dtype=float)
        flags.append("tau_acf_failed")

    if acf_vals.size > tau_min:
        for tau in range(tau_min, min(tau_hi, len(acf_vals) - 1) + 1):
            prev = float(acf_vals[tau - 1]) if tau - 1 >= 0 else np.nan
            cur = float(acf_vals[tau])
            if np.isfinite(cur) and ((cur <= 0.0) or (np.isfinite(prev) and prev > 0.0 and cur <= 0.0)):
                return int(tau), "acf_zero", diagnostics, flags

        threshold = float(np.exp(-1.0))
        for tau in range(tau_min, min(tau_hi, len(acf_vals) - 1) + 1):
            cur = float(acf_vals[tau])
            if np.isfinite(cur) and cur < threshold:
                return int(tau), "acf_einv", diagnostics, flags

    flags.append("tau_fallback_default")
    return tau_min, "fallback_default", diagnostics, flags


def _phase_space_reconstruct(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = int(x.size)
    n_vec = n - (m - 1) * tau
    if n_vec <= 1:
        return np.empty((0, m), dtype=float)
    cols = [x[i * tau : i * tau + n_vec] for i in range(m)]
    return np.column_stack(cols)


def select_embedding_dimension_fnn_ratios(
    ratios: Dict[int, float],
    emb_cfg: Dict[str, Any],
) -> Tuple[float, str, float]:
    if not ratios:
        return np.nan, "failed", np.nan

    fnn_threshold = float(emb_cfg.get("fnn_threshold", 0.05))
    plateau_tol = float(emb_cfg.get("fnn_plateau_tol", 0.01))
    min_improvement = float(emb_cfg.get("fnn_min_improvement", 0.03))

    items = sorted((int(m), float(v)) for m, v in ratios.items() if np.isfinite(v))
    if not items:
        return np.nan, "failed", np.nan

    ms = [m for m, _ in items]
    vals = [v for _, v in items]
    min_fraction = float(np.min(vals))

    strict = [m for m, v in items if v <= fnn_threshold]
    if strict:
        return float(strict[0]), "strict_threshold", min_fraction

    if len(items) >= 3:
        for i in range(1, len(items) - 1):
            cur_v = vals[i]
            tail_vals = vals[i:]
            tail_diffs = [abs(tail_vals[j] - tail_vals[j - 1]) for j in range(1, len(tail_vals))]
            remaining_drop = max(0.0, cur_v - min(tail_vals))
            if tail_diffs and max(tail_diffs) <= plateau_tol and remaining_drop <= min_improvement:
                return float(ms[i]), "stable_plateau", min_fraction

    best_m = min(items, key=lambda kv: kv[1])[0]
    return float(best_m), "best_available", min_fraction


def select_embedding_dimension_fnn(
    x: np.ndarray,
    tau: int,
    emb_cfg: Dict[str, Any],
) -> Tuple[float, bool, str, float, Dict[int, float], List[str]]:
    m_min = max(2, int(emb_cfg.get("m_min", 2)))
    m_max = max(m_min, int(emb_cfg.get("m_max", 10)))
    fnn_rtol = float(emb_cfg.get("fnn_rtol", 10.0))
    fnn_atol_std = float(emb_cfg.get("fnn_atol_std", 2.0))

    flags: List[str] = []
    ratios: Dict[int, float] = {}
    sigma = float(np.std(x, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan, False, "failed", np.nan, ratios, ["fnn_zero_std", "fnn_unstable"]

    for m in range(m_min, m_max + 1):
        x_m = _phase_space_reconstruct(x, m=m, tau=tau)
        n_aligned = int(x.size - m * tau)
        if x_m.shape[0] < 5 or n_aligned < 5:
            continue
        x_m = x_m[:n_aligned, :]
        tail = x[m * tau : m * tau + n_aligned]

        try:
            tree = cKDTree(x_m)
            dist, idx = tree.query(x_m, k=2)
        except Exception:
            flags.append("fnn_tree_query_failed")
            continue

        nn_dist = np.asarray(dist[:, 1], dtype=float)
        nn_idx = np.asarray(idx[:, 1], dtype=int)
        valid_nn = (nn_idx >= 0) & (nn_idx < n_aligned) & np.isfinite(nn_dist)
        if np.count_nonzero(valid_nn) < 5:
            continue

        delta = np.abs(tail[valid_nn] - tail[nn_idx[valid_nn]])
        base_dist = np.maximum(nn_dist[valid_nn], 1e-12)
        crit1 = delta / base_dist > fnn_rtol
        crit2 = delta / sigma > fnn_atol_std
        fnn_ratio = float(np.mean(crit1 | crit2))
        if np.isfinite(fnn_ratio):
            ratios[m] = fnn_ratio

    if not ratios:
        return np.nan, False, "failed", np.nan, ratios, ["fnn_unavailable", "fnn_unstable"]

    m, mode, min_fraction = select_embedding_dimension_fnn_ratios(ratios, emb_cfg=emb_cfg)
    if mode == "failed":
        flags.append("fnn_unstable")
        return np.nan, False, mode, min_fraction, ratios, flags

    fnn_success = mode in {"strict_threshold", "stable_plateau", "best_available"}
    return float(m), fnn_success, mode, float(min_fraction), ratios, flags


def estimate_correlation_dimension(
    x: np.ndarray,
    tau: int,
    m: int,
    corr_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float], List[str]]:
    flags: List[str] = []
    diagnostics: Dict[str, float] = {}
    if not bool(corr_cfg.get("enabled", True)):
        return np.nan, diagnostics, ["correlation_dimension_disabled"]

    x_m = _phase_space_reconstruct(x, m=m, tau=tau)
    if x_m.shape[0] < 20:
        return np.nan, diagnostics, ["correlation_dimension_too_short"]

    max_points = max(50, int(corr_cfg.get("max_points", 800)))
    if x_m.shape[0] > max_points:
        idx = np.linspace(0, x_m.shape[0] - 1, num=max_points, dtype=int)
        x_m = x_m[idx]

    dists = pdist(x_m)
    dists = dists[np.isfinite(dists) & (dists > 0)]
    if dists.size < 50:
        return np.nan, diagnostics, ["correlation_dimension_insufficient_pairs"]

    q_low = float(corr_cfg.get("radius_quantile_low", 0.05))
    q_high = float(corr_cfg.get("radius_quantile_high", 0.8))
    q_low = min(max(q_low, 0.001), 0.5)
    q_high = min(max(q_high, q_low + 0.01), 0.99)
    r_lo = float(np.quantile(dists, q_low))
    r_hi = float(np.quantile(dists, q_high))
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_lo <= 0 or r_hi <= r_lo:
        return np.nan, diagnostics, ["correlation_dimension_invalid_radius_range"]

    grid_size = max(8, int(corr_cfg.get("radius_grid_size", 24)))
    radii = np.logspace(np.log10(r_lo), np.log10(r_hi), num=grid_size)
    cvals = np.asarray([np.mean(dists < r) for r in radii], dtype=float)

    valid = np.isfinite(cvals) & (cvals > 0) & (cvals < 1)
    if np.count_nonzero(valid) < int(corr_cfg.get("min_scaling_points", 6)):
        return np.nan, diagnostics, ["correlation_dimension_insufficient_correlation_points"]

    log_r = np.log(radii[valid])
    log_c = np.log(cvals[valid])

    min_pts = max(3, int(corr_cfg.get("min_scaling_points", 6)))
    min_r2 = float(corr_cfg.get("min_r2", 0.9))
    best: Tuple[float, float, int] | None = None
    for i in range(0, log_r.size - min_pts + 1):
        xw = log_r[i : i + min_pts]
        yw = log_c[i : i + min_pts]
        fit = linregress(xw, yw)
        if not np.isfinite(fit.slope) or not np.isfinite(fit.rvalue):
            continue
        r2 = float(fit.rvalue**2)
        if (best is None) or (r2 > best[1]):
            best = (float(fit.slope), r2, i)

    if best is None:
        return np.nan, diagnostics, ["correlation_dimension_scaling_fit_failed"]
    slope, r2, start_idx = best
    diagnostics["corr_dim_fit_r2"] = float(r2)
    diagnostics["corr_dim_fit_start_idx"] = float(start_idx)
    diagnostics["corr_dim_fit_points"] = float(min_pts)
    if r2 < min_r2:
        return np.nan, diagnostics, ["correlation_dimension_no_reasonable_scaling_region"]
    if not np.isfinite(slope):
        return np.nan, diagnostics, ["correlation_dimension_nonfinite"]
    if slope < 0:
        flags.append("correlation_dimension_negative")
    return float(slope), diagnostics, flags


def estimate_largest_lyapunov_rosenstein(
    x: np.ndarray,
    tau: int,
    m: int,
    lya_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float], List[str]]:
    flags: List[str] = []
    diagnostics: Dict[str, float] = {}

    method = str(lya_cfg.get("method", "rosenstein")).strip().lower()
    if method != "rosenstein":
        return np.nan, diagnostics, ["lyapunov_method_unsupported"]

    x_m = _phase_space_reconstruct(x, m=m, tau=tau)
    if x_m.shape[0] < 30:
        return np.nan, diagnostics, ["lyapunov_too_short"]

    trajectory_len = max(3, int(lya_cfg.get("trajectory_len", 20)))
    theiler = max(0, int(lya_cfg.get("theiler_window", 10)))
    min_pairs = max(5, int(lya_cfg.get("min_pairs", 20)))
    neighbor_k = max(2, int(lya_cfg.get("neighbor_search_k", 50)))

    valid_n = x_m.shape[0] - trajectory_len
    if valid_n <= min_pairs:
        return np.nan, diagnostics, ["lyapunov_insufficient_valid_trajectories"]

    base = x_m[:valid_n]
    tree = cKDTree(base)
    kq = min(neighbor_k, valid_n)
    if kq <= 1:
        return np.nan, diagnostics, ["lyapunov_neighbor_search_too_small"]

    dists, nbrs = tree.query(base, k=kq)
    pairs: List[Tuple[int, int]] = []
    for i in range(valid_n):
        for pos in range(1, kq):
            j = int(nbrs[i, pos])
            dij = float(dists[i, pos])
            if j < 0 or j >= valid_n:
                continue
            if abs(i - j) <= theiler:
                continue
            if not np.isfinite(dij) or dij <= 0:
                continue
            pairs.append((i, j))
            break

    if len(pairs) < min_pairs:
        return np.nan, diagnostics, ["lyapunov_insufficient_neighbor_pairs"]

    diagnostics["lyapunov_pairs"] = float(len(pairs))
    mean_log_div = np.full(trajectory_len, np.nan, dtype=float)
    for k in range(trajectory_len):
        vals: List[float] = []
        for i, j in pairs:
            i2 = i + k
            j2 = j + k
            if i2 >= x_m.shape[0] or j2 >= x_m.shape[0]:
                continue
            d = float(np.linalg.norm(x_m[i2] - x_m[j2]))
            if np.isfinite(d) and d > 0:
                vals.append(np.log(d))
        if len(vals) >= min_pairs:
            mean_log_div[k] = float(np.mean(vals))

    fit_start = max(0, int(lya_cfg.get("fit_start", 1)))
    fit_end = min(trajectory_len - 1, int(lya_cfg.get("fit_end", 10)))
    if fit_end <= fit_start:
        return np.nan, diagnostics, ["lyapunov_invalid_fit_window"]

    idx = np.arange(fit_start, fit_end + 1)
    y = mean_log_div[idx]
    mask = np.isfinite(y)
    min_fit_points = max(3, int(lya_cfg.get("min_fit_points", 5)))
    if np.count_nonzero(mask) < min_fit_points:
        return np.nan, diagnostics, ["lyapunov_insufficient_fit_points"]

    fit = linregress(idx[mask], y[mask])
    if not np.isfinite(fit.slope):
        return np.nan, diagnostics, ["lyapunov_nonfinite"]

    diagnostics["lyapunov_fit_r2"] = float(fit.rvalue**2) if np.isfinite(fit.rvalue) else np.nan
    diagnostics["lyapunov_fit_points"] = float(np.count_nonzero(mask))
    return float(fit.slope), diagnostics, flags


def compute_lyapunov_time(lle: float, min_exponent: float = 1e-3) -> Tuple[float, List[str]]:
    if not np.isfinite(lle):
        return np.nan, ["lyapunov_time_lle_nonfinite"]
    if lle <= 0:
        return np.nan, ["lyapunov_time_nonpositive_lle"]
    if lle < float(min_exponent):
        return np.nan, ["lyapunov_time_near_zero_lle"]
    value = 1.0 / lle
    if not np.isfinite(value):
        return np.nan, ["lyapunov_time_nonfinite"]
    return float(value), []


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
        feature_block="D",
        n_obs_used=int(x.size),
    )
    for flag in clean_flags:
        result.add_warning(flag)

    metric_columns = _feature_metric_columns(cfg)
    for col in metric_columns:
        if col == "tau_selection_method":
            result.metrics[col] = ""
        elif col == "fnn_selection_mode":
            result.metrics[col] = "failed"
        elif col == "fnn_success_flag":
            result.metrics[col] = False
        else:
            result.metrics[col] = np.nan

    min_len = int(cfg.get("metrics", {}).get("minimum_series_length", 300))
    if x.size < min_len:
        result.add_warning("series_too_short_for_block_d")
        return result

    tau, tau_method, _tau_diag, tau_flags = select_delay_tau(x, cfg.get("metrics", {}).get("delay_selection", {}))
    result.metrics["selected_delay_tau"] = float(tau)
    result.metrics["tau_selection_method"] = tau_method
    for flag in tau_flags:
        result.add_warning(flag)

    m, fnn_success, fnn_mode, fnn_min_fraction, _ratios, fnn_flags = select_embedding_dimension_fnn(
        x,
        tau=tau,
        emb_cfg=cfg.get("metrics", {}).get("embedding", {}),
    )
    result.metrics["fnn_success_flag"] = bool(fnn_success)
    result.metrics["fnn_selection_mode"] = str(fnn_mode)
    result.metrics["fnn_min_fraction"] = float(fnn_min_fraction) if np.isfinite(fnn_min_fraction) else np.nan
    result.metrics["embedding_dimension"] = float(m) if np.isfinite(m) else np.nan
    for flag in fnn_flags:
        result.add_warning(flag)

    if not np.isfinite(m):
        result.add_warning("embedding_dimension_unavailable")
        return result
    m_int = int(m)

    corr_dim, _corr_diag, corr_flags = estimate_correlation_dimension(
        x,
        tau=tau,
        m=m_int,
        corr_cfg=cfg.get("metrics", {}).get("correlation_dimension", {}),
    )
    result.metrics["correlation_dimension"] = corr_dim
    for flag in corr_flags:
        result.add_warning(flag)

    lle, _lya_diag, lya_flags = estimate_largest_lyapunov_rosenstein(
        x,
        tau=tau,
        m=m_int,
        lya_cfg=cfg.get("metrics", {}).get("lyapunov", {}),
    )
    result.metrics["largest_lyapunov_exponent"] = lle
    for flag in lya_flags:
        result.add_warning(flag)

    lya_cfg = cfg.get("metrics", {}).get("lyapunov", {})
    l_time, l_time_flags = compute_lyapunov_time(
        lle,
        min_exponent=float(lya_cfg.get("lyapunov_time_min_exponent", 1e-3)),
    )
    result.metrics["lyapunov_time"] = l_time
    for flag in l_time_flags:
        result.add_warning(flag)

    return result


def _metric_ranges(features_df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in columns:
        vals = pd.to_numeric(features_df[col], errors="coerce")
        out[col] = {
            "min": float(vals.min()) if vals.notna().any() else np.nan,
            "max": float(vals.max()) if vals.notna().any() else np.nan,
            "mean": float(vals.mean()) if vals.notna().any() else np.nan,
            "median": float(vals.median()) if vals.notna().any() else np.nan,
        }
    return out


def _summarize_feature_table(features_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m_min = int(cfg.get("metrics", {}).get("embedding", {}).get("m_min", 2))
    m_max = int(cfg.get("metrics", {}).get("embedding", {}).get("m_max", 10))

    tau_methods = ["ami", "acf_zero", "acf_einv", "fallback_default"]
    tau_breakdown = {
        method: int((features_df["tau_selection_method"].astype(str) == method).sum()) for method in tau_methods
    }

    key_cols = [
        "selected_delay_tau",
        "embedding_dimension",
        "fnn_min_fraction",
        "correlation_dimension",
        "largest_lyapunov_exponent",
        "lyapunov_time",
    ]
    ranges = _metric_ranges(features_df, key_cols)
    nan_counts = {f"nan_{col}": int(pd.to_numeric(features_df[col], errors="coerce").isna().sum()) for col in key_cols}

    lle = pd.to_numeric(features_df["largest_lyapunov_exponent"], errors="coerce")
    cdim = pd.to_numeric(features_df["correlation_dimension"], errors="coerce")
    emb = pd.to_numeric(features_df["embedding_dimension"], errors="coerce")
    fnn_modes = ["strict_threshold", "stable_plateau", "best_available", "failed"]
    fnn_breakdown = {
        mode: int((features_df["fnn_selection_mode"].astype(str) == mode).sum()) for mode in fnn_modes
    }
    warning_counter: Counter[str] = Counter()
    for raw in features_df["feature_warning_flags"].astype(str):
        if not raw:
            continue
        for token in raw.split(";"):
            token = token.strip()
            if token:
                warning_counter[token] += 1

    summary: Dict[str, Any] = {
        "series_total": int(len(features_df)),
        "series_successful": int((features_df["feature_status"] == "success").sum()),
        "series_with_warnings": int((features_df["feature_status"] != "success").sum()),
        "warning_rows": int((features_df["feature_warning_flags"].astype(str) != "").sum()),
        "warning_flag_breakdown": dict(sorted(warning_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tau_selection_breakdown": tau_breakdown,
        "fnn_selection_breakdown": fnn_breakdown,
        "exponent_nonpositive": int((lle <= 0).sum()),
        "lyapunov_time_nonpositive_lle": int(warning_counter.get("lyapunov_time_nonpositive_lle", 0)),
        "lyapunov_time_near_zero_lle": int(warning_counter.get("lyapunov_time_near_zero_lle", 0)),
        "correlation_dimension_negative": int((cdim < 0).sum()),
        "embedding_dimension_out_of_bounds": int(((emb < m_min) | (emb > m_max)).sum()),
        "ranges": ranges,
        **nan_counts,
    }
    return summary


def run_feature_block_d_pipeline(config_path: str = "configs/features_block_D_v1.yaml") -> FeatureBlockDOutputs:
    cfg = load_feature_block_config(config_path)
    run_name = str(cfg.get("run_name", "feature_block_D_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    logger.info("Feature block D started")
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
                feature_block="D",
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

    excel_out = export_features_block_d_excel(
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
        config_params=cfg.get("metrics", {}),
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Output series: %d", int(len(features_df)))
    logger.info("Series successful: %d", summary.get("series_successful", 0))
    logger.info("Series with warnings: %d", summary.get("series_with_warnings", 0))
    logger.info("Warning rows: %d", summary.get("warning_rows", 0))
    logger.info("Warning flag breakdown: %s", summary.get("warning_flag_breakdown", {}))
    logger.info("NaN embedding_dimension: %d", summary.get("nan_embedding_dimension", 0))
    logger.info("NaN fnn_min_fraction: %d", summary.get("nan_fnn_min_fraction", 0))
    logger.info("NaN correlation_dimension: %d", summary.get("nan_correlation_dimension", 0))
    logger.info("NaN largest_lyapunov_exponent: %d", summary.get("nan_largest_lyapunov_exponent", 0))
    logger.info("NaN lyapunov_time: %d", summary.get("nan_lyapunov_time", 0))
    logger.info("Tau selection breakdown: %s", summary.get("tau_selection_breakdown", {}))
    logger.info("FNN selection breakdown: %s", summary.get("fnn_selection_breakdown", {}))
    logger.info("largest_lyapunov_exponent <= 0: %d", summary.get("exponent_nonpositive", 0))
    logger.info("lyapunov_time_nonpositive_lle: %d", summary.get("lyapunov_time_nonpositive_lle", 0))
    logger.info("lyapunov_time_near_zero_lle: %d", summary.get("lyapunov_time_near_zero_lle", 0))
    logger.info("correlation_dimension < 0: %d", summary.get("correlation_dimension_negative", 0))
    logger.info("embedding_dimension outside bounds: %d", summary.get("embedding_dimension_out_of_bounds", 0))
    logger.info("Ranges: %s", summary.get("ranges", {}))
    logger.info("Processing errors: %d", errors)
    logger.info("Parquet path: %s", parquet_path)
    logger.info("Excel path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_block_D",
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

    return FeatureBlockDOutputs(
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
