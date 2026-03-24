from __future__ import annotations

import numpy as np


def build_direct_horizon_target(log_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Compute y_t^(h)=sum_{k=1..h} r_{t+k} for valid t positions."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    values = np.asarray(log_returns, dtype=np.float64)
    n = values.shape[0]
    if n <= horizon:
        return np.asarray([], dtype=np.float64)

    cs = np.concatenate(([0.0], np.cumsum(values)))
    # Valid anchors: t in [0, n-h-1]
    t_idx = np.arange(0, n - horizon, dtype=np.int64)
    start = t_idx + 1
    end = t_idx + horizon + 1
    target = cs[end] - cs[start]
    return target.astype(np.float64)


def build_same_horizon_feature_series(log_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Compute trailing h-step returns aligned on timestamp t: sum_{k=0..h-1} r_{t-k}."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    values = np.asarray(log_returns, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < horizon:
        return out

    cs = np.concatenate(([0.0], np.cumsum(values)))
    idx = np.arange(horizon - 1, n, dtype=np.int64)
    end = idx + 1
    start = end - horizon
    out[idx] = cs[end] - cs[start]
    return out
