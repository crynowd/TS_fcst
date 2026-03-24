from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


METRIC_COLUMNS = [
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


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(numer / denom)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true)
    mask = denom > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def _huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    e = y_true - y_pred
    ae = np.abs(e)
    quad = np.minimum(ae, delta)
    lin = ae - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if y_train.size < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(y_train)))
    mae = mean_absolute_error(y_true, y_pred)
    return _safe_div(float(mae), float(scale))


def _r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ref = float(np.nanmean(y_train)) if y_train.size else 0.0
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_ref = float(np.sum((y_true - ref) ** 2))
    return 1.0 - _safe_div(sse_model, sse_ref)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    logger: Any | None = None,
) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)

    result: dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}

    try:
        result["mae"] = float(mean_absolute_error(y_t, y_p))
    except Exception as exc:
        if logger:
            logger.warning("metric mae failed: %s", exc)

    try:
        result["mse"] = float(mean_squared_error(y_t, y_p))
        result["rmse"] = float(np.sqrt(result["mse"]))
    except Exception as exc:
        if logger:
            logger.warning("metric mse/rmse failed: %s", exc)

    try:
        result["mase"] = _mase(y_t, y_p, y_tr)
    except Exception as exc:
        if logger:
            logger.warning("metric mase failed: %s", exc)

    try:
        result["mape"] = _mape(y_t, y_p)
    except Exception as exc:
        if logger:
            logger.warning("metric mape failed: %s", exc)

    try:
        result["smape"] = _smape(y_t, y_p)
    except Exception as exc:
        if logger:
            logger.warning("metric smape failed: %s", exc)

    try:
        result["huber"] = _huber(y_t, y_p)
    except Exception as exc:
        if logger:
            logger.warning("metric huber failed: %s", exc)

    try:
        result["directional_accuracy"] = _directional_accuracy(y_t, y_p)
    except Exception as exc:
        if logger:
            logger.warning("metric directional_accuracy failed: %s", exc)

    try:
        result["medae"] = float(median_absolute_error(y_t, y_p))
    except Exception as exc:
        if logger:
            logger.warning("metric medae failed: %s", exc)

    try:
        result["bias"] = float(np.mean(y_p - y_t))
    except Exception as exc:
        if logger:
            logger.warning("metric bias failed: %s", exc)

    try:
        result["r2"] = float(r2_score(y_t, y_p)) if y_t.size > 1 else float("nan")
    except Exception as exc:
        if logger:
            logger.warning("metric r2 failed: %s", exc)

    try:
        result["r2_oos"] = _r2_oos(y_t, y_p, y_tr)
    except Exception as exc:
        if logger:
            logger.warning("metric r2_oos failed: %s", exc)

    return result
