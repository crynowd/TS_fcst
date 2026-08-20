from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.targets import build_same_horizon_feature_series


@dataclass
class SupervisedWindowData:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    feature_start_idx: np.ndarray
    feature_end_idx: np.ndarray
    target_start_idx: np.ndarray
    target_end_idx: np.ndarray


@dataclass
class FoldSlice:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


TEMPORAL_SPLIT_POLICY_VERSION = "target_end_lte_right_origin_v1"


@dataclass
class LeakageSafeFoldSlice:
    fold_id: int
    outer_train_idx_before_purge: np.ndarray
    outer_train_idx: np.ndarray
    fit_idx_before_purge: np.ndarray
    fit_idx: np.ndarray
    validation_idx_before_purge: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray


def build_supervised_windows(
    series_df: pd.DataFrame,
    horizon: int,
    window_size: int,
) -> SupervisedWindowData:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    returns = series_df["log_return"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(series_df["date"]).to_numpy()

    feat_h = build_same_horizon_feature_series(returns, horizon=horizon)

    start_t = (horizon - 1) + (window_size - 1)
    end_t = len(returns) - horizon - 1

    if end_t < start_t:
        return SupervisedWindowData(
            X=np.empty((0, window_size), dtype=np.float64),
            y=np.empty((0,), dtype=np.float64),
            timestamps=np.empty((0,), dtype="datetime64[ns]"),
            feature_start_idx=np.empty((0,), dtype=np.int64),
            feature_end_idx=np.empty((0,), dtype=np.int64),
            target_start_idx=np.empty((0,), dtype=np.int64),
            target_end_idx=np.empty((0,), dtype=np.int64),
        )

    t_arr = np.arange(start_t, end_t + 1, dtype=np.int64)
    X = np.stack([feat_h[t - window_size + 1 : t + 1] for t in t_arr], axis=0)

    cs = np.concatenate(([0.0], np.cumsum(returns)))
    start = t_arr + 1
    end = t_arr + horizon + 1
    y = (cs[end] - cs[start]).astype(np.float64)

    timestamps = dates[t_arr]

    feature_start_idx = t_arr - window_size + 1
    feature_end_idx = t_arr.copy()
    target_start_idx = t_arr + 1
    target_end_idx = t_arr + horizon

    if not np.all(feature_end_idx < target_start_idx):
        raise AssertionError("supervised windows leak target observations into features")
    if not np.all(target_end_idx < len(returns)):
        raise AssertionError("target window exceeds available series history")

    return SupervisedWindowData(
        X=X,
        y=y,
        timestamps=timestamps,
        feature_start_idx=feature_start_idx.astype(np.int64),
        feature_end_idx=feature_end_idx.astype(np.int64),
        target_start_idx=target_start_idx.astype(np.int64),
        target_end_idx=target_end_idx.astype(np.int64),
    )


def build_rolling_origin_folds(n_samples: int, n_folds: int) -> list[FoldSlice]:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if n_samples <= n_folds:
        return []

    test_size = max(1, n_samples // (n_folds + 1))
    initial_train = n_samples - (n_folds * test_size)
    if initial_train < 1:
        return []

    folds: list[FoldSlice] = []
    for fold_id in range(1, n_folds + 1):
        train_end = initial_train + (fold_id - 1) * test_size
        test_end = min(train_end + test_size, n_samples)
        if train_end <= 0 or test_end <= train_end:
            continue
        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_idx = np.arange(train_end, test_end, dtype=np.int64)
        if len(np.intersect1d(train_idx, test_idx)) > 0:
            raise AssertionError("rolling-origin fold has overlapping train/test indices")
        if train_idx.size and test_idx.size and train_idx.max() >= test_idx.min():
            raise AssertionError("rolling-origin fold is not time ordered")
        folds.append(FoldSlice(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx))
    return folds


def purge_left_partition(
    supervised: SupervisedWindowData,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> np.ndarray:
    """Keep left samples whose targets are observable by the first right origin."""
    left = np.asarray(left_idx, dtype=np.int64)
    right = np.asarray(right_idx, dtype=np.int64)
    if left.size == 0 or right.size == 0:
        return left.copy()

    boundary_origin = int(supervised.feature_end_idx[right[0]])
    keep = supervised.target_end_idx[left] <= boundary_origin
    return left[keep]


def build_leakage_safe_fold(
    supervised: SupervisedWindowData,
    fold: FoldSlice,
    validation_fraction: float = 0.2,
) -> LeakageSafeFoldSlice:
    """Build target-aware outer train and chronological neural fit/validation slices."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    outer_before = np.asarray(fold.train_idx, dtype=np.int64)
    test_idx = np.asarray(fold.test_idx, dtype=np.int64)
    outer_train_idx = purge_left_partition(supervised, outer_before, test_idx)

    val_size = max(1, int(validation_fraction * len(outer_train_idx))) if len(outer_train_idx) else 0
    if len(outer_train_idx) - val_size < 1:
        val_size = 0

    if val_size > 0:
        fit_before = outer_train_idx[:-val_size]
        validation_before = outer_train_idx[-val_size:]
        validation_idx = purge_left_partition(supervised, validation_before, test_idx)
        fit_idx = purge_left_partition(supervised, fit_before, validation_idx)
    else:
        fit_before = outer_train_idx.copy()
        fit_idx = outer_train_idx.copy()
        validation_before = np.empty((0,), dtype=np.int64)
        validation_idx = np.empty((0,), dtype=np.int64)

    return LeakageSafeFoldSlice(
        fold_id=int(fold.fold_id),
        outer_train_idx_before_purge=outer_before.copy(),
        outer_train_idx=outer_train_idx,
        fit_idx_before_purge=fit_before,
        fit_idx=fit_idx,
        validation_idx_before_purge=validation_before,
        validation_idx=validation_idx,
        test_idx=test_idx.copy(),
    )
