from __future__ import annotations

from typing import Any

import numpy as np

from src.forecasting.adapters.base import FitContext, ForecastModelAdapter


class NaiveZeroAdapter(ForecastModelAdapter):
    def __init__(self) -> None:
        self._name = "naive_zero"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        return None

    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)

    def get_model_name(self) -> str:
        return self._name

    def get_model_config(self) -> dict[str, Any]:
        return {}


class NaiveMeanAdapter(ForecastModelAdapter):
    def __init__(self) -> None:
        self._name = "naive_mean"
        self._mean = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        self._mean = float(np.nanmean(y_train)) if y_train.size else 0.0

    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        return np.full(X.shape[0], self._mean, dtype=np.float64)

    def get_model_name(self) -> str:
        return self._name

    def get_model_config(self) -> dict[str, Any]:
        return {"train_mean": self._mean}
