from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.forecasting.adapters.base import FitContext, ForecastModelAdapter


class RidgeLagAdapter(ForecastModelAdapter):
    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self._name = "ridge_lag"
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        return self.model.predict(X).astype(np.float64)

    def get_model_name(self) -> str:
        return self._name

    def get_model_config(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "random_state": self.random_state}
