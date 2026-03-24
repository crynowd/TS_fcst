from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class TaskTimeoutError(TimeoutError):
    """Raised when model training or prediction exceeds task-level timeout."""


@dataclass
class FitContext:
    max_train_seconds: float | None = None
    max_predict_seconds: float | None = None
    max_epochs: int = 20
    early_stopping_patience: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    verbose: bool = False


class ForecastModelAdapter(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_model_config(self) -> dict[str, Any]:
        raise NotImplementedError
