from __future__ import annotations

from typing import Any

import numpy as np

from src.forecasting.adapters.base import FitContext, ForecastModelAdapter
from src.forecasting.architectures.esn import ChaoticESN, ESN
from src.forecasting.architectures.transient_esn import TransientChaoticESN


class _BaseESNAdapter(ForecastModelAdapter):
    def __init__(self, name: str, model: ESN) -> None:
        self._name = name
        self.model = model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        self.model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        return self.model.predict(X.astype(np.float32)).astype(np.float64)

    def get_model_name(self) -> str:
        return self._name

    def get_model_config(self) -> dict[str, Any]:
        return {
            "n_inputs": int(self.model.n_inputs),
            "n_reservoir": int(self.model.n_reservoir),
            "spectral_radius": float(self.model.spectral_radius),
            "input_scale": float(self.model.input_scale),
            "leak_rate": float(self.model.leak_rate),
            "ridge_alpha": float(self.model.ridge_alpha),
        }


class ESNAdapter(_BaseESNAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="esn", model=ESN(**kwargs))


class ChaoticESNAdapter(_BaseESNAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="chaotic_esn", model=ChaoticESN(**kwargs))


class TransientChaoticESNAdapter(_BaseESNAdapter):
    def __init__(self, **kwargs: Any) -> None:
        model = TransientChaoticESN(**kwargs)
        super().__init__(name="transient_chaotic_esn", model=model)

    def get_model_config(self) -> dict[str, Any]:
        cfg = super().get_model_config()
        cfg.update(
            {
                "g_start": float(self.model.g_start),
                "g_end": float(self.model.g_end),
            }
        )
        return cfg
