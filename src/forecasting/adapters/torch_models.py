from __future__ import annotations

import time
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.adapters.base import FitContext, ForecastModelAdapter, TaskTimeoutError
from src.forecasting.architectures.chaotic_logistic import ChaoticLogisticNet
from src.forecasting.architectures.lstm import LSTMForecast, LSTMForecastChaotic
from src.forecasting.architectures.mlp import ChaoticMLP, VanillaMLP

InputMode = Literal["flat", "sequence"]


def _split_runtime_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_keys = {"seed", "device", "logger", "epoch_log_interval"}
    runtime = {k: kwargs[k] for k in runtime_keys if k in kwargs}
    model = {k: v for k, v in kwargs.items() if k not in runtime_keys}
    return runtime, model


class TorchRegressorAdapter(ForecastModelAdapter):
    def __init__(
        self,
        name: str,
        model_builder: Callable[[int], nn.Module],
        model_kwargs: dict[str, Any] | None = None,
        input_mode: InputMode = "flat",
        seed: int = 42,
        device: str = "cpu",
        logger: Any | None = None,
        epoch_log_interval: int = 5,
    ) -> None:
        self._name = name
        self._model_builder = model_builder
        self._model_kwargs = dict(model_kwargs or {})
        self.input_mode = input_mode
        self.seed = seed
        self.device = device
        self.logger = logger
        self.epoch_log_interval = epoch_log_interval
        self.model: nn.Module | None = None
        self._last_training_diagnostics: dict[str, Any] = {}

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        if self.input_mode == "flat":
            return torch.from_numpy(X.astype(np.float32))
        return torch.from_numpy(X.astype(np.float32)).unsqueeze(-1)

    def _ensure_model(self, input_size: int) -> None:
        if self.model is not None:
            return
        torch.manual_seed(self.seed)
        self.model = self._model_builder(input_size, **self._model_kwargs).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        context: FitContext | None = None,
    ) -> None:
        ctx = context or FitContext()
        X_tensor = self._prepare_input(X_train)
        self._ensure_model(int(X_tensor.shape[-1]))
        assert self.model is not None

        y_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1)
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = max(1, int(ctx.batch_size))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_X_tensor = self._prepare_input(X_val) if X_val is not None else None
        val_y_tensor = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(-1) if y_val is not None else None

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(ctx.learning_rate),
            weight_decay=float(ctx.weight_decay),
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        patience_left = int(ctx.early_stopping_patience)
        t0 = time.monotonic()
        train_loss_history: list[float] = []
        val_loss_history: list[float | None] = []
        epochs_completed = 0
        early_stopping_reason = "max_epochs_reached"

        for epoch in range(1, int(ctx.max_epochs) + 1):
            epochs_completed = epoch
            self.model.train()
            running = 0.0
            seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * xb.shape[0]
                seen += int(xb.shape[0])

            train_loss = running / max(1, seen)
            val_loss = float("nan")
            train_loss_history.append(float(train_loss))

            if val_X_tensor is not None and val_y_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    vx = val_X_tensor.to(self.device)
                    vy = val_y_tensor.to(self.device)
                    vpred = self.model(vx)
                    val_loss = float(criterion(vpred, vy).item())

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = int(ctx.early_stopping_patience)
                else:
                    patience_left -= 1
                val_loss_history.append(float(val_loss))
            else:
                val_loss_history.append(None)

            if self.logger is not None and (epoch == 1 or epoch % self.epoch_log_interval == 0):
                self.logger.info(
                    "epoch_progress model=%s epoch=%d/%d train_loss=%.6f val_loss=%s patience_left=%d",
                    self._name,
                    epoch,
                    int(ctx.max_epochs),
                    train_loss,
                    f"{val_loss:.6f}" if np.isfinite(val_loss) else "nan",
                    patience_left,
                )

            elapsed = time.monotonic() - t0
            if ctx.max_train_seconds is not None and elapsed > float(ctx.max_train_seconds):
                self._last_training_diagnostics = {
                    "epochs_completed": int(epochs_completed),
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                    "early_stopping_reason": "timeout",
                    "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
                    "final_train_loss": float(train_loss_history[-1]) if train_loss_history else None,
                    "final_val_loss": val_loss_history[-1] if val_loss_history else None,
                    "batch_size": int(batch_size),
                    "learning_rate": float(ctx.learning_rate),
                    "weight_decay": float(ctx.weight_decay),
                    "shuffle": True,
                    "device": self.device,
                }
                raise TaskTimeoutError(f"Training timeout after {elapsed:.2f}s")

            if val_X_tensor is not None and val_y_tensor is not None and patience_left <= 0:
                early_stopping_reason = "patience_exhausted"
                if self.logger is not None:
                    self.logger.info(
                        "early_stopping model=%s epoch=%d best_val_loss=%.6f",
                        self._name,
                        epoch,
                        best_val,
                    )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._last_training_diagnostics = {
            "epochs_completed": int(epochs_completed),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "early_stopping_reason": early_stopping_reason,
            "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
            "final_train_loss": float(train_loss_history[-1]) if train_loss_history else None,
            "final_val_loss": val_loss_history[-1] if val_loss_history else None,
            "batch_size": int(batch_size),
            "learning_rate": float(ctx.learning_rate),
            "weight_decay": float(ctx.weight_decay),
            "shuffle": True,
            "device": self.device,
        }

    def predict(self, X: np.ndarray, context: FitContext | None = None) -> np.ndarray:
        ctx = context or FitContext()
        assert self.model is not None, "Call fit() first"
        t0 = time.monotonic()
        X_tensor = self._prepare_input(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(X_tensor).squeeze(-1).detach().cpu().numpy().astype(np.float64)
        elapsed = time.monotonic() - t0
        if ctx.max_predict_seconds is not None and elapsed > float(ctx.max_predict_seconds):
            raise TaskTimeoutError(f"Prediction timeout after {elapsed:.2f}s")
        return y

    def get_model_name(self) -> str:
        return self._name

    def get_model_config(self) -> dict[str, Any]:
        return {
            "model_kwargs": self._model_kwargs,
            "input_mode": self.input_mode,
            "seed": self.seed,
            "device": self.device,
        }

    def get_training_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_training_diagnostics)


class VanillaMLPAdapter(TorchRegressorAdapter):
    def __init__(self, **kwargs: Any) -> None:
        runtime_kwargs, model_kwargs = _split_runtime_kwargs(kwargs)
        super().__init__(
            name="vanilla_mlp",
            model_builder=VanillaMLP,
            model_kwargs=model_kwargs,
            input_mode="flat",
            **runtime_kwargs,
        )


class ChaoticMLPAdapter(TorchRegressorAdapter):
    def __init__(self, **kwargs: Any) -> None:
        runtime_kwargs, model_kwargs = _split_runtime_kwargs(kwargs)
        super().__init__(
            name="chaotic_mlp",
            model_builder=ChaoticMLP,
            model_kwargs=model_kwargs,
            input_mode="flat",
            **runtime_kwargs,
        )


class ChaoticLogisticNetAdapter(TorchRegressorAdapter):
    def __init__(self, **kwargs: Any) -> None:
        runtime_kwargs, model_kwargs = _split_runtime_kwargs(kwargs)
        super().__init__(
            name="chaotic_logistic_net",
            model_builder=self._build_model,
            model_kwargs=model_kwargs,
            input_mode="flat",
            **runtime_kwargs,
        )

    @staticmethod
    def _build_model(input_size: int, **model_kwargs: Any) -> nn.Module:
        return ChaoticLogisticNet(window_size=input_size, **model_kwargs)


class LSTMForecastAdapter(TorchRegressorAdapter):
    def __init__(self, **kwargs: Any) -> None:
        runtime_kwargs, model_kwargs = _split_runtime_kwargs(kwargs)
        super().__init__(
            name="lstm_forecast",
            model_builder=LSTMForecast,
            model_kwargs=model_kwargs,
            input_mode="sequence",
            **runtime_kwargs,
        )


class ChaoticLSTMForecastAdapter(TorchRegressorAdapter):
    def __init__(self, **kwargs: Any) -> None:
        runtime_kwargs, model_kwargs = _split_runtime_kwargs(kwargs)
        super().__init__(
            name="chaotic_lstm_forecast",
            model_builder=LSTMForecastChaotic,
            model_kwargs=model_kwargs,
            input_mode="sequence",
            **runtime_kwargs,
        )
