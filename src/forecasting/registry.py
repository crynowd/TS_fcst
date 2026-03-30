from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.forecasting.adapters import (
    ChaoticESNAdapter,
    ChaoticLSTMForecastAdapter,
    ChaoticLogisticNetAdapter,
    ChaoticMLPAdapter,
    ESNAdapter,
    ForecastModelAdapter,
    LSTMForecastAdapter,
    NaiveMeanAdapter,
    NaiveZeroAdapter,
    RidgeLagAdapter,
    TransientChaoticESNAdapter,
    VanillaMLPAdapter,
)


@dataclass
class ModelSpec:
    name: str
    family: str
    enabled_default: bool
    smoke_default: bool
    factory: Callable[[dict[str, Any]], ForecastModelAdapter]
    notes: str


def _pop_runtime_keys(overrides: dict[str, Any]) -> dict[str, Any]:
    drop_keys = {
        "max_train_seconds_per_task",
        "max_predict_seconds_per_task",
        "max_epochs",
        "early_stopping_patience",
        "batch_size",
        "learning_rate",
        "enabled",
        "run",
    }
    return {k: v for k, v in overrides.items() if k not in drop_keys}


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "naive_zero": ModelSpec(
            name="naive_zero",
            family="baseline",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: NaiveZeroAdapter(),
            notes="direct",
        ),
        "naive_mean": ModelSpec(
            name="naive_mean",
            family="baseline",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: NaiveMeanAdapter(),
            notes="direct",
        ),
        "ridge_lag": ModelSpec(
            name="ridge_lag",
            family="linear",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: RidgeLagAdapter(**kwargs),
            notes="adapter over sklearn.Ridge",
        ),
        "esn": ModelSpec(
            name="esn",
            family="reservoir",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: ESNAdapter(**kwargs),
            notes="direct architecture fit/predict",
        ),
        "chaotic_esn": ModelSpec(
            name="chaotic_esn",
            family="reservoir",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: ChaoticESNAdapter(**kwargs),
            notes="direct architecture fit/predict",
        ),
        "transient_chaotic_esn": ModelSpec(
            name="transient_chaotic_esn",
            family="reservoir",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: TransientChaoticESNAdapter(**kwargs),
            notes="direct architecture fit/predict",
        ),
        "vanilla_mlp": ModelSpec(
            name="vanilla_mlp",
            family="torch",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: VanillaMLPAdapter(**kwargs),
            notes="torch trainer wrapper",
        ),
        "chaotic_mlp": ModelSpec(
            name="chaotic_mlp",
            family="torch",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: ChaoticMLPAdapter(**kwargs),
            notes="torch trainer wrapper",
        ),
        "chaotic_logistic_net": ModelSpec(
            name="chaotic_logistic_net",
            family="torch",
            enabled_default=True,
            smoke_default=True,
            factory=lambda kwargs: ChaoticLogisticNetAdapter(**kwargs),
            notes="torch trainer wrapper",
        ),
        "lstm_forecast": ModelSpec(
            name="lstm_forecast",
            family="torch",
            enabled_default=True,
            smoke_default=False,
            factory=lambda kwargs: LSTMForecastAdapter(**kwargs),
            notes="supported, inactive in smoke by config",
        ),
        "chaotic_lstm_forecast": ModelSpec(
            name="chaotic_lstm_forecast",
            family="torch",
            enabled_default=True,
            smoke_default=False,
            factory=lambda kwargs: ChaoticLSTMForecastAdapter(**kwargs),
            notes="supported, inactive in smoke by config",
        ),
    }


def build_model(model_name: str, config: dict[str, Any], logger: Any | None = None) -> ForecastModelAdapter:
    specs = get_model_specs()
    if model_name not in specs:
        raise KeyError(f"Model not registered: {model_name}")

    model_cfg = dict(config.get("model_overrides", {}).get(model_name, {}))
    clean_cfg = _pop_runtime_keys(model_cfg)
    if specs[model_name].family == "torch":
        clean_cfg.setdefault("device", str(config.get("device", "cpu")))
        clean_cfg.setdefault("logger", logger)
    return specs[model_name].factory(clean_cfg)


def build_model_registry_table(
    active_models: list[str],
    inactive_models: list[str],
    model_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs = get_model_specs()
    metadata = model_metadata or {}
    rows: list[dict[str, Any]] = []
    active = set(active_models)
    inactive = set(inactive_models)
    for name in sorted(specs.keys()):
        spec = specs[name]
        if name in active:
            run_status = "active"
        elif name in inactive:
            run_status = "inactive_but_supported"
        else:
            run_status = "registered_not_selected"
        rows.append(
            {
                "model_name": name,
                "family": spec.family,
                "run_status": run_status,
                "smoke_default": spec.smoke_default,
                "integration": spec.notes,
                "candidate_id": str(metadata.get(name, {}).get("candidate_id", "")),
                "selection_role": str(metadata.get(name, {}).get("selection_role", "")),
            }
        )
    return rows
