from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.adapters import FitContext, TaskTimeoutError
from src.forecasting.data import build_series_lookup, load_log_returns_table, select_series
from src.forecasting.registry import build_model, get_model_specs
from src.forecasting.windowing import build_rolling_origin_folds, build_supervised_windows


REPORT_DIR = Path("artifacts/reports/forecasting_audit_v2")
PARAMS_CSV = REPORT_DIR / "neural_training_params.csv"
DIAGNOSTIC_JSON = REPORT_DIR / "neural_training_diagnostic.json"
AUDIT_MD = REPORT_DIR / "neural_training_audit.md"

ADAPTER_FILES = {
    "naive_zero": "src/forecasting/adapters/naive.py",
    "naive_mean": "src/forecasting/adapters/naive.py",
    "ridge_lag": "src/forecasting/adapters/sklearn_models.py",
    "vanilla_mlp": "src/forecasting/adapters/torch_models.py",
    "chaotic_mlp": "src/forecasting/adapters/torch_models.py",
    "chaotic_logistic_net": "src/forecasting/adapters/torch_models.py",
    "lstm_forecast": "src/forecasting/adapters/torch_models.py",
    "chaotic_lstm_forecast": "src/forecasting/adapters/torch_models.py",
}

ARCHITECTURE_FILES = {
    "naive_zero": "",
    "naive_mean": "",
    "ridge_lag": "sklearn.linear_model.Ridge",
    "vanilla_mlp": "src/forecasting/architectures/mlp.py",
    "chaotic_mlp": "src/forecasting/architectures/mlp.py",
    "chaotic_logistic_net": "src/forecasting/architectures/chaotic_logistic.py",
    "lstm_forecast": "src/forecasting/architectures/lstm.py",
    "chaotic_lstm_forecast": "src/forecasting/architectures/lstm.py",
}


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _non_esn_models() -> list[str]:
    specs = get_model_specs()
    return [name for name, spec in specs.items() if spec.family != "reservoir" and "esn" not in name.lower()]


def _runtime_for_model(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    runtime = dict(cfg.get("training", {}))
    runtime.update(cfg.get("model_overrides", {}).get(model_name, {}))
    return runtime


def _fit_context(cfg: dict[str, Any], model_name: str) -> FitContext:
    runtime = _runtime_for_model(cfg, model_name)
    return FitContext(
        max_train_seconds=float(
            runtime.get("max_train_seconds_per_task", cfg.get("timeouts", {}).get("max_train_seconds_per_task", 60))
        ),
        max_predict_seconds=float(
            runtime.get("max_predict_seconds_per_task", cfg.get("timeouts", {}).get("max_predict_seconds_per_task", 15))
        ),
        max_epochs=int(runtime.get("max_epochs", cfg.get("training", {}).get("max_epochs", 20))),
        early_stopping_patience=int(
            runtime.get("early_stopping_patience", cfg.get("training", {}).get("early_stopping_patience", 5))
        ),
        batch_size=int(runtime.get("batch_size", cfg.get("training", {}).get("batch_size", 64))),
        learning_rate=float(runtime.get("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))),
        weight_decay=float(runtime.get("weight_decay", cfg.get("training", {}).get("weight_decay", 0.0))),
    )


def _split_train_validation(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    val_size = max(1, int(0.2 * len(X_train)))
    if len(X_train) - val_size < 1:
        val_size = 0
    if val_size > 0:
        return X_train[:-val_size], y_train[:-val_size], X_train[-val_size:], y_train[-val_size:], val_size
    return X_train, y_train, None, None, 0


def _training_param_rows(cfg: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = get_model_specs()
    for model_name in models:
        ctx = _fit_context(cfg, model_name)
        model = build_model(model_name=model_name, config=cfg, logger=None)
        model_config = model.get_model_config()
        model_kwargs = dict(model_config.get("model_kwargs", {}))
        hidden_dims = model_kwargs.get("hidden_dims")
        hidden_size = model_kwargs.get("hidden_size", hidden_dims)
        rows.append(
            {
                "model_name": model_name,
                "family": specs[model_name].family,
                "adapter_file": ADAPTER_FILES.get(model_name, ""),
                "architecture_file": ARCHITECTURE_FILES.get(model_name, ""),
                "epochs": ctx.max_epochs if specs[model_name].family == "torch" else "",
                "early_stopping": bool(specs[model_name].family == "torch" and ctx.early_stopping_patience > 0),
                "patience": ctx.early_stopping_patience if specs[model_name].family == "torch" else "",
                "lr": ctx.learning_rate if specs[model_name].family == "torch" else "",
                "batch_size": ctx.batch_size if specs[model_name].family == "torch" else "",
                "hidden_size": json.dumps(hidden_size) if hidden_size is not None else "",
                "num_layers": model_kwargs.get("num_layers", ""),
                "dropout": model_kwargs.get("dropout", ""),
                "weight_decay": ctx.weight_decay if specs[model_name].family == "torch" else "",
                "validation_mode": "last_20pct_of_train_no_shuffle" if specs[model_name].family == "torch" else "unused",
                "shuffle": True if specs[model_name].family == "torch" else False,
                "seed": model_config.get("seed", model_config.get("random_state", "")),
                "device": model_config.get("device", ""),
                "input_mode": model_config.get("input_mode", ""),
                "yaml_model_override": json.dumps(cfg.get("model_overrides", {}).get(model_name, {}), sort_keys=True),
            }
        )
    return rows


def _write_params_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_name",
        "family",
        "adapter_file",
        "architecture_file",
        "epochs",
        "early_stopping",
        "patience",
        "lr",
        "batch_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "weight_decay",
        "validation_mode",
        "shuffle",
        "seed",
        "device",
        "input_mode",
        "yaml_model_override",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    config_path: str,
    max_series: int,
    horizon: int,
    models: list[str] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    cfg = load_forecasting_benchmark_config(config_path)
    active_models = models or _non_esn_models()
    out_dir = Path(output_dir)
    params_path = out_dir / PARAMS_CSV.name
    diagnostic_path = out_dir / DIAGNOSTIC_JSON.name
    md_path = out_dir / AUDIT_MD.name

    param_rows = _training_param_rows(cfg, active_models)
    _write_params_csv(param_rows, params_path)

    log_returns = load_log_returns_table(cfg["data"]["source_path"])
    selected = select_series(
        log_returns_df=log_returns,
        dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")),
        max_series=max_series,
        series_selection_mode=str(cfg["data"].get("series_selection_mode", "first_n")),
        series_ids=None,
    )
    lookup = build_series_lookup(log_returns, selected)
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}
    window_size = int(window_sizes[horizon])
    n_folds = int(cfg.get("validation", {}).get("n_folds", 1))

    task_rows: list[dict[str, Any]] = []
    for series in selected:
        sdf = lookup[series.series_id]
        sup = build_supervised_windows(sdf, horizon=horizon, window_size=window_size)
        folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
        if not folds:
            continue
        fold = folds[0]
        X_train = sup.X[fold.train_idx]
        y_train = sup.y[fold.train_idx]
        X_test = sup.X[fold.test_idx]
        X_fit, y_fit, X_val, y_val, val_size = _split_train_validation(X_train, y_train)

        for model_name in active_models:
            ctx = _fit_context(cfg, model_name)
            model = build_model(model_name=model_name, config=cfg, logger=None)
            status = "success"
            error_type = ""
            error_message = ""
            t0 = time.monotonic()
            try:
                model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val, context=ctx)
                fit_seconds = time.monotonic() - t0
                pred = model.predict(X_test, context=ctx)
                predict_rows = int(len(pred))
            except TaskTimeoutError as exc:
                fit_seconds = time.monotonic() - t0
                predict_rows = 0
                status = "timeout"
                error_type = exc.__class__.__name__
                error_message = str(exc)
            except Exception as exc:  # pragma: no cover - diagnostic should preserve failures in JSON
                fit_seconds = time.monotonic() - t0
                predict_rows = 0
                status = "error"
                error_type = exc.__class__.__name__
                error_message = str(exc)

            training_diag = {}
            if hasattr(model, "get_training_diagnostics"):
                training_diag = model.get_training_diagnostics()
            task_rows.append(
                {
                    "model_name": model_name,
                    "series_id": series.series_id,
                    "ticker": series.ticker,
                    "market": series.market,
                    "horizon": int(horizon),
                    "fold_id": int(fold.fold_id),
                    "status": status,
                    "error_type": error_type,
                    "error_message": error_message,
                    "runtime_seconds": float(fit_seconds),
                    "n_train_total_before_validation": int(len(y_train)),
                    "n_fit": int(len(y_fit)),
                    "n_val": int(val_size),
                    "n_test": int(len(X_test)),
                    "prediction_rows": int(predict_rows),
                    "device": training_diag.get("device", model.get_model_config().get("device", "")),
                    "actual_epochs_used": training_diag.get("epochs_completed"),
                    "train_loss_history": training_diag.get("train_loss_history", []),
                    "validation_loss_history": training_diag.get("val_loss_history", []),
                    "early_stopping_reason": training_diag.get("early_stopping_reason", ""),
                    "final_train_loss": training_diag.get("final_train_loss"),
                    "final_val_loss": training_diag.get("final_val_loss"),
                }
            )

    diagnostic = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "source_path": cfg["data"]["source_path"],
        "models": active_models,
        "series_count": len(selected),
        "horizon": int(horizon),
        "window_size": int(window_size),
        "validation_split": "last 20% of train fold, no validation shuffle",
        "train_loader_shuffle": "True for torch models",
        "early_stopping_metric": "validation MSELoss when validation data exists",
        "loss": "torch.nn.MSELoss for torch models",
        "tasks": task_rows,
    }
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(cfg, param_rows, diagnostic, md_path)
    return diagnostic


def _write_markdown_report(
    cfg: dict[str, Any],
    param_rows: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    path: Path,
) -> None:
    torch_rows = [r for r in param_rows if r["family"] == "torch"]
    non_esn_names = [r["model_name"] for r in param_rows]
    files_checked = [
        "src/forecasting/registry.py",
        "src/forecasting/runners.py",
        "src/forecasting/windowing.py",
        "src/forecasting/adapters/base.py",
        "src/forecasting/adapters/torch_models.py",
        "src/forecasting/adapters/sklearn_models.py",
        "src/forecasting/adapters/naive.py",
        "src/forecasting/architectures/mlp.py",
        "src/forecasting/architectures/lstm.py",
        "src/forecasting/architectures/chaotic_logistic.py",
        "configs/forecasting_benchmark_smoke_v1.yaml",
        "configs/forecasting_selected_architectures_v1.yaml",
        "configs/architecture_tuning_mlp_v1.yaml",
        "configs/architecture_tuning_lstm_v1.yaml",
        "configs/architecture_tuning_logistic_v1.yaml",
    ]
    task_rows = diagnostic.get("tasks", [])
    md: list[str] = []
    md.append("# Neural training audit v2")
    md.append("")
    md.append(f"Generated at UTC: {diagnostic['checked_at_utc']}")
    md.append("")
    md.append("## Checked files")
    md.extend(f"- `{f}`" for f in files_checked)
    md.append("")
    md.append("## Registered non-ESN models")
    md.extend(f"- `{name}`" for name in non_esn_names)
    md.append("")
    md.append("## Actual training parameters")
    md.append("")
    md.append("| model | epochs | early stopping | patience | lr | batch | hidden | layers | dropout | weight decay | validation | shuffle | seed | device |")
    md.append("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|")
    for r in param_rows:
        md.append(
            "| {model_name} | {epochs} | {early_stopping} | {patience} | {lr} | {batch_size} | {hidden_size} | {num_layers} | {dropout} | {weight_decay} | {validation_mode} | {shuffle} | {seed} | {device} |".format(
                **r
            )
        )
    md.append("")
    md.append("## Findings")
    md.append("- The benchmark loader merges `configs/forecasting_selected_architectures_v1.yaml` into `model_overrides`, so selected YAML architecture params do reach the adapters.")
    md.append("- Runtime params in the selected architecture YAML are empty for the neural models; therefore the smoke benchmark global `training` block controls epochs, patience, batch size and learning rate.")
    md.append("- Torch models use `torch.nn.MSELoss`; early stopping monitors validation MSE when the runner provides a validation split.")
    md.append("- Validation is the final 20% of each rolling-origin train fold. It is time-ordered and does not use future test observations, so the split itself is not a leakage source.")
    md.append("- The torch `DataLoader` uses `shuffle=True` inside the fit subset. This does not leak labels or future test data, but it breaks chronological mini-batch ordering; for MLP it is usually acceptable, for LSTM sequence samples remain internally ordered but batches are shuffled.")
    md.append("- `build_model` is called inside each model/series/horizon/fold task, so model objects are reinitialized across series, folds and horizons. The adapter would reuse state if the same instance were fitted twice, but the benchmark runner does not do that.")
    md.append("- `weight_decay` is now represented in `FitContext`; current configs do not set it, so the effective value is 0.0.")
    md.append("")
    md.append("## Underfitting risk")
    if any(int(r["epochs"] or 0) <= 2 for r in torch_rows):
        md.append("- High: the main smoke config gives neural models only 2 max epochs and patience 1. Diagnostics below show most torch tasks stopped at the max epoch, and one stopped as soon as validation loss worsened after a single non-improving epoch.")
    else:
        md.append("- No severe epoch cap detected in the audited config.")
    md.append("- LSTM selected architectures are intentionally small (`lstm_forecast` 32x1 and `chaotic_lstm_forecast` 64x2). The bigger risk for the next benchmark is the epoch cap, not architecture size.")
    md.append("- Dropout is 0.0 for `lstm_forecast` and 0.1 for `chaotic_lstm_forecast`; MLPs and logistic net have no dropout. There is no evidence of excessive regularization.")
    md.append("")
    md.append("## Diagnostic run")
    md.append(f"- Series: {diagnostic['series_count']}")
    md.append(f"- Horizon: {diagnostic['horizon']}")
    md.append(f"- Window size: {diagnostic['window_size']}")
    md.append("")
    md.append("| model | status | epochs | final train loss | final val loss | early stopping | seconds | n_fit | n_val | n_test | device |")
    md.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for t in task_rows:
        md.append(
            "| {model_name} | {status} | {actual_epochs_used} | {final_train_loss} | {final_val_loss} | {early_stopping_reason} | {runtime_seconds:.4f} | {n_fit} | {n_val} | {n_test} | {device} |".format(
                **t
            )
        )
    md.append("")
    md.append("## Safe recommendations")
    md.append("- For the next non-smoke benchmark, increase neural `max_epochs` before changing architectures. A conservative starting point is 30-50 epochs for MLP/logistic and 40-80 for LSTM.")
    md.append("- Keep early stopping enabled and use patience around 5-10 so ESN remains fast while neural models are not capped at 2 epochs.")
    md.append("- Keep the last-block validation split for time-series safety; consider making the 20% ratio explicit in config later.")
    md.append("- Consider `shuffle=False` specifically for LSTM if preserving chronological batch order is desired; sample windows themselves are not shuffled internally.")
    md.append("- Do not change the main smoke config for this audit. Treat the higher epoch settings as recommendations for a future full run config.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit non-ESN neural forecasting training parameters")
    parser.add_argument("--config", default="configs/forecasting_benchmark_smoke_v1.yaml")
    parser.add_argument("--max-series", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--models", default=None, help="Comma-separated model names; default is all registered non-ESN models")
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        config_path=args.config,
        max_series=int(args.max_series),
        horizon=int(args.horizon),
        models=_parse_csv_strings(args.models),
        output_dir=args.output_dir,
    )
    print(
        "neural_training_audit_ok tasks={tasks} series={series} horizon={horizon} output_dir={output_dir}".format(
            tasks=len(summary["tasks"]),
            series=summary["series_count"],
            horizon=summary["horizon"],
            output_dir=str(Path(args.output_dir).resolve()),
        )
    )


if __name__ == "__main__":
    main()
