from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.forecasting.adapters import FitContext, TaskTimeoutError
from src.forecasting.data import build_series_lookup, load_log_returns_table, select_series
from src.forecasting.io import (
    FOLD_METRICS_COLUMNS,
    RAW_PREDICTION_COLUMNS,
    SPLIT_METADATA_COLUMNS,
    TASK_AUDIT_COLUMNS,
    ensure_table_schema,
    export_forecasting_benchmark_v2_excel,
    export_forecasting_summary_excel,
    save_tables,
)
from src.forecasting.metrics import METRIC_COLUMNS, compute_regression_metrics
from src.forecasting.registry import build_model, build_model_registry_table, get_model_specs, resolve_torch_device
from src.forecasting.windowing import build_rolling_origin_folds, build_supervised_windows
from src.utils.manifest import get_git_commit, write_manifest


def _task_id(series_id: str, horizon: int, model_name: str, fold_id: int, config_hash: str | None = None) -> str:
    base = f"{series_id}|h{horizon}|{model_name}|f{fold_id}"
    return f"{base}|cfg{config_hash[:12]}" if config_hash else base


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: pd.NA for c in columns}]).iloc[0:0].copy()


def _json_hash(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _aggregate_series_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(
            columns=["run_id", "model_name", "series_id", "ticker", "market", "horizon", "n_folds"]
        )

    keys = ["run_id", "model_name", "series_id", "ticker", "market", "horizon"]
    metric_cols = [m for m in METRIC_COLUMNS if m in fold_df.columns]
    rows: list[dict[str, Any]] = []
    for key_vals, sdf in fold_df.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key_vals))
        row["n_folds"] = int(len(sdf))
        for m in metric_cols:
            vals = pd.to_numeric(sdf[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{m}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_model_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model_name, horizon), sdf in fold_df.groupby(["model_name", "horizon"], dropna=False, sort=False):
        row: dict[str, Any] = {"model_name": model_name, "horizon": horizon, "count": int(len(sdf))}
        for metric in ["rmse", "directional_accuracy"]:
            vals = pd.to_numeric(sdf.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"mean_{metric}"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"std_{metric}"] = float(vals.std(ddof=0)) if not vals.empty else np.nan
            row[f"variance_{metric}"] = float(vals.var(ddof=0)) if not vals.empty else np.nan
            row[f"median_{metric}"] = float(vals.median()) if not vals.empty else np.nan
            row[f"q25_{metric}"] = float(vals.quantile(0.25)) if not vals.empty else np.nan
            row[f"q75_{metric}"] = float(vals.quantile(0.75)) if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_model_series_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["series_id", "ticker", "market", "model_name", "horizon"]
    for key_vals, sdf in fold_df.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key_vals))
        row["count"] = int(len(sdf))
        for metric in ["rmse", "directional_accuracy"]:
            vals = pd.to_numeric(sdf.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"mean_{metric}"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"std_{metric}"] = float(vals.std(ddof=0)) if not vals.empty else np.nan
            row[f"median_{metric}"] = float(vals.median()) if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["series_id", "horizon", "mean_rmse"], kind="stable")


def _resolve_execution_filters(cfg: dict[str, Any]) -> tuple[list[str], list[int], list[str] | None]:
    active_models = list(cfg.get("models", {}).get("active", []))
    if cfg.get("filters", {}).get("active_models"):
        allowed = set(cfg["filters"]["active_models"])
        active_models = [m for m in active_models if m in allowed]

    horizons = [int(h) for h in cfg.get("horizons", [1, 5, 20])]
    if cfg.get("filters", {}).get("horizons"):
        hset = {int(h) for h in cfg["filters"]["horizons"]}
        horizons = [h for h in horizons if h in hset]

    series_ids = cfg.get("filters", {}).get("series_ids") or cfg.get("data", {}).get("series_ids")
    if series_ids:
        series_ids = [str(s) for s in series_ids]
    return active_models, horizons, series_ids


def _load_existing_task_status(task_audit_path: str | Path) -> dict[tuple[str, str], str]:
    path = Path(task_audit_path)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if "task_id" not in df.columns or "status" not in df.columns:
        return {}
    if "config_hash" not in df.columns:
        df["config_hash"] = ""
    dedup = df.drop_duplicates(subset=["task_id", "config_hash"], keep="last")
    return {(str(r.task_id), str(r.config_hash)): str(r.status) for r in dedup.itertuples(index=False)}


def _training_params_for_model(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    training = dict(cfg.get("training", {}))
    by_model = dict(training.pop("by_model", {}) or {})
    params = training
    params.update(dict(by_model.get(model_name, {}) or {}))
    overrides = dict(cfg.get("model_overrides", {}).get(model_name, {}) or {})
    for key in [
        "max_train_seconds_per_task",
        "max_predict_seconds_per_task",
        "max_epochs",
        "early_stopping_patience",
        "batch_size",
        "learning_rate",
        "weight_decay",
    ]:
        if key in overrides:
            params[key] = overrides[key]
    return params


def _task_config_hash(
    cfg: dict[str, Any],
    model_name: str,
    horizon: int,
    fold_id: int,
    window_size: int,
    n_folds: int,
) -> str:
    payload = {
        "model_name": model_name,
        "model_params": cfg.get("model_overrides", {}).get(model_name, {}),
        "training_params": _training_params_for_model(cfg, model_name),
        "horizon": int(horizon),
        "fold_id": int(fold_id),
        "n_folds": int(n_folds),
        "validation": cfg.get("validation", {}),
        "window_size": int(window_size),
        "random_seed": int(cfg.get("random_seed", 42)),
        "dataset_profile": cfg.get("data", {}).get("dataset_profile", ""),
        "dataset_version": cfg.get("dataset_version", ""),
        "source_path": cfg.get("data", {}).get("source_path", ""),
    }
    return _json_hash(payload)


def _format_eta(elapsed: float, done: int, total: int) -> tuple[float, float]:
    avg = elapsed / done if done else 0.0
    remaining = max(0, total - done) * avg if done else 0.0
    return avg, remaining


def _split_metadata_row(run_id: str, s: Any, horizon: int, fold: Any, sup: Any) -> dict[str, Any]:
    train_ts = pd.to_datetime(sup.timestamps[fold.train_idx])
    test_ts = pd.to_datetime(sup.timestamps[fold.test_idx])
    return {
        "run_id": run_id,
        "series_id": s.series_id,
        "ticker": s.ticker,
        "market": s.market,
        "fold_id": int(fold.fold_id),
        "horizon": int(horizon),
        "train_start": train_ts.min() if len(train_ts) else pd.NaT,
        "train_end": train_ts.max() if len(train_ts) else pd.NaT,
        "test_start": test_ts.min() if len(test_ts) else pd.NaT,
        "test_end": test_ts.max() if len(test_ts) else pd.NaT,
        "n_train": int(len(fold.train_idx)),
        "n_test": int(len(fold.test_idx)),
    }


def _write_v2_outputs(
    cfg: dict[str, Any],
    run_id: str,
    raw_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    split_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, str]:
    outputs_cfg = cfg.get("outputs", {})
    out_paths: dict[str, str] = {}

    metrics_path = Path(outputs_cfg["metrics_long_path"]).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_parquet(metrics_path, index=False)
    out_paths["metrics_long"] = str(metrics_path)

    if outputs_cfg.get("metrics_long_csv_path"):
        csv_path = Path(outputs_cfg["metrics_long_csv_path"]).resolve()
        fold_df.to_csv(csv_path, index=False)
        out_paths["metrics_long_csv"] = str(csv_path)

    split_path = Path(outputs_cfg["split_metadata_path"]).resolve()
    split_df.to_parquet(split_path, index=False)
    out_paths["split_metadata"] = str(split_path)

    errors = audit_df[audit_df["status"].isin(["error", "timeout"])].copy() if not audit_df.empty else audit_df
    errors_path = Path(outputs_cfg["errors_csv_path"]).resolve()
    errors.to_csv(errors_path, index=False)
    out_paths["errors_csv"] = str(errors_path)

    if bool(cfg.get("save_predictions", False)):
        pred_path = Path(outputs_cfg["predictions_path"]).resolve()
        raw_df.to_parquet(pred_path, index=False)
        out_paths["predictions"] = str(pred_path)

    manifest_path = Path(outputs_cfg["run_manifest_path"]).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out_paths["run_manifest"] = str(manifest_path)

    snapshot_path = Path(outputs_cfg["config_snapshot_path"]).resolve()
    snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out_paths["config_snapshot"] = str(snapshot_path)
    return out_paths


def _build_v2_excel_sheets(
    cfg: dict[str, Any],
    run_id: str,
    selected: list[Any],
    active_models: list[str],
    horizons: list[int],
    fold_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    split_df: pd.DataFrame,
    model_registry_df: pd.DataFrame,
    device_info: dict[str, Any],
    total_elapsed: float,
) -> dict[str, pd.DataFrame]:
    model_summary = _aggregate_model_summary(fold_df)
    model_series_summary = _aggregate_model_series_summary(fold_df)
    winners_rows: list[dict[str, Any]] = []
    if not model_summary.empty:
        for horizon, sdf in model_summary.groupby("horizon", dropna=False, sort=False):
            rmse_w = sdf.sort_values("mean_rmse", ascending=True).head(1)
            da_w = sdf.sort_values("mean_directional_accuracy", ascending=False).head(1)
            if not rmse_w.empty:
                winners_rows.append({"horizon": horizon, "metric": "rmse", **rmse_w.iloc[0].to_dict()})
            if not da_w.empty:
                winners_rows.append({"horizon": horizon, "metric": "directional_accuracy", **da_w.iloc[0].to_dict()})

    if not fold_df.empty:
        speed = (
            fold_df.assign(
                total_seconds=pd.to_numeric(fold_df["fit_seconds"], errors="coerce").fillna(0)
                + pd.to_numeric(fold_df["predict_seconds"], errors="coerce").fillna(0)
            )
            .groupby("model_name", dropna=False)["total_seconds"]
            .agg(["count", "mean", "median", "sum"])
            .reset_index()
        )
    else:
        speed = pd.DataFrame()

    resume_counts = audit_df["status"].value_counts().rename_axis("status").reset_index(name="count") if not audit_df.empty else pd.DataFrame()
    error_summary = (
        audit_df[audit_df["status"].isin(["error", "timeout"])]
        .groupby(["model_name", "error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        if not audit_df.empty
        else pd.DataFrame()
    )
    device_rows = []
    specs = get_model_specs()
    for model_name in active_models:
        if specs.get(model_name) and specs[model_name].family == "torch":
            device_rows.append({"model_name": model_name, **device_info})
    config_summary = pd.DataFrame(
        [{"key": k, "value": json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v} for k, v in cfg.items() if k != "meta"]
    )
    return {
        "run_info": pd.DataFrame(
            [
                {"key": "run_id", "value": run_id},
                {"key": "elapsed_seconds", "value": total_elapsed},
                {"key": "series_count", "value": len(selected)},
                {"key": "model_count", "value": len(active_models)},
                {"key": "horizons", "value": ",".join(map(str, horizons))},
                {"key": "n_folds", "value": cfg.get("validation", {}).get("n_folds", "")},
            ]
        ),
        "dataset_summary": pd.DataFrame([s.__dict__ for s in selected]),
        "model_summary_by_horizon": model_summary,
        "model_series_summary": model_series_summary,
        "fold_stability": split_df,
        "winners_by_horizon_metric": pd.DataFrame(winners_rows),
        "error_summary": error_summary,
        "resume_summary": resume_counts,
        "speed_summary": speed,
        "device_summary": pd.DataFrame(device_rows),
        "config_summary": config_summary,
        "model_registry": model_registry_df,
    }


def run_forecasting_benchmark(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    run_name = str(cfg.get("outputs", {}).get("run_name", "forecasting_benchmark_smoke_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id") or cfg.get("outputs", {}).get("run_id") or f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    stage_name = str(cfg.get("stage", "forecasting_benchmark_smoke"))
    start_ts = datetime.now(timezone.utc)
    started_at = monotonic()

    np.random.seed(int(cfg.get("random_seed", 42)))
    device_info = resolve_torch_device(str(cfg.get("device", "cpu")))
    logger.info(
        "device requested_device=%s resolved_device=%s cuda_available=%s gpu_name=%s",
        device_info["requested_device"],
        device_info["resolved_device"],
        device_info["cuda_available"],
        device_info["gpu_name"],
    )

    log_returns = load_log_returns_table(cfg["data"]["source_path"])
    active_models, horizons, filter_series_ids = _resolve_execution_filters(cfg)
    inactive_supported = list(cfg.get("models", {}).get("inactive_but_supported", []))
    selected_arch_meta = dict(cfg.get("meta", {}).get("selected_architectures", {}))
    selected_model_metadata = dict(selected_arch_meta.get("model_metadata", {}))

    selected = select_series(
        log_returns_df=log_returns,
        dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")),
        max_series=int(cfg["data"].get("max_series", cfg["data"].get("dataset_limit", 10))) if cfg["data"].get("max_series", cfg["data"].get("dataset_limit")) else None,
        series_selection_mode=str(cfg["data"].get("series_selection_mode", "first_n")),
        series_ids=filter_series_ids,
    )
    dataset_profile = str(cfg["data"].get("dataset_profile", "core_balanced"))
    lookup = build_series_lookup(log_returns, selected, dataset_profile=dataset_profile)

    n_folds = int(cfg.get("validation", {}).get("n_folds", 3))
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}
    outputs_cfg = cfg.get("outputs", {})
    resume = bool(cfg.get("resume", cfg.get("filters", {}).get("resume_failed_only", False)))
    progress_every_n = max(1, int(cfg.get("progress_every_n", 1)))
    save_predictions = bool(cfg.get("save_predictions", True))
    v2_outputs = bool(outputs_cfg.get("metrics_long_path"))

    task_status_existing: dict[tuple[str, str], str] = {}
    if resume and outputs_cfg.get("task_audit_path"):
        task_status_existing = _load_existing_task_status(outputs_cfg["task_audit_path"])

    planned_tasks = 0
    split_rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for s in selected:
        sdf = lookup.get(s.series_id, pd.DataFrame())
        for h in horizons:
            w = window_sizes.get(h)
            if not w:
                continue
            sup = build_supervised_windows(sdf, horizon=h, window_size=w)
            folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
            for fold in folds:
                split_rows_by_key[(s.series_id, h, fold.fold_id)] = _split_metadata_row(run_id, s, h, fold, sup)
            planned_tasks += len(folds) * len(active_models)

    logger.info(
        "forecasting_benchmark start run_id=%s series=%d models=%d horizons=%d folds=%d planned_tasks=%d",
        run_id,
        len(selected),
        len(active_models),
        len(horizons),
        n_folds,
        planned_tasks,
    )

    raw_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    completed_count = 0
    skipped_count = 0
    timeout_count = 0
    error_count = 0
    task_counter = 0
    series_seconds: dict[str, float] = {}

    for s in selected:
        series_t0 = monotonic()
        series_df = lookup.get(s.series_id)
        if series_df is None or series_df.empty:
            logger.warning("skip series_id=%s: no rows", s.series_id)
            continue

        for horizon in horizons:
            window_size = window_sizes.get(horizon)
            if not window_size:
                logger.warning("skip horizon=%d: missing window_size", horizon)
                continue

            sup = build_supervised_windows(series_df, horizon=horizon, window_size=window_size)
            folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
            if not folds:
                logger.warning("skip series_id=%s horizon=%d: insufficient samples after windowing (n=%d)", s.series_id, horizon, len(sup.y))
                continue

            for model_name in active_models:
                for fold in folds:
                    task_counter += 1
                    config_hash = _task_config_hash(cfg, model_name, horizon, fold.fold_id, window_size, n_folds)
                    task_id = _task_id(s.series_id, horizon, model_name, fold.fold_id, config_hash)

                    old_status = task_status_existing.get((task_id, config_hash))
                    if resume and old_status == "success":
                        skipped_count += 1
                        elapsed = monotonic() - started_at
                        avg, eta = _format_eta(elapsed, task_counter, planned_tasks)
                        if skipped_count == 1 or task_counter % progress_every_n == 0:
                            logger.info(
                                "task_progress processed=%d/%d completed=%d skipped=%d failed=%d current=%s/%s/h%d/f%d elapsed=%.1fs avg_sec=%.2f eta_sec=%.1f",
                                task_counter,
                                planned_tasks,
                                completed_count,
                                skipped_count,
                                timeout_count + error_count,
                                s.series_id,
                                model_name,
                                horizon,
                                fold.fold_id,
                                elapsed,
                                avg,
                                eta,
                            )
                        continue

                    elapsed = monotonic() - started_at
                    avg, eta = _format_eta(elapsed, max(1, task_counter - 1), planned_tasks)
                    if task_counter == 1 or task_counter % progress_every_n == 0:
                        logger.info(
                            "task_progress processed=%d/%d completed=%d skipped=%d failed=%d current=%s/%s/h%d/f%d elapsed=%.1fs avg_sec=%.2f eta_sec=%.1f",
                            task_counter,
                            planned_tasks,
                            completed_count,
                            skipped_count,
                            timeout_count + error_count,
                            s.series_id,
                            model_name,
                            horizon,
                            fold.fold_id,
                            elapsed,
                            avg,
                            eta,
                        )

                    X_train = sup.X[fold.train_idx]
                    y_train = sup.y[fold.train_idx]
                    X_test = sup.X[fold.test_idx]
                    y_test = sup.y[fold.test_idx]
                    ts_test = sup.timestamps[fold.test_idx]

                    if len(X_train) < 2 or len(X_test) < 1:
                        task_status = "error"
                        error_type = "insufficient_data"
                        notes = "insufficient fold samples"
                        fit_seconds = np.nan
                        predict_seconds = np.nan
                        metrics = {m: np.nan for m in METRIC_COLUMNS}
                        error_count += 1
                    else:
                        val_size = max(1, int(0.2 * len(X_train)))
                        if len(X_train) - val_size < 1:
                            val_size = 0
                        if val_size > 0:
                            X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
                            y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
                        else:
                            X_fit, y_fit = X_train, y_train
                            X_val, y_val = None, None

                        model = build_model(model_name=model_name, config=cfg, logger=logger)
                        model_runtime = _training_params_for_model(cfg, model_name)
                        ctx = FitContext(
                            max_train_seconds=float(model_runtime.get("max_train_seconds_per_task", cfg.get("timeouts", {}).get("max_train_seconds_per_task", 60))),
                            max_predict_seconds=float(model_runtime.get("max_predict_seconds_per_task", cfg.get("timeouts", {}).get("max_predict_seconds_per_task", 15))),
                            max_epochs=int(model_runtime.get("max_epochs", cfg.get("training", {}).get("max_epochs", 20))),
                            early_stopping_patience=int(model_runtime.get("early_stopping_patience", cfg.get("training", {}).get("early_stopping_patience", 5))),
                            batch_size=int(model_runtime.get("batch_size", cfg.get("training", {}).get("batch_size", 64))),
                            learning_rate=float(model_runtime.get("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))),
                            weight_decay=float(model_runtime.get("weight_decay", cfg.get("training", {}).get("weight_decay", 0.0))),
                        )

                        task_status = "success"
                        error_type = ""
                        notes = ""
                        fit_seconds = np.nan
                        predict_seconds = np.nan
                        y_pred = np.full_like(y_test, np.nan, dtype=np.float64)
                        metrics = {m: np.nan for m in METRIC_COLUMNS}

                        try:
                            t_fit = monotonic()
                            model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val, context=ctx)
                            fit_seconds = monotonic() - t_fit
                            t_pred = monotonic()
                            y_pred = model.predict(X_test, context=ctx)
                            predict_seconds = monotonic() - t_pred
                            metrics = compute_regression_metrics(y_true=y_test, y_pred=y_pred, y_train=y_train, logger=logger)
                            completed_count += 1
                            if hasattr(model, "get_training_diagnostics"):
                                diag = model.get_training_diagnostics()
                                if diag:
                                    notes = json.dumps({"training": diag}, ensure_ascii=False, default=str)
                        except TaskTimeoutError as exc:
                            task_status = "timeout"
                            error_type = exc.__class__.__name__
                            notes = str(exc)
                            timeout_count += 1
                            logger.error("task timeout task=%s details=%s", task_id, exc)
                        except Exception as exc:
                            task_status = "error"
                            error_type = exc.__class__.__name__
                            notes = str(exc)
                            error_count += 1
                            logger.error("task error task=%s details=%s", task_id, exc)
                            logger.debug("traceback: %s", traceback.format_exc())

                        if save_predictions:
                            train_mean = float(np.nanmean(y_train)) if len(y_train) else np.nan
                            for i in range(len(y_test)):
                                raw_rows.append(
                                    {
                                        "run_id": run_id,
                                        "model_name": model_name,
                                        "series_id": s.series_id,
                                        "ticker": s.ticker,
                                        "market": s.market,
                                        "horizon": horizon,
                                        "fold_id": fold.fold_id,
                                        "timestamp": pd.Timestamp(ts_test[i]),
                                        "y_true": float(y_test[i]),
                                        "y_pred": float(y_pred[i]) if i < len(y_pred) else np.nan,
                                        "status": task_status,
                                        "y_train_mean": train_mean,
                                        "y_naive_baseline": train_mean,
                                    }
                                )

                    fold_row = {
                        "run_id": run_id,
                        "model_name": model_name,
                        "series_id": s.series_id,
                        "ticker": s.ticker,
                        "market": s.market,
                        "horizon": horizon,
                        "fold_id": fold.fold_id,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "fit_seconds": float(fit_seconds) if np.isfinite(fit_seconds) else np.nan,
                        "predict_seconds": float(predict_seconds) if np.isfinite(predict_seconds) else np.nan,
                        "status": task_status,
                        "config_hash": config_hash,
                    }
                    fold_row.update(metrics)
                    fold_rows.append(fold_row)
                    audit_rows.append(
                        {
                            "task_id": task_id,
                            "config_hash": config_hash,
                            "run_id": run_id,
                            "model_name": model_name,
                            "series_id": s.series_id,
                            "ticker": s.ticker,
                            "market": s.market,
                            "horizon": horizon,
                            "fold_id": fold.fold_id,
                            "status": task_status,
                            "error_type": error_type,
                            "fit_seconds": float(fit_seconds) if np.isfinite(fit_seconds) else np.nan,
                            "predict_seconds": float(predict_seconds) if np.isfinite(predict_seconds) else np.nan,
                            "notes": notes,
                            "requested_device": device_info["requested_device"],
                            "resolved_device": device_info["resolved_device"] if get_model_specs()[model_name].family == "torch" else "",
                        }
                    )
                    logger.info(
                        "task_result model=%s series=%s horizon=%d fold=%d status=%s completed=%d skipped=%d failed=%d",
                        model_name,
                        s.series_id,
                        horizon,
                        fold.fold_id,
                        task_status,
                        completed_count,
                        skipped_count,
                        timeout_count + error_count,
                    )
        series_seconds[s.series_id] = monotonic() - series_t0

    raw_df = pd.DataFrame(raw_rows) if raw_rows else _empty_frame(RAW_PREDICTION_COLUMNS)
    fold_df = pd.DataFrame(fold_rows) if fold_rows else _empty_frame(FOLD_METRICS_COLUMNS + ["config_hash"])
    audit_df = pd.DataFrame(audit_rows) if audit_rows else _empty_frame(TASK_AUDIT_COLUMNS)
    split_df = pd.DataFrame(split_rows_by_key.values()) if split_rows_by_key else _empty_frame(SPLIT_METADATA_COLUMNS)

    if resume and outputs_cfg.get("task_audit_path") and Path(outputs_cfg["task_audit_path"]).exists():
        old_audit = pd.read_parquet(outputs_cfg["task_audit_path"])
        if "config_hash" not in old_audit.columns:
            old_audit["config_hash"] = ""
        if not old_audit.empty:
            audit_df = pd.concat([old_audit, audit_df], ignore_index=True)
            audit_df = audit_df.drop_duplicates(subset=["task_id", "config_hash"], keep="last")
        if outputs_cfg.get("fold_metrics_path") and Path(outputs_cfg["fold_metrics_path"]).exists():
            old_fold = pd.read_parquet(outputs_cfg["fold_metrics_path"])
            if "config_hash" not in old_fold.columns:
                old_fold["config_hash"] = ""
            fold_df = pd.concat([old_fold, fold_df], ignore_index=True)
            fold_df = fold_df.drop_duplicates(subset=["model_name", "series_id", "horizon", "fold_id", "config_hash"], keep="last")
        if save_predictions and outputs_cfg.get("raw_predictions_path") and Path(outputs_cfg["raw_predictions_path"]).exists():
            old_raw = pd.read_parquet(outputs_cfg["raw_predictions_path"])
            raw_df = pd.concat([old_raw, raw_df], ignore_index=True)
            raw_df = raw_df.drop_duplicates(subset=["model_name", "series_id", "horizon", "fold_id", "timestamp"], keep="last")

    series_df = _aggregate_series_metrics(fold_df)
    end_ts = datetime.now(timezone.utc)
    total_elapsed = monotonic() - started_at

    model_registry_rows = build_model_registry_table(active_models=active_models, inactive_models=inactive_supported, model_metadata=selected_model_metadata)
    model_registry_df = pd.DataFrame(model_registry_rows)

    summary = {
        "series_count": len(selected),
        "model_count": len(active_models),
        "horizon_count": len(horizons),
        "task_count": int(len(audit_df)),
        "success_count": int((audit_df["status"] == "success").sum() if not audit_df.empty else 0),
        "skipped_count": int(skipped_count),
        "timeout_count": int((audit_df["status"] == "timeout").sum() if not audit_df.empty else 0),
        "error_count": int((audit_df["status"] == "error").sum() if not audit_df.empty else 0),
        "elapsed_seconds": float(total_elapsed),
        "series_seconds": series_seconds,
    }
    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg.get("meta", {}).get("project_root", Path.cwd()))),
        "config_path": cfg.get("meta", {}).get("config_path", ""),
        "config_hash": cfg.get("meta", {}).get("config_hash", ""),
        "device": device_info,
        "input_sources": {"log_returns": cfg["data"]["source_path"], "dataset_profile": cfg["data"].get("dataset_profile", "")},
        "outputs": {},
        "summary": summary,
    }

    if v2_outputs:
        outputs_cfg["task_audit_path"] = str(Path(outputs_cfg.get("task_audit_path", Path(outputs_cfg["output_dir"]) / "task_audit.parquet")).resolve())
        outputs_cfg["fold_metrics_path"] = str(Path(outputs_cfg.get("fold_metrics_path", outputs_cfg["metrics_long_path"])).resolve())
        outputs_cfg["raw_predictions_path"] = str(Path(outputs_cfg.get("raw_predictions_path", outputs_cfg["predictions_path"])).resolve())
        Path(outputs_cfg["task_audit_path"]).parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_parquet(outputs_cfg["task_audit_path"], index=False)
        out_paths = _write_v2_outputs(cfg, run_id, raw_df, fold_df, audit_df, split_df, manifest)
        out_paths["task_audit"] = outputs_cfg["task_audit_path"]
        excel_sheets = _build_v2_excel_sheets(cfg, run_id, selected, active_models, horizons, fold_df, audit_df, split_df, model_registry_df, device_info, total_elapsed)
        excel_path = export_forecasting_benchmark_v2_excel(outputs_cfg["excel_report_path"], excel_sheets)
    else:
        outputs = {
            "raw_predictions": outputs_cfg["raw_predictions_path"],
            "fold_metrics": outputs_cfg["fold_metrics_path"],
            "series_metrics": outputs_cfg["series_metrics_path"],
            "task_audit": outputs_cfg["task_audit_path"],
        }
        out_paths = save_tables(raw_df=raw_df, fold_df=fold_df, series_df=series_df, audit_df=audit_df, outputs=outputs)
        summary_df = pd.DataFrame([{"metric": k, "value": json.dumps(v, default=str) if isinstance(v, dict) else v} for k, v in summary.items()])
        readme_df = pd.DataFrame(
            [
                {"key": "run_id", "value": run_id},
                {"key": "stage", "value": stage_name},
                {"key": "config_path", "value": cfg.get("meta", {}).get("config_path", "")},
                {"key": "input_source", "value": cfg["data"]["source_path"]},
                {"key": "dataset_profile", "value": cfg["data"].get("dataset_profile", "")},
            ]
        )
        excel_path = export_forecasting_summary_excel(outputs_cfg["excel_report_path"], summary_df, model_registry_df, audit_df, fold_df, series_df, readme_df)

    manifest["outputs"] = {**out_paths, "excel_report": str(excel_path), "log": cfg.get("meta", {}).get("log_path", "")}
    manifest_path = (
        Path(out_paths["run_manifest"])
        if "run_manifest" in out_paths
        else write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)
    )
    if "run_manifest" in out_paths:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    logger.info(
        "forecasting_benchmark finished run_id=%s tasks=%d success=%d skipped=%d timeout=%d error=%d elapsed=%.1fs",
        run_id,
        len(audit_df),
        summary["success_count"],
        skipped_count,
        summary["timeout_count"],
        summary["error_count"],
        total_elapsed,
    )

    return {
        "run_id": run_id,
        "raw_predictions_path": out_paths.get("raw_predictions", out_paths.get("predictions", "")),
        "fold_metrics_path": out_paths.get("fold_metrics", out_paths.get("metrics_long", "")),
        "series_metrics_path": out_paths.get("series_metrics", ""),
        "task_audit_path": out_paths.get("task_audit", ""),
        "split_metadata_path": out_paths.get("split_metadata", ""),
        "excel_report_path": str(excel_path),
        "manifest_path": str(manifest_path),
        "n_tasks": int(len(audit_df)),
        "success_count": summary["success_count"],
        "skipped_count": int(skipped_count),
        "timeout_count": summary["timeout_count"],
        "error_count": summary["error_count"],
        "elapsed_seconds": float(total_elapsed),
        "series_seconds": series_seconds,
        "model_registry": model_registry_rows,
    }
