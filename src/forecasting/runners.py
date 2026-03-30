from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.adapters import FitContext, TaskTimeoutError
from src.forecasting.data import build_series_lookup, load_log_returns_table, select_series
from src.forecasting.io import (
    FOLD_METRICS_COLUMNS,
    RAW_PREDICTION_COLUMNS,
    TASK_AUDIT_COLUMNS,
    export_forecasting_summary_excel,
    save_tables,
)
from src.forecasting.metrics import METRIC_COLUMNS, compute_regression_metrics
from src.forecasting.registry import build_model, build_model_registry_table
from src.forecasting.windowing import build_rolling_origin_folds, build_supervised_windows
from src.utils.manifest import get_git_commit, write_manifest


def _task_id(series_id: str, horizon: int, model_name: str, fold_id: int) -> str:
    return f"{series_id}|h{horizon}|{model_name}|f{fold_id}"


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: pd.NA for c in columns}]).iloc[0:0].copy()


def _aggregate_series_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(
            columns=["run_id", "model_name", "series_id", "ticker", "market", "horizon", "n_folds"]
        )

    keys = ["run_id", "model_name", "series_id", "ticker", "market", "horizon"]
    metric_cols = [m for m in METRIC_COLUMNS if m in fold_df.columns]

    grouped = fold_df.groupby(keys, dropna=False, sort=False)
    rows: list[dict[str, Any]] = []
    for key_vals, sdf in grouped:
        row = dict(zip(keys, key_vals))
        row["n_folds"] = int(len(sdf))
        for m in metric_cols:
            vals = pd.to_numeric(sdf[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{m}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_execution_filters(cfg: dict[str, Any]) -> tuple[list[str], list[int], list[str] | None]:
    active_models = list(cfg.get("models", {}).get("active", []))
    if cfg.get("filters", {}).get("active_models"):
        allowed = set(cfg["filters"]["active_models"])
        active_models = [m for m in active_models if m in allowed]

    horizons = [int(h) for h in cfg.get("horizons", [1, 5, 20])]
    if cfg.get("filters", {}).get("horizons"):
        hset = {int(h) for h in cfg["filters"]["horizons"]}
        horizons = [h for h in horizons if h in hset]

    series_ids = cfg.get("filters", {}).get("series_ids")
    if series_ids:
        series_ids = [str(s) for s in series_ids]
    return active_models, horizons, series_ids


def _load_existing_task_status(task_audit_path: str | Path) -> dict[str, str]:
    path = Path(task_audit_path)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if "task_id" not in df.columns or "status" not in df.columns:
        return {}
    dedup = df.drop_duplicates(subset=["task_id"], keep="last")
    return {str(r.task_id): str(r.status) for r in dedup.itertuples(index=False)}


def run_forecasting_benchmark(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    run_name = str(cfg.get("outputs", {}).get("run_name", "forecasting_benchmark_smoke_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    stage_name = "forecasting_benchmark_smoke"
    start_ts = datetime.now(timezone.utc)

    log_returns = load_log_returns_table(cfg["data"]["source_path"])

    active_models, horizons, filter_series_ids = _resolve_execution_filters(cfg)
    inactive_supported = list(cfg.get("models", {}).get("inactive_but_supported", []))
    selected_arch_meta = dict(cfg.get("meta", {}).get("selected_architectures", {}))
    selected_model_metadata = dict(selected_arch_meta.get("model_metadata", {}))

    if selected_arch_meta.get("applied"):
        logger.info(
            "selected_architectures source=%s requested=%s resolved=%s",
            selected_arch_meta.get("source_config_path", ""),
            selected_arch_meta.get("requested_candidate_ids", []),
            selected_arch_meta.get("resolved_candidate_ids", []),
        )
        missing_candidate_ids = selected_arch_meta.get("missing_candidate_ids", [])
        if missing_candidate_ids:
            logger.warning("selected_architectures missing_candidate_ids=%s", missing_candidate_ids)
        mapping_rows = []
        for model_name in active_models:
            row = selected_model_metadata.get(model_name, {})
            if row:
                mapping_rows.append(
                    f"{model_name}->{row.get('candidate_id', '')}:{row.get('selection_role', '')}"
                )
        if mapping_rows:
            logger.info("selected_architectures model_mapping=%s", ", ".join(mapping_rows))

    selected = select_series(
        log_returns_df=log_returns,
        dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")),
        max_series=int(cfg["data"].get("max_series", 10)) if cfg["data"].get("max_series") else None,
        series_selection_mode=str(cfg["data"].get("series_selection_mode", "first_n")),
        series_ids=filter_series_ids,
    )
    lookup = build_series_lookup(log_returns, selected)

    n_folds = int(cfg.get("validation", {}).get("n_folds", 3))
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}

    task_status_existing: dict[str, str] = {}
    outputs_cfg = cfg.get("outputs", {})
    resume_failed_only = bool(cfg.get("filters", {}).get("resume_failed_only", False))
    if resume_failed_only and outputs_cfg.get("task_audit_path"):
        task_status_existing = _load_existing_task_status(outputs_cfg["task_audit_path"])

    planned_tasks = 0
    for s in selected:
        sdf = lookup.get(s.series_id, pd.DataFrame())
        for h in horizons:
            w = window_sizes.get(h)
            if not w:
                continue
            sup = build_supervised_windows(sdf, horizon=h, window_size=w)
            folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
            planned_tasks += len(folds) * len(active_models)

    logger.info(
        "forecasting_benchmark start run_id=%s series=%d models=%d horizons=%d planned_tasks=%d",
        run_id,
        len(selected),
        len(active_models),
        len(horizons),
        planned_tasks,
    )

    raw_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    task_counter = 0
    success_count = 0
    timeout_count = 0
    error_count = 0
    started_at = monotonic()

    for s in selected:
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
                logger.warning(
                    "skip series_id=%s horizon=%d: insufficient samples after windowing (n=%d)",
                    s.series_id,
                    horizon,
                    len(sup.y),
                )
                continue

            for model_name in active_models:
                for fold in folds:
                    task_counter += 1
                    task_id = _task_id(s.series_id, horizon, model_name, fold.fold_id)

                    old_status = task_status_existing.get(task_id)
                    if resume_failed_only and old_status == "success":
                        logger.info("resume_failed_only skip task=%s", task_id)
                        continue

                    logger.info(
                        "task_progress %d/%d model=%s series=%s horizon=%d fold=%d elapsed=%.1fs",
                        task_counter,
                        planned_tasks,
                        model_name,
                        s.series_id,
                        horizon,
                        fold.fold_id,
                        monotonic() - started_at,
                    )

                    X_train = sup.X[fold.train_idx]
                    y_train = sup.y[fold.train_idx]
                    X_test = sup.X[fold.test_idx]
                    y_test = sup.y[fold.test_idx]
                    ts_test = sup.timestamps[fold.test_idx]

                    if len(X_train) < 2 or len(X_test) < 1:
                        notes = "insufficient fold samples"
                        audit_rows.append(
                            {
                                "task_id": task_id,
                                "run_id": run_id,
                                "model_name": model_name,
                                "series_id": s.series_id,
                                "ticker": s.ticker,
                                "market": s.market,
                                "horizon": horizon,
                                "fold_id": fold.fold_id,
                                "status": "error",
                                "error_type": "insufficient_data",
                                "fit_seconds": np.nan,
                                "predict_seconds": np.nan,
                                "notes": notes,
                            }
                        )
                        error_count += 1
                        continue

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

                    model_runtime = dict(cfg.get("training", {}))
                    model_runtime.update(cfg.get("model_overrides", {}).get(model_name, {}))
                    ctx = FitContext(
                        max_train_seconds=float(
                            model_runtime.get(
                                "max_train_seconds_per_task",
                                cfg.get("timeouts", {}).get("max_train_seconds_per_task", 60),
                            )
                        ),
                        max_predict_seconds=float(
                            model_runtime.get(
                                "max_predict_seconds_per_task",
                                cfg.get("timeouts", {}).get("max_predict_seconds_per_task", 15),
                            )
                        ),
                        max_epochs=int(model_runtime.get("max_epochs", cfg.get("training", {}).get("max_epochs", 20))),
                        early_stopping_patience=int(
                            model_runtime.get(
                                "early_stopping_patience",
                                cfg.get("training", {}).get("early_stopping_patience", 5),
                            )
                        ),
                        batch_size=int(model_runtime.get("batch_size", cfg.get("training", {}).get("batch_size", 64))),
                        learning_rate=float(
                            model_runtime.get("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))
                        ),
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
                        success_count += 1

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
                    }
                    fold_row.update(metrics)
                    fold_rows.append(fold_row)

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

                    audit_rows.append(
                        {
                            "task_id": task_id,
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
                        }
                    )

                    logger.info(
                        "task_result model=%s series=%s horizon=%d fold=%d status=%s success=%d timeout=%d error=%d",
                        model_name,
                        s.series_id,
                        horizon,
                        fold.fold_id,
                        task_status,
                        success_count,
                        timeout_count,
                        error_count,
                    )

    raw_df = pd.DataFrame(raw_rows) if raw_rows else _empty_frame(RAW_PREDICTION_COLUMNS)
    fold_df = pd.DataFrame(fold_rows) if fold_rows else _empty_frame(FOLD_METRICS_COLUMNS)
    audit_df = pd.DataFrame(audit_rows) if audit_rows else _empty_frame(TASK_AUDIT_COLUMNS)

    if resume_failed_only:
        old_parts: dict[str, pd.DataFrame] = {}
        for key in ["raw_predictions", "fold_metrics", "task_audit"]:
            p = outputs_cfg.get(f"{key}_path")
            if p and Path(p).exists():
                old_parts[key] = pd.read_parquet(p)

        if "raw_predictions" in old_parts and not old_parts["raw_predictions"].empty:
            raw_df = pd.concat([old_parts["raw_predictions"], raw_df], ignore_index=True)
            raw_df = raw_df.drop_duplicates(
                subset=["model_name", "series_id", "horizon", "fold_id", "timestamp"], keep="last"
            )

        if "fold_metrics" in old_parts and not old_parts["fold_metrics"].empty:
            fold_df = pd.concat([old_parts["fold_metrics"], fold_df], ignore_index=True)
            fold_df = fold_df.drop_duplicates(subset=["model_name", "series_id", "horizon", "fold_id"], keep="last")

        if "task_audit" in old_parts and not old_parts["task_audit"].empty:
            audit_df = pd.concat([old_parts["task_audit"], audit_df], ignore_index=True)
            audit_df = audit_df.drop_duplicates(subset=["task_id"], keep="last")

    series_df = _aggregate_series_metrics(fold_df)

    outputs = {
        "raw_predictions": outputs_cfg["raw_predictions_path"],
        "fold_metrics": outputs_cfg["fold_metrics_path"],
        "series_metrics": outputs_cfg["series_metrics_path"],
        "task_audit": outputs_cfg["task_audit_path"],
    }
    out_paths = save_tables(raw_df=raw_df, fold_df=fold_df, series_df=series_df, audit_df=audit_df, outputs=outputs)

    model_registry_rows = build_model_registry_table(
        active_models=active_models,
        inactive_models=inactive_supported,
        model_metadata=selected_model_metadata,
    )
    model_registry_df = pd.DataFrame(model_registry_rows)

    summary_rows = [
        {"metric": "run_id", "value": run_id},
        {"metric": "series_count", "value": len(selected)},
        {"metric": "model_count", "value": len(active_models)},
        {"metric": "horizon_count", "value": len(horizons)},
        {"metric": "fold_tasks", "value": int(len(audit_df))},
        {"metric": "success_count", "value": int((audit_df["status"] == "success").sum() if not audit_df.empty else 0)},
        {"metric": "timeout_count", "value": int((audit_df["status"] == "timeout").sum() if not audit_df.empty else 0)},
        {"metric": "error_count", "value": int((audit_df["status"] == "error").sum() if not audit_df.empty else 0)},
    ]

    if not fold_df.empty:
        fit_seconds_numeric = pd.to_numeric(fold_df["fit_seconds"], errors="coerce")
        med_runtime = (
            fold_df.assign(_fit_seconds_numeric=fit_seconds_numeric)
            .groupby("model_name", dropna=False)["_fit_seconds_numeric"]
            .median()
            .sort_index()
            .to_dict()
        )
        for model_name, med_sec in med_runtime.items():
            summary_rows.append({"metric": f"median_fit_seconds_{model_name}", "value": float(med_sec)})

        for m in ["mae", "rmse", "mase", "directional_accuracy", "r2_oos"]:
            vals = pd.to_numeric(fold_df[m], errors="coerce") if m in fold_df.columns else pd.Series(dtype=float)
            summary_rows.append(
                {
                    "metric": f"{m}_mean_over_tasks",
                    "value": float(vals.mean()) if not vals.empty and vals.notna().any() else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": stage_name},
            {"key": "config_path", "value": cfg.get("meta", {}).get("config_path", "")},
            {"key": "input_source", "value": cfg["data"]["source_path"]},
            {"key": "dataset_profile", "value": cfg["data"].get("dataset_profile", "")},
            {"key": "notes", "value": "raw predictions allow future metric recomputation without retraining"},
        ]
    )

    excel_path = export_forecasting_summary_excel(
        excel_path=outputs_cfg["excel_report_path"],
        summary_df=summary_df,
        model_registry_df=model_registry_df,
        task_audit_df=audit_df,
        fold_metrics_df=fold_df,
        series_metrics_df=series_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg.get("meta", {}).get("project_root", Path.cwd()))),
        "config_path": cfg.get("meta", {}).get("config_path", ""),
        "input_sources": {
            "log_returns": cfg["data"]["source_path"],
            "dataset_profile": cfg["data"].get("dataset_profile", ""),
        },
        "outputs": {
            **out_paths,
            "excel_report": str(excel_path),
            "log": cfg.get("meta", {}).get("log_path", ""),
        },
        "summary": {
            "series_count": len(selected),
            "model_count": len(active_models),
            "horizon_count": len(horizons),
            "task_count": int(len(audit_df)),
            "success_count": int((audit_df["status"] == "success").sum() if not audit_df.empty else 0),
            "timeout_count": int((audit_df["status"] == "timeout").sum() if not audit_df.empty else 0),
            "error_count": int((audit_df["status"] == "error").sum() if not audit_df.empty else 0),
        },
    }

    manifest_path = write_manifest(
        manifest=manifest,
        manifests_dir=cfg["artifacts"]["manifests"],
        run_id=run_id,
    )

    logger.info(
        "forecasting_benchmark finished run_id=%s tasks=%d success=%d timeout=%d error=%d elapsed=%.1fs",
        run_id,
        len(audit_df),
        int((audit_df["status"] == "success").sum() if not audit_df.empty else 0),
        int((audit_df["status"] == "timeout").sum() if not audit_df.empty else 0),
        int((audit_df["status"] == "error").sum() if not audit_df.empty else 0),
        (end_ts - start_ts).total_seconds(),
    )

    return {
        "run_id": run_id,
        "raw_predictions_path": out_paths["raw_predictions"],
        "fold_metrics_path": out_paths["fold_metrics"],
        "series_metrics_path": out_paths["series_metrics"],
        "task_audit_path": out_paths["task_audit"],
        "excel_report_path": str(excel_path),
        "manifest_path": str(manifest_path),
        "n_tasks": int(len(audit_df)),
        "success_count": int((audit_df["status"] == "success").sum() if not audit_df.empty else 0),
        "timeout_count": int((audit_df["status"] == "timeout").sum() if not audit_df.empty else 0),
        "error_count": int((audit_df["status"] == "error").sum() if not audit_df.empty else 0),
        "model_registry": model_registry_rows,
    }
