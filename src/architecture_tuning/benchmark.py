from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.architecture_tuning.dataset import (
    list_external_csv_files,
    load_external_log_returns_series,
    sample_external_series,
    save_selected_series_csv,
    selected_series_to_frame,
)
from src.forecasting.adapters import FitContext, TaskTimeoutError
from src.forecasting.metrics import METRIC_COLUMNS, compute_regression_metrics
from src.forecasting.registry import get_model_specs
from src.forecasting.windowing import build_rolling_origin_folds, build_supervised_windows
from src.reporting.excel_export import export_architecture_tuning_benchmark_excel
from src.utils.manifest import get_git_commit, write_manifest


SERIES_LEVEL_COLUMNS = [
    "run_id",
    "model_name",
    "candidate_id",
    "series_id",
    "horizon",
    "mae",
    "rmse",
    "mase",
    "directional_accuracy",
    "fit_time_sec",
    "predict_time_sec",
    "status",
]

CANDIDATE_LEVEL_COLUMNS = [
    "run_id",
    "model_name",
    "candidate_id",
    "compare_group_id",
    "n_reservoir",
    "spectral_radius",
    "chaotic_spectral_radius",
    "input_scale",
    "leak_rate",
    "ridge_alpha",
    "g_start",
    "g_end",
    "params_json",
    "params_repr",
    "n_series",
    "horizon",
    "mae_mean",
    "rmse_mean",
    "mase_mean",
    "directional_accuracy_mean",
    "fit_time_sec_mean",
    "predict_time_sec_mean",
    "total_runtime_sec",
    "status",
    "notes",
]

PAIR_COMPARISON_COLUMNS = [
    "run_id",
    "horizon",
    "compare_group_id",
    "esn_candidate_id",
    "chaotic_esn_candidate_id",
    "transient_chaotic_esn_candidate_id",
    "esn_mae_mean",
    "chaotic_esn_mae_mean",
    "transient_chaotic_esn_mae_mean",
    "chaotic_esn_delta_mae_vs_esn",
    "transient_chaotic_esn_delta_mae_vs_esn",
    "esn_runtime_sec_mean",
    "chaotic_esn_runtime_sec_mean",
    "transient_chaotic_esn_runtime_sec_mean",
    "chaotic_esn_runtime_delta_sec_vs_esn",
    "transient_chaotic_esn_runtime_delta_sec_vs_esn",
    "esn_status",
    "chaotic_esn_status",
    "transient_chaotic_esn_status",
]

BEST_CANDIDATE_SUMMARY_COLUMNS = [
    "run_id",
    "model_name",
    "best_candidate_id",
    "mae_mean_over_horizons",
    "rmse_mean_over_horizons",
    "mase_mean_over_horizons",
    "directional_accuracy_mean_over_horizons",
    "fit_time_sec_mean_over_horizons",
    "predict_time_sec_mean_over_horizons",
    "runtime_sec_mean_over_horizons",
    "compare_group_id",
    "status",
]


def _resolve_device(requested: str) -> str:
    value = str(requested).strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value or "cpu"


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _candidate_runtime_params(cfg: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(cfg.get("training", {}))
    runtime.update(cfg.get("timeouts", {}))
    runtime.update(candidate.get("runtime_params", {}))
    return runtime


def _build_fit_context(runtime: dict[str, Any]) -> FitContext:
    return FitContext(
        max_train_seconds=float(runtime.get("max_train_seconds_per_task", 60)),
        max_predict_seconds=float(runtime.get("max_predict_seconds_per_task", 15)),
        max_epochs=int(runtime.get("max_epochs", 20)),
        early_stopping_patience=int(runtime.get("early_stopping_patience", 5)),
        batch_size=int(runtime.get("batch_size", 64)),
        learning_rate=float(runtime.get("learning_rate", 1e-3)),
    )


def _instantiate_model(
    model_name: str,
    model_params: dict[str, Any],
    device: str,
    logger: Any,
):
    specs = get_model_specs()
    if model_name not in specs:
        raise KeyError(f"Model not registered: {model_name}")
    kwargs = dict(model_params)
    if specs[model_name].family == "torch":
        kwargs.setdefault("device", device)
        kwargs.setdefault("logger", logger)
    return specs[model_name].factory(kwargs)


def _candidate_params_repr(candidate: dict[str, Any]) -> tuple[str, str]:
    payload = {
        "model_params": candidate.get("model_params", {}),
        "runtime_params": candidate.get("runtime_params", {}),
        "compare_group_id": candidate.get("compare_group_id", ""),
    }
    params_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return params_json, str(payload)


def _extract_candidate_core_params(candidate: dict[str, Any]) -> dict[str, Any]:
    model_params = dict(candidate.get("model_params", {}))
    return {
        "n_reservoir": model_params.get("n_reservoir"),
        "spectral_radius": model_params.get("spectral_radius"),
        "chaotic_spectral_radius": model_params.get("chaotic_spectral_radius"),
        "input_scale": model_params.get("input_scale"),
        "leak_rate": model_params.get("leak_rate"),
        "ridge_alpha": model_params.get("ridge_alpha"),
        "g_start": model_params.get("g_start"),
        "g_end": model_params.get("g_end"),
    }


def build_best_by_model_table(candidate_level_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_level_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "horizon",
                "best_candidate_id",
                "mae_mean",
                "rmse_mean",
                "mase_mean",
                "directional_accuracy_mean",
                "fit_time_sec_mean",
                "predict_time_sec_mean",
                "status",
            ]
        )

    rows: list[dict[str, Any]] = []
    keys = ["model_name", "horizon"]
    for key_vals, sdf in candidate_level_df.groupby(keys, sort=True):
        clean = sdf.copy()
        clean["mae_mean"] = pd.to_numeric(clean["mae_mean"], errors="coerce")
        clean = clean.sort_values(["mae_mean", "candidate_id"], ascending=[True, True], kind="stable")
        best = clean.iloc[0]
        rows.append(
            {
                "model_name": key_vals[0],
                "horizon": int(key_vals[1]),
                "best_candidate_id": str(best["candidate_id"]),
                "mae_mean": float(best["mae_mean"]) if pd.notna(best["mae_mean"]) else np.nan,
                "rmse_mean": float(best["rmse_mean"]) if pd.notna(best["rmse_mean"]) else np.nan,
                "mase_mean": float(best["mase_mean"]) if pd.notna(best["mase_mean"]) else np.nan,
                "directional_accuracy_mean": (
                    float(best["directional_accuracy_mean"]) if pd.notna(best["directional_accuracy_mean"]) else np.nan
                ),
                "fit_time_sec_mean": float(best["fit_time_sec_mean"]) if pd.notna(best["fit_time_sec_mean"]) else np.nan,
                "predict_time_sec_mean": (
                    float(best["predict_time_sec_mean"]) if pd.notna(best["predict_time_sec_mean"]) else np.nan
                ),
                "status": str(best.get("status", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def build_best_candidate_summary_table(candidate_level_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_level_df.empty:
        return pd.DataFrame(columns=BEST_CANDIDATE_SUMMARY_COLUMNS)

    metrics_to_avg = [
        "mae_mean",
        "rmse_mean",
        "mase_mean",
        "directional_accuracy_mean",
        "fit_time_sec_mean",
        "predict_time_sec_mean",
        "total_runtime_sec",
    ]
    base_cols = [
        "run_id",
        "model_name",
        "candidate_id",
        "compare_group_id",
        "status",
    ]
    clean = candidate_level_df[base_cols + metrics_to_avg].copy()
    for col in metrics_to_avg:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["status_priority"] = (clean["status"].astype(str) != "success").astype(int)

    grouped = (
        clean.groupby(["run_id", "model_name", "candidate_id", "compare_group_id"], dropna=False, sort=False)
        .agg(
            mae_mean=("mae_mean", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            mase_mean=("mase_mean", "mean"),
            directional_accuracy_mean=("directional_accuracy_mean", "mean"),
            fit_time_sec_mean=("fit_time_sec_mean", "mean"),
            predict_time_sec_mean=("predict_time_sec_mean", "mean"),
            total_runtime_sec=("total_runtime_sec", "mean"),
            status_priority=("status_priority", "min"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for (run_id, model_name), sdf in grouped.groupby(["run_id", "model_name"], sort=True):
        ranked = sdf.sort_values(["status_priority", "mae_mean", "candidate_id"], ascending=[True, True, True], kind="stable")
        best = ranked.iloc[0]
        rows.append(
            {
                "run_id": str(run_id),
                "model_name": str(model_name),
                "best_candidate_id": str(best["candidate_id"]),
                "mae_mean_over_horizons": float(best["mae_mean"]) if pd.notna(best["mae_mean"]) else np.nan,
                "rmse_mean_over_horizons": float(best["rmse_mean"]) if pd.notna(best["rmse_mean"]) else np.nan,
                "mase_mean_over_horizons": float(best["mase_mean"]) if pd.notna(best["mase_mean"]) else np.nan,
                "directional_accuracy_mean_over_horizons": (
                    float(best["directional_accuracy_mean"]) if pd.notna(best["directional_accuracy_mean"]) else np.nan
                ),
                "fit_time_sec_mean_over_horizons": (
                    float(best["fit_time_sec_mean"]) if pd.notna(best["fit_time_sec_mean"]) else np.nan
                ),
                "predict_time_sec_mean_over_horizons": (
                    float(best["predict_time_sec_mean"]) if pd.notna(best["predict_time_sec_mean"]) else np.nan
                ),
                "runtime_sec_mean_over_horizons": (
                    float(best["total_runtime_sec"]) if pd.notna(best["total_runtime_sec"]) else np.nan
                ),
                "compare_group_id": str(best["compare_group_id"]) if pd.notna(best["compare_group_id"]) else "",
                "status": "success" if int(best["status_priority"]) == 0 else "failed",
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, BEST_CANDIDATE_SUMMARY_COLUMNS)


def build_pair_comparison_table(candidate_level_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_level_df.empty or "compare_group_id" not in candidate_level_df.columns:
        return pd.DataFrame(columns=PAIR_COMPARISON_COLUMNS)

    focus_models = {"esn", "chaotic_esn", "transient_chaotic_esn"}
    df = candidate_level_df[candidate_level_df["model_name"].astype(str).isin(focus_models)].copy()
    if df.empty:
        return pd.DataFrame(columns=PAIR_COMPARISON_COLUMNS)

    df["compare_group_id"] = df["compare_group_id"].astype(str)
    df = df[df["compare_group_id"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=PAIR_COMPARISON_COLUMNS)

    for col in ["mae_mean", "fit_time_sec_mean", "predict_time_sec_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["runtime_sec_mean"] = df["fit_time_sec_mean"] + df["predict_time_sec_mean"]
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].astype(str)
    else:
        df["run_id"] = ""

    rows: list[dict[str, Any]] = []
    group_keys = ["run_id", "horizon", "compare_group_id"]
    for (run_id, horizon, compare_group_id), sdf in df.groupby(group_keys, sort=True, dropna=False):
        rec: dict[str, Any] = {
            "run_id": str(run_id),
            "horizon": int(horizon),
            "compare_group_id": str(compare_group_id),
        }
        for model_name in ["esn", "chaotic_esn", "transient_chaotic_esn"]:
            mdf = sdf[sdf["model_name"] == model_name].copy()
            if mdf.empty:
                rec[f"{model_name}_candidate_id"] = ""
                rec[f"{model_name}_mae_mean"] = np.nan
                rec[f"{model_name}_runtime_sec_mean"] = np.nan
                rec[f"{model_name}_status"] = "missing"
                continue
            ranked = mdf.sort_values(["mae_mean", "candidate_id"], ascending=[True, True], kind="stable")
            best = ranked.iloc[0]
            rec[f"{model_name}_candidate_id"] = str(best["candidate_id"])
            rec[f"{model_name}_mae_mean"] = float(best["mae_mean"]) if pd.notna(best["mae_mean"]) else np.nan
            rec[f"{model_name}_runtime_sec_mean"] = (
                float(best["runtime_sec_mean"]) if pd.notna(best["runtime_sec_mean"]) else np.nan
            )
            rec[f"{model_name}_status"] = str(best.get("status", ""))

        esn_mae = rec.get("esn_mae_mean", np.nan)
        chaotic_mae = rec.get("chaotic_esn_mae_mean", np.nan)
        transient_mae = rec.get("transient_chaotic_esn_mae_mean", np.nan)
        esn_runtime = rec.get("esn_runtime_sec_mean", np.nan)
        chaotic_runtime = rec.get("chaotic_esn_runtime_sec_mean", np.nan)
        transient_runtime = rec.get("transient_chaotic_esn_runtime_sec_mean", np.nan)
        rec["chaotic_esn_delta_mae_vs_esn"] = chaotic_mae - esn_mae if pd.notna(chaotic_mae) and pd.notna(esn_mae) else np.nan
        rec["transient_chaotic_esn_delta_mae_vs_esn"] = (
            transient_mae - esn_mae if pd.notna(transient_mae) and pd.notna(esn_mae) else np.nan
        )
        rec["chaotic_esn_runtime_delta_sec_vs_esn"] = (
            chaotic_runtime - esn_runtime if pd.notna(chaotic_runtime) and pd.notna(esn_runtime) else np.nan
        )
        rec["transient_chaotic_esn_runtime_delta_sec_vs_esn"] = (
            transient_runtime - esn_runtime if pd.notna(transient_runtime) and pd.notna(esn_runtime) else np.nan
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    return _ensure_columns(out, PAIR_COMPARISON_COLUMNS)


def _build_summary(
    cfg: dict[str, Any],
    run_id: str,
    n_available_files: int,
    selected_series_df: pd.DataFrame,
    candidate_level_df: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    success_count = int((candidate_level_df["status"] == "success").sum()) if not candidate_level_df.empty else 0
    failure_count = int((candidate_level_df["status"] != "success").sum()) if not candidate_level_df.empty else 0
    rows = [
        {"metric": "run_id", "value": run_id},
        {"metric": "external_data_dir", "value": cfg["external_data_dir"]},
        {"metric": "random_seed", "value": int(cfg["random_seed"])},
        {"metric": "sample_size", "value": int(cfg["sample_size"])},
        {"metric": "n_available_files", "value": int(n_available_files)},
        {"metric": "n_selected_files", "value": int(len(selected_series_df))},
        {"metric": "selected_models", "value": ", ".join(cfg["selected_models"])},
        {"metric": "horizons", "value": ", ".join(str(h) for h in cfg["horizons"])},
        {"metric": "device", "value": device},
        {"metric": "success_count", "value": success_count},
        {"metric": "failed_count", "value": failure_count},
    ]
    if not candidate_level_df.empty:
        mae = pd.to_numeric(candidate_level_df["mae_mean"], errors="coerce")
        runtime = pd.to_numeric(candidate_level_df["total_runtime_sec"], errors="coerce")
        compare_groups = (
            candidate_level_df["compare_group_id"].astype(str).replace("", pd.NA).dropna().nunique()
            if "compare_group_id" in candidate_level_df.columns
            else 0
        )
        rows.extend(
            [
                {"metric": "mae_mean_over_candidates", "value": float(mae.mean()) if mae.notna().any() else np.nan},
                {"metric": "mae_best", "value": float(mae.min()) if mae.notna().any() else np.nan},
                {"metric": "runtime_sec_mean", "value": float(runtime.mean()) if runtime.notna().any() else np.nan},
                {"metric": "runtime_sec_median", "value": float(runtime.median()) if runtime.notna().any() else np.nan},
                {"metric": "n_compare_groups", "value": int(compare_groups)},
            ]
        )
    return pd.DataFrame(rows)


def run_architecture_tuning_benchmark(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    stage_name = str(cfg.get("stage", "architecture_tuning_benchmark"))
    run_name = str(cfg.get("run_name", "architecture_tuning_benchmark_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    start_ts = datetime.now(timezone.utc)
    started_at = monotonic()

    device = _resolve_device(str(cfg.get("device", "cpu")))
    logger.info("architecture_tuning_benchmark start run_id=%s", run_id)
    logger.info("config_path=%s", cfg.get("meta", {}).get("config_path", ""))
    logger.info("external_data_dir=%s", cfg["external_data_dir"])
    logger.info("sample_size=%d random_seed=%d", int(cfg["sample_size"]), int(cfg["random_seed"]))

    files = list_external_csv_files(cfg["external_data_dir"], file_pattern=str(cfg.get("file_pattern", "*.csv")))
    logger.info("found_csv_files=%d", len(files))
    selected = sample_external_series(
        files=files,
        sample_size=int(cfg["sample_size"]),
        random_seed=int(cfg["random_seed"]),
    )
    selected_df = selected_series_to_frame(selected)
    selected_series_path = save_selected_series_csv(selected_df, cfg["outputs"]["selected_series_csv"])
    logger.info("selected_files=%d", len(selected_df))
    logger.info("selected_files_list=%s", "; ".join(selected_df["file_name"].astype(str).tolist()))

    horizons = [int(h) for h in cfg.get("horizons", [1])]
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}
    n_folds = int(cfg.get("validation", {}).get("n_folds", 3))
    selected_models = [str(m) for m in cfg.get("selected_models", [])]
    candidate_map = dict(cfg.get("candidates", {}))

    logger.info("models=%s", ", ".join(selected_models))
    logger.info("horizons=%s", ", ".join(str(h) for h in horizons))

    series_lookup: dict[str, pd.DataFrame] = {}
    skipped_series: list[tuple[str, str]] = []
    for row in selected_df.itertuples(index=False):
        try:
            series_lookup[str(row.series_id)] = load_external_log_returns_series(
                source_path=str(row.source_path),
                date_column=str(cfg["data_transforms"].get("date_column", "Date")),
                close_column=str(cfg["data_transforms"].get("close_column", "Close")),
                min_history=int(cfg["data_transforms"].get("min_history", 120)),
            )
        except Exception as exc:
            skipped_series.append((str(row.series_id), str(exc)))
            logger.warning("skip_series series_id=%s reason=%s", str(row.series_id), exc)

    series_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    n_candidates = 0

    for model_name in selected_models:
        candidates = list(candidate_map.get(model_name, []))
        if not candidates:
            candidates = [{"candidate_id": "default", "model_params": {}, "runtime_params": {}}]

        for candidate in candidates:
            n_candidates += 1
            candidate_id = str(candidate.get("candidate_id", f"{model_name}_candidate_{n_candidates:03d}"))
            params_json, params_repr = _candidate_params_repr(candidate)
            candidate_core_params = _extract_candidate_core_params(candidate)
            runtime_cfg = _candidate_runtime_params(cfg, candidate)
            fit_ctx = _build_fit_context(runtime_cfg)
            model_params = dict(candidate.get("model_params", {}))
            compare_group_id = str(candidate.get("compare_group_id", ""))

            logger.info(
                "candidate_start model=%s candidate_id=%s compare_group_id=%s",
                model_name,
                candidate_id,
                compare_group_id,
            )
            candidate_started_at = monotonic()

            per_horizon_runtime: dict[int, float] = {h: 0.0 for h in horizons}
            horizon_success_flags: dict[int, bool] = {h: False for h in horizons}

            for series_id, sdf in series_lookup.items():
                for horizon in horizons:
                    window_size = window_sizes.get(horizon)
                    if not window_size:
                        continue
                    sup = build_supervised_windows(sdf, horizon=horizon, window_size=window_size)
                    folds = build_rolling_origin_folds(n_samples=len(sup.y), n_folds=n_folds)

                    if len(sup.y) == 0 or not folds:
                        series_rows.append(
                            {
                                "run_id": run_id,
                                "model_name": model_name,
                                "candidate_id": candidate_id,
                                "series_id": series_id,
                                "horizon": horizon,
                                "mae": np.nan,
                                "rmse": np.nan,
                                "mase": np.nan,
                                "directional_accuracy": np.nan,
                                "fit_time_sec": np.nan,
                                "predict_time_sec": np.nan,
                                "status": "failed",
                            }
                        )
                        continue

                    fold_metrics_rows: list[dict[str, float]] = []
                    fold_fit_times: list[float] = []
                    fold_predict_times: list[float] = []
                    status = "success"
                    notes = ""

                    for fold in folds:
                        X_train = sup.X[fold.train_idx]
                        y_train = sup.y[fold.train_idx]
                        X_test = sup.X[fold.test_idx]
                        y_test = sup.y[fold.test_idx]

                        val_size = max(1, int(0.2 * len(X_train)))
                        if len(X_train) - val_size < 1:
                            val_size = 0
                        if val_size > 0:
                            X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
                            y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
                        else:
                            X_fit, y_fit = X_train, y_train
                            X_val, y_val = None, None

                        fit_seconds = np.nan
                        predict_seconds = np.nan
                        try:
                            model = _instantiate_model(
                                model_name=model_name,
                                model_params=model_params,
                                device=device,
                                logger=logger,
                            )
                            t0 = monotonic()
                            model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val, context=fit_ctx)
                            fit_seconds = monotonic() - t0

                            t1 = monotonic()
                            y_pred = model.predict(X_test, context=fit_ctx)
                            predict_seconds = monotonic() - t1

                            metrics = compute_regression_metrics(
                                y_true=y_test,
                                y_pred=y_pred,
                                y_train=y_train,
                                logger=logger,
                            )
                            fold_metrics_rows.append(metrics)
                            fold_fit_times.append(float(fit_seconds))
                            fold_predict_times.append(float(predict_seconds))
                        except TaskTimeoutError as exc:
                            status = "failed"
                            notes = str(exc)
                            logger.warning(
                                "candidate_fold_timeout model=%s candidate=%s series=%s horizon=%d fold=%d err=%s",
                                model_name,
                                candidate_id,
                                series_id,
                                horizon,
                                fold.fold_id,
                                exc,
                            )
                            break
                        except Exception as exc:
                            status = "failed"
                            notes = f"{exc.__class__.__name__}: {exc}"
                            logger.warning(
                                "candidate_fold_error model=%s candidate=%s series=%s horizon=%d fold=%d err=%s",
                                model_name,
                                candidate_id,
                                series_id,
                                horizon,
                                fold.fold_id,
                                exc,
                            )
                            logger.debug("traceback: %s", traceback.format_exc())
                            break

                    if status == "success" and fold_metrics_rows:
                        metrics_df = pd.DataFrame(fold_metrics_rows)
                        row = {
                            "run_id": run_id,
                            "model_name": model_name,
                            "candidate_id": candidate_id,
                            "series_id": series_id,
                            "horizon": horizon,
                            "mae": float(pd.to_numeric(metrics_df["mae"], errors="coerce").mean()),
                            "rmse": float(pd.to_numeric(metrics_df["rmse"], errors="coerce").mean()),
                            "mase": float(pd.to_numeric(metrics_df["mase"], errors="coerce").mean()),
                            "directional_accuracy": float(
                                pd.to_numeric(metrics_df["directional_accuracy"], errors="coerce").mean()
                            ),
                            "fit_time_sec": float(np.mean(fold_fit_times)) if fold_fit_times else np.nan,
                            "predict_time_sec": float(np.mean(fold_predict_times)) if fold_predict_times else np.nan,
                            "status": "success",
                        }
                        horizon_success_flags[horizon] = True
                        per_horizon_runtime[horizon] += float(np.nansum(fold_fit_times) + np.nansum(fold_predict_times))
                    else:
                        row = {
                            "run_id": run_id,
                            "model_name": model_name,
                            "candidate_id": candidate_id,
                            "series_id": series_id,
                            "horizon": horizon,
                            "mae": np.nan,
                            "rmse": np.nan,
                            "mase": np.nan,
                            "directional_accuracy": np.nan,
                            "fit_time_sec": np.nan,
                            "predict_time_sec": np.nan,
                            "status": "failed",
                        }
                        if notes:
                            row["notes"] = notes
                    series_rows.append(row)

            for horizon in horizons:
                horizon_df = pd.DataFrame(
                    [
                        r
                        for r in series_rows
                        if r["model_name"] == model_name
                        and r["candidate_id"] == candidate_id
                        and int(r["horizon"]) == int(horizon)
                    ]
                )

                success_mask = horizon_df["status"].astype(str) == "success" if not horizon_df.empty else pd.Series(dtype=bool)
                candidate_status = "success" if bool(success_mask.any()) else "failed"
                notes = ""
                if not horizon_success_flags[horizon]:
                    notes = "no successful series/folds for this candidate-horizon"

                candidate_rows.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "candidate_id": candidate_id,
                        "compare_group_id": compare_group_id,
                        **candidate_core_params,
                        "params_json": params_json,
                        "params_repr": params_repr,
                        "n_series": int(len(horizon_df)),
                        "horizon": horizon,
                        "mae_mean": (
                            float(pd.to_numeric(horizon_df.loc[success_mask, "mae"], errors="coerce").mean())
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "rmse_mean": (
                            float(pd.to_numeric(horizon_df.loc[success_mask, "rmse"], errors="coerce").mean())
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "mase_mean": (
                            float(pd.to_numeric(horizon_df.loc[success_mask, "mase"], errors="coerce").mean())
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "directional_accuracy_mean": (
                            float(
                                pd.to_numeric(
                                    horizon_df.loc[success_mask, "directional_accuracy"],
                                    errors="coerce",
                                ).mean()
                            )
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "fit_time_sec_mean": (
                            float(pd.to_numeric(horizon_df.loc[success_mask, "fit_time_sec"], errors="coerce").mean())
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "predict_time_sec_mean": (
                            float(
                                pd.to_numeric(horizon_df.loc[success_mask, "predict_time_sec"], errors="coerce").mean()
                            )
                            if not horizon_df.empty and success_mask.any()
                            else np.nan
                        ),
                        "total_runtime_sec": float(per_horizon_runtime[horizon]),
                        "status": candidate_status,
                        "notes": notes,
                    }
                )
                if candidate_status == "success":
                    success_count += 1
                else:
                    failure_count += 1

            logger.info(
                "candidate_done model=%s candidate_id=%s elapsed=%.2fs",
                model_name,
                candidate_id,
                monotonic() - candidate_started_at,
            )

    series_level_df = _ensure_columns(pd.DataFrame(series_rows), SERIES_LEVEL_COLUMNS)
    candidate_level_df = _ensure_columns(pd.DataFrame(candidate_rows), CANDIDATE_LEVEL_COLUMNS)
    best_by_model_df = build_best_by_model_table(candidate_level_df)
    pair_comparison_df = build_pair_comparison_table(candidate_level_df)
    best_candidate_summary_df = build_best_candidate_summary_table(candidate_level_df)

    out_cfg = cfg["outputs"]
    candidate_level_parquet = Path(out_cfg["candidate_level_results_parquet"]).resolve()
    candidate_level_csv = Path(out_cfg["candidate_level_results_csv"]).resolve()
    series_level_parquet = Path(out_cfg["series_level_results_parquet"]).resolve()
    pair_comparison_csv = Path(out_cfg["pair_comparison_csv"]).resolve()
    best_candidate_summary_csv = Path(out_cfg["best_candidate_summary_csv"]).resolve()
    for out_path in [
        candidate_level_parquet,
        candidate_level_csv,
        series_level_parquet,
        pair_comparison_csv,
        best_candidate_summary_csv,
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_level_df.to_parquet(candidate_level_parquet, index=False)
    candidate_level_df.to_csv(candidate_level_csv, index=False)
    series_level_df.to_parquet(series_level_parquet, index=False)
    pair_comparison_df.to_csv(pair_comparison_csv, index=False)
    best_candidate_summary_df.to_csv(best_candidate_summary_csv, index=False)

    summary_df = _build_summary(
        cfg=cfg,
        run_id=run_id,
        n_available_files=len(files),
        selected_series_df=selected_df,
        candidate_level_df=candidate_level_df,
        device=device,
    )
    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": stage_name},
            {"key": "config_path", "value": cfg.get("meta", {}).get("config_path", "")},
            {"key": "selection_artifact", "value": str(selected_series_path)},
            {
                "key": "candidate_results",
                "value": f"{candidate_level_parquet}; {candidate_level_csv}",
            },
            {"key": "series_results", "value": str(series_level_parquet)},
            {"key": "pair_comparison_csv", "value": str(pair_comparison_csv)},
            {"key": "best_candidate_summary_csv", "value": str(best_candidate_summary_csv)},
            {
                "key": "pair_comparison_support",
                "value": "candidate payload supports compare_group_id and separate model/runtime params",
            },
            {
                "key": "metrics_available",
                "value": ", ".join(METRIC_COLUMNS),
            },
        ]
    )
    excel_path = export_architecture_tuning_benchmark_excel(
        excel_path=out_cfg["excel_report_path"],
        summary_df=summary_df,
        selected_series_df=selected_df,
        candidate_level_df=candidate_level_df,
        best_by_model_df=best_by_model_df,
        pair_comparison_df=pair_comparison_df,
        best_candidate_summary_df=best_candidate_summary_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg["meta"]["config_path"],
        "input_sources": {
            "external_data_dir": cfg["external_data_dir"],
            "selected_series_csv": str(selected_series_path),
        },
        "outputs": {
            "selected_series_csv": str(selected_series_path),
            "candidate_level_results_parquet": str(candidate_level_parquet),
            "candidate_level_results_csv": str(candidate_level_csv),
            "series_level_results_parquet": str(series_level_parquet),
            "pair_comparison_csv": str(pair_comparison_csv),
            "best_candidate_summary_csv": str(best_candidate_summary_csv),
            "excel_report": str(excel_path),
            "log": cfg.get("meta", {}).get("log_path", ""),
        },
        "summary": {
            "external_data_dir": cfg["external_data_dir"],
            "sample_size": int(cfg["sample_size"]),
            "random_seed": int(cfg["random_seed"]),
            "n_available_files": int(len(files)),
            "n_selected_files": int(len(selected_df)),
            "n_models": int(len(selected_models)),
            "n_candidates": int(n_candidates),
            "n_horizons": int(len(horizons)),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "skipped_series": skipped_series,
            "runtime_sec": float((end_ts - start_ts).total_seconds()),
        },
    }
    manifest_path = write_manifest(
        manifest=manifest,
        manifests_dir=cfg["artifacts"]["manifests"],
        run_id=run_id,
    )

    logger.info("selected_series_path=%s", selected_series_path)
    logger.info("candidate_level_results_parquet=%s", candidate_level_parquet)
    logger.info("candidate_level_results_csv=%s", candidate_level_csv)
    logger.info("series_level_results_parquet=%s", series_level_parquet)
    logger.info("pair_comparison_csv=%s", pair_comparison_csv)
    logger.info("best_candidate_summary_csv=%s", best_candidate_summary_csv)
    logger.info("excel_report_path=%s", excel_path)
    logger.info("manifest_path=%s", manifest_path)
    logger.info("total_candidates=%d success=%d failed=%d", n_candidates, success_count, failure_count)
    logger.info("architecture_tuning_benchmark finished elapsed=%.2fs", monotonic() - started_at)

    return {
        "run_id": run_id,
        "selected_series_path": str(selected_series_path),
        "candidate_level_results_parquet": str(candidate_level_parquet),
        "candidate_level_results_csv": str(candidate_level_csv),
        "series_level_results_parquet": str(series_level_parquet),
        "pair_comparison_csv": str(pair_comparison_csv),
        "best_candidate_summary_csv": str(best_candidate_summary_csv),
        "excel_report_path": str(excel_path),
        "manifest_path": str(manifest_path),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "n_candidates": int(n_candidates),
        "n_models": int(len(selected_models)),
        "n_horizons": int(len(horizons)),
    }
