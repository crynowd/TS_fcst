from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.adapters.naive import NaiveMeanAdapter, NaiveZeroAdapter
from src.forecasting.data import build_series_lookup, load_log_returns_table, select_series
from src.forecasting.metrics import compute_regression_metrics
from src.forecasting.registry import build_model, get_model_specs
from src.forecasting.windowing import SupervisedWindowData, build_rolling_origin_folds, build_supervised_windows


BASELINE_MODELS = ["naive_zero", "naive_mean", "ridge_lag"]


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [str(x.strip()) for x in value.split(",") if x.strip()]


def _esn_like_models() -> list[str]:
    specs = get_model_specs()
    return [name for name, spec in specs.items() if spec.family == "reservoir" or "esn" in name.lower()]


def _assert_window_alignment(
    sup: SupervisedWindowData,
    returns: np.ndarray,
    horizon: int,
) -> None:
    if sup.X.shape[0] != sup.y.shape[0] or sup.y.shape[0] != sup.timestamps.shape[0]:
        raise AssertionError("X/y/timestamp row counts differ")
    if not np.all(sup.feature_end_idx < sup.target_start_idx):
        raise AssertionError("feature window overlaps or reaches target window")
    if not np.all((sup.target_end_idx - sup.target_start_idx + 1) == horizon):
        raise AssertionError("target window length does not match horizon")

    cs = np.concatenate(([0.0], np.cumsum(returns.astype(np.float64))))
    expected_y = cs[sup.target_end_idx + 1] - cs[sup.target_start_idx]
    np.testing.assert_allclose(sup.y, expected_y, rtol=1e-10, atol=1e-12)


def _assert_fold_integrity(sup: SupervisedWindowData, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(np.intersect1d(train_idx, test_idx)) > 0:
        raise AssertionError("train/test sample indices overlap")
    if train_idx.size == 0 or test_idx.size == 0:
        raise AssertionError("empty train or test split")
    if int(train_idx.max()) >= int(test_idx.min()):
        raise AssertionError("train/test sample indices are not rolling-origin ordered")
    if int(sup.feature_end_idx[train_idx].max()) >= int(sup.target_start_idx[test_idx].min()):
        # This is allowed for walk-forward validation: train targets can be later than
        # the first test feature anchor. The diagnostic records it instead of failing.
        return


def _prediction_order_invariant(model_name: str, cfg: dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> bool:
    if len(X_test) < 2:
        return True
    model = build_model(model_name=model_name, config=cfg, logger=None)
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=np.float64).reshape(-1)
    rev_idx = np.arange(len(X_test) - 1, -1, -1, dtype=np.int64)
    pred_rev = np.asarray(model.predict(X_test[rev_idx]), dtype=np.float64).reshape(-1)[rev_idx]
    return bool(np.allclose(pred, pred_rev, rtol=1e-5, atol=1e-6, equal_nan=True))


def _run_target_permutation_check(
    model_name: str,
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if len(y_train) < 4 or len(y_test) < 1:
        return {"status": "skipped", "reason": "insufficient samples"}

    normal_model = build_model(model_name=model_name, config=cfg, logger=None)
    normal_model.fit(X_train, y_train)
    normal_pred = normal_model.predict(X_test)
    normal_rmse = compute_regression_metrics(y_test, normal_pred, y_train).get("rmse", np.nan)

    rng = np.random.default_rng(seed)
    permuted_y = np.asarray(y_train, dtype=np.float64).copy()
    rng.shuffle(permuted_y)
    perm_model = build_model(model_name=model_name, config=cfg, logger=None)
    perm_model.fit(X_train, permuted_y)
    perm_pred = perm_model.predict(X_test)
    perm_rmse = compute_regression_metrics(y_test, perm_pred, y_train).get("rmse", np.nan)

    ratio = float(perm_rmse / normal_rmse) if np.isfinite(normal_rmse) and normal_rmse > 0 else np.nan
    if not np.isfinite(ratio):
        status = "warning"
    elif ratio >= 1.05:
        status = "pass"
    else:
        status = "warning"

    return {
        "status": status,
        "normal_rmse": float(normal_rmse) if np.isfinite(normal_rmse) else None,
        "permuted_rmse": float(perm_rmse) if np.isfinite(perm_rmse) else None,
        "permuted_to_normal_rmse_ratio": ratio if np.isfinite(ratio) else None,
    }


def run_diagnostics(
    config_path: str,
    max_series: int,
    horizons: list[int] | None,
    models: list[str] | None,
    output_path: str | Path,
) -> dict[str, Any]:
    cfg = load_forecasting_benchmark_config(config_path)
    active_models = models or list(dict.fromkeys(BASELINE_MODELS + _esn_like_models()))
    selected_horizons = horizons or [int(h) for h in cfg.get("horizons", [1, 5, 20])]
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}

    log_returns = load_log_returns_table(cfg["data"]["source_path"])
    selected = select_series(
        log_returns_df=log_returns,
        dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")),
        max_series=max_series,
        series_selection_mode=str(cfg["data"].get("series_selection_mode", "first_n")),
        series_ids=None,
    )
    lookup = build_series_lookup(log_returns, selected)

    n_folds = int(cfg.get("validation", {}).get("n_folds", 1))
    task_rows: list[dict[str, Any]] = []
    split_signatures: dict[tuple[str, int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    permutation_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []

    for series in selected:
        sdf = lookup[series.series_id]
        returns = sdf["log_return"].to_numpy(dtype=np.float64)
        for horizon in selected_horizons:
            window_size = window_sizes.get(int(horizon))
            if not window_size:
                continue
            sup = build_supervised_windows(sdf, horizon=int(horizon), window_size=int(window_size))
            _assert_window_alignment(sup, returns=returns, horizon=int(horizon))
            folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
            if not folds:
                continue
            fold = folds[0]
            _assert_fold_integrity(sup, fold.train_idx, fold.test_idx)
            split_signatures[(series.series_id, int(horizon), int(fold.fold_id))] = (
                tuple(int(x) for x in fold.train_idx),
                tuple(int(x) for x in fold.test_idx),
            )

            X_train = sup.X[fold.train_idx]
            y_train = sup.y[fold.train_idx]
            X_test = sup.X[fold.test_idx]
            y_test = sup.y[fold.test_idx]

            if "naive_zero" in active_models:
                z = NaiveZeroAdapter()
                z.fit(X_train, y_train)
                pred = z.predict(X_test)
                baseline_rows.append({"model_name": "naive_zero", "check": "all_zero", "status": bool(np.all(pred == 0.0))})
                if not np.all(pred == 0.0):
                    raise AssertionError("naive_zero produced non-zero predictions")

            if "naive_mean" in active_models:
                m = NaiveMeanAdapter()
                m.fit(X_train, y_train)
                expected = float(np.nanmean(y_train)) if y_train.size else 0.0
                pred = m.predict(X_test)
                ok = bool(np.allclose(pred, expected, rtol=0.0, atol=1e-12))
                baseline_rows.append({"model_name": "naive_mean", "check": "train_mean_only", "status": ok})
                if not ok:
                    raise AssertionError("naive_mean prediction differs from train-only mean")

            for model_name in active_models:
                model = build_model(model_name=model_name, config=cfg, logger=None)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                if pred.shape[0] != y_test.shape[0]:
                    raise AssertionError(f"{model_name} prediction length differs from y_test length")
                metrics = compute_regression_metrics(y_true=y_test, y_pred=pred, y_train=y_train)
                task_rows.append(
                    {
                        "series_id": series.series_id,
                        "horizon": int(horizon),
                        "fold_id": int(fold.fold_id),
                        "model_name": model_name,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "train_idx_min": int(fold.train_idx.min()),
                        "train_idx_max": int(fold.train_idx.max()),
                        "test_idx_min": int(fold.test_idx.min()),
                        "test_idx_max": int(fold.test_idx.max()),
                        "test_timestamp_min": str(pd.Timestamp(sup.timestamps[fold.test_idx][0])),
                        "test_timestamp_max": str(pd.Timestamp(sup.timestamps[fold.test_idx][-1])),
                        "rmse": float(metrics["rmse"]) if np.isfinite(metrics["rmse"]) else None,
                        "directional_accuracy": (
                            float(metrics["directional_accuracy"])
                            if np.isfinite(metrics["directional_accuracy"])
                            else None
                        ),
                    }
                )

                if model_name in _esn_like_models():
                    order_ok = _prediction_order_invariant(model_name, cfg, X_train, y_train, X_test)
                    order_rows.append({"model_name": model_name, "series_id": series.series_id, "horizon": int(horizon), "status": order_ok})
                    if not order_ok:
                        raise AssertionError(f"{model_name} predictions depend on test-row order")

            for model_name in _esn_like_models():
                if model_name in active_models:
                    perm = _run_target_permutation_check(model_name, cfg, X_train, y_train, X_test, y_test, seed=1234)
                    perm.update({"model_name": model_name, "series_id": series.series_id, "horizon": int(horizon), "fold_id": int(fold.fold_id)})
                    permutation_rows.append(perm)

    expected_task_count = len(task_rows)
    if expected_task_count == 0:
        raise AssertionError("diagnostic run produced no tasks")

    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "source_path": cfg["data"]["source_path"],
        "models": active_models,
        "esn_like_models_registered": _esn_like_models(),
        "horizons": selected_horizons,
        "series_count": len(selected),
        "task_count": expected_task_count,
        "split_signature_count": len(split_signatures),
        "baseline_checks": baseline_rows,
        "order_invariance_checks": order_rows,
        "target_permutation_checks": permutation_rows,
        "tasks": task_rows,
    }

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESN/baseline leakage diagnostics on a small forecasting sample")
    parser.add_argument("--config", default="configs/forecasting_benchmark_smoke_v1.yaml")
    parser.add_argument("--max-series", type=int, default=1)
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons, e.g. 1,5,20")
    parser.add_argument("--models", default=None, help="Comma-separated model names; default is baselines plus ESN-like models")
    parser.add_argument("--output", default="reports/esn_baseline_leakage_diagnostic.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_diagnostics(
        config_path=args.config,
        max_series=int(args.max_series),
        horizons=_parse_csv_ints(args.horizons),
        models=_parse_csv_strings(args.models),
        output_path=args.output,
    )
    warnings = [
        row
        for row in summary["target_permutation_checks"]
        if row.get("status") not in {"pass", "skipped"}
    ]
    print(
        "diagnostic_ok tasks={tasks} series={series} models={models} permutation_warnings={warnings} output={output}".format(
            tasks=summary["task_count"],
            series=summary["series_count"],
            models=len(summary["models"]),
            warnings=len(warnings),
            output=str(Path(args.output).resolve()),
        )
    )


if __name__ == "__main__":
    main()
