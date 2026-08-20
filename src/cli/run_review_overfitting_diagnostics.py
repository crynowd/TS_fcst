from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.adapters import FitContext, TaskTimeoutError
from src.forecasting.data import build_series_lookup, load_log_returns_table, select_series
from src.forecasting.metrics import compute_regression_metrics
from src.forecasting.registry import build_model, get_model_specs, resolve_torch_device
from src.forecasting.runners import _task_config_hash, _training_params_for_model
from src.forecasting.windowing import build_rolling_origin_folds, build_supervised_windows
from src.utils.manifest import get_git_commit


CHAOTIC_MODELS = {
    "chaotic_mlp",
    "chaotic_lstm_forecast",
    "chaotic_esn",
    "transient_chaotic_esn",
    "chaotic_logistic_net",
}
COUNTERPART_MODELS = {"vanilla_mlp", "lstm_forecast", "esn", "naive_zero", "naive_mean", "ridge_lag"}
PRESENTATION_NAMES = {"chaotic_mlp": "chaoticMLP / chaotic_mlp"}
METRIC_PREFIXES = ["train", "validation", "test"]
REPORT_METRICS = [
    "train_RMSE",
    "validation_RMSE",
    "test_RMSE",
    "RMSE_test_train_gap",
    "RMSE_test_val_gap",
    "train_MAE",
    "validation_MAE",
    "test_MAE",
    "MAE_test_train_gap",
    "MAE_test_val_gap",
    "train_DA",
    "validation_DA",
    "test_DA",
    "DA_train_test_gap",
    "DA_val_test_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full train/validation/test overfitting diagnostics for benchmark models.")
    parser.add_argument("--config", default="configs/forecasting_benchmark_v2.yaml")
    parser.add_argument("--output-dir", default="artifacts/review_overfitting_diagnostics_full")
    parser.add_argument("--models", default="", help="Comma-separated model names. Defaults to chaotic benchmark models.")
    parser.add_argument("--include-counterparts", action="store_true", help="Also run non-chaotic counterpart/baseline models.")
    parser.add_argument("--max-series", type=int, default=0, help="Optional explicit cap for technical/debug runs. Default is full config.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write checkpoint after this many newly computed tasks.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _task_key(model_name: str, series_id: str, horizon: int, fold_id: int, config_hash: str) -> str:
    return f"{model_name}|{series_id}|h{int(horizon)}|f{int(fold_id)}|cfg{config_hash[:12]}"


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _split_bounds(timestamps: np.ndarray, idx: np.ndarray) -> tuple[Any, Any]:
    if idx is None or len(idx) == 0:
        return pd.NaT, pd.NaT
    ts = pd.to_datetime(timestamps[idx])
    return ts.min(), ts.max()


def _metric_fields(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_RMSE": _float_or_nan(metrics.get("rmse")),
        f"{prefix}_MAE": _float_or_nan(metrics.get("mae")),
        f"{prefix}_DA": _float_or_nan(metrics.get("directional_accuracy")),
    }


def _gap_fields(row: dict[str, Any]) -> dict[str, float]:
    return {
        "RMSE_test_train_gap": _float_or_nan(row.get("test_RMSE")) - _float_or_nan(row.get("train_RMSE")),
        "RMSE_test_val_gap": _float_or_nan(row.get("test_RMSE")) - _float_or_nan(row.get("validation_RMSE")),
        "MAE_test_train_gap": _float_or_nan(row.get("test_MAE")) - _float_or_nan(row.get("train_MAE")),
        "MAE_test_val_gap": _float_or_nan(row.get("test_MAE")) - _float_or_nan(row.get("validation_MAE")),
        "DA_train_test_gap": _float_or_nan(row.get("train_DA")) - _float_or_nan(row.get("test_DA")),
        "DA_val_test_gap": _float_or_nan(row.get("validation_DA")) - _float_or_nan(row.get("test_DA")),
    }


def _training_diag_fields(model: Any, ctx: FitContext, model_name: str) -> dict[str, Any]:
    if not hasattr(model, "get_training_diagnostics"):
        return {
            "epochs_trained": "not_available",
            "best_epoch": "not_available",
            "final_train_loss": "not_available",
            "best_validation_loss": "not_available",
            "early_stopped": "not_available",
            "patience": ctx.early_stopping_patience,
            "max_epochs": ctx.max_epochs,
            "learning_rate": ctx.learning_rate,
            "batch_size": ctx.batch_size,
        }
    diag = model.get_training_diagnostics() or {}
    val_hist = diag.get("val_loss_history") or []
    finite_val = [(i + 1, float(v)) for i, v in enumerate(val_hist) if v is not None and np.isfinite(float(v))]
    best_epoch = min(finite_val, key=lambda x: x[1])[0] if finite_val else "not_available"
    reason = str(diag.get("early_stopping_reason", ""))
    return {
        "epochs_trained": diag.get("epochs_completed", "not_available"),
        "best_epoch": best_epoch,
        "final_train_loss": diag.get("final_train_loss", "not_available"),
        "best_validation_loss": diag.get("best_val_loss", "not_available"),
        "early_stopped": bool(reason == "patience_exhausted") if reason else "not_available",
        "patience": ctx.early_stopping_patience,
        "max_epochs": ctx.max_epochs,
        "learning_rate": ctx.learning_rate,
        "batch_size": ctx.batch_size,
        "training_diagnostics_json": json.dumps(diag, ensure_ascii=False, default=str) if diag else "not_available",
        "model_family": get_model_specs()[model_name].family,
    }


def _resolve_models(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    active = list(dict.fromkeys(str(m) for m in cfg.get("models", {}).get("active", [])))
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if requested:
        selected = [m for m in active if m in set(requested)]
        missing = sorted(set(requested).difference(active))
        if missing:
            raise ValueError(f"Requested models are not active in benchmark config: {missing}")
        return selected
    wanted = set(CHAOTIC_MODELS)
    if args.include_counterparts:
        wanted.update(COUNTERPART_MODELS)
    return [m for m in active if m in wanted]


def _load_checkpoint(checkpoint_csv: Path, checkpoint_parquet: Path, state_path: Path) -> pd.DataFrame:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            latest = Path(str(state.get("latest_checkpoint_parquet", "")))
            if latest.exists():
                return pd.read_parquet(latest)
        except Exception:
            pass
    versioned = sorted(checkpoint_parquet.parent.glob(checkpoint_parquet.name + ".*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if versioned:
        return pd.read_parquet(versioned[0])
    if checkpoint_parquet.exists():
        return pd.read_parquet(checkpoint_parquet)
    if checkpoint_csv.exists():
        try:
            return pd.read_csv(checkpoint_csv)
        except Exception:
            pass
    return pd.DataFrame()


def _write_checkpoint(df: pd.DataFrame, checkpoint_csv: Path, checkpoint_parquet: Path, state_path: Path) -> tuple[Path, Path | None]:
    checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
    previous_latest = ""
    if state_path.exists():
        try:
            previous_latest = str(json.loads(state_path.read_text(encoding="utf-8")).get("latest_checkpoint_parquet", ""))
        except Exception:
            previous_latest = ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tmp_csv = checkpoint_csv.with_name(f"{checkpoint_csv.name}.{os.getpid()}.{stamp}.csv")
    tmp_parquet = checkpoint_parquet.with_name(f"{checkpoint_parquet.name}.{os.getpid()}.{stamp}.parquet")
    parquet_df = df.copy()
    for col in parquet_df.columns:
        if parquet_df[col].dtype == "object":
            parquet_df[col] = parquet_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    parquet_df.to_parquet(tmp_parquet, index=False)
    csv_path: Path | None = None
    try:
        df.to_csv(tmp_csv, index=False)
        csv_path = tmp_csv
    except PermissionError:
        try:
            tmp_csv.unlink(missing_ok=True)
        except Exception:
            pass
    success = int((df.get("status", pd.Series(dtype=str)) == "success").sum()) if not df.empty else 0
    failed = int((df.get("status", pd.Series(dtype=str)).isin(["error", "timeout"])).sum()) if not df.empty else 0
    state = {
        "updated_at_utc": _utc_now(),
        "rows": int(len(df)),
        "success": success,
        "failed": failed,
        "latest_checkpoint_parquet": str(tmp_parquet.resolve()),
        "latest_checkpoint_csv": str(csv_path.resolve()) if csv_path and csv_path.exists() else "",
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if previous_latest and previous_latest != str(tmp_parquet.resolve()):
        try:
            Path(previous_latest).unlink(missing_ok=True)
        except Exception:
            pass
    return tmp_parquet, csv_path


def _summary_stats(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {k: np.nan for k in ["mean", "median", "std", "p05", "p25", "p75", "p95", "min", "max"]} | {"n": 0}
    return {
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std(ddof=0)),
        "p05": float(vals.quantile(0.05)),
        "p25": float(vals.quantile(0.25)),
        "p75": float(vals.quantile(0.75)),
        "p95": float(vals.quantile(0.95)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": int(vals.size),
    }


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys)
    rows: list[dict[str, Any]] = []
    ok = df[df["status"] == "success"].copy()
    for key_vals, sdf in ok.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        row["n_observations"] = int(len(sdf))
        row["n_series"] = int(sdf["series_id"].nunique()) if "series_id" in sdf.columns else np.nan
        for metric in REPORT_METRICS:
            stats = _summary_stats(sdf[metric]) if metric in sdf.columns else _summary_stats(pd.Series(dtype=float))
            for stat_name, stat_value in stats.items():
                if stat_name != "n":
                    row[f"{metric}_{stat_name}"] = stat_value
        rows.append(row)
    return pd.DataFrame(rows)


def _gap_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = df[df["status"] == "success"].copy() if not df.empty else df
    for metric in [m for m in REPORT_METRICS if "gap" in m]:
        stats = _summary_stats(ok[metric]) if metric in ok.columns else _summary_stats(pd.Series(dtype=float))
        rows.append({"gap_metric": metric, **stats})
    return pd.DataFrame(rows)


def _config_used_rows(cfg: dict[str, Any], models: list[str], selected_count: int, output_dir: Path) -> pd.DataFrame:
    selected_meta = cfg.get("meta", {}).get("selected_architectures", {})
    rows = [
        {"key": "benchmark_config", "value": cfg.get("meta", {}).get("config_path", "")},
        {"key": "selected_architectures_config", "value": selected_meta.get("source_config_path", "")},
        {"key": "dataset_source", "value": cfg.get("data", {}).get("source_path", "")},
        {"key": "dataset_profile", "value": cfg.get("data", {}).get("dataset_profile", "")},
        {"key": "dataset_max_series", "value": cfg.get("data", {}).get("max_series", "")},
        {"key": "actual_series_count", "value": selected_count},
        {"key": "horizons", "value": json.dumps(cfg.get("horizons", []), ensure_ascii=False)},
        {"key": "window_sizes", "value": json.dumps(cfg.get("window_sizes", {}), ensure_ascii=False)},
        {"key": "validation", "value": json.dumps(cfg.get("validation", {}), ensure_ascii=False)},
        {"key": "training", "value": json.dumps(cfg.get("training", {}), ensure_ascii=False)},
        {"key": "models", "value": json.dumps(models, ensure_ascii=False)},
        {"key": "model_overrides", "value": json.dumps({m: cfg.get("model_overrides", {}).get(m, {}) for m in models}, ensure_ascii=False)},
        {"key": "diagnostic_output_dir", "value": str(output_dir.resolve())},
    ]
    return pd.DataFrame(rows)


def _format_md_value(value: Any, floatfmt: str = ".6f") -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), floatfmt)
    return str(value).replace("|", "\\|")


def _to_markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(_format_md_value(row[c], floatfmt=floatfmt) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep, *body])


def _build_excel_and_report(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    models: list[str],
    selected_count: int,
    output_dir: Path,
    started_at_utc: str,
) -> tuple[Path, Path]:
    ok = df[df["status"] == "success"].copy() if not df.empty else df
    errors = df[df["status"].isin(["error", "timeout"])].copy() if not df.empty else pd.DataFrame()
    model_horizon_summary = _aggregate(df, ["presentation_model_name", "model_name", "horizon"])
    model_summary = _aggregate(df, ["presentation_model_name", "model_name"])
    horizon_summary = _aggregate(df, ["horizon"])
    model_horizon_fold_summary = _aggregate(df, ["presentation_model_name", "model_name", "horizon", "fold"])
    gap_distribution = _gap_distribution(df)
    early_cols = [
        "model_name",
        "horizon",
        "epochs_trained",
        "best_epoch",
        "final_train_loss",
        "best_validation_loss",
        "early_stopped",
        "patience",
        "max_epochs",
        "learning_rate",
        "batch_size",
    ]
    early_stopping_summary = df[[c for c in early_cols if c in df.columns]].copy() if not df.empty else pd.DataFrame(columns=early_cols)
    config_used = _config_used_rows(cfg, models, selected_count, output_dir)

    summary_cols = [
        "presentation_model_name",
        "model_name",
        "horizon",
        "n_observations",
        "n_series",
    ] + [f"{m}_mean" for m in REPORT_METRICS]
    summary_for_review = model_horizon_summary[[c for c in summary_cols if c in model_horizon_summary.columns]].copy()

    xlsx_path = output_dir / "chaotic_overfitting_diagnostics_full.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_diagnostics", index=False)
        model_horizon_summary.to_excel(writer, sheet_name="model_horizon_summary", index=False)
        model_summary.to_excel(writer, sheet_name="model_summary", index=False)
        horizon_summary.to_excel(writer, sheet_name="horizon_summary", index=False)
        model_horizon_fold_summary.to_excel(writer, sheet_name="model_horizon_fold_summary", index=False)
        gap_distribution.to_excel(writer, sheet_name="gap_distribution", index=False)
        early_stopping_summary.to_excel(writer, sheet_name="early_stopping_summary", index=False)
        errors.to_excel(writer, sheet_name="errors", index=False)
        config_used.to_excel(writer, sheet_name="config_used", index=False)
        summary_for_review.to_excel(writer, sheet_name="summary_for_review", index=False)

    expected = selected_count * len(models) * len(cfg.get("horizons", [])) * int(cfg.get("validation", {}).get("n_folds", 3))
    success = int((df["status"] == "success").sum()) if not df.empty else 0
    failed = int((df["status"].isin(["error", "timeout"])).sum()) if not df.empty else 0
    main_table = summary_for_review.copy()
    rename = {f"{m}_mean": m for m in REPORT_METRICS}
    main_table = main_table.rename(columns=rename)
    keep = ["presentation_model_name", "horizon", "n_observations"] + REPORT_METRICS
    table_md = _to_markdown_table(main_table[[c for c in keep if c in main_table.columns]], floatfmt=".6f")

    top_rmse = ok.sort_values("RMSE_test_train_gap", ascending=False).head(1) if "RMSE_test_train_gap" in ok.columns else pd.DataFrame()
    top_da = ok.sort_values("DA_train_test_gap", ascending=False).head(1) if "DA_train_test_gap" in ok.columns else pd.DataFrame()
    gap_dist = {r["gap_metric"]: r for r in gap_distribution.to_dict("records")}
    median_rmse = gap_dist.get("RMSE_test_train_gap", {}).get("median", np.nan)
    p75_rmse = gap_dist.get("RMSE_test_train_gap", {}).get("p75", np.nan)
    median_da = gap_dist.get("DA_train_test_gap", {}).get("median", np.nan)
    p75_da = gap_dist.get("DA_train_test_gap", {}).get("p75", np.nan)

    if np.isfinite(median_rmse) and np.isfinite(p75_rmse) and p75_rmse < 0.01 and np.isfinite(p75_da) and p75_da < 0.03:
        interpretation = "Диагностика не выявила выраженного систематического train-test gap для большинства хаотических моделей."
        reviewer_text = (
            "В основном эксперименте контроль переобучения обеспечивался хронологическим rolling-origin разбиением, "
            "отложенными test-folds, validation/early stopping для нейросетевых моделей и расчетом признаков только на обучающем участке. "
            "Дополнительно была проведена диагностика train/validation/test-разрывов для хаотических моделей в воспроизведенной benchmark-схеме. "
            "Она не выявила выраженного систематического ухудшения на test относительно validation/train для большинства хаотических архитектур. "
            "При этом полное отсутствие переобучения не утверждается; вывод о хаотических моделях формулируется ограниченно."
        )
    elif np.isfinite(p75_rmse) and p75_rmse > 0.03:
        interpretation = "Диагностика показала систематический train-test gap у хаотических моделей; результаты нужно трактовать ограниченно."
        reviewer_text = (
            "Диагностика показала систематический train-test gap у хаотических моделей. "
            "Поэтому результаты по cNN следует трактовать как ограниченные: они демонстрируют потенциальную применимость в отдельных случаях, "
            "но не являются доказательством устойчивого превосходства. Основной вклад работы в этом случае состоит не в утверждении превосходства "
            "хаотических моделей, а в единой схеме сравнения и постановке feature-based model selection."
        )
    else:
        interpretation = "Диагностика выявила train-test gap у части хаотических моделей; выводы о них следует формулировать ограниченно."
        reviewer_text = (
            "Дополнительная диагностика выявила train-test gap у части хаотических моделей, наиболее выраженный для отдельных моделей/горизонтов. "
            "Это подтверждает чувствительность хаотических архитектур к параметрам и необходимость контроля переобучения. "
            "Поэтому в работе не утверждается универсальное преимущество cNN: вывод ограничен тем, что хаотические модели могут быть конкурентоспособны "
            "в отдельных постановках, особенно по Directional Accuracy, но требуют аккуратной validation/test-проверки."
        )

    sources = _to_markdown_table(config_used, floatfmt=".6f")
    md = f"""# Диагностика переобучения хаотических моделей

Generated at UTC: {_utc_now()}

## 1. Цель диагностики
Проверить, наблюдается ли у хаотических forecasting-моделей выраженный систематический train-test или validation-test gap, который мог бы указывать на переобучение. Диагностика не доказывает полное отсутствие переобучения; она проверяет наличие выраженного разрыва в воспроизведенной benchmark-схеме.

## 2. Источники истины
{sources}

## 3. Воспроизведенная экспериментальная схема
- Dataset/profile: `{cfg.get("data", {}).get("source_path", "")}`, `{cfg.get("data", {}).get("dataset_profile", "")}`.
- Число series: {selected_count}.
- Horizons: `{cfg.get("horizons", [])}`.
- Rolling-origin folds: `{cfg.get("validation", {}).get("n_folds", "")}`.
- Window sizes: `{cfg.get("window_sizes", {})}`.
- Validation: последний 20% блок train-fold, как в benchmark runner; для torch-моделей используется для early stopping.
- Models: `{models}`.
- Seeds/repeats: seed из benchmark config и model overrides; отдельных repeats в forecasting benchmark не найдено.
- Metric definitions: RMSE/MAE/Directional Accuracy из `src.forecasting.metrics.compute_regression_metrics`; gaps рассчитаны как test-train/test-validation для ошибок и train-test/validation-test для DA.

## 4. Объем расчета
- Ожидалось комбинаций: {expected}.
- Успешно рассчитано: {success}.
- Failed/timeout: {failed}.
- Запуск начат: {started_at_utc}.
- Checkpoint files: `{output_dir / "raw_diagnostics_checkpoint.csv"}`, `{output_dir / "raw_diagnostics_checkpoint.parquet"}`.

## 5. Главная таблица model x horizon
{table_md}

## 6. Интерпретация
{interpretation}

Overall gap distribution:
- RMSE test-train median: {median_rmse:.6f}; p75: {p75_rmse:.6f}.
- DA train-test median: {median_da:.6f}; p75: {p75_da:.6f}.

"""
    if not top_rmse.empty:
        r = top_rmse.iloc[0]
        md += f"- Largest single RMSE test-train gap: `{r['model_name']}`, series `{r['series_id']}`, h={int(r['horizon'])}, fold={int(r['fold'])}: {float(r['RMSE_test_train_gap']):.6f}.\n"
    if not top_da.empty:
        r = top_da.iloc[0]
        md += f"- Largest single DA train-test gap: `{r['model_name']}`, series `{r['series_id']}`, h={int(r['horizon'])}, fold={int(r['fold'])}: {float(r['DA_train_test_gap']):.6f}.\n"
    md += f"""

Validation-test gaps are interpreted separately from train-test gaps. A large time-series gap can reflect overfitting, but can also reflect a regime shift; therefore the conclusion is diagnostic and deliberately limited.

## 7. Ограничения
- Диагностика не доказывает невозможность переобучения.
- Train/test gap во временных рядах может отражать не только overfitting, но и изменение рыночного режима.
- Метрики train рассчитаны на fit-подвыборке после отделения validation-блока; validation-метрики рассчитаны на последнем 20% блоке train-fold.
- Для reservoir-моделей validation-блок не управляет обучением, но сохранен как дополнительный хронологический holdout.

## 8. Готовый текст для ответа рецензенту
{reviewer_text}

## Backup slide
- Expected combinations: {expected}; successful: {success}; failed: {failed}.
- Series: {selected_count}; horizons: {len(cfg.get("horizons", []))}; folds: {cfg.get("validation", {}).get("n_folds", "")}.
- RMSE test-train median/p75: {median_rmse:.6f}/{p75_rmse:.6f}.
- DA train-test median/p75: {median_da:.6f}/{p75_da:.6f}.
- Main conclusion: {interpretation}
"""
    md_path = output_dir / "chaotic_overfitting_diagnostics_full.md"
    md_path.write_text(md, encoding="utf-8")
    return xlsx_path, md_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at_utc = _utc_now()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output_dir / "diagnostics.log", encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("review_overfitting_diagnostics")
    cfg = load_forecasting_benchmark_config(args.config)
    if args.max_series and args.max_series > 0:
        cfg["data"]["max_series"] = int(args.max_series)
        cfg["data"]["dataset_limit"] = int(args.max_series)

    np.random.seed(int(cfg.get("random_seed", 42)))
    device_info = resolve_torch_device(str(cfg.get("device", "cpu")))
    logger.info("device=%s", device_info)

    models = _resolve_models(cfg, args)
    if not models:
        raise ValueError("No diagnostic models resolved from benchmark config.")

    log_returns = load_log_returns_table(cfg["data"]["source_path"])
    selected = select_series(
        log_returns_df=log_returns,
        dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")),
        max_series=int(cfg["data"].get("max_series", cfg["data"].get("dataset_limit", 0))) or None,
        series_selection_mode=str(cfg["data"].get("series_selection_mode", "first_n")),
        series_ids=cfg.get("filters", {}).get("series_ids") or cfg.get("data", {}).get("series_ids") or None,
    )
    lookup = build_series_lookup(log_returns, selected, dataset_profile=str(cfg["data"].get("dataset_profile", "core_balanced")))
    horizons = [int(h) for h in cfg.get("horizons", [1, 5, 20])]
    window_sizes = {int(k): int(v) for k, v in dict(cfg.get("window_sizes", {})).items()}
    n_folds = int(cfg.get("validation", {}).get("n_folds", 3))

    checkpoint_csv = output_dir / "raw_diagnostics_checkpoint.csv"
    checkpoint_parquet = output_dir / "raw_diagnostics_checkpoint.parquet"
    state_path = output_dir / "progress_state.json"
    existing = _load_checkpoint(checkpoint_csv, checkpoint_parquet, state_path) if not args.no_resume else pd.DataFrame()
    done_keys = set(existing.loc[existing["status"] == "success", "diagnostic_task_key"].astype(str)) if not existing.empty and "diagnostic_task_key" in existing.columns else set()
    rows = existing.to_dict("records") if not existing.empty else []

    expected = len(selected) * len(models) * len(horizons) * n_folds
    logger.info("diagnostic_start series=%d models=%d horizons=%d folds=%d expected=%d resume_rows=%d", len(selected), len(models), len(horizons), n_folds, expected, len(rows))
    t0 = monotonic()
    new_since_checkpoint = 0
    processed = 0

    for s in selected:
        series_df = lookup.get(s.series_id)
        if series_df is None or series_df.empty:
            continue
        for horizon in horizons:
            window_size = window_sizes.get(horizon)
            if not window_size:
                continue
            sup = build_supervised_windows(series_df, horizon=horizon, window_size=window_size)
            folds = build_rolling_origin_folds(len(sup.y), n_folds=n_folds)
            for model_name in models:
                for fold in folds:
                    config_hash = _task_config_hash(cfg, model_name, horizon, fold.fold_id, window_size, n_folds)
                    task_key = _task_key(model_name, s.series_id, horizon, fold.fold_id, config_hash)
                    processed += 1
                    if task_key in done_keys:
                        continue

                    train_idx = fold.train_idx
                    val_size = max(1, int(0.2 * len(train_idx)))
                    if len(train_idx) - val_size < 1:
                        val_size = 0
                    fit_idx = train_idx[:-val_size] if val_size > 0 else train_idx
                    val_idx = train_idx[-val_size:] if val_size > 0 else np.array([], dtype=np.int64)
                    test_idx = fold.test_idx

                    row: dict[str, Any] = {
                        "diagnostic_task_key": task_key,
                        "run_id": "review_overfitting_diagnostics_full",
                        "model_name": model_name,
                        "presentation_model_name": PRESENTATION_NAMES.get(model_name, model_name),
                        "series_id": s.series_id,
                        "instrument_id": s.series_id,
                        "ticker": s.ticker,
                        "market": s.market,
                        "horizon": int(horizon),
                        "fold": int(fold.fold_id),
                        "seed": cfg.get("model_overrides", {}).get(model_name, {}).get("seed", cfg.get("random_seed", "")),
                        "window_size": int(window_size),
                        "config_hash": config_hash,
                        "status": "success",
                        "error_message": "",
                    }
                    row["train_start"], row["train_end"] = _split_bounds(sup.timestamps, fit_idx)
                    row["validation_start"], row["validation_end"] = _split_bounds(sup.timestamps, val_idx)
                    row["test_start"], row["test_end"] = _split_bounds(sup.timestamps, test_idx)

                    try:
                        if len(fit_idx) < 2 or len(test_idx) < 1:
                            raise ValueError("insufficient fold samples")
                        model = build_model(model_name=model_name, config=cfg, logger=logger)
                        runtime = _training_params_for_model(cfg, model_name)
                        ctx = FitContext(
                            max_train_seconds=float(runtime.get("max_train_seconds_per_task", cfg.get("timeouts", {}).get("max_train_seconds_per_task", 300))),
                            max_predict_seconds=float(runtime.get("max_predict_seconds_per_task", cfg.get("timeouts", {}).get("max_predict_seconds_per_task", 30))),
                            max_epochs=int(runtime.get("max_epochs", cfg.get("training", {}).get("max_epochs", 20))),
                            early_stopping_patience=int(runtime.get("early_stopping_patience", cfg.get("training", {}).get("early_stopping_patience", 5))),
                            batch_size=int(runtime.get("batch_size", cfg.get("training", {}).get("batch_size", 128))),
                            learning_rate=float(runtime.get("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))),
                            weight_decay=float(runtime.get("weight_decay", cfg.get("training", {}).get("weight_decay", 0.0))),
                        )
                        fit_start = monotonic()
                        model.fit(
                            sup.X[fit_idx],
                            sup.y[fit_idx],
                            X_val=sup.X[val_idx] if len(val_idx) else None,
                            y_val=sup.y[val_idx] if len(val_idx) else None,
                            context=ctx,
                        )
                        row["fit_seconds"] = monotonic() - fit_start
                        pred_start = monotonic()
                        pred_train = model.predict(sup.X[fit_idx], context=ctx)
                        pred_val = model.predict(sup.X[val_idx], context=ctx) if len(val_idx) else np.array([], dtype=np.float64)
                        pred_test = model.predict(sup.X[test_idx], context=ctx)
                        row["predict_seconds"] = monotonic() - pred_start

                        row.update(_metric_fields("train", compute_regression_metrics(sup.y[fit_idx], pred_train, sup.y[fit_idx], logger=logger)))
                        if len(val_idx):
                            row.update(_metric_fields("validation", compute_regression_metrics(sup.y[val_idx], pred_val, sup.y[fit_idx], logger=logger)))
                        else:
                            row.update({f"validation_{m}": np.nan for m in ["RMSE", "MAE", "DA"]})
                        row.update(_metric_fields("test", compute_regression_metrics(sup.y[test_idx], pred_test, sup.y[fit_idx], logger=logger)))
                        row.update(_gap_fields(row))
                        row.update(_training_diag_fields(model, ctx, model_name))
                    except TaskTimeoutError as exc:
                        row["status"] = "timeout"
                        row["error_message"] = str(exc)
                        row.update({m: np.nan for m in REPORT_METRICS})
                    except Exception as exc:
                        row["status"] = "error"
                        row["error_message"] = f"{exc.__class__.__name__}: {exc}"
                        row["traceback"] = traceback.format_exc()
                        row.update({m: np.nan for m in REPORT_METRICS})

                    rows.append(row)
                    new_since_checkpoint += 1
                    elapsed = monotonic() - t0
                    logger.info(
                        "task_result processed=%d/%d model=%s series=%s h=%d fold=%d status=%s elapsed=%.1fs",
                        processed,
                        expected,
                        model_name,
                        s.series_id,
                        horizon,
                        fold.fold_id,
                        row["status"],
                        elapsed,
                    )
                    if new_since_checkpoint >= max(1, int(args.checkpoint_every)):
                        current = pd.DataFrame(rows).drop_duplicates(subset=["diagnostic_task_key"], keep="last")
                        _write_checkpoint(current, checkpoint_csv, checkpoint_parquet, state_path)
                        new_since_checkpoint = 0

    final_df = pd.DataFrame(rows).drop_duplicates(subset=["diagnostic_task_key"], keep="last") if rows else pd.DataFrame()
    latest_parquet, latest_csv = _write_checkpoint(final_df, checkpoint_csv, checkpoint_parquet, state_path)
    raw_final_parquet = output_dir / "raw_diagnostics.parquet"
    raw_final_csv = output_dir / "raw_diagnostics.csv"
    final_df.to_parquet(raw_final_parquet, index=False)
    final_df.to_csv(raw_final_csv, index=False)
    xlsx_path, md_path = _build_excel_and_report(final_df, cfg, models, len(selected), output_dir, started_at_utc)
    manifest = {
        "generated_at_utc": _utc_now(),
        "git_commit": get_git_commit(Path(cfg.get("meta", {}).get("project_root", Path.cwd()))),
        "config_path": cfg.get("meta", {}).get("config_path", ""),
        "outputs": {
            "raw_csv": str(raw_final_csv),
            "raw_parquet": str(raw_final_parquet),
            "checkpoint_csv": str(latest_csv) if latest_csv else "",
            "checkpoint_parquet": str(latest_parquet),
            "xlsx": str(xlsx_path),
            "md": str(md_path),
            "state": str(state_path),
        },
    }
    (output_dir / "diagnostics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
