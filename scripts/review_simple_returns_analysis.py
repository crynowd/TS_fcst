from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "forecasting" / "forecasting_benchmark_v2"
DEFAULT_OUT = ROOT / "artifacts" / "review_simple_returns_analysis"

KEYS = ["run_id", "model_name", "series_id", "ticker", "market", "horizon", "fold_id"]
CASE_KEYS = ["series_id", "ticker", "market", "horizon", "fold_id"]
METRICS = ["rmse", "mae", "directional_accuracy"]
LOW_IS_BETTER = {"rmse": True, "mae": True, "directional_accuracy": False}
PLOT_MODELS_HINT = [
    "chaotic_esn",
    "transient_chaotic_esn",
    "chaotic_mlp",
    "chaotic_lstm_forecast",
    "vanilla_mlp",
    "lstm_forecast",
    "esn",
]


def _fmt(x: Any, digits: int = 6) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}g}"


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    lines = ["| " + " | ".join(map(str, work.columns)) + " |"]
    lines.append("| " + " | ".join("---" for _ in work.columns) + " |")
    for _, row in work.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(_fmt(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _rank(metric: str, values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=LOW_IS_BETTER[metric])


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    ordered = df.sort_values([metric, "model_name"], ascending=[LOW_IS_BETTER[metric], True], kind="stable")
    return ordered.iloc[0]


def _schema_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [int(df[col].notna().sum()) for col in df.columns],
            "n_rows": len(df),
        }
    )


def _compute_simple_task_metrics(predictions_path: Path) -> pd.DataFrame:
    cols = KEYS + ["y_true", "y_pred"]
    pred = pd.read_parquet(predictions_path, columns=cols)
    simple_true = np.expm1(pred["y_true"].to_numpy(dtype=np.float64))
    simple_pred = np.expm1(pred["y_pred"].to_numpy(dtype=np.float64))
    err = simple_pred - simple_true
    pred["_abs_err"] = np.abs(err)
    pred["_sq_err"] = err * err
    pred["_dir_ok"] = (np.sign(simple_true) == np.sign(simple_pred)).astype(np.float64)

    grouped = (
        pred.groupby(KEYS, sort=False, observed=True)
        .agg(
            n_obs=("y_true", "size"),
            mae_sum=("_abs_err", "sum"),
            mse_sum=("_sq_err", "sum"),
            directional_accuracy=("_dir_ok", "mean"),
        )
        .reset_index()
    )
    grouped["mae"] = grouped["mae_sum"] / grouped["n_obs"]
    grouped["mse"] = grouped["mse_sum"] / grouped["n_obs"]
    grouped["rmse"] = np.sqrt(grouped["mse"])
    return grouped[KEYS + ["n_obs", "mae", "mse", "rmse", "directional_accuracy"]]


def _aggregate_by_horizon_model(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["horizon", "model_name"], sort=True, observed=True)
        .agg(
            n_tasks=("series_id", "size"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
    )


def _comparison_tables(log_hm: pd.DataFrame, simple_hm: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log = log_hm.copy()
    simple = simple_hm.copy()
    log["return_space"] = "log"
    simple["return_space"] = "simple"
    mean_by_horizon = pd.concat([log, simple], ignore_index=True)
    mean_by_horizon = (
        mean_by_horizon.groupby(["return_space", "horizon"], sort=True)
        .agg(
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
    )
    mean_overall = (
        pd.concat([log, simple], ignore_index=True)
        .groupby("return_space", sort=True)
        .agg(
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
    )

    best_rows: list[dict[str, Any]] = []
    for space, df in [("log", log_hm), ("simple", simple_hm)]:
        for (horizon, hdf) in df.groupby("horizon", sort=True):
            for metric in METRICS:
                row = _best_row(hdf, metric)
                best_rows.append(
                    {
                        "return_space": space,
                        "horizon": int(horizon),
                        "metric": metric,
                        "best_model": row["model_name"],
                        "best_value": float(row[metric]),
                    }
                )
    best_models = pd.DataFrame(best_rows)

    rank_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    merged = log_hm.merge(simple_hm, on=["horizon", "model_name"], suffixes=("_log", "_simple"))
    for (horizon, hdf) in merged.groupby("horizon", sort=True):
        for metric in METRICS:
            work = hdf[["horizon", "model_name", f"{metric}_log", f"{metric}_simple"]].dropna().copy()
            work["rank_log"] = _rank(metric, work[f"{metric}_log"])
            work["rank_simple"] = _rank(metric, work[f"{metric}_simple"])
            if len(work) > 1:
                rho, pvalue = stats.spearmanr(work["rank_log"], work["rank_simple"], nan_policy="omit")
            else:
                rho, pvalue = np.nan, np.nan
            rank_rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    "n_models": int(len(work)),
                }
            )
            work["rank_shift_abs"] = (work["rank_simple"] - work["rank_log"]).abs()
            max_shift = work.sort_values(["rank_shift_abs", "model_name"], ascending=[False, True], kind="stable").iloc[0]
            shift_rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "model_name": max_shift["model_name"],
                    "rank_log": float(max_shift["rank_log"]),
                    "rank_simple": float(max_shift["rank_simple"]),
                    "rank_shift_abs": float(max_shift["rank_shift_abs"]),
                }
            )
    rank_correlations = pd.DataFrame(rank_rows)
    max_rank_shifts = pd.DataFrame(shift_rows)
    return {
        "mean_metrics_by_horizon": mean_by_horizon,
        "mean_metrics_overall": mean_overall,
        "best_models_by_horizon_metric": best_models,
        "rank_correlations": rank_correlations,
        "max_rank_shifts": max_rank_shifts,
    }


def _select_cases(simple_task_metrics: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for metric in ["rmse", "directional_accuracy"]:
        rows = []
        for case, cdf in simple_task_metrics.groupby(CASE_KEYS, sort=False):
            best = _best_row(cdf, metric)
            rows.append(
                {
                    **dict(zip(CASE_KEYS, case, strict=True)),
                    "selection_metric": metric,
                    "best_model": best["model_name"],
                    "best_value": float(best[metric]),
                }
            )
        cases = pd.DataFrame(rows)
        cases = cases.dropna(subset=["best_value"])
        asc = LOW_IS_BETTER[metric]
        best_case = cases.sort_values(["best_value", "series_id", "fold_id"], ascending=[asc, True, True], kind="stable").iloc[0]
        bad_case = cases.sort_values(["best_value", "series_id", "fold_id"], ascending=[not asc, True, True], kind="stable").iloc[0]
        median_value = float(cases["best_value"].median())
        cases["_dist"] = (cases["best_value"] - median_value).abs()
        median_case = cases.sort_values(["_dist", "series_id", "fold_id"], ascending=[True, True, True], kind="stable").iloc[0]
        for label, row in [("best", best_case), ("bad", bad_case), ("median", median_case)]:
            rec = row.drop(labels=[c for c in ["_dist"] if c in row.index]).to_dict()
            rec["case_type"] = label
            selected.append(rec)
    return pd.DataFrame(selected)


def _case_filter(df: pd.DataFrame, case: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key in CASE_KEYS:
        mask &= df[key] == case[key]
    return mask


def _models_for_case(task_metrics: pd.DataFrame, case: pd.Series) -> list[str]:
    cdf = task_metrics.loc[_case_filter(task_metrics, case)].copy()
    models: list[str] = ["naive_zero"]
    for metric in ["rmse", "directional_accuracy"]:
        model = str(_best_row(cdf, metric)["model_name"])
        if model not in models:
            models.append(model)
    if len(models) < 4:
        for candidate in PLOT_MODELS_HINT:
            if candidate in set(cdf["model_name"]) and candidate not in models:
                models.append(candidate)
                break
    return models[:4]


def _plot_case(
    pred_case: pd.DataFrame,
    task_case: pd.DataFrame,
    case: pd.Series,
    models: list[str],
    output_path: Path,
) -> None:
    work = pred_case[pred_case["model_name"].isin(models)].copy()
    work["simple_true"] = np.expm1(work["y_true"].to_numpy(dtype=np.float64))
    work["simple_pred"] = np.expm1(work["y_pred"].to_numpy(dtype=np.float64))
    work = work.sort_values(["model_name", "timestamp"], kind="stable")
    actual = work[["timestamp", "simple_true"]].drop_duplicates("timestamp").sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].plot(actual["timestamp"], actual["simple_true"], color="black", linewidth=1.8, label="actual simple return")
    colors = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        mdf = work[work["model_name"] == model]
        if mdf.empty:
            continue
        task_row = task_case[task_case["model_name"] == model].iloc[0]
        label = f"{model} (RMSE={_fmt(task_row['rmse'], 4)}, DA={_fmt(task_row['directional_accuracy'], 4)})"
        axes[0].plot(mdf["timestamp"], mdf["simple_pred"], linewidth=1.25, color=colors(idx), label=label)
        axes[1].plot(
            mdf["timestamp"],
            mdf["simple_pred"].to_numpy(dtype=float) - mdf["simple_true"].to_numpy(dtype=float),
            linewidth=1.0,
            color=colors(idx),
            label=model,
        )

    title = (
        f"{case['case_type'].upper()} by {case['selection_metric']} | "
        f"{case['series_id']} h={int(case['horizon'])} fold={int(case['fold_id'])} | "
        f"best={case['best_model']} value={_fmt(case['best_value'], 5)}"
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Cumulative simple return")
    axes[1].set_ylabel("Prediction minus actual")
    axes[1].set_xlabel("Timestamp")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_two_model_comparison(
    pred: pd.DataFrame,
    task_metrics: pd.DataFrame,
    case: pd.Series,
    output_path: Path,
    metric_label: str,
) -> None:
    cdf = task_metrics.loc[_case_filter(task_metrics, case)].copy()
    rmse_model = str(_best_row(cdf, "rmse")["model_name"])
    da_model = str(_best_row(cdf, "directional_accuracy")["model_name"])
    models = list(dict.fromkeys([rmse_model, da_model]))
    work = pred.loc[_case_filter(pred, case) & pred["model_name"].isin(models)].copy()
    work["simple_true"] = np.expm1(work["y_true"].to_numpy(dtype=np.float64))
    work["simple_pred"] = np.expm1(work["y_pred"].to_numpy(dtype=np.float64))
    actual = work[["timestamp", "simple_true"]].drop_duplicates("timestamp").sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].plot(actual["timestamp"], actual["simple_true"], color="black", linewidth=1.8, label="actual simple return")
    colors = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        mdf = work[work["model_name"] == model].sort_values("timestamp")
        task_row = cdf[cdf["model_name"] == model].iloc[0]
        label = f"{model} (RMSE={_fmt(task_row['rmse'], 4)}, DA={_fmt(task_row['directional_accuracy'], 4)})"
        axes[0].plot(mdf["timestamp"], mdf["simple_pred"], linewidth=1.35, color=colors(idx), label=label)
        axes[1].plot(
            mdf["timestamp"],
            mdf["simple_pred"].to_numpy(dtype=float) - mdf["simple_true"].to_numpy(dtype=float),
            linewidth=1.0,
            color=colors(idx),
            label=model,
        )
    axes[0].set_title(
        f"RMSE-best vs DA-best | {metric_label} | {case['series_id']} h={int(case['horizon'])} fold={int(case['fold_id'])}"
    )
    axes[0].set_ylabel("Cumulative simple return")
    axes[1].set_ylabel("Prediction minus actual")
    axes[1].set_xlabel("Timestamp")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _make_plots(
    predictions_path: Path,
    task_metrics: pd.DataFrame,
    selected_cases: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    cols = KEYS + ["timestamp", "y_true", "y_pred"]
    pred = pd.read_parquet(predictions_path, columns=cols)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for _, case in selected_cases.iterrows():
        models = _models_for_case(task_metrics, case)
        pred_case = pred.loc[_case_filter(pred, case) & pred["model_name"].isin(models)].copy()
        task_case = task_metrics.loc[_case_filter(task_metrics, case)].copy()
        filename = (
            f"case_{case['selection_metric']}_{case['case_type']}_"
            f"{case['series_id'].replace(':', '_')}_h{int(case['horizon'])}_f{int(case['fold_id'])}.png"
        )
        path = figures_dir / filename
        _plot_case(pred_case, task_case, case, models, path)
        paths.append(path)

    diff_cases = []
    for _, row in selected_cases.iterrows():
        cdf = task_metrics.loc[_case_filter(task_metrics, row)]
        if str(_best_row(cdf, "rmse")["model_name"]) != str(_best_row(cdf, "directional_accuracy")["model_name"]):
            diff_cases.append(row)
    if len(diff_cases) < 2:
        case_rows = []
        for case, cdf in task_metrics.groupby(CASE_KEYS, sort=False):
            rmse_model = str(_best_row(cdf, "rmse")["model_name"])
            da_model = str(_best_row(cdf, "directional_accuracy")["model_name"])
            if rmse_model == da_model:
                continue
            case_rows.append(
                {
                    **dict(zip(CASE_KEYS, case, strict=True)),
                    "rmse_model": rmse_model,
                    "da_model": da_model,
                    "gap": abs(float(_best_row(cdf, "directional_accuracy")["directional_accuracy"]) - float(cdf[cdf["model_name"] == rmse_model]["directional_accuracy"].iloc[0])),
                }
            )
        extra = pd.DataFrame(case_rows).sort_values(["gap", "series_id"], ascending=[False, True], kind="stable")
        for _, row in extra.head(2 - len(diff_cases)).iterrows():
            diff_cases.append(row)

    for idx, row in enumerate(diff_cases[:2], start=1):
        path = figures_dir / f"two_model_rmse_vs_da_{idx}_{row['series_id'].replace(':', '_')}_h{int(row['horizon'])}_f{int(row['fold_id'])}.png"
        _plot_two_model_comparison(pred, task_metrics, row, path, metric_label=f"example {idx}")
        paths.append(path)
    return paths


def _write_summary(
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
    selected_cases: pd.DataFrame,
    figure_paths: list[Path],
    input_dir: Path,
) -> None:
    rank = tables["rank_correlations"]
    shifts = tables["max_rank_shifts"]
    min_rho = float(rank["spearman_rho"].min()) if not rank.empty else np.nan
    max_shift = float(shifts["rank_shift_abs"].max()) if not shifts.empty else np.nan
    best = tables["best_models_by_horizon_metric"]
    pivot_best = best.pivot_table(index=["horizon", "metric"], columns="return_space", values="best_model", aggfunc="first").reset_index()
    changed_best = int((pivot_best.get("log") != pivot_best.get("simple")).sum()) if {"log", "simple"}.issubset(pivot_best.columns) else 0

    conclusion = (
        "Переход к обычным накопленным доходностям не меняет основные выводы: "
        "ранжирования моделей почти полностью совпадают, а Directional Accuracy остается той же по знаку."
    )
    if changed_best > 0 or (np.isfinite(max_shift) and max_shift >= 2) or (np.isfinite(min_rho) and min_rho < 0.95):
        conclusion = (
            "Переход к обычным накопленным доходностям частично меняет локальные выводы: "
            "есть изменения победителей или заметные сдвиги рангов, поэтому для защиты стоит явно оговорить пространство доходностей."
        )

    rel_figs = [str(p.relative_to(out_dir)) for p in figure_paths]
    text = f"""# Simple-Returns Post-Hoc Analysis

Input artifacts: `{input_dir.as_posix()}`.

No models were retrained. Existing forecast-level `y_true` and `y_pred` were transformed as:

```text
simple_true = exp(y_true) - 1
simple_pred = exp(y_pred) - 1
```

## Forecast Artifact Schema

Full schema was saved to `schema_predictions.csv`; `metrics_long.parquet` schema was saved to `schema_metrics_long.csv`.

## Mean Metrics

Overall averages across `horizon x model_name`:

{_markdown_table(tables["mean_metrics_overall"])}

By horizon:

{_markdown_table(tables["mean_metrics_by_horizon"])}

## Best Models

{_markdown_table(tables["best_models_by_horizon_metric"])}

## Rank Stability

Spearman rank correlations:

{_markdown_table(tables["rank_correlations"])}

Maximum rank shifts:

{_markdown_table(tables["max_rank_shifts"])}

## Selected Forecast Cases

{_markdown_table(selected_cases)}

## Figures

{chr(10).join(f'- `{p}`' for p in rel_figs)}

## Conclusion

{conclusion}

For the defense, the usable point is that RMSE/MAE change scale after the nonlinear `exp(.) - 1` transform, but the broad model ordering is stable. DA is effectively invariant because the transform preserves the sign around zero.
"""
    (out_dir / "summary.md").write_text(text, encoding="utf-8")


def run(input_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = input_dir / "predictions.parquet"
    metrics_path = input_dir / "metrics_long.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    _schema_frame(predictions_path).to_csv(out_dir / "schema_predictions.csv", index=False)
    _schema_frame(metrics_path).to_csv(out_dir / "schema_metrics_long.csv", index=False)

    simple_task = _compute_simple_task_metrics(predictions_path)
    simple_task.to_csv(out_dir / "simple_task_metrics.csv", index=False)
    simple_hm = _aggregate_by_horizon_model(simple_task)
    simple_hm.to_csv(out_dir / "simple_metrics_by_horizon_model.csv", index=False)

    log_task = pd.read_parquet(metrics_path, columns=KEYS + METRICS)
    log_task.to_csv(out_dir / "log_task_metrics_from_artifact.csv", index=False)
    log_hm = _aggregate_by_horizon_model(log_task)
    log_hm.to_csv(out_dir / "log_metrics_by_horizon_model.csv", index=False)

    tables = _comparison_tables(log_hm, simple_hm)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    selected_cases = _select_cases(simple_task)
    selected_cases.to_csv(out_dir / "selected_plot_cases.csv", index=False)
    figure_paths = _make_plots(predictions_path, simple_task, selected_cases, out_dir)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "created_files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_summary(out_dir, tables, selected_cases, figure_paths, input_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc simple-return analysis for saved forecasting predictions.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
