from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "review_statistical_analysis"

HORIZONS = [1, 5, 20]
METRIC_DIRECTIONS = {"rmse": "min", "directional_accuracy": "max"}
CHAOS_MODELS = {
    "chaotic_esn",
    "transient_chaotic_esn",
    "chaotic_mlp",
    "chaotic_logistic_net",
    "chaotic_lstm_forecast",
}
BASELINE_MODELS = {"naive_zero", "naive_mean"}


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _to_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: float, digits: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}g}"


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    cols = list(work.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in work.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(_fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _normal_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    se = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    if vals.size > 1 and np.isfinite(se):
        tcrit = float(stats.t.ppf(1 - alpha / 2, vals.size - 1))
        mean = float(np.mean(vals))
        return mean - tcrit * se, mean + tcrit * se, se
    return float("nan"), float("nan"), se


def _row_bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def _hierarchical_cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    work = df[["split_seed", "series_id", value_col]].dropna().copy()
    if work.empty:
        return float("nan"), float("nan")
    seed_groups: dict[Any, dict[Any, np.ndarray]] = {}
    for seed, sdf in work.groupby("split_seed", sort=False):
        seed_groups[seed] = {
            sid: g[value_col].to_numpy(dtype=float)
            for sid, g in sdf.groupby("series_id", sort=False)
        }
    seeds = np.array(list(seed_groups.keys()), dtype=object)
    if seeds.size == 0:
        return float("nan"), float("nan")

    means: list[float] = []
    for _ in range(n_boot):
        sample_vals: list[np.ndarray] = []
        sampled_seeds = rng.choice(seeds, size=seeds.size, replace=True)
        for seed in sampled_seeds:
            series_keys = np.array(list(seed_groups[seed].keys()), dtype=object)
            sampled_series = rng.choice(series_keys, size=series_keys.size, replace=True)
            sample_vals.extend(seed_groups[seed][sid] for sid in sampled_series)
        if sample_vals:
            means.append(float(np.mean(np.concatenate(sample_vals))))
    if not means:
        return float("nan"), float("nan")
    return tuple(np.quantile(np.asarray(means), [0.025, 0.975]).astype(float))


def _match_best_config(routing: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    mask = (
        (routing["horizon"].astype(int) == int(config["horizon"]))
        & (routing["target_metric"].astype(str) == str(config["target_metric"]))
        & (routing["model"].astype(str) == str(config["model"]))
        & (routing["candidate_set"].astype(str) == str(config["candidate_set"]))
        & (routing["feature_set"].astype(str) == str(config["feature_set"]))
        & (routing["balancing_mode"].astype(str) == str(config["balancing_mode"]))
        & (routing["decision_rule"].astype(str) == str(config["decision_rule"]))
    )
    threshold = _to_float(config.get("confidence_threshold", np.nan))
    if np.isfinite(threshold):
        mask &= np.isclose(pd.to_numeric(routing["confidence_threshold"], errors="coerce"), threshold, equal_nan=False)
    else:
        mask &= pd.to_numeric(routing["confidence_threshold"], errors="coerce").isna()
    return routing.loc[mask].copy()


def load_best_routing(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_cfg = pd.read_csv(paths["best_config"])
    needed_cols = [
        "run_id",
        "repeat_id",
        "split_seed",
        "method",
        "model",
        "horizon",
        "target_metric",
        "feature_set",
        "candidate_set",
        "object_id",
        "series_id",
        "fold_id",
        "balancing_mode",
        "decision_rule",
        "confidence_threshold",
        "selected_model",
        "oracle_model",
        "best_single_model",
        "achieved_metric",
        "oracle_metric",
        "baseline_metric",
        "improvement_vs_best_single",
        "gap_to_oracle",
    ]
    routing = pd.read_parquet(paths["routing"], columns=needed_cols)
    frames = []
    for _, cfg in best_cfg.iterrows():
        sub = _match_best_config(routing, cfg)
        sub["selected_meta_model"] = str(cfg["model"])
        frames.append(sub)
    best_routing = pd.concat(frames, ignore_index=True)
    for c in ["achieved_metric", "oracle_metric", "baseline_metric", "improvement_vs_best_single", "gap_to_oracle"]:
        best_routing[c] = pd.to_numeric(best_routing[c], errors="coerce")
    return best_cfg, best_routing


def directional_accuracy_tests(best_routing: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    work = best_routing[best_routing["target_metric"] == "directional_accuracy"].copy()
    for horizon, hdf in work.groupby("horizon", sort=True):
        hdf = hdf.copy()
        hdf["diff"] = hdf["achieved_metric"] - hdf["baseline_metric"]
        diff = hdf["diff"].dropna().to_numpy(dtype=float)
        ci_low, ci_high, se = _normal_ci(diff)
        boot_low, boot_high = _row_bootstrap_ci(diff, rng, n_boot)
        cl_low, cl_high = _hierarchical_cluster_bootstrap_ci(hdf, "diff", rng, n_boot)

        t_p = stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue if diff.size > 1 else np.nan
        nonzero = diff[np.abs(diff) > 1e-15]
        wilcoxon_p = stats.wilcoxon(nonzero, alternative="two-sided").pvalue if nonzero.size > 0 else np.nan
        pos = int(np.sum(diff > 1e-15))
        neg = int(np.sum(diff < -1e-15))
        sign_p = stats.binomtest(pos, pos + neg, 0.5, alternative="two-sided").pvalue if (pos + neg) > 0 else np.nan
        seed_summary = (
            hdf.groupby("split_seed", sort=True)["diff"].mean().reset_index(name="seed_diff_mean")
        )

        rows.append(
            {
                "horizon": int(horizon),
                "metric": "directional_accuracy",
                "metamodel_mean": float(hdf["achieved_metric"].mean()),
                "fixed_mean": float(hdf["baseline_metric"].mean()),
                "oracle_mean": float(hdf["oracle_metric"].mean()),
                "diff_mean": float(np.mean(diff)),
                "diff_std": float(np.std(diff, ddof=1)) if diff.size > 1 else np.nan,
                "diff_se": se,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "bootstrap_ci_low": boot_low,
                "bootstrap_ci_high": boot_high,
                "clustered_bootstrap_ci_low": cl_low,
                "clustered_bootstrap_ci_high": cl_high,
                "p_ttest": float(t_p),
                "p_wilcoxon": float(wilcoxon_p),
                "p_sign": float(sign_p),
                "n_obs": int(len(hdf)),
                "n_instruments": int(hdf["series_id"].nunique()),
                "n_seeds": int(hdf["split_seed"].nunique()),
                "n_seed_configs_positive": int((seed_summary["seed_diff_mean"] > 0).sum()),
                "n_seed_configs_total": int(len(seed_summary)),
                "positive_obs": pos,
                "negative_obs": neg,
                "zero_obs": int(np.sum(np.abs(diff) <= 1e-15)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "directional_accuracy_significance.csv", index=False)
    return out


def metric_confidence_intervals(best_routing: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 17)
    metric_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for (metric, horizon), hdf in best_routing.groupby(["target_metric", "horizon"], sort=True):
        aliases = {
            "metamodel": "achieved_metric",
            "best_fixed": "baseline_metric",
            "oracle": "oracle_metric",
        }
        for label, col in aliases.items():
            vals = hdf[col].dropna().to_numpy(dtype=float)
            ci_low, ci_high, se = _normal_ci(vals)
            boot_low, boot_high = _row_bootstrap_ci(vals, rng, n_boot)
            cl_low, cl_high = _hierarchical_cluster_bootstrap_ci(hdf.rename(columns={col: "_value"}), "_value", rng, n_boot)
            metric_rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "estimator": label,
                    "mean": float(np.mean(vals)) if vals.size else np.nan,
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                    "se": se,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                    "clustered_bootstrap_ci_low": cl_low,
                    "clustered_bootstrap_ci_high": cl_high,
                    "n_obs": int(len(hdf)),
                    "n_instruments": int(hdf["series_id"].nunique()),
                    "n_seeds": int(hdf["split_seed"].nunique()),
                    "direction": METRIC_DIRECTIONS[str(metric)],
                }
            )

        for label, col in {
            "metamodel_minus_best_fixed": "improvement_vs_best_single",
            "metamodel_minus_oracle": None,
        }.items():
            ddf = hdf.copy()
            if col is None:
                if metric == "rmse":
                    ddf["_diff"] = ddf["oracle_metric"] - ddf["achieved_metric"]
                else:
                    ddf["_diff"] = ddf["achieved_metric"] - ddf["oracle_metric"]
            else:
                ddf["_diff"] = ddf[col]
            vals = ddf["_diff"].dropna().to_numpy(dtype=float)
            ci_low, ci_high, se = _normal_ci(vals)
            boot_low, boot_high = _row_bootstrap_ci(vals, rng, n_boot)
            cl_low, cl_high = _hierarchical_cluster_bootstrap_ci(ddf, "_diff", rng, n_boot)
            t_p = stats.ttest_1samp(vals, 0.0, nan_policy="omit").pvalue if vals.size > 1 else np.nan
            nonzero = vals[np.abs(vals) > 1e-15]
            w_p = stats.wilcoxon(nonzero).pvalue if nonzero.size else np.nan
            pos = int(np.sum(vals > 1e-15))
            neg = int(np.sum(vals < -1e-15))
            s_p = stats.binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) else np.nan
            diff_rows.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "comparison": label,
                    "diff_mean": float(np.mean(vals)) if vals.size else np.nan,
                    "diff_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                    "diff_se": se,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                    "clustered_bootstrap_ci_low": cl_low,
                    "clustered_bootstrap_ci_high": cl_high,
                    "p_ttest": float(t_p),
                    "p_wilcoxon": float(w_p),
                    "p_sign": float(s_p),
                    "n_obs": int(len(ddf)),
                    "n_instruments": int(ddf["series_id"].nunique()),
                    "n_seeds": int(ddf["split_seed"].nunique()),
                }
            )
    ci_df = pd.DataFrame(metric_rows)
    diff_df = pd.DataFrame(diff_rows)
    ci_df.to_csv(out_dir / "ci_metrics_by_horizon.csv", index=False)
    diff_df.to_csv(out_dir / "ci_differences_by_horizon.csv", index=False)
    return ci_df, diff_df


def rmse_mae_comparison(paths: dict[str, Path], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_parquet(paths["forecast_metrics"])
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    metrics["family"] = metrics["model_name"].map(model_family)

    comp = (
        metrics.groupby(["horizon", "model_name", "family"], sort=True)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            da_mean=("directional_accuracy", "mean"),
            n_tasks=("rmse", "size"),
        )
        .reset_index()
    )
    comp["rmse_rank"] = comp.groupby("horizon")["rmse_mean"].rank(method="min", ascending=True)
    comp["mae_rank"] = comp.groupby("horizon")["mae_mean"].rank(method="min", ascending=True)
    comp["rmse_mae_rank_delta"] = comp["mae_rank"] - comp["rmse_rank"]

    winner_rows = []
    for metric in ["rmse", "mae"]:
        idx = metrics.groupby(["horizon", "series_id", "fold_id"], sort=False)[metric].idxmin()
        winners = metrics.loc[idx, ["horizon", "series_id", "fold_id", "model_name"]].copy()
        dist = winners.groupby(["horizon", "model_name"], sort=True).size().reset_index(name=f"winner_count_{metric}")
        total = winners.groupby("horizon", sort=True).size().rename(f"winner_total_{metric}").reset_index()
        dist = dist.merge(total, on="horizon")
        dist[f"winner_share_{metric}"] = dist[f"winner_count_{metric}"] / dist[f"winner_total_{metric}"]
        winner_rows.append(dist)
    winners_df = winner_rows[0].merge(winner_rows[1], on=["horizon", "model_name"], how="outer").fillna(0)

    corr_rows = []
    for horizon, hdf in metrics.groupby("horizon", sort=True):
        corr_rows.append(
            {
                "horizon": int(horizon),
                "pearson_rmse_mae": float(hdf[["rmse", "mae"]].corr(method="pearson").iloc[0, 1]),
                "spearman_rmse_mae": float(hdf[["rmse", "mae"]].corr(method="spearman").iloc[0, 1]),
                "n_model_tasks": int(len(hdf)),
                "best_fixed_rmse": str(comp[comp["horizon"] == horizon].sort_values("rmse_mean").iloc[0]["model_name"]),
                "best_fixed_mae": str(comp[comp["horizon"] == horizon].sort_values("mae_mean").iloc[0]["model_name"]),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    comp = comp.merge(corr_df, on="horizon", how="left")

    comp.to_csv(out_dir / "rmse_mae_model_comparison.csv", index=False)
    winners_df.to_csv(out_dir / "winner_distribution_rmse_vs_mae.csv", index=False)
    corr_df.to_csv(out_dir / "rmse_mae_correlation_by_horizon.csv", index=False)
    return comp, winners_df, corr_df


def model_family(model_name: str) -> str:
    name = str(model_name)
    if name in BASELINE_MODELS:
        return "baseline"
    if name in CHAOS_MODELS:
        return "chaos_inspired"
    if name in {"vanilla_mlp", "lstm_forecast", "esn"}:
        return "neural_or_reservoir"
    if name == "ridge_lag":
        return "linear"
    return "other"


def prediction_outlier_sensitivity(paths: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    pred_cols = ["model_name", "series_id", "horizon", "fold_id", "y_true", "y_pred", "status"]
    preds = pd.read_parquet(paths["predictions"], columns=pred_cols)
    preds = preds[preds["status"].astype(str) == "success"].copy()
    preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")
    preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors="coerce")
    preds = preds.dropna(subset=["y_true", "y_pred"])
    preds["abs_y_true"] = preds["y_true"].abs()
    err = preds["y_pred"] - preds["y_true"]
    preds["sq_err"] = err * err

    rows: list[dict[str, Any]] = []
    scenarios = [("original", None, None)]
    for basis in ["abs_y_true", "sq_err"]:
        for pct in [0.01, 0.05]:
            scenarios.append((f"trim_top_{int(pct * 100)}pct_by_{basis}", basis, pct))
            scenarios.append((f"winsorize_top_{int(pct * 100)}pct_by_{basis}", basis, -pct))

    for horizon, hdf in preds.groupby("horizon", sort=True):
        thresholds = {
            (basis, pct): float(hdf[basis].quantile(1 - pct))
            for basis in ["abs_y_true", "sq_err"]
            for pct in [0.01, 0.05]
        }
        total_sq = float(hdf["sq_err"].sum())
        for basis in ["abs_y_true", "sq_err"]:
            for pct in [0.01, 0.05]:
                th = thresholds[(basis, pct)]
                mask = hdf[basis] >= th
                rows.append(
                    {
                        "horizon": int(horizon),
                        "scenario": f"contribution_top_{int(pct * 100)}pct_by_{basis}",
                        "model_name": "__all_models__",
                        "family": "all",
                        "rmse": float(np.sqrt(hdf.loc[mask, "sq_err"].mean())) if mask.any() else np.nan,
                        "n_predictions": int(mask.sum()),
                        "share_squared_error": float(hdf.loc[mask, "sq_err"].sum() / total_sq) if total_sq > 0 else np.nan,
                        "winner_rank": np.nan,
                    }
                )

        for scenario, basis, pct in scenarios:
            sdf = hdf
            sq_col = "sq_err"
            if basis is not None and pct is not None:
                th = thresholds[(basis, abs(pct))]
                if pct > 0:
                    sdf = hdf[hdf[basis] < th].copy()
                else:
                    sdf = hdf.copy()
                    if basis == "sq_err":
                        sdf[sq_col] = np.minimum(sdf[sq_col], th)
                    else:
                        cap_mask = sdf["abs_y_true"] > th
                        # Recompute squared error after capping only the observed return magnitude.
                        capped_true = np.sign(sdf.loc[cap_mask, "y_true"]) * th
                        capped_err = sdf.loc[cap_mask, "y_pred"] - capped_true
                        sdf.loc[cap_mask, sq_col] = capped_err * capped_err
            agg = (
                sdf.groupby("model_name", sort=True)
                .agg(rmse=(sq_col, lambda x: float(np.sqrt(np.mean(x)))), n_predictions=(sq_col, "size"))
                .reset_index()
            )
            agg["winner_rank"] = agg["rmse"].rank(method="min", ascending=True)
            for r in agg.itertuples(index=False):
                rows.append(
                    {
                        "horizon": int(horizon),
                        "scenario": scenario,
                        "model_name": str(r.model_name),
                        "family": model_family(str(r.model_name)),
                        "rmse": float(r.rmse),
                        "n_predictions": int(r.n_predictions),
                        "share_squared_error": np.nan,
                        "winner_rank": float(r.winner_rank),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "outlier_sensitivity_rmse.csv", index=False)
    return out


def overfitting_diagnostics(paths: dict[str, Path], out_dir: Path) -> tuple[pd.DataFrame, str]:
    metrics = pd.read_parquet(paths["forecast_metrics"])
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    metrics["family"] = metrics["model_name"].map(model_family)
    test_var = (
        metrics.groupby(["model_name", "family", "horizon"], sort=True)
        .agg(
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_da_mean=("directional_accuracy", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            n_tasks=("rmse", "size"),
        )
        .reset_index()
    )
    test_var["rmse_cv_across_tasks"] = test_var["test_rmse_std"] / test_var["test_rmse_mean"].replace(0, np.nan)

    diag_path = paths["neural_diag"]
    train_rows: list[dict[str, Any]] = []
    if diag_path.exists():
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
        for task in payload.get("tasks", []):
            train_loss = _to_float(task.get("final_train_loss"))
            val_loss = _to_float(task.get("final_val_loss"))
            train_rows.append(
                {
                    "model_name": task.get("model_name", ""),
                    "series_id": task.get("series_id", ""),
                    "horizon": task.get("horizon", np.nan),
                    "fold_id": task.get("fold_id", np.nan),
                    "diagnostic_source": str(diag_path),
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "validation_train_gap": val_loss - train_loss if np.isfinite(train_loss) and np.isfinite(val_loss) else np.nan,
                    "actual_epochs_used": task.get("actual_epochs_used", np.nan),
                    "early_stopping_reason": task.get("early_stopping_reason", ""),
                }
            )
    train_df = pd.DataFrame(train_rows)
    out = test_var.merge(
        train_df[["model_name", "final_train_loss", "final_val_loss", "validation_train_gap", "actual_epochs_used", "early_stopping_reason"]],
        on="model_name",
        how="left",
    )
    out.to_csv(out_dir / "overfitting_diagnostics.csv", index=False)

    family_summary = (
        test_var.groupby("family", sort=True)
        .agg(
            rmse_mean=("test_rmse_mean", "mean"),
            rmse_cv_mean=("rmse_cv_across_tasks", "mean"),
            da_mean=("test_da_mean", "mean"),
            n_models=("model_name", "nunique"),
        )
        .reset_index()
    )
    lines = [
        "# Overfitting diagnostics summary",
        "",
        "Full train/validation/test metric artifacts for every forecasting task were not found. "
        "The direct train-test gap can therefore not be computed without rerunning the forecasting models.",
        "",
        "Available evidence used here:",
        "- test-fold metrics from `forecasting_benchmark_v2/metrics_long.parquet`;",
        "- one lightweight neural diagnostic from `neural_training_diagnostic.json`, containing final train and validation losses for one series/horizon/fold;",
        "- neural training parameter audit from `neural_training_params.csv`.",
        "",
        "Family-level test summary:",
        _markdown_table(family_summary),
        "",
        "Interpretation: the saved artifacts support only indirect overfitting diagnostics across all tasks. "
        "Chaos-inspired models can be compared by test performance dispersion, but systematic train-test overfitting cannot be proven from the saved benchmark outputs alone.",
    ]
    text = "\n".join(lines)
    (out_dir / "overfitting_diagnostics_summary.md").write_text(text, encoding="utf-8")
    return out, text


def _psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 5 or y.size < 5:
        return np.nan
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.nan
    x_counts, _ = np.histogram(x, bins=edges)
    y_counts, _ = np.histogram(y, bins=edges)
    eps = 1e-6
    xp = x_counts / max(x_counts.sum(), 1) + eps
    yp = y_counts / max(y_counts.sum(), 1) + eps
    return float(np.sum((yp - xp) * np.log(yp / xp)))


def feature_stability(paths: dict[str, Path], out_dir: Path) -> tuple[pd.DataFrame, str]:
    feat = pd.read_parquet(paths["features"])
    catalog = pd.read_parquet(paths["series_catalog"], columns=["series_id", "market"]) if paths["series_catalog"].exists() else pd.DataFrame()
    if "market" not in feat.columns and not catalog.empty:
        feat = feat.merge(catalog.drop_duplicates("series_id"), on="series_id", how="left")
    id_cols = {"series_id", "horizon", "fold_id", "train_start", "train_end", "n_train", "feature_status", "feature_warning_flags", "market"}
    feature_cols = [c for c in feat.columns if c not in id_cols and pd.api.types.is_numeric_dtype(feat[c])]
    rows: list[dict[str, Any]] = []
    scopes = [("all", feat)]
    if "market" in feat.columns:
        scopes.extend((str(m), g.copy()) for m, g in feat.groupby("market", sort=True))

    for scope, sdf in scopes:
        for horizon, hdf in sdf.groupby("horizon", sort=True):
            for feature in feature_cols:
                wide = hdf.pivot_table(index="series_id", columns="fold_id", values=feature, aggfunc="mean")
                if not {1, 2, 3}.issubset(set(wide.columns)):
                    continue
                vals = wide[[1, 2, 3]].dropna()
                if vals.empty:
                    continue
                pair_corrs = []
                rank_corrs = []
                psi_vals = []
                for a, b in [(1, 2), (2, 3), (1, 3)]:
                    pair_corrs.append(vals[a].corr(vals[b], method="pearson"))
                    rank_corrs.append(vals[a].corr(vals[b], method="spearman"))
                    psi_vals.append(_psi(vals[a], vals[b]))
                abs_change = pd.concat(
                    [(vals[2] - vals[1]).abs(), (vals[3] - vals[2]).abs(), (vals[3] - vals[1]).abs()],
                    ignore_index=True,
                )
                row_mean = vals.mean(axis=1).replace(0, np.nan)
                cv = vals.std(axis=1) / row_mean.abs()
                rows.append(
                    {
                        "scope": scope,
                        "horizon": int(horizon),
                        "feature": feature,
                        "pearson_corr_mean": float(np.nanmean(pair_corrs)),
                        "spearman_corr_mean": float(np.nanmean(rank_corrs)),
                        "mean_abs_change": float(abs_change.mean()),
                        "median_abs_change": float(abs_change.median()),
                        "coefficient_of_variation_mean": float(cv.replace([np.inf, -np.inf], np.nan).mean()),
                        "psi_mean": float(np.nanmean(psi_vals)),
                        "n_series": int(len(vals)),
                    }
                )
    out = pd.DataFrame(rows)
    out["stability_score"] = out["spearman_corr_mean"] - out["coefficient_of_variation_mean"].clip(upper=10) * 0.05 - out["psi_mean"].fillna(0).clip(upper=10) * 0.05
    out.to_csv(out_dir / "feature_stability_by_fold.csv", index=False)

    all_scope = out[out["scope"] == "all"].copy()
    summary_rows = (
        all_scope.groupby("feature", sort=True)
        .agg(
            spearman_corr_mean=("spearman_corr_mean", "mean"),
            mean_abs_change=("mean_abs_change", "mean"),
            cv_mean=("coefficient_of_variation_mean", "mean"),
            psi_mean=("psi_mean", "mean"),
            stability_score=("stability_score", "mean"),
        )
        .reset_index()
        .sort_values("stability_score", ascending=False)
    )
    high = summary_rows.head(8)
    low = summary_rows.tail(8).sort_values("stability_score")
    text = "\n".join(
        [
            "# Feature stability summary",
            "",
            "Feature stability is evaluated across rolling-origin folds using Pearson/Spearman correlation, absolute changes, coefficient of variation, and PSI.",
            "",
            "Most stable features:",
            _markdown_table(high),
            "",
            "Least stable features:",
            _markdown_table(low),
        ]
    )
    (out_dir / "feature_stability_summary.md").write_text(text, encoding="utf-8")
    return out, text


def write_report(
    out_dir: Path,
    da: pd.DataFrame,
    ci_metrics: pd.DataFrame,
    ci_diff: pd.DataFrame,
    rmse_mae: pd.DataFrame,
    winners: pd.DataFrame,
    outliers: pd.DataFrame,
    overfit_text: str,
    feature_text: str,
    paths: dict[str, Path],
) -> None:
    da_show = da.copy()
    ci_da = ci_metrics[ci_metrics["metric"].astype(str) == "directional_accuracy"].copy()
    ci_rmse = ci_metrics[ci_metrics["metric"].astype(str) == "rmse"].copy()
    best_fixed_compare = (
        rmse_mae.groupby("horizon", sort=True)
        .apply(lambda g: pd.Series({
            "best_rmse_model": g.sort_values("rmse_mean").iloc[0]["model_name"],
            "best_mae_model": g.sort_values("mae_mean").iloc[0]["model_name"],
            "rmse_mae_pearson": g.iloc[0]["pearson_rmse_mae"],
            "rmse_mae_spearman": g.iloc[0]["spearman_rmse_mae"],
        }), include_groups=False)
        .reset_index()
    )
    original_winners = outliers[(outliers["scenario"] == "original") & (outliers["winner_rank"] == 1)].copy()
    contribution = outliers[outliers["scenario"].astype(str).str.startswith("contribution_top_")].copy()
    scenario_winners = outliers[
        (outliers["winner_rank"] == 1)
        & (outliers["scenario"].astype(str).isin(
            [
                "original",
                "trim_top_1pct_by_sq_err",
                "trim_top_5pct_by_sq_err",
                "winsorize_top_1pct_by_sq_err",
                "winsorize_top_5pct_by_sq_err",
            ]
        ))
    ].copy()

    lines = [
        "# Additional statistical analysis for reviewer questions",
        "",
        "## 1. Data and artifacts used",
        "",
        "Used existing artifacts only; no forecasting or meta-learning models were retrained.",
        "",
        "| artifact | purpose |",
        "|---|---|",
        f"| `{paths['routing'].relative_to(ROOT)}` | route-level paired metamodel/best-fixed/oracle rows |",
        f"| `{paths['best_config'].relative_to(ROOT)}` | selected best meta-learning configuration per horizon and metric |",
        f"| `{paths['forecast_metrics'].relative_to(ROOT)}` | forecasting RMSE, MAE and DA by model/series/fold/horizon |",
        f"| `{paths['predictions'].relative_to(ROOT)}` | forecast-level y_true/y_pred for outlier sensitivity |",
        f"| `{paths['features'].relative_to(ROOT)}` | fold-aware train-only feature matrix |",
        f"| `{paths['neural_diag'].relative_to(ROOT)}` | limited train/validation neural diagnostic |",
        "",
        "Full train/validation/test metrics for every forecasting task were not found; the overfitting section therefore separates direct evidence from indirect diagnostics.",
        "",
        "## 2. Directional Accuracy: metamodel vs best fixed",
        "",
        _markdown_table(da_show),
        "",
        "Interpretation: the point estimates are positive for all horizons. Classical paired tests on all route rows are optimistic because folds and repeated split seeds are dependent; the clustered bootstrap intervals are the more conservative reference.",
        "For h=1 and h=5, the clustered bootstrap CI for the metamodel minus best-fixed DA difference remains positive. For h=20, the clustered bootstrap CI includes zero, so the gain is better described as empirically positive but not robustly proven under clustered resampling.",
        "",
        "## 3. Confidence intervals for RMSE and DA",
        "",
        "Directional Accuracy CIs:",
        _markdown_table(ci_da),
        "",
        "RMSE CIs:",
        _markdown_table(ci_rmse),
        "",
        "Difference tests:",
        _markdown_table(ci_diff),
        "",
        "## 4. RMSE, MAE and outlier sensitivity",
        "",
        "Best fixed comparison by RMSE and MAE:",
        _markdown_table(best_fixed_compare),
        "",
        "The best-fixed comparison above uses the unweighted task-level means stored in `metrics_long.parquet`, matching the benchmark/meta-learning aggregation unit. Under both RMSE and MAE, `naive_zero` remains the best fixed model for h=1, h=5 and h=20.",
        "",
        "Original forecast-level RMSE winners after recomputing from predictions:",
        _markdown_table(original_winners[["horizon", "model_name", "family", "rmse", "n_predictions"]]),
        "",
        "The forecast-level table is weighted by the number of prediction rows, so it can differ slightly from the unweighted fold/task-level benchmark means.",
        "",
        "Extreme-observation contribution to squared error:",
        _markdown_table(contribution[["horizon", "scenario", "rmse", "n_predictions", "share_squared_error"]]),
        "",
        "RMSE winners under trimming/winsorization by squared error:",
        _markdown_table(scenario_winners[["horizon", "scenario", "model_name", "family", "rmse", "n_predictions"]]),
        "",
        "Winner distributions for RMSE vs MAE are saved in `winner_distribution_rmse_vs_mae.csv`; outlier trimming/winsorization results are saved in `outlier_sensitivity_rmse.csv`.",
        "",
        "## 5. Overfitting diagnostics",
        "",
        overfit_text.replace("# Overfitting diagnostics summary\n\n", ""),
        "",
        "## 6. Feature stability over time",
        "",
        feature_text.replace("# Feature stability summary\n\n", ""),
        "",
        "## 7. Short thesis-defense bullets",
        "",
        "- RMSE был выбран как стандартная метрика ошибки величины прогноза; дополнительно проверена MAE на тех же forecast-level артефактах.",
        "- Улучшение Directional Accuracy у метамодели положительно на всех горизонтах, но для вывода о статистической значимости нужно опираться на кластерные интервалы, а не только на обычные paired tests.",
        "- Сравнения metamodel, best fixed и oracle построены парно на одних и тех же test meta-observations.",
        "- Rolling-origin folds и held-out instruments уменьшают риск переобучения метамодели; split seeds явно сохранены и использованы в bootstrap.",
        "- Прямой train-test gap для всех forecasting candidates не сохранён, поэтому переобучение сложных моделей диагностировано ограниченно по test-дисперсии и имеющемуся neural audit.",
        "- Chaos-inspired модели не показывают устойчивого доминирования по RMSE/MAE, но могут выигрывать на отдельных инструментах и горизонтах.",
        "- Fold-aware train-only признаки позволяют проверить стабильность характеристик временных рядов между rolling-origin folds.",
        "- Основное ограничение: часть статистических тестов всё равно чувствительна к зависимости folds внутри одного инструмента.",
        "",
    ]
    (out_dir / "review_statistical_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Additional statistical analysis for reviewer questions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--skip-predictions", action="store_true", help="Skip forecast-level outlier analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "routing": ROOT / "artifacts" / "meta_modeling" / "routing_rows_v2.parquet",
        "best_config": ROOT / "artifacts" / "meta_modeling" / "best_config_per_task_v2.csv",
        "forecast_metrics": ROOT / "artifacts" / "forecasting" / "forecasting_benchmark_v2" / "metrics_long.parquet",
        "predictions": ROOT / "artifacts" / "forecasting" / "forecasting_benchmark_v2" / "predictions.parquet",
        "features": ROOT / "artifacts" / "features" / "fold_aware_features_v2" / "final_train_only_features_by_fold.parquet",
        "series_catalog": ROOT / "artifacts" / "processed" / "series_catalog_v1.parquet",
        "neural_diag": ROOT / "artifacts" / "reports" / "forecasting_audit_v2" / "neural_training_diagnostic.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    best_cfg, best_routing = load_best_routing(paths)
    best_cfg.to_csv(out_dir / "selected_best_configs_used.csv", index=False)
    best_routing.to_csv(out_dir / "paired_best_config_routing_rows.csv", index=False)

    da = directional_accuracy_tests(best_routing, out_dir, args.n_boot, args.seed)
    ci_metrics, ci_diff = metric_confidence_intervals(best_routing, out_dir, args.n_boot, args.seed)
    rmse_mae, winners, _ = rmse_mae_comparison(paths, out_dir)
    if args.skip_predictions:
        outliers = pd.DataFrame(
            [{"scenario": "not_run", "notes": "forecast-level analysis skipped by --skip-predictions"}]
        )
        outliers.to_csv(out_dir / "outlier_sensitivity_rmse.csv", index=False)
    else:
        outliers = prediction_outlier_sensitivity(paths, out_dir)
    _, overfit_text = overfitting_diagnostics(paths, out_dir)
    _, feature_text = feature_stability(paths, out_dir)
    write_report(out_dir, da, ci_metrics, ci_diff, rmse_mae, winners, outliers, overfit_text, feature_text, paths)

    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "inputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "outputs_dir": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote review statistical analysis to {out_dir}")


if __name__ == "__main__":
    main()
