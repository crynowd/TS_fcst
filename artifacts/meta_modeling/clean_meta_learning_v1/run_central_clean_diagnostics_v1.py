"""Central diagnostics for the frozen clean meta-learning test decisions only.

This script intentionally reads only routing_rows_v2.parquet produced by the
validation-selected, frozen-test clean experiment.  It does not fit models,
change labels, or access forecasting/features inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
ROUTING_PATH = ROOT / "routing_rows_v2.parquet"
BOOTSTRAP_DRAWS = 10_000
RANDOM_SEED = 20260819
EPS = 1e-12


def _gain(frame: pd.DataFrame) -> pd.Series:
    """Positive means selected routing improves on the train-defined fixed model."""
    return np.where(
        frame["target_metric"].eq("rmse"),
        frame["baseline_metric"] - frame["achieved_metric"],
        frame["achieved_metric"] - frame["baseline_metric"],
    )


def _cluster_bootstrap(gains: pd.DataFrame, rng: np.random.Generator) -> dict[str, float | int]:
    grouped = gains.groupby("series_id", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(grouped)
    sample_idx = rng.integers(0, n_clusters, size=(BOOTSTRAP_DRAWS, n_clusters))
    sampled_sums = sums[sample_idx].sum(axis=1)
    sampled_counts = counts[sample_idx].sum(axis=1)
    draws = sampled_sums / sampled_counts
    return {
        "n_series_clusters": int(n_clusters),
        "bootstrap_draws": int(BOOTSTRAP_DRAWS),
        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_draw_share_gain_gt_zero": float(np.mean(draws > 0)),
        "bootstrap_draw_share_gain_ge_zero": float(np.mean(draws >= 0)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
    }


def _cluster_tests(gains: pd.DataFrame, rng: np.random.Generator) -> dict[str, float | int | str]:
    # One series-level mean retains all of a series' folds/repeat appearances,
    # avoiding the invalid treatment of those rows as independent observations.
    cluster_means = gains.groupby("series_id", sort=True)["gain"].mean().to_numpy(dtype=float)
    observed = float(np.mean(cluster_means))
    nonzero = cluster_means[np.abs(cluster_means) > EPS]
    result: dict[str, float | int | str] = {
        "cluster_mean_gain": observed,
        "test_n_series_clusters": int(len(cluster_means)),
        "test_n_nonzero_series_gains": int(len(nonzero)),
    }
    if len(nonzero) == 0:
        result.update({"wilcoxon_p_one_sided_greater": np.nan, "signflip_p_one_sided_greater": np.nan})
        return result

    try:
        result["wilcoxon_p_one_sided_greater"] = float(
            wilcoxon(nonzero, alternative="greater", zero_method="wilcox", method="auto").pvalue
        )
    except ValueError:
        result["wilcoxon_p_one_sided_greater"] = np.nan

    signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOTSTRAP_DRAWS, len(cluster_means)))
    null_draws = (signs * cluster_means).mean(axis=1)
    result["signflip_p_one_sided_greater"] = float((1 + np.sum(null_draws >= observed)) / (BOOTSTRAP_DRAWS + 1))
    return result


def main() -> None:
    routing = pd.read_parquet(ROUTING_PATH).copy()
    required = {
        "evaluation_partition", "selected_by_validation", "series_id", "repeat_id", "horizon", "target_metric",
        "achieved_metric", "baseline_metric", "oracle_metric", "selected_model", "best_single_model",
        "gap_to_oracle", "best_score_tie_flag", "first_second_score_margin",
    }
    missing = sorted(required - set(routing.columns))
    if missing:
        raise ValueError(f"Frozen routing rows are missing required columns: {missing}")
    if set(routing["evaluation_partition"].astype(str)) != {"test"}:
        raise ValueError("Diagnostics must use frozen test rows only")
    if not routing["selected_by_validation"].eq(1).all():
        raise ValueError("Diagnostics found rows not selected by validation")

    routing["gain"] = _gain(routing).astype(float)
    routing["selector_deviation_from_fixed"] = routing["selected_model"].astype(str).ne(routing["best_single_model"].astype(str))
    routing["fixed_to_oracle_gap"] = np.where(
        routing["target_metric"].eq("rmse"),
        routing["baseline_metric"] - routing["oracle_metric"],
        routing["oracle_metric"] - routing["baseline_metric"],
    )
    routing["selected_to_oracle_gap"] = routing["gap_to_oracle"].astype(float)
    group_cols = ["horizon", "target_metric"]
    rng = np.random.default_rng(RANDOM_SEED)

    paired_rows: list[dict[str, object]] = []
    repeat_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []

    for (horizon, metric), g in routing.groupby(group_cols, sort=True):
        bootstrap = _cluster_bootstrap(g[["series_id", "gain"]], rng)
        tests = _cluster_tests(g[["series_id", "gain"]], rng)
        paired_rows.append(
            {
                "horizon": int(horizon),
                "target_metric": metric,
                "n_test_decision_rows": int(len(g)),
                "mean_paired_gain": float(g["gain"].mean()),
                "median_paired_gain": float(g["gain"].median()),
                **bootstrap,
                **tests,
            }
        )

        for repeat_id, rg in g.groupby("repeat_id", sort=True):
            repeat_rows.append(
                {
                    "horizon": int(horizon),
                    "target_metric": metric,
                    "repeat_id": int(repeat_id),
                    "split_seed": int(rg["split_seed"].iloc[0]),
                    "n_test_decision_rows": int(len(rg)),
                    "gain": float(rg["gain"].mean()),
                }
            )

        fixed_gap = float(g["fixed_to_oracle_gap"].mean())
        selected_gap = float(g["selected_to_oracle_gap"].mean())
        mean_gain = float(g["gain"].mean())
        selector_rows.append(
            {
                "horizon": int(horizon),
                "target_metric": metric,
                "n_test_decision_rows": int(len(g)),
                "selector_deviation_from_fixed_rate": float(g["selector_deviation_from_fixed"].mean()),
                "mean_regret_relative_to_oracle": selected_gap,
                "mean_fixed_to_oracle_gap": fixed_gap,
                "mean_selected_to_oracle_gap": selected_gap,
                "mean_gain": mean_gain,
                "oracle_gap_closed_ratio": float(mean_gain / fixed_gap) if abs(fixed_gap) > EPS else np.nan,
                "oracle_gap_closed_ratio_interpretable": bool(abs(fixed_gap) > EPS),
                "gain_when_selector_stayed_fixed": float(g.loc[~g["selector_deviation_from_fixed"], "gain"].mean()),
                "gain_when_selector_deviated": float(g.loc[g["selector_deviation_from_fixed"], "gain"].mean()),
                "n_selector_stayed_fixed": int((~g["selector_deviation_from_fixed"]).sum()),
                "n_selector_deviated": int(g["selector_deviation_from_fixed"].sum()),
            }
        )

        distribution = (
            g.groupby("selected_model", sort=True)
            .size()
            .rename("n_test_decision_rows")
            .reset_index()
        )
        distribution["horizon"] = int(horizon)
        distribution["target_metric"] = metric
        distribution["share_test_decision_rows"] = distribution["n_test_decision_rows"] / len(g)
        model_rows.extend(distribution[["horizon", "target_metric", "selected_model", "n_test_decision_rows", "share_test_decision_rows"]].to_dict("records"))

        if metric == "directional_accuracy":
            da = g.copy()
            da["tie_group"] = np.where(da["best_score_tie_flag"].astype(int).eq(1), "best_score_tie", "unique_best")
            for tie_group, tg in da.groupby("tie_group", sort=True):
                tie_rows.append(
                    {
                        "diagnostic": "tie_status",
                        "horizon": int(horizon),
                        "target_metric": metric,
                        "group": tie_group,
                        "n_test_decision_rows": int(len(tg)),
                        "selected_vs_fixed_gain": float(tg["gain"].mean()),
                        "selector_deviation_from_fixed_rate": float(tg["selector_deviation_from_fixed"].mean()),
                    }
                )
            zero = da["first_second_score_margin"].abs() <= EPS
            nonzero = da.loc[~zero, "first_second_score_margin"].astype(float)
            cutoff = float(nonzero.quantile(0.5)) if len(nonzero) else np.nan
            da["margin_group"] = np.where(
                zero,
                "exact_tie",
                np.where(da["first_second_score_margin"] <= cutoff, "low_nonzero_margin", "higher_nonzero_margin"),
            )
            for margin_group, mg in da.groupby("margin_group", sort=False):
                tie_rows.append(
                    {
                        "diagnostic": "margin_group",
                        "horizon": int(horizon),
                        "target_metric": metric,
                        "group": margin_group,
                        "n_test_decision_rows": int(len(mg)),
                        "selected_vs_fixed_gain": float(mg["gain"].mean()),
                        "selector_deviation_from_fixed_rate": float(mg["selector_deviation_from_fixed"].mean()),
                        "nonzero_margin_median_cutoff": cutoff,
                    }
                )

    paired_df = pd.DataFrame(paired_rows)
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_summary = repeat_df.groupby(group_cols, sort=True)["gain"].agg(
        positive_repeats=lambda x: int((x > 0).sum()),
        n_repeats="size",
        gain_min="min",
        gain_max="max",
        gain_mean="mean",
        gain_sd="std",
    ).reset_index()
    repeat_df = repeat_df.merge(repeat_summary, on=group_cols, how="left", validate="many_to_one")
    selector_df = pd.DataFrame(selector_rows)
    model_df = pd.DataFrame(model_rows)
    tie_df = pd.DataFrame(tie_rows)

    paired_df.to_csv(ROOT / "paired_uncertainty_clustered_by_series_v1.csv", index=False)
    repeat_df.to_csv(ROOT / "repeat_stability_v1.csv", index=False)
    selector_df.to_csv(ROOT / "minimal_selector_diagnostics_v1.csv", index=False)
    model_df.to_csv(ROOT / "selected_model_distribution_v1.csv", index=False)
    tie_df.to_csv(ROOT / "da_tie_margin_diagnostics_v1.csv", index=False)
    with (ROOT / "central_clean_diagnostics_metadata_v1.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(ROUTING_PATH.resolve()),
                "source_constraints": "frozen validation-selected test decisions only",
                "unit_of_analysis": "one frozen test decision row (series_id, fold_id, repeat_id, horizon, metric)",
                "dependence_handling": "primary percentile bootstrap resamples series_id clusters with replacement and retains every row/fold/repeat appearance of each sampled series",
                "secondary_tests": "one series-level average gain per cluster; one-sided Wilcoxon and Monte Carlo sign-flip tests",
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "random_seed": RANDOM_SEED,
                "gain_definition": {"directional_accuracy": "selected - fixed", "rmse": "fixed - selected"},
                "tie_scope": "winner/margin fields stored in frozen routing rows for each validation-selected candidate set",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
