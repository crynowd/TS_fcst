"""Cheap consistency checks for the current ICDM paper artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_icdm"
BENCH = ROOT / "artifacts/forecasting/forecasting_benchmark_v2_clean_batched"
FEATURES = ROOT / "artifacts/features/fold_aware_features_v2_clean_batched"
META = ROOT / "artifacts/meta_modeling/clean_meta_learning_v1"
ARCH = ROOT / "configs/forecasting_selected_architectures_v1.yaml"
ABLATION = ROOT / "artifacts/meta_modeling/clean_meta_learning_feature_ablation_v1_repaired"

MODELS = {
    "naive_zero",
    "naive_mean",
    "ridge_lag",
    "esn",
    "chaotic_esn",
    "transient_chaotic_esn",
    "vanilla_mlp",
    "chaotic_mlp",
    "chaotic_logistic_net",
    "lstm_forecast",
    "chaotic_lstm_forecast",
}
HORIZONS = {1, 5, 20}
FOLDS = {1, 2, 3}
REPEATS = {1, 2, 3, 4, 5}


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
            print(f"[OK] {message}")
        else:
            self.failed += 1
            print(f"[FAIL] {message}")

    def equal(self, actual: object, expected: object, message: str) -> None:
        self.check(actual == expected, f"{message}: actual={actual!r}, expected={expected!r}")

    def close(self, actual: float, expected: float, tolerance: float, message: str) -> None:
        self.check(abs(float(actual) - expected) <= tolerance, f"{message}: actual={float(actual):.10f}, expected={expected:.10f}")


def require_files(checks: Checks) -> bool:
    required = [
        BENCH / "metrics_long.parquet",
        BENCH / "predictions.parquet",
        BENCH / "split_metadata.parquet",
        BENCH / "task_audit.parquet",
        FEATURES / "final_train_only_features_by_fold.parquet",
        META / "feature_list_v2.csv",
        META / "split_assignments_v2.csv",
        META / "best_config_per_task_v2.csv",
        META / "selected_test_results_v2.csv",
        META / "selector_decisions_by_repeat_v1.csv",
        META / "routing_rows_v2.parquet",
        META / "paired_uncertainty_clustered_by_series_v1.csv",
        ARCH,
        ABLATION / "feature_ablation_summary.csv",
        ABLATION / "repair_manifest.json",
        PAPER / "model_family_mapping.csv",
    ]
    for path in required:
        checks.check(path.is_file(), f"required source exists: {path.relative_to(ROOT).as_posix()}")
    return all(path.is_file() for path in required)


def check_benchmark(checks: Checks) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_parquet(BENCH / "metrics_long.parquet")
    audit = pd.read_parquet(BENCH / "task_audit.parquet")
    split = pd.read_parquet(BENCH / "split_metadata.parquet")
    checks.equal(len(metrics), 41382, "clean metrics row count")
    checks.equal(len(audit), 41382, "clean task-audit row count")
    checks.equal(int((metrics["status"].astype(str) == "success").sum()), 41382, "successful metric tasks")
    checks.equal(int((audit["status"].astype(str) == "success").sum()), 41382, "successful audited tasks")
    checks.equal(set(metrics["model_name"].astype(str)), MODELS, "clean benchmark model set")
    checks.equal(set(metrics["horizon"].astype(int)), HORIZONS, "clean benchmark horizons")
    checks.equal(set(metrics["fold_id"].astype(int)), FOLDS, "clean benchmark folds")
    checks.equal(len(split), 3762, "clean split-metadata row count")
    checks.equal(set(split["split_policy"].astype(str)), {"target_end_lte_right_origin_v1"}, "target-window policy")
    expected_removed = {1: 0, 5: 4, 20: 19}
    for horizon, removed in expected_removed.items():
        subset = split[split["horizon"] == horizon]
        checks.equal(set(subset["outer_train_n_samples_removed"].astype(int)), {removed}, f"h={horizon} outer-train purge")
        checks.equal(set(subset["fit_n_samples_removed"].astype(int)), {removed}, f"h={horizon} fit purge")
    violations = int(
        (split["outer_train_last_target_end_idx"] > split["test_first_forecast_origin_idx"]).sum()
        + (split["fit_last_target_end_idx"] > split["validation_first_forecast_origin_idx"]).sum()
        + (split["validation_last_target_end_idx"] > split["test_first_forecast_origin_idx"]).sum()
    )
    checks.equal(violations, 0, "target-window boundary violations")

    batch_paths = sorted(ROOT.glob("configs/forecasting_benchmark_v2_clean_batched_batch_*.yaml"))
    batch_configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in batch_paths]
    checks.equal(len(batch_configs), 4, "clean benchmark batch manifests")
    protocol_signatures = {
        json.dumps(
            {
                "dataset_version": config.get("dataset_version"),
                "horizons": config.get("horizons"),
                "window_sizes": config.get("window_sizes"),
                "validation": config.get("validation"),
                "selected_architectures": config.get("selected_architectures"),
                "training": config.get("training"),
            },
            sort_keys=True,
        )
        for config in batch_configs
    }
    checks.equal(len(protocol_signatures), 1, "shared protocol across four clean batches")
    batch_series = [set(config["filters"]["series_ids"]) for config in batch_configs]
    checks.equal(sum(len(values) for values in batch_series), 418, "sum of clean batch instrument counts")
    checks.equal(len(set().union(*batch_series)), 418, "clean batch instrument union")
    return metrics, split


def check_features(checks: Checks) -> None:
    feature_list = pd.read_csv(META / "feature_list_v2.csv")
    feature_matrix = pd.read_parquet(FEATURES / "final_train_only_features_by_fold.parquet")
    feature_names = feature_list["feature_name"].astype(str).tolist()
    checks.equal(len(feature_names), 25, "clean feature-list size")
    checks.equal(len(feature_matrix), 3762, "clean fold-aware feature rows")
    checks.equal(len(feature_matrix[["series_id", "horizon", "fold_id"]].drop_duplicates()), 3762, "unique fold-aware feature keys")
    checks.equal(sorted(set(feature_names) - set(feature_matrix.columns)), [], "feature-list columns present in clean matrix")
    finite = feature_matrix[feature_names].notna().all().all()
    checks.check(bool(finite), "all clean feature values are non-missing")


def check_architecture_tuning(checks: Checks, split: pd.DataFrame) -> None:
    with ARCH.open("r", encoding="utf-8") as handle:
        arch = yaml.safe_load(handle)
    selected = {row["model_name"]: row["model_params"] for row in arch["selected_models"]}
    checks.equal(selected["esn"]["n_reservoir"], 64, "final ESN units")
    checks.close(selected["esn"]["spectral_radius"], 0.9, 1e-12, "final ESN spectral radius")
    checks.equal(selected["chaotic_esn"]["n_reservoir"], 96, "final chaotic ESN units")
    checks.close(selected["chaotic_esn"]["spectral_radius"], 0.95, 1e-12, "final chaotic ESN base radius")
    checks.close(selected["chaotic_esn"]["chaotic_spectral_radius"], 1.25, 1e-12, "final chaotic ESN chaotic radius")
    checks.equal(selected["chaotic_lstm_forecast"]["hidden_size"], 64, "final chaotic LSTM units")
    checks.equal(selected["chaotic_lstm_forecast"]["num_layers"], 1, "final chaotic LSTM layers")
    checks.close(selected["chaotic_lstm_forecast"]["dropout"], 0.0, 1e-12, "final chaotic LSTM dropout")

    tuning_paths = sorted((ROOT / "artifacts/architecture_tuning").glob("*_v1/selected_series_*.csv"))
    tuning_sets = [set(pd.read_csv(path)["series_id"].astype(str)) for path in tuning_paths]
    checks.equal(len(tuning_paths), 4, "architecture-tuning family panels")
    checks.check(bool(tuning_sets) and all(values == tuning_sets[0] for values in tuning_sets), "all tuning families use the same panel")
    checks.equal(len(tuning_sets[0]), 12, "independent tuning instrument count")
    checks.equal(len(tuning_sets[0] & set(split["ticker"].astype(str))), 0, "tuning/benchmark instrument overlap")


def check_meta_protocol(checks: Checks) -> pd.DataFrame:
    assignments = pd.read_csv(META / "split_assignments_v2.csv")
    decisions = pd.read_csv(META / "selector_decisions_by_repeat_v1.csv")
    selected = pd.read_csv(META / "selected_test_results_v2.csv")
    best = pd.read_csv(META / "best_config_per_task_v2.csv")
    routing = pd.read_parquet(META / "routing_rows_v2.parquet")
    bootstrap = pd.read_csv(META / "paired_uncertainty_clustered_by_series_v1.csv")
    key = ["repeat_id", "horizon", "target_metric"]

    checks.equal(set(assignments["repeat_id"].astype(int)), REPEATS, "meta repeat set")
    leakage = assignments.groupby(key + ["series_id"])["split"].nunique().gt(1).sum()
    checks.equal(int(leakage), 0, "instrument-level split leakage")
    counts = assignments.groupby(key + ["split"])["object_id"].nunique().unstack(fill_value=0)
    checks.check(bool((counts["train"] == 876).all()), "meta-train has 876 fold objects per task")
    checks.check(bool((counts["validation"] == 126).all()), "validation has 126 fold objects per task")
    checks.check(bool((counts["test"] == 252).all()), "test has 252 fold objects per task")
    checks.equal(len(decisions), 30, "selector decision rows")
    checks.equal(len(selected), 30, "frozen selected-test rows")
    checks.equal(len(best), 30, "validation-selected configuration rows")
    checks.equal(set(map(tuple, decisions[key].to_numpy())), set(map(tuple, selected[key].to_numpy())), "decision/frozen-test keys")
    checks.check(not best["test_used_for_selection"].astype(bool).any(), "test is never used for configuration selection")
    checks.check(bool((best["evaluation_partition"].astype(str) == "validation").all()), "all configurations are selected on validation")
    checks.check(bool((selected["test_evaluations_for_task"] == 1).all()), "one frozen test per selected task")
    checks.equal(len(routing), 7560, "frozen routing row count")
    figure_rows = routing[
        (routing["horizon"] == 5)
        & (routing["target_metric"].astype(str) == "directional_accuracy")
        & (routing["evaluation_partition"].astype(str) == "test")
        & (routing["selected_by_validation"].astype(int) == 1)
    ]
    checks.equal(len(figure_rows), 1260, "Figure 2 pooled frozen-test N")
    checks.equal(len(bootstrap), 6, "cluster-bootstrap task summaries")
    checks.equal(set(bootstrap["bootstrap_draws"].astype(int)), {10000}, "cluster-bootstrap draws")
    return decisions


def check_ablation(checks: Checks) -> None:
    summary = pd.read_csv(ABLATION / "feature_ablation_summary.csv")
    manifest = json.loads((ABLATION / "repair_manifest.json").read_text(encoding="utf-8"))
    checks.equal(set(summary["feature_set"].astype(str)), {"full_25", "standard", "without_phase_space", "nonlinear_only"}, "feature-family ablation sets")
    checks.equal(len(summary), 24, "feature-family ablation summary rows")
    checks.equal(manifest.get("validation_rows"), 14400, "complete ablation validation grid")
    checks.equal(manifest.get("frozen_tests_recomputed"), 120, "recomputed ablation frozen tests")


def check_tables(checks: Checks, metrics: pd.DataFrame, decisions: pd.DataFrame) -> None:
    table_names = [
        "table_i_features.csv",
        "table_ii_candidates.csv",
        "table_iii_protocol.csv",
        "table_iv_direct_forecasting.csv",
        "table_v_winner_family_counts.csv",
        "table_v_winner_family_counts_by_horizon.csv",
        "table_vi_meta_selection_results.csv",
    ]
    for name in table_names:
        checks.check((PAPER / "tables" / name).is_file(), f"generated paper table exists: {name}")
    checks.equal(list((PAPER / "tables").glob("*_NEEDS_SOURCE.csv")), [], "no stale NEEDS_SOURCE outputs")

    table_ii = pd.read_csv(PAPER / "tables/table_ii_candidates.csv")
    params = {row.model_name: json.loads(row.model_params) for row in table_ii.itertuples()}
    checks.equal(params["esn"]["n_reservoir"], 64, "Table II ESN units")
    checks.close(params["chaotic_esn"]["chaotic_spectral_radius"], 1.25, 1e-12, "Table II chaotic ESN radius")
    checks.equal(params["chaotic_lstm_forecast"]["num_layers"], 1, "Table II chaotic LSTM layers")
    checks.close(params["chaotic_lstm_forecast"]["dropout"], 0.0, 1e-12, "Table II chaotic LSTM dropout")

    table_iv = pd.read_csv(PAPER / "tables/table_iv_direct_forecasting.csv").set_index("model_name")
    direct = metrics.groupby(["model_name", "horizon"])[["rmse", "directional_accuracy"]].mean()
    for model in MODELS:
        for horizon in HORIZONS:
            checks.close(table_iv.loc[model, f"RMSE h={horizon}"], direct.loc[(model, horizon), "rmse"] * 100, 1e-10, f"Table IV {model} RMSE h={horizon}")
            checks.close(table_iv.loc[model, f"DA h={horizon}"], direct.loc[(model, horizon), "directional_accuracy"] * 100, 1e-10, f"Table IV {model} DA h={horizon}")

    table_v = pd.read_csv(PAPER / "tables/table_v_winner_family_counts.csv").set_index("Model family")
    expected_v = {
        "Zero/mean baselines": (856, 177),
        "Non-chaotic models": (215, 420),
        "Chaos-inspired models": (183, 657),
    }
    for family, (rmse_wins, da_wins) in expected_v.items():
        checks.equal(int(table_v.loc[family, "RMSE wins"]), rmse_wins, f"Table V {family} RMSE wins")
        checks.equal(int(table_v.loc[family, "DA wins"]), da_wins, f"Table V {family} DA wins")
    checks.check(table_v["aggregation"].str.contains("predefined clean candidate order", regex=False).all(), "Table V records predefined tie-break order")

    table_vi = pd.read_csv(PAPER / "tables/table_vi_meta_selection_results.csv").set_index(["h", "Metric"])
    expected_vi = {
        (1, "directional_accuracy"): (49.2571, 49.2024, -0.0547, 51.2840),
        (5, "directional_accuracy"): (50.3454, 50.6594, 0.3140, 53.0760),
        (20, "directional_accuracy"): (51.9146, 52.0401, 0.1255, 54.8731),
        (1, "rmse"): (2.7127294, 2.7127306, 0.0, 2.7043726),
        (5, "rmse"): (5.7902990, 5.7916141, -0.0013152, 5.7699774),
        (20, "rmse"): (11.1258041, 11.1302112, -0.0044071, 10.9999569),
    }
    for key_value, expected in expected_vi.items():
        row = table_vi.loc[key_value]
        for column, value in zip(["Fixed", "Selected", "Gain", "Oracle"], expected):
            tolerance = 5e-5 if key_value[1] == "directional_accuracy" else 5e-6
            checks.close(row[column], value, tolerance, f"Table VI {key_value} {column}")
    checks.check(bool((table_vi["configuration_selection_partition"] == "validation").all()), "Table VI records validation-only selection")
    checks.check(not table_vi["test_used_for_selection"].astype(bool).any(), "Table VI records no test selection")
    checks.check(bool((table_vi["test_evaluations_per_selected_task"] == 1).all()), "Table VI records one frozen test")


def check_figures_and_docs(checks: Checks) -> None:
    current = [
        PAPER / "figures/figure_2_pooled_confusion_h5_da.png",
        PAPER / "figures/figure_2_pooled_confusion_h5_da.pdf",
        PAPER / "figures/figure_2_pooled_confusion_h5_da_counts.csv",
    ]
    for path in current:
        checks.check(path.is_file() and path.stat().st_size > 0, f"current figure artifact exists: {path.name}")
    counts = pd.read_csv(current[2])
    checks.equal(int(counts["row_total"].sum()), 1260, "Figure 2 support-table total")
    checks.equal(set(counts["pooled_N"].astype(int)), {1260}, "Figure 2 support-table N annotation")

    stale = [
        "figure_2_rmse_fixed_meta_oracle",
        "figure_3_da_fixed_meta_oracle",
        "figure_4_winner_distribution",
    ]
    for stem in stale:
        checks.check(not (PAPER / "figures" / f"{stem}.png").exists() and not (PAPER / "figures" / f"{stem}.pdf").exists(), f"stale figure removed: {stem}")

    readme = (PAPER / "README.md").read_text(encoding="utf-8")
    required_phrases = [
        "target_end_lte_right_origin_v1",
        "12 U.S. ETFs",
        "Five repeated instrument-level partitions",
        "validation",
        "cluster bootstrap",
        "Market heterogeneity",
        "Feature-family ablation",
        "Price-scale-transition sensitivity",
        "Feature stability",
        "family-level sensitivity diagnostic",
        "subsequently retuned",
    ]
    for phrase in required_phrases:
        checks.check(phrase.lower() in readme.lower(), f"README documents: {phrase}")

    table_builder = (PAPER / "scripts/build_paper_tables.py").read_text(encoding="utf-8")
    figure_builder = (PAPER / "scripts/build_paper_figures.py").read_text(encoding="utf-8")
    checks.check("artifacts/forecasting/forecasting_benchmark_v2/" not in table_builder, "table builder has no legacy benchmark path")
    checks.check("meta_modeling_experiments_v2" not in table_builder, "table builder has no legacy meta-report path")
    checks.check("forecasting_benchmark_v2/metrics_long" not in figure_builder, "figure builder has no legacy benchmark path")


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        print(f"[FAIL] Run from repository root: {ROOT}", file=sys.stderr)
        return 2
    checks = Checks()
    if not require_files(checks):
        print(f"Summary: {checks.passed} passed, {checks.failed} failed")
        return 1
    try:
        metrics, split = check_benchmark(checks)
        check_features(checks)
        check_architecture_tuning(checks, split)
        decisions = check_meta_protocol(checks)
        check_ablation(checks)
        check_tables(checks, metrics, decisions)
        check_figures_and_docs(checks)
    except Exception as exc:
        checks.check(False, f"checker raised {type(exc).__name__}: {exc}")
    print(f"Summary: {checks.passed} passed, {checks.failed} failed")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
