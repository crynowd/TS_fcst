"""Build the current ICDM paper tables from final clean artifacts only.

This script never trains forecasting models or metamodels. It only aggregates
completed clean/leakage-safe outputs into paper-facing CSV files.

Run from the repository root:

    python paper_icdm/scripts/build_paper_tables.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "paper_icdm" / "tables"

CLEAN_BENCHMARK_DIR = REPO_ROOT / "artifacts/forecasting/forecasting_benchmark_v2_clean_batched"
CLEAN_FEATURE_DIR = REPO_ROOT / "artifacts/features/fold_aware_features_v2_clean_batched"
CLEAN_META_DIR = REPO_ROOT / "artifacts/meta_modeling/clean_meta_learning_v1"

FEATURE_LIST = CLEAN_META_DIR / "feature_list_v2.csv"
FEATURE_MATRIX = CLEAN_FEATURE_DIR / "final_train_only_features_by_fold.parquet"
FORECAST_CONFIGS = tuple(
    REPO_ROOT / f"configs/forecasting_benchmark_v2_clean_batched_batch_{batch:02d}.yaml"
    for batch in range(1, 5)
)
FORECAST_CONFIG = FORECAST_CONFIGS[0]
ARCH_CONFIG = REPO_ROOT / "configs/forecasting_selected_architectures_v1.yaml"
META_CONFIG = REPO_ROOT / "configs/meta_modeling_clean_v1.yaml"
SPLIT_METADATA = CLEAN_BENCHMARK_DIR / "split_metadata.parquet"
TASK_AUDIT = CLEAN_BENCHMARK_DIR / "task_audit.parquet"
METRICS_LONG = CLEAN_BENCHMARK_DIR / "metrics_long.parquet"
SPLIT_ASSIGNMENTS = CLEAN_META_DIR / "split_assignments_v2.csv"
META_DECISIONS = CLEAN_META_DIR / "selector_decisions_by_repeat_v1.csv"
META_SELECTED_TEST = CLEAN_META_DIR / "selected_test_results_v2.csv"
META_BEST_CONFIG = CLEAN_META_DIR / "best_config_per_task_v2.csv"
CLUSTER_BOOTSTRAP = CLEAN_META_DIR / "paired_uncertainty_clustered_by_series_v1.csv"
FAMILY_MAPPING = REPO_ROOT / "paper_icdm/model_family_mapping.csv"

HORIZONS = [1, 5, 20]
METRICS = ["rmse", "directional_accuracy"]
EXPECTED_SERIES = 418
EXPECTED_TASKS = EXPECTED_SERIES * len(HORIZONS)

# Presentation order for Table II.
MODEL_ORDERED = [
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
]

# Predeclared class/candidate order used by the clean meta-learning path and by
# the rmse__/da__ columns in clean_fold_aware_meta_inputs.parquet. Do not replace
# this with a runtime sort by model_name: exact DA ties must follow this order.
WINNER_TIE_BREAK_ORDER = [
    "chaotic_esn",
    "chaotic_logistic_net",
    "chaotic_lstm_forecast",
    "chaotic_mlp",
    "esn",
    "lstm_forecast",
    "naive_mean",
    "naive_zero",
    "ridge_lag",
    "transient_chaotic_esn",
    "vanilla_mlp",
]

FEATURE_ROWS = [
    ("Long memory", "Rescaled-range Hurst exponent", "hurst_rs"),
    ("Long memory", "DFA Hurst exponent", "hurst_dfa"),
    ("Linear dependence", "Return autocorrelation at lag 2", "acf_lag_2"),
    ("Linear dependence", "Return autocorrelation at lag 5", "acf_lag_5"),
    ("Linear dependence", "Return autocorrelation at lag 10", "acf_lag_10"),
    ("Linear dependence", "Return autocorrelation at lag 25", "acf_lag_25"),
    ("Linear dependence", "Return autocorrelation at lag 50", "acf_lag_50"),
    ("Linear dependence", "Return autocorrelation at lag 100", "acf_lag_100"),
    ("Linear dependence", "Lo-MacKinlay variance ratio with q = 10", "vr_q10"),
    ("Linear dependence", "Ljung-Box statistic up to lag 50", "lb_ret_stat_50"),
    ("Volatility dependence", "Absolute-return autocorrelation at lag 2", "abs_acf_lag_2"),
    ("Volatility dependence", "Absolute-return autocorrelation at lag 5", "abs_acf_lag_5"),
    ("Volatility dependence", "Absolute-return autocorrelation at lag 10", "abs_acf_lag_10"),
    ("Volatility dependence", "Absolute-return autocorrelation at lag 25", "abs_acf_lag_25"),
    ("Volatility dependence", "Absolute-return autocorrelation at lag 50", "abs_acf_lag_50"),
    ("Complexity and spectrum", "Normalized Lempel-Ziv complexity", "lz_complexity"),
    ("Complexity and spectrum", "Normalized permutation entropy", "permutation_entropy"),
    ("Complexity and spectrum", "Welch spectral flatness", "spectral_flatness"),
    ("Distribution and tails", "Fisher excess kurtosis", "kurtosis"),
    ("Distribution and tails", "Moors robust kurtosis", "robust_kurtosis"),
    ("Distribution and tails", "Upper-tail quantile ratio", "tail_ratio_upper"),
    ("Distribution and tails", "Hill tail index", "hill_tail_index"),
    ("Phase-space structure", "Average-mutual-information delay", "selected_delay_tau"),
    ("Phase-space structure", "False-nearest-neighbor embedding dimension", "embedding_dimension"),
    ("Phase-space structure", "Grassberger-Procaccia correlation dimension", "correlation_dimension"),
]


class TableBuilder:
    def __init__(self) -> None:
        self.created: list[Path] = []
        self.sources: dict[str, list[str]] = {}

    @staticmethod
    def rel(path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix()

    def require(self, table: str, sources: list[Path]) -> None:
        missing = [self.rel(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{table}: required clean source(s) missing: {missing}")

    def write(self, table: str, frame: pd.DataFrame, name: str, sources: list[Path]) -> None:
        self.require(table, sources)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        path = TABLE_DIR / name
        frame.to_csv(path, index=False)
        self.created.append(path)
        self.sources[table] = [self.rel(source) for source in sources]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def jsonish(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def display_scale(metric: str) -> tuple[float, str, str]:
    if metric == "rmse":
        return 100.0, "percentage log-return points", "lower_is_better; gain=fixed-selected"
    return 100.0, "percentage points", "higher_is_better; gain=selected-fixed"


def build_table_i(builder: TableBuilder) -> None:
    sources = [FEATURE_LIST, FEATURE_MATRIX]
    builder.require("Table I", sources)
    listed = set(pd.read_csv(FEATURE_LIST)["feature_name"].astype(str))
    matrix_columns = set(pd.read_parquet(FEATURE_MATRIX).columns)
    rows = []
    for group, label, column in FEATURE_ROWS:
        if column not in listed or column not in matrix_columns:
            raise ValueError(f"Table I: expected feature missing from clean artifacts: {column}")
        rows.append({"feature_group": group, "feature_name": label, "internal_feature_column": column})
    if listed != {row[2] for row in FEATURE_ROWS}:
        raise ValueError("Table I: clean feature list does not equal the declared 25-feature paper set")
    builder.write("Table I", pd.DataFrame(rows), "table_i_features.csv", sources)


def build_table_ii(builder: TableBuilder) -> None:
    sources = [ARCH_CONFIG, *FORECAST_CONFIGS, FAMILY_MAPPING]
    builder.require("Table II", sources)
    arch = load_yaml(ARCH_CONFIG)
    forecast = load_yaml(FORECAST_CONFIG)
    family = pd.read_csv(FAMILY_MAPPING)
    display_by_model = dict(zip(family["model_name"], family["paper_display_name"]))
    selected_by_model = {row["model_name"]: row for row in arch.get("selected_models", [])}
    training = forecast.get("training", {})
    baselines = {
        "naive_zero": {},
        "naive_mean": {},
        "ridge_lag": forecast.get("model_overrides", {}).get("ridge_lag", {}),
    }
    rows = []
    for model_name in MODEL_ORDERED:
        selected = selected_by_model.get(model_name)
        if model_name not in baselines and selected is None:
            raise ValueError(f"Table II: {model_name} absent from final selected-architecture config")
        model_params = baselines[model_name] if model_name in baselines else selected.get("model_params", {})
        if model_name in baselines:
            train_params: dict[str, Any] = {}
            family_name = "baseline"
            role = "baseline"
            candidate_id = ""
            source = builder.rel(FORECAST_CONFIG)
        else:
            train_params = training.get("by_model", {}).get(model_name, {})
            family_name = str(selected.get("family", ""))
            role = str(selected.get("selection_role", ""))
            candidate_id = str(selected.get("candidate_id", ""))
            source = builder.rel(ARCH_CONFIG)
        rows.append(
            {
                "model_name": model_name,
                "paper_display_name": display_by_model.get(model_name, model_name),
                "candidate_id": candidate_id,
                "family": family_name,
                "selection_role": role,
                "model_params": jsonish(model_params),
                "training_params": jsonish(train_params),
                "source": source,
            }
        )
    builder.write("Table II", pd.DataFrame(rows), "table_ii_candidates.csv", sources)


def tuning_overlap_with_benchmark(split_metadata: pd.DataFrame) -> tuple[int, int]:
    tuning_files = sorted((REPO_ROOT / "artifacts/architecture_tuning").glob("*_v1/selected_series_*.csv"))
    if not tuning_files:
        raise FileNotFoundError("Table III: architecture-tuning selected-series files are missing")
    tuning_sets = [set(pd.read_csv(path)["series_id"].astype(str)) for path in tuning_files]
    if any(values != tuning_sets[0] for values in tuning_sets[1:]):
        raise ValueError("Table III: architecture-tuning families do not use the same instrument set")
    benchmark_tickers = set(split_metadata["ticker"].astype(str))
    return len(tuning_sets[0]), len(tuning_sets[0] & benchmark_tickers)


def build_table_iii(builder: TableBuilder) -> None:
    tuning_sources = sorted((REPO_ROOT / "artifacts/architecture_tuning").glob("*_v1/selected_series_*.csv"))
    sources = [
        *FORECAST_CONFIGS,
        META_CONFIG,
        SPLIT_METADATA,
        TASK_AUDIT,
        SPLIT_ASSIGNMENTS,
        META_BEST_CONFIG,
        CLUSTER_BOOTSTRAP,
        *tuning_sources,
    ]
    builder.require("Table III", sources)
    forecast = load_yaml(FORECAST_CONFIG)
    meta = load_yaml(META_CONFIG)
    splits = pd.read_parquet(SPLIT_METADATA)
    audit = pd.read_parquet(TASK_AUDIT)
    assignments = pd.read_csv(SPLIT_ASSIGNMENTS)
    bootstrap = pd.read_csv(CLUSTER_BOOTSTRAP)
    tuning_n, overlap_n = tuning_overlap_with_benchmark(splits)

    split_counts = (
        assignments.groupby(["repeat_id", "horizon", "target_metric", "split"])["object_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    expected_split_counts = {"train": 876, "validation": 126, "test": 252}
    if any((split_counts[name] != count).any() for name, count in expected_split_counts.items()):
        raise ValueError("Table III: clean instrument split counts are inconsistent")

    rows = [
        ("Markets", jsonish(splits.groupby("market")["series_id"].nunique().sort_index().to_dict()), SPLIT_METADATA),
        ("Benchmark instruments", int(splits["series_id"].nunique()), SPLIT_METADATA),
        ("Series representation", forecast.get("dataset_version", ""), FORECAST_CONFIG),
        ("Forecast horizons", jsonish(forecast.get("horizons", [])), FORECAST_CONFIG),
        ("Input windows", jsonish(forecast.get("window_sizes", {})), FORECAST_CONFIG),
        ("Rolling-origin folds", int(forecast.get("validation", {}).get("n_folds", 0)), FORECAST_CONFIG),
        ("Target-window separation", str(splits["split_policy"].dropna().unique().tolist()), SPLIT_METADATA),
        ("Boundary purge by horizon", "h=1:0; h=5:4; h=20:19 samples from outer-train and fit boundaries", SPLIT_METADATA),
        ("Successful forecasting tasks", int((audit["status"].astype(str) == "success").sum()), TASK_AUDIT),
        ("Fold-aware meta-observations", int(splits[["series_id", "horizon", "fold_id"]].drop_duplicates().shape[0]), SPLIT_METADATA),
        ("Meta split unit", "instrument; all three folds remain together", SPLIT_ASSIGNMENTS),
        ("Meta split counts per repeat/horizon/metric", jsonish(expected_split_counts), SPLIT_ASSIGNMENTS),
        ("Repeated evaluation", f"{int(meta.get('n_repeats', 0))} instrument-level repeats", META_CONFIG),
        ("Configuration protocol", "meta-train fit -> validation selection per repeat x horizon x metric -> one frozen test evaluation", META_CONFIG),
        ("Test use in selection", "none", META_BEST_CONFIG),
        ("Candidate sets", jsonish(meta.get("candidate_selection", {}).get("top_k_values", [])), META_CONFIG),
        ("Metamodels", jsonish(meta.get("classification_models", [])), META_CONFIG),
        ("Architecture tuning panel", f"{tuning_n} US ETFs; overlap with benchmark={overlap_n}", tuning_sources[0]),
        ("Cluster bootstrap", f"{int(bootstrap['bootstrap_draws'].min())} draws; clustered by series_id", CLUSTER_BOOTSTRAP),
    ]
    frame = pd.DataFrame(
        {"protocol_component": component, "value": value, "source": builder.rel(source)}
        for component, value, source in rows
    )
    builder.write("Table III", frame, "table_iii_protocol.csv", sources)


def build_table_iv(builder: TableBuilder) -> None:
    sources = [METRICS_LONG, FAMILY_MAPPING]
    builder.require("Table IV", sources)
    metrics = pd.read_parquet(METRICS_LONG)
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    names = pd.read_csv(FAMILY_MAPPING)
    display_by_model = dict(zip(names["model_name"], names["paper_display_name"]))
    grouped = metrics.groupby(["model_name", "horizon"], sort=False).agg(
        rmse=("rmse", "mean"), directional_accuracy=("directional_accuracy", "mean")
    )
    rows = []
    for model_name in MODEL_ORDERED:
        row: dict[str, Any] = {"model_name": model_name, "paper_display_name": display_by_model[model_name]}
        for horizon in HORIZONS:
            values = grouped.loc[(model_name, horizon)]
            row[f"RMSE h={horizon}"] = float(values["rmse"]) * 100.0
            row[f"RMSE raw h={horizon}"] = float(values["rmse"])
            row[f"DA h={horizon}"] = float(values["directional_accuracy"]) * 100.0
        row["rmse_scale"] = "percentage log-return points"
        row["da_scale"] = "percent"
        rows.append(row)
    builder.write("Table IV", pd.DataFrame(rows), "table_iv_direct_forecasting.csv", sources)


def winner_tables(metrics: pd.DataFrame, family: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_by_model = dict(zip(family["model_name"], family["family"]))
    rank = {model: idx for idx, model in enumerate(WINNER_TIE_BREAK_ORDER)}
    observed = set(metrics["model_name"].astype(str).unique())
    if observed != set(WINNER_TIE_BREAK_ORDER):
        raise ValueError(f"Table V: candidate-order mismatch: observed={sorted(observed)}")
    series_level = metrics.groupby(["series_id", "horizon", "model_name"], sort=False).agg(
        rmse=("rmse", "mean"), directional_accuracy=("directional_accuracy", "mean")
    ).reset_index()
    series_level["candidate_order"] = series_level["model_name"].map(rank)
    winners: list[dict[str, Any]] = []
    for (series_id, horizon), group in series_level.groupby(["series_id", "horizon"], sort=False):
        rmse_row = group.sort_values(["rmse", "candidate_order"], ascending=[True, True], kind="stable").iloc[0]
        da_row = group.sort_values(
            ["directional_accuracy", "candidate_order"], ascending=[False, True], kind="stable"
        ).iloc[0]
        winners.extend(
            [
                {"series_id": series_id, "horizon": int(horizon), "metric": "rmse", "model_name": rmse_row["model_name"]},
                {"series_id": series_id, "horizon": int(horizon), "metric": "directional_accuracy", "model_name": da_row["model_name"]},
            ]
        )
    winners_df = pd.DataFrame(winners)
    winners_df["family"] = winners_df["model_name"].map(family_by_model)
    if winners_df["family"].isna().any():
        raise ValueError("Table V: family mapping is incomplete")

    family_order = ["Zero/mean baselines", "Non-chaotic models", "Chaos-inspired models"]
    aggregate_rows = []
    horizon_rows = []
    tie_note = "mean across 3 folds; exact ties use predefined clean candidate order"
    for family_name in family_order:
        rmse_n = int(((winners_df["metric"] == "rmse") & (winners_df["family"] == family_name)).sum())
        da_n = int(((winners_df["metric"] == "directional_accuracy") & (winners_df["family"] == family_name)).sum())
        aggregate_rows.append(
            {
                "Model family": family_name,
                "RMSE wins": rmse_n,
                "DA wins": da_n,
                "RMSE share": rmse_n / EXPECTED_TASKS,
                "DA share": da_n / EXPECTED_TASKS,
                "task_count": EXPECTED_TASKS,
                "aggregation": tie_note,
                "tie_break_candidate_order": json.dumps(WINNER_TIE_BREAK_ORDER),
            }
        )
    for metric in METRICS:
        for horizon in HORIZONS:
            subset = winners_df[(winners_df["metric"] == metric) & (winners_df["horizon"] == horizon)]
            if len(subset) != EXPECTED_SERIES:
                raise ValueError(f"Table V: {metric}, h={horizon} has {len(subset)} winners")
            for family_name in family_order:
                count = int((subset["family"] == family_name).sum())
                horizon_rows.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "Model family": family_name,
                        "winner_count": count,
                        "winner_share": count / EXPECTED_SERIES,
                        "task_count": EXPECTED_SERIES,
                        "aggregation": tie_note,
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(horizon_rows)


def build_table_v(builder: TableBuilder) -> None:
    sources = [METRICS_LONG, FAMILY_MAPPING]
    builder.require("Table V", sources)
    metrics = pd.read_parquet(METRICS_LONG)
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    aggregate, by_horizon = winner_tables(metrics, pd.read_csv(FAMILY_MAPPING))
    builder.write("Table V", aggregate, "table_v_winner_family_counts.csv", sources)
    builder.write("Table V support", by_horizon, "table_v_winner_family_counts_by_horizon.csv", sources)


def build_table_vi(builder: TableBuilder) -> None:
    sources = [META_DECISIONS, META_SELECTED_TEST, META_BEST_CONFIG]
    builder.require("Table VI", sources)
    decisions = pd.read_csv(META_DECISIONS)
    selected_test = pd.read_csv(META_SELECTED_TEST)
    best_config = pd.read_csv(META_BEST_CONFIG)

    key_columns = ["repeat_id", "horizon", "target_metric"]
    if len(decisions) != 30 or len(selected_test) != 30 or len(best_config) != 30:
        raise ValueError("Table VI: expected 30 repeat x horizon x metric rows in each clean selection artifact")
    if set(map(tuple, decisions[key_columns].to_numpy())) != set(map(tuple, selected_test[key_columns].to_numpy())):
        raise ValueError("Table VI: selector decisions and frozen tests have different task keys")
    if best_config["test_used_for_selection"].astype(bool).any():
        raise ValueError("Table VI: clean selection artifact reports test use during configuration selection")
    if not (selected_test["test_evaluations_for_task"] == 1).all():
        raise ValueError("Table VI: frozen test was not evaluated exactly once for every selected task")

    rows = []
    for horizon in HORIZONS:
        for metric in ["directional_accuracy", "rmse"]:
            group = decisions[(decisions["horizon"] == horizon) & (decisions["target_metric"] == metric)].copy()
            if set(group["repeat_id"].astype(int)) != {1, 2, 3, 4, 5}:
                raise ValueError(f"Table VI: incomplete repeats for h={horizon}, metric={metric}")
            fixed = pd.to_numeric(group["fixed_test_score"], errors="raise")
            selected = pd.to_numeric(group["selected_test_score"], errors="raise")
            oracle = pd.to_numeric(group["oracle_test_score"], errors="raise")
            gain = selected - fixed if metric == "directional_accuracy" else fixed - selected
            multiplier, unit, gain_definition = display_scale(metric)
            fixed_models = sorted(group["fixed_model"].astype(str).unique())
            rows.append(
                {
                    "h": horizon,
                    "Metric": metric,
                    "Fixed model": ", ".join(fixed_models),
                    "Fixed": fixed.mean() * multiplier,
                    "Selected": selected.mean() * multiplier,
                    "Gain": gain.mean() * multiplier,
                    "Oracle": oracle.mean() * multiplier,
                    "fixed_raw_mean": fixed.mean(),
                    "fixed_raw_std": fixed.std(ddof=1),
                    "selected_raw_mean": selected.mean(),
                    "selected_raw_std": selected.std(ddof=1),
                    "gain_raw_mean": gain.mean(),
                    "gain_raw_std": gain.std(ddof=1),
                    "oracle_raw_mean": oracle.mean(),
                    "oracle_raw_std": oracle.std(ddof=1),
                    "display_unit": unit,
                    "gain_definition": gain_definition,
                    "n_repeats": group["repeat_id"].nunique(),
                    "n_test_rows": 1260,
                    "selection_scope": "repeat_x_horizon_x_metric",
                    "configuration_selection_partition": "validation",
                    "test_evaluations_per_selected_task": 1,
                    "test_used_for_selection": False,
                    "selected_config_ids": json.dumps(group["config_id"].astype(str).tolist()),
                }
            )
    builder.write("Table VI", pd.DataFrame(rows), "table_vi_meta_selection_results.csv", sources)


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        print(f"[ERROR] Run from repository root: {REPO_ROOT}", file=sys.stderr)
        return 2
    builder = TableBuilder()
    try:
        build_table_i(builder)
        build_table_ii(builder)
        build_table_iii(builder)
        build_table_iv(builder)
        build_table_v(builder)
        build_table_vi(builder)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("Created files:")
    for path in builder.created:
        print(f"- {builder.rel(path)}")
    print("Source files used:")
    for table, sources in builder.sources.items():
        print(f"- {table}: {', '.join(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
