"""Read-only ICDM paper artifact consistency checker.

Run from the repository root:

    python paper_icdm/scripts/check_artifacts.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_MODELS = {
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
EXPECTED_HORIZONS = {1, 5, 20}
EXPECTED_FOLDS = {1, 2, 3}
EXPECTED_REPEATS = {1, 2, 3, 4, 5}
EXPECTED_TOP_K = {3, 4, 5, 6}
EXPECTED_FEATURE_COUNT = 25
EXPECTED_SERIES = 418
EXPECTED_MARKETS = {"RU": 209, "US": 209}
EXPECTED_FAMILIES = {
    "Zero/mean baselines",
    "Non-chaotic models",
    "Chaos-inspired models",
}


REQUIRED_ARTIFACTS = [
    "artifacts/processed/log_returns_v1.parquet",
    "artifacts/processed/series_catalog_v1.parquet",
    "artifacts/processed/dataset_profiles_v1.parquet",
    "artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet",
    "artifacts/meta_modeling/feature_list_v2.csv",
    "artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet",
    "artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json",
    "artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet",
    "artifacts/meta_modeling/split_assignments_v2.csv",
    "artifacts/meta_modeling/task_results_v2.parquet",
    "artifacts/meta_modeling/model_order_mapping_v2.csv",
    "artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2.xlsx",
    "paper_icdm/model_family_mapping.csv",
]

OPTIONAL_ARTIFACTS = [
    "artifacts/meta_modeling/routing_rows_v2.parquet",
    "artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet",
    "artifacts/reports.zip",
]


@dataclass
class Check:
    status: str
    section: str
    message: str
    critical: bool = False


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.critical_missing: list[str] = []
        self.recommended_uploads: list[str] = []

    def add(self, status: str, section: str, message: str, critical: bool = False) -> None:
        self.checks.append(Check(status, section, message, critical))
        print(f"[{status}] {section}: {message}")

    def ok(self, section: str, message: str) -> None:
        self.add("OK", section, message)

    def warning(self, section: str, message: str) -> None:
        self.add("WARNING", section, message)

    def fail(self, section: str, message: str) -> None:
        self.add("FAIL", section, message, critical=True)

    def skipped(self, section: str, message: str) -> None:
        self.add("SKIPPED", section, message)

    def counts(self) -> dict[str, int]:
        statuses = {"OK": 0, "WARNING": 0, "FAIL": 0, "SKIPPED": 0}
        for check in self.checks:
            statuses[check.status] += 1
        return statuses

    def has_failures(self) -> bool:
        return any(check.status == "FAIL" for check in self.checks)


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def tracked_files() -> set[str]:
    output = run_git(["ls-files"])
    if output is None:
        return set()
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def status_porcelain() -> dict[str, str]:
    output = run_git(["status", "--porcelain"])
    if output is None:
        return {}
    statuses: dict[str, str] = {}
    for line in output.splitlines():
        if len(line) >= 4:
            statuses[line[3:].replace("\\", "/")] = line[:2]
    return statuses


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"could not read {rel(path)}: {exc}") from exc
    raise RuntimeError(f"unsupported table extension for {rel(path)}")


def require_file(reporter: Reporter, relative_path: str, section: str) -> bool:
    path = repo_path(relative_path)
    if not path.exists():
        reporter.critical_missing.append(relative_path)
        reporter.fail(section, f"{relative_path} missing")
        return False
    if not path.is_file():
        reporter.fail(section, f"{relative_path} exists but is not a file")
        return False
    reporter.ok(section, f"{relative_path} present ({file_size(path)})")
    return True


def require_columns(
    reporter: Reporter, section: str, df: pd.DataFrame, required: Iterable[str]
) -> bool:
    required_set = set(required)
    missing = sorted(required_set - set(df.columns))
    if missing:
        reporter.fail(
            section,
            f"missing columns {missing}; actual columns: {list(df.columns)}",
        )
        return False
    reporter.ok(section, f"required columns present: {sorted(required_set)}")
    return True


def values_as_set(series: pd.Series) -> set:
    return set(series.dropna().unique().tolist())


def candidate_columns(df: pd.DataFrame, fragments: Iterable[str]) -> list[str]:
    fragments_lower = [fragment.lower() for fragment in fragments]
    return [
        column
        for column in df.columns
        if any(fragment in column.lower() for fragment in fragments_lower)
    ]


def check_membership(
    reporter: Reporter,
    section: str,
    label: str,
    observed: set,
    expected: set,
    fail_on_missing: bool = True,
) -> bool:
    missing = expected - observed
    if missing:
        message = f"{label} missing {sorted(missing)}; observed {sorted(observed)}"
        if fail_on_missing:
            reporter.fail(section, message)
        else:
            reporter.warning(section, message)
        return False
    reporter.ok(section, f"{label} contains expected values {sorted(expected)}")
    return True


def check_processed_data(reporter: Reporter) -> None:
    section = "Processed data"
    profiles_path = repo_path("artifacts/processed/dataset_profiles_v1.parquet")
    returns_path = repo_path("artifacts/processed/log_returns_v1.parquet")
    if not require_file(reporter, str(rel(profiles_path)), section):
        return
    try:
        profiles = read_table(profiles_path)
    except RuntimeError as exc:
        reporter.fail(section, str(exc))
        profiles = None
    if profiles is not None and require_columns(
        reporter,
        section,
        profiles,
        ["series_id", "market", "dataset_profile", "selected_length"],
    ):
        core = profiles[profiles["dataset_profile"] == "core_balanced"]
        if core.empty:
            reporter.fail(section, "dataset_profile core_balanced not found")
        else:
            series_count = core["series_id"].nunique()
            market_counts = core.drop_duplicates("series_id")["market"].value_counts().to_dict()
            if series_count == EXPECTED_SERIES and all(
                market_counts.get(market, 0) == count
                for market, count in EXPECTED_MARKETS.items()
            ):
                reporter.ok(
                    section,
                    f"core_balanced has {series_count} series "
                    f"(RU={market_counts.get('RU', 0)}, US={market_counts.get('US', 0)})",
                )
            else:
                reporter.fail(
                    section,
                    f"core_balanced counts differ: series={series_count}, markets={market_counts}",
                )
            min_length = int(core["selected_length"].min())
            exact_2000 = int((core["selected_length"] == 2000).sum())
            shorter = int((core["selected_length"] < 2000).sum())
            if min_length >= 1500:
                reporter.ok(
                    "Length policy",
                    f"min selected_length={min_length}, target 2000, "
                    f"exact_2000={exact_2000}, shorter={shorter}",
                )
            else:
                reporter.fail(
                    "Length policy",
                    f"min selected_length={min_length}, expected at least 1500",
                )

    if not require_file(reporter, str(rel(returns_path)), section):
        return
    try:
        returns = read_table(returns_path)
    except RuntimeError as exc:
        reporter.fail(section, str(exc))
        return
    require_columns(
        reporter,
        section,
        returns,
        ["series_id", "ticker", "market", "date", "log_return", "dataset_profile"],
    )


def check_feature_artifacts(reporter: Reporter) -> None:
    section = "Feature artifacts"
    feature_list_path = repo_path("artifacts/meta_modeling/feature_list_v2.csv")
    matrix_path = repo_path(
        "artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet"
    )
    feature_names: list[str] = []
    if require_file(reporter, str(rel(feature_list_path)), section):
        try:
            feature_list = read_table(feature_list_path)
            name_col = "feature_name" if "feature_name" in feature_list.columns else feature_list.columns[0]
            feature_names = feature_list[name_col].dropna().astype(str).tolist()
            if len(feature_names) == EXPECTED_FEATURE_COUNT:
                reporter.ok(section, "feature_list_v2.csv contains 25 features")
            else:
                reporter.fail(section, f"feature list has {len(feature_names)} rows, expected 25")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))

    if not require_file(reporter, str(rel(matrix_path)), section):
        return
    try:
        matrix = read_table(matrix_path)
    except RuntimeError as exc:
        reporter.fail(section, str(exc))
        return
    require_columns(reporter, section, matrix, ["series_id", "horizon", "fold_id"])
    if len(matrix) == EXPECTED_SERIES * 3 * 3:
        reporter.ok(section, f"feature matrix has {len(matrix)} rows")
    else:
        reporter.warning(section, f"feature matrix has {len(matrix)} rows, expected 3762")
    check_membership(reporter, section, "horizons", values_as_set(matrix["horizon"]), EXPECTED_HORIZONS)
    check_membership(reporter, section, "folds", values_as_set(matrix["fold_id"]), EXPECTED_FOLDS)
    if feature_names:
        missing = sorted(set(feature_names) - set(matrix.columns))
        if missing:
            reporter.fail(section, f"feature columns missing from matrix: {missing}")
        else:
            reporter.ok(section, "all 25 feature_list_v2 features are present in matrix")
    else:
        key_cols = {"series_id", "horizon", "fold_id", "train_start", "train_end", "n_train"}
        feature_cols = [col for col in matrix.columns if col not in key_cols and not col.startswith("feature_")]
        if len(feature_cols) == EXPECTED_FEATURE_COUNT:
            reporter.ok(section, "matrix appears to contain 25 feature columns")
        else:
            reporter.warning(section, f"could not unambiguously identify 25 feature columns: {feature_cols}")


def check_forecasting_artifacts(reporter: Reporter) -> None:
    section = "Forecasting artifacts"
    metrics_path = repo_path("artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet")
    manifest_path = repo_path("artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json")
    split_path = repo_path("artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet")

    metrics = None
    if require_file(reporter, str(rel(metrics_path)), section):
        try:
            metrics = read_table(metrics_path)
        except RuntimeError as exc:
            reporter.fail(section, str(exc))
    if metrics is not None:
        require_columns(reporter, section, metrics, ["model_name", "horizon", "fold_id"])
        check_membership(reporter, section, "horizons", values_as_set(metrics["horizon"]), EXPECTED_HORIZONS)
        check_membership(reporter, section, "folds", values_as_set(metrics["fold_id"]), EXPECTED_FOLDS)
        check_membership(
            reporter,
            section,
            "model_name",
            set(metrics["model_name"].dropna().astype(str).unique()),
            EXPECTED_MODELS,
        )
        rmse_cols = candidate_columns(metrics, ["rmse"])
        da_cols = candidate_columns(metrics, ["directional_accuracy"])
        if rmse_cols and da_cols:
            reporter.ok(section, f"metric columns found: RMSE={rmse_cols}, directional accuracy={da_cols}")
        else:
            reporter.fail(section, f"metric columns incomplete: RMSE={rmse_cols}, directional accuracy={da_cols}")
        status_cols = candidate_columns(metrics, ["status", "success", "error", "failed"])
        if status_cols:
            failed = pd.Series(False, index=metrics.index)
            for col in status_cols:
                values = metrics[col]
                if values.dtype == bool:
                    failed = failed | (~values)
                else:
                    lowered = values.astype(str).str.lower()
                    failed = failed | lowered.isin({"fail", "failed", "error", "false", "0"})
            if failed.any():
                reporter.fail(section, f"explicit failed task rows found: {int(failed.sum())}")
            else:
                reporter.ok(section, f"no explicit failed tasks in status columns {status_cols}")
        else:
            reporter.skipped(section, "no explicit task status column found")
        expected_rows = EXPECTED_SERIES * 3 * 3 * len(EXPECTED_MODELS)
        if len(metrics) == expected_rows:
            reporter.ok(section, f"metrics_long has expected fold-level row count {expected_rows}")
        else:
            reporter.warning(section, f"metrics_long has {len(metrics)} rows, expected {expected_rows}")

    if require_file(reporter, str(rel(manifest_path)), section):
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if isinstance(manifest, dict):
                reporter.ok(section, f"run_manifest.json top-level keys: {sorted(manifest.keys())}")
            else:
                reporter.warning(section, f"run_manifest.json is {type(manifest).__name__}, expected object")
        except Exception as exc:
            reporter.fail(section, f"could not parse run_manifest.json: {exc}")

    if require_file(reporter, str(rel(split_path)), section):
        try:
            split = read_table(split_path)
            require_columns(reporter, section, split, ["fold_id"])
            check_membership(reporter, section, "split_metadata folds", values_as_set(split["fold_id"]), EXPECTED_FOLDS)
            window_cols = candidate_columns(split, ["train_start", "train_end", "test_start", "test_end"])
            if window_cols:
                reporter.ok(section, f"temporal window columns found: {window_cols}")
            else:
                reporter.warning(section, f"no temporal window columns found; actual columns: {list(split.columns)}")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))


def parse_top_k(value: object) -> int | None:
    text = str(value).lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def check_meta_artifacts(reporter: Reporter) -> None:
    section = "Meta-learning artifacts"
    split_path = repo_path("artifacts/meta_modeling/split_assignments_v2.csv")
    task_path = repo_path("artifacts/meta_modeling/task_results_v2.parquet")
    routing_path = repo_path("artifacts/meta_modeling/routing_rows_v2.parquet")
    mapping_path = repo_path("artifacts/meta_modeling/model_order_mapping_v2.csv")

    if require_file(reporter, str(rel(split_path)), section):
        try:
            splits = read_table(split_path)
            require_columns(reporter, section, splits, ["split"])
            repeat_col = "repeat_id" if "repeat_id" in splits.columns else None
            metric_col = "target_metric" if "target_metric" in splits.columns else None
            horizon_col = "horizon" if "horizon" in splits.columns else None
            if repeat_col:
                check_membership(reporter, section, "repeats", values_as_set(splits[repeat_col]), EXPECTED_REPEATS)
            if horizon_col:
                check_membership(reporter, section, "horizons", values_as_set(splits[horizon_col]), EXPECTED_HORIZONS)
            if metric_col:
                metrics = {str(value).lower() for value in splits[metric_col].dropna().unique()}
                if "rmse" in metrics and any("direction" in value for value in metrics):
                    reporter.ok(section, f"split assignment metrics include RMSE and directional accuracy: {sorted(metrics)}")
                else:
                    reporter.fail(section, f"split assignment metrics incomplete: {sorted(metrics)}")
            split_values = {str(value).lower() for value in splits["split"].dropna().unique()}
            check_membership(reporter, section, "split labels", split_values, {"train", "validation", "test"})
            group_cols = [col for col in [repeat_col, horizon_col, metric_col] if col]
            if group_cols:
                counts = splits.groupby(group_cols + ["split"]).size().unstack(fill_value=0)
                expected = {"train": 876, "validation": 126, "test": 252}
                mismatched = []
                for split_name, expected_count in expected.items():
                    if split_name in counts.columns:
                        bad = counts[counts[split_name] != expected_count]
                        if not bad.empty:
                            mismatched.append(split_name)
                    else:
                        mismatched.append(split_name)
                if mismatched:
                    reporter.warning(section, f"split counts differ for {mismatched}; observed sample: {counts.head().to_dict()}")
                else:
                    reporter.ok(section, "split counts match 876/126/252 per repeat/horizon/metric")
            instrument_col = next((col for col in ["series_id", "ticker", "instrument_id", "object_id"] if col in splits.columns), None)
            if instrument_col and group_cols:
                leakage = 0
                for _, group in splits.groupby(group_cols):
                    by_instrument = group.groupby(instrument_col)["split"].nunique()
                    leakage += int((by_instrument > 1).sum())
                if leakage:
                    reporter.fail(section, f"instrument split leakage candidates found: {leakage}")
                else:
                    reporter.ok(section, f"no instrument-level split leakage using {instrument_col}")
            else:
                reporter.skipped(section, "could not identify instrument key or grouping columns for leakage check")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))

    if require_file(reporter, str(rel(task_path)), section):
        try:
            tasks = read_table(task_path)
            metric_cols = candidate_columns(tasks, ["metric"])
            model_cols = candidate_columns(tasks, ["model", "classifier"])
            topk_cols = candidate_columns(tasks, ["candidate_set", "top"])
            reporter.ok(section, f"task_results candidate columns: metrics={metric_cols}, models={model_cols}, top-k={topk_cols}")
            metrics_joined = " ".join(str(value).lower() for col in metric_cols for value in tasks[col].dropna().unique())
            if "rmse" in metrics_joined and "direction" in metrics_joined:
                reporter.ok(section, "task_results contains RMSE and directional accuracy tasks")
            else:
                reporter.fail(section, f"task_results metrics incomplete in columns {metric_cols}")
            top_k_observed = {
                parsed
                for col in topk_cols
                for parsed in (parse_top_k(value) for value in tasks[col].dropna().unique())
                if parsed is not None
            }
            check_membership(reporter, section, "top-k values", top_k_observed, EXPECTED_TOP_K)
            model_values = " ".join(
                str(value).lower().replace("_", " ")
                for col in model_cols
                for value in tasks[col].dropna().unique()
            )
            required_classifiers = ["logistic regression", "random forest", "catboost"]
            missing_classifiers = [name for name in required_classifiers if name not in model_values]
            if missing_classifiers:
                reporter.fail(section, f"classifiers missing: {missing_classifiers}")
            else:
                reporter.ok(section, "classifiers include logistic regression, random forest, and CatBoost")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))

    if routing_path.exists():
        reporter.warning(
            section,
            f"{rel(routing_path)} present ({file_size(routing_path)}); large optional route-level source",
        )
        try:
            routing = read_table(routing_path)
            relevant = candidate_columns(routing, ["selected", "fixed", "best", "oracle", "metric", "model", "score"])
            reporter.ok(section, f"routing_rows relevant columns: {relevant}")
            required_fragments = ["selected", "best", "oracle"]
            missing = [fragment for fragment in required_fragments if not candidate_columns(routing, [fragment])]
            if missing:
                reporter.warning(section, f"routing_rows may not contain route reconstruction fields for {missing}")
            else:
                reporter.ok(section, "routing_rows contains selected/best/oracle route fields")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))
    else:
        reporter.warning(
            section,
            "routing_rows_v2.parquet absent; optional external source for detailed route reconstruction",
        )

    if require_file(reporter, str(rel(mapping_path)), section):
        try:
            mapping = read_table(mapping_path)
            model_col = next((col for col in ["model_name", "model", "candidate_model"] if col in mapping.columns), None)
            if model_col:
                observed = set(mapping[model_col].dropna().astype(str).unique())
                check_membership(reporter, section, "model_order_mapping models", observed, EXPECTED_MODELS, fail_on_missing=False)
                reporter.ok(section, f"model_order_mapping unique model/candidate count: {len(observed)}")
            else:
                reporter.warning(section, f"no model-name candidate column found; actual columns: {list(mapping.columns)}")
        except RuntimeError as exc:
            reporter.fail(section, str(exc))


def check_model_family_mapping(reporter: Reporter) -> None:
    section = "Model-family mapping"
    mapping_path = repo_path("paper_icdm/model_family_mapping.csv")
    metrics_path = repo_path("artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet")
    if not require_file(reporter, str(rel(mapping_path)), section):
        return
    try:
        mapping = read_table(mapping_path)
    except RuntimeError as exc:
        reporter.fail(section, str(exc))
        return
    if not require_columns(
        reporter,
        section,
        mapping,
        ["model_name", "paper_display_name", "family"],
    ):
        return
    observed_models = set(mapping["model_name"].dropna().astype(str).unique())
    observed_families = set(mapping["family"].dropna().astype(str).unique())
    extra_models = observed_models - EXPECTED_MODELS
    missing_models = EXPECTED_MODELS - observed_models
    if missing_models or extra_models:
        reporter.fail(
            section,
            f"model mapping mismatch; missing={sorted(missing_models)}, extra={sorted(extra_models)}",
        )
    else:
        reporter.ok(section, "mapping contains exactly the 11 expected models")
    if observed_families == EXPECTED_FAMILIES:
        reporter.ok(section, f"mapping families match expected set: {sorted(EXPECTED_FAMILIES)}")
    else:
        reporter.fail(
            section,
            f"family mismatch; expected={sorted(EXPECTED_FAMILIES)}, observed={sorted(observed_families)}",
        )
    duplicate_models = mapping["model_name"][mapping["model_name"].duplicated()].tolist()
    if duplicate_models:
        reporter.fail(section, f"duplicate model_name entries: {duplicate_models}")
    else:
        reporter.ok(section, "no duplicate model_name entries")
    if metrics_path.exists():
        try:
            metrics = read_table(metrics_path)
        except RuntimeError as exc:
            reporter.fail(section, str(exc))
            return
        if "model_name" not in metrics.columns:
            reporter.fail(section, "metrics_long.parquet has no model_name column")
            return
        metrics_models = set(metrics["model_name"].dropna().astype(str).unique())
        missing_from_mapping = metrics_models - observed_models
        missing_from_metrics = observed_models - metrics_models
        if missing_from_mapping or missing_from_metrics:
            reporter.warning(
                section,
                "metrics/model-family mapping names differ; "
                f"missing_from_mapping={sorted(missing_from_mapping)}, "
                f"missing_from_metrics={sorted(missing_from_metrics)}",
            )
        else:
            reporter.ok(section, "Table V and Figure 4 coverage has metrics_long plus family mapping")
    else:
        reporter.warning(section, "metrics_long.parquet unavailable; Table V and Figure 4 remain incomplete")


def artifact_recommendation(path: Path, optional: bool, tracked: bool) -> str:
    if optional:
        return "optional"
    if tracked:
        return "keep tracked"
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 10 * 1024 * 1024:
        return "track directly"
    if size < 100 * 1024 * 1024:
        return "use Git LFS or GitHub Release"
    return "use Git LFS, GitHub Release, or external archive"


def check_tracking(reporter: Reporter) -> None:
    section = "Tracking/upload status"
    tracked = tracked_files()
    porcelain = status_porcelain()
    for relative_path in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
        path = repo_path(relative_path)
        is_optional = relative_path in OPTIONAL_ARTIFACTS
        exists = path.exists()
        is_tracked = relative_path in tracked
        recommendation = artifact_recommendation(path, is_optional, is_tracked)
        status = "OK" if exists and (is_tracked or is_optional) else "WARNING"
        if not exists and not is_optional:
            status = "FAIL"
            reporter.critical_missing.append(relative_path)
        message = (
            f"{relative_path}: exists={'yes' if exists else 'no'}, "
            f"tracked={'yes' if is_tracked else 'no'}, size={file_size(path)}, "
            f"git_status={porcelain.get(relative_path, 'clean/unknown')}, "
            f"recommendation={recommendation}"
        )
        reporter.add(status, section, message, critical=(status == "FAIL"))
        if exists and not is_tracked and recommendation not in {"optional", "missing"}:
            reporter.recommended_uploads.append(f"{relative_path} -> {recommendation}")
    for large_artifact in OPTIONAL_ARTIFACTS:
        path = repo_path(large_artifact)
        if path.exists() and large_artifact in tracked:
            reporter.warning(section, f"{large_artifact} is tracked; prefer LFS/release/external archive")
        elif path.exists():
            reporter.warning(section, f"{large_artifact} present but not tracked; archive externally if needed")
        else:
            reporter.warning(section, f"{large_artifact} absent; optional external artifact")


def check_paper_coverage(reporter: Reporter) -> None:
    section = "Paper output coverage"
    exists = {relative_path: repo_path(relative_path).exists() for relative_path in REQUIRED_ARTIFACTS}
    config_forecasting = repo_path("configs/forecasting_benchmark_v2.yaml").exists()
    config_arch = repo_path("configs/forecasting_selected_architectures_v1.yaml").exists()
    config_meta = repo_path("configs/meta_modeling_experiments_v2.yaml").exists()
    family_mapping = repo_path("paper_icdm/model_family_mapping.csv").exists()
    rows = [
        ("Table I", "feature_list_v2 + feature matrix", exists["artifacts/meta_modeling/feature_list_v2.csv"] and exists["artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet"], "OK"),
        ("Table II", "selected architectures config + forecasting config", config_forecasting and config_arch, "OK"),
        ("Table III", "forecasting config + meta config + manifest + split metadata", config_forecasting and config_meta and exists["artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json"] and exists["artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet"], "OK"),
        ("Table IV", "metrics_long", exists["artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet"], "OK"),
        ("Table VI", "meta_modeling_experiments_v2.xlsx summary sheet + compact lineage files", exists["artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2.xlsx"] and exists["artifacts/meta_modeling/task_results_v2.parquet"] and exists["artifacts/meta_modeling/split_assignments_v2.csv"] and exists["artifacts/meta_modeling/model_order_mapping_v2.csv"], "OK"),
        ("Figure 2", "task_results compact metrics", exists["artifacts/meta_modeling/task_results_v2.parquet"], "OK"),
        ("Figure 3", "task_results compact metrics", exists["artifacts/meta_modeling/task_results_v2.parquet"], "OK"),
    ]
    print("\nCoverage matrix:")
    print("Paper item | Source coverage | Status")
    for item, source, covered, ok_status in rows:
        status = ok_status if covered else "FAIL"
        print(f"{item} | {source} | {status}")
        reporter.add(status, section, f"{item}: {source}", critical=(status == "FAIL"))
    for item in ["Table V", "Figure 4"]:
        if exists["artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet"] and not family_mapping:
            message = (
                f"{item}: source metrics exist, but family mapping is not yet "
                "documented as a file"
            )
            print(f"{item} | metrics_long + model family mapping needed | WARNING")
            reporter.warning(section, message)
        elif family_mapping:
            print(f"{item} | metrics_long + model family mapping needed | OK")
            reporter.ok(section, f"{item}: metrics and family mapping file available")
        else:
            print(f"{item} | metrics_long + model family mapping needed | FAIL")
            reporter.fail(section, f"{item}: metrics_long missing")
    print("Figure 1 | schematic/manual source | WARNING")
    reporter.warning(section, "Figure 1: schematic/manual source; not data-generated in this stage")


def print_summary(reporter: Reporter) -> None:
    counts = reporter.counts()
    print("\nSummary:")
    for status in ["OK", "WARNING", "FAIL", "SKIPPED"]:
        print(f"  {status}: {counts[status]}")
    print("\nCritical missing artifacts:")
    if reporter.critical_missing:
        for item in sorted(set(reporter.critical_missing)):
            print(f"  {item}")
    else:
        print("  none")
    print("\nRecommended uploads:")
    if reporter.recommended_uploads:
        for item in reporter.recommended_uploads:
            print(f"  {item}")
    else:
        print("  none")


def main() -> int:
    reporter = Reporter()
    print("ICDM artifact validation checker")
    print(f"Repository root: {REPO_ROOT}")
    if Path.cwd().resolve() != REPO_ROOT:
        reporter.warning("Invocation", "script is intended to be run from repository root")
    else:
        reporter.ok("Invocation", "running from repository root")

    for artifact in REQUIRED_ARTIFACTS:
        require_file(reporter, artifact, "Required artifact presence")
    for artifact in OPTIONAL_ARTIFACTS:
        path = repo_path(artifact)
        if path.exists():
            reporter.warning("Optional artifact presence", f"{artifact} present ({file_size(path)})")
        else:
            reporter.warning("Optional artifact presence", f"{artifact} absent; optional, not a failure")

    check_processed_data(reporter)
    check_feature_artifacts(reporter)
    check_forecasting_artifacts(reporter)
    check_meta_artifacts(reporter)
    check_model_family_mapping(reporter)
    check_paper_coverage(reporter)
    check_tracking(reporter)
    print_summary(reporter)
    return 1 if reporter.has_failures() else 0


if __name__ == "__main__":
    sys.exit(main())
