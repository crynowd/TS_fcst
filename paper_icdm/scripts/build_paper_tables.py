"""Build compact ICDM paper tables from existing tracked artifacts.

Run from the repository root:

    python paper_icdm/scripts/build_paper_tables.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "paper_icdm" / "tables"

FEATURE_LIST = REPO_ROOT / "artifacts/meta_modeling/feature_list_v2.csv"
FEATURE_MATRIX = REPO_ROOT / "artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet"
FORECAST_CONFIG = REPO_ROOT / "configs/forecasting_benchmark_v2.yaml"
ARCH_CONFIG = REPO_ROOT / "configs/forecasting_selected_architectures_v1.yaml"
META_CONFIG = REPO_ROOT / "configs/meta_modeling_experiments_v2.yaml"
RUN_MANIFEST = REPO_ROOT / "artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json"
SPLIT_METADATA = REPO_ROOT / "artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet"
SPLIT_ASSIGNMENTS = REPO_ROOT / "artifacts/meta_modeling/split_assignments_v2.csv"
METRICS_LONG = REPO_ROOT / "artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet"
FAMILY_MAPPING = REPO_ROOT / "paper_icdm/model_family_mapping.csv"
TASK_RESULTS = REPO_ROOT / "artifacts/meta_modeling/task_results_v2.parquet"
MODEL_ORDER = REPO_ROOT / "artifacts/meta_modeling/model_order_mapping_v2.csv"
ROUTING_ROWS = REPO_ROOT / "artifacts/meta_modeling/routing_rows_v2.parquet"
META_EXCEL = REPO_ROOT / "artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2.xlsx"


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
        self.warnings: list[str] = []
        self.manual: dict[str, list[str]] = {}

    def rel(self, path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix()

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARNING] {message}")

    def write_csv(self, table_name: str, df: pd.DataFrame, out_name: str, sources: list[Path]) -> Path:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TABLE_DIR / out_name
        df.to_csv(out_path, index=False)
        self.created.append(out_path)
        self.sources[table_name] = [self.rel(path) for path in sources]
        if "needs_manual_verification" in df.columns:
            values = df["needs_manual_verification"].fillna(False).astype(bool)
            if bool(values.any()):
                reasons = []
                if "verification_note" in df.columns:
                    reasons = sorted(set(df.loc[values, "verification_note"].dropna().astype(str)))
                self.manual[table_name] = reasons or ["one or more rows flagged"]
        return out_path

    def missing_source_csv(self, table_name: str, out_name: str, missing: list[Path], note: str) -> Path:
        df = pd.DataFrame(
            [
                {
                    "table": table_name,
                    "status": "needs_manual_verification",
                    "missing_source": self.rel(path),
                    "reason": note,
                    "needs_manual_verification": True,
                }
                for path in missing
            ]
        )
        self.warn(f"{table_name}: {note}: {', '.join(self.rel(path) for path in missing)}")
        return self.write_csv(table_name, df, out_name, missing)


def require_sources(builder: TableBuilder, table_name: str, sources: list[Path], out_name: str) -> bool:
    missing = [path for path in sources if not path.exists()]
    if missing:
        builder.missing_source_csv(table_name, out_name, missing, "required source missing")
        return False
    return True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def git_ls_files(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.relative_to(REPO_ROOT).as_posix()],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0


def jsonish(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def fmt_metric(value: float, scale: float) -> float:
    return float(value) * scale


def build_table_i(builder: TableBuilder) -> None:
    table = "Table I"
    sources = [FEATURE_LIST, FEATURE_MATRIX]
    if not require_sources(builder, table, sources, "table_i_features_NEEDS_SOURCE.csv"):
        return
    features = pd.read_csv(FEATURE_LIST)
    feature_names = set(features["feature_name"].astype(str))
    matrix_cols = set(pd.read_parquet(FEATURE_MATRIX, columns=[]).columns)
    # columns=[] only checks file readability in pyarrow; read one row for actual columns.
    matrix_cols = set(pd.read_parquet(FEATURE_MATRIX).columns)
    rows = []
    for group, display_name, feature_name in FEATURE_ROWS:
        found = feature_name in feature_names and feature_name in matrix_cols
        rows.append(
            {
                "feature_group": group,
                "feature_name": display_name,
                "internal_feature_column": feature_name if found else "",
                "needs_manual_verification": not found,
                "verification_note": "" if found else f"{feature_name} not found in feature list and feature matrix",
            }
        )
    extra = sorted(feature_names - {row[2] for row in FEATURE_ROWS})
    if extra:
        builder.warn(f"{table}: feature_list_v2.csv contains unmapped features: {extra}")
    builder.write_csv(table, pd.DataFrame(rows), "table_i_features.csv", sources)


def build_table_ii(builder: TableBuilder) -> None:
    table = "Table II"
    sources = [ARCH_CONFIG, FORECAST_CONFIG]
    if not require_sources(builder, table, sources, "table_ii_candidates_NEEDS_SOURCE.csv"):
        return
    arch = load_yaml(ARCH_CONFIG)
    forecast = load_yaml(FORECAST_CONFIG)
    family_df = pd.read_csv(FAMILY_MAPPING) if FAMILY_MAPPING.exists() else pd.DataFrame()
    display_by_model = dict(zip(family_df.get("model_name", []), family_df.get("paper_display_name", [])))
    selected_by_model = {row["model_name"]: row for row in arch.get("selected_models", [])}
    baseline_params = {
        "naive_zero": {},
        "naive_mean": {},
        "ridge_lag": forecast.get("model_overrides", {}).get("ridge_lag", {}),
    }
    training_cfg = forecast.get("training", {})
    rows = []
    for model_name in MODEL_ORDERED:
        selected = selected_by_model.get(model_name, {})
        is_baseline = model_name in baseline_params
        model_params = baseline_params.get(model_name, selected.get("model_params", {}))
        runtime_params = selected.get("runtime_params", {})
        train_params = training_cfg.get("by_model", {}).get(model_name, training_cfg if model_name not in {"naive_zero", "naive_mean", "ridge_lag"} else {})
        missing = not is_baseline and not selected
        source = "configs/forecasting_benchmark_v2.yaml"
        if selected:
            source = "configs/forecasting_selected_architectures_v1.yaml"
        rows.append(
            {
                "model_name": model_name,
                "paper_display_name": display_by_model.get(model_name, model_name),
                "candidate_id": selected.get("candidate_id", ""),
                "family": selected.get("family", "baseline" if is_baseline else ""),
                "selection_role": selected.get("selection_role", "baseline" if is_baseline else ""),
                "model_params": jsonish(model_params),
                "runtime_params": jsonish(runtime_params),
                "training_params": jsonish(train_params),
                "source": source,
                "needs_manual_verification": missing,
                "verification_note": "" if not missing else "model parameters not found in selected architecture config",
            }
        )
    builder.write_csv(table, pd.DataFrame(rows), "table_ii_candidates.csv", sources + ([FAMILY_MAPPING] if FAMILY_MAPPING.exists() else []))


def build_table_iii(builder: TableBuilder) -> None:
    table = "Table III"
    sources = [FORECAST_CONFIG, META_CONFIG, RUN_MANIFEST, SPLIT_METADATA, SPLIT_ASSIGNMENTS]
    if not require_sources(builder, table, sources, "table_iii_protocol_NEEDS_SOURCE.csv"):
        return
    forecast = load_yaml(FORECAST_CONFIG)
    meta = load_yaml(META_CONFIG)
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    split_meta = pd.read_parquet(SPLIT_METADATA)
    split_assign = pd.read_csv(SPLIT_ASSIGNMENTS)
    rows = [
        ("Markets", ", ".join(f"{k}={v}" for k, v in split_meta.groupby("market")["series_id"].nunique().sort_index().items()), SPLIT_METADATA),
        ("Series representation", forecast.get("dataset_version", ""), FORECAST_CONFIG),
        ("Main dataset size", split_meta["series_id"].nunique(), SPLIT_METADATA),
        ("Return history length", f"n_train range {int(split_meta['n_train'].min())}-{int(split_meta['n_train'].max())}; n_test={int(split_meta['n_test'].median())}", SPLIT_METADATA),
        ("Forecast horizons", forecast.get("horizons", []), FORECAST_CONFIG),
        ("Input windows", forecast.get("window_sizes", {}), FORECAST_CONFIG),
        ("Temporal evaluation", f"{forecast.get('validation', {}).get('method')} with {forecast.get('validation', {}).get('n_folds')} folds", FORECAST_CONFIG),
        ("Meta-observations", split_assign["object_id"].nunique(), SPLIT_ASSIGNMENTS),
        ("Meta split", split_assign.groupby("split")["object_id"].nunique().to_dict(), SPLIT_ASSIGNMENTS),
        ("Per-horizon meta-observations", split_assign.groupby("horizon")["object_id"].nunique().to_dict(), SPLIT_ASSIGNMENTS),
        ("Candidate sets", meta.get("candidate_selection", {}).get("top_k_values", []), META_CONFIG),
        ("Metamodels", meta.get("classification_models", []), META_CONFIG),
        ("Grid evaluation", {"balancing_modes": meta.get("balancing_modes", []), "decision_rules": meta.get("decision_rules", []), "confidence_thresholds": meta.get("confidence_thresholds", [])}, META_CONFIG),
        ("Fallback thresholds", meta.get("confidence_thresholds", []), META_CONFIG),
        ("Reported metamodel result", "best configuration per (horizon, target_metric) selected by improvement_mean then gap_mean", META_CONFIG),
        ("Repeated evaluation", {"n_repeats": meta.get("n_repeats"), "split_random_seed": meta.get("split", {}).get("random_seed")}, META_CONFIG),
        ("Fixed-model baseline", "best_single_metric in meta-modeling task results", TASK_RESULTS),
    ]
    data = []
    for component, value, source in rows:
        needs = source == TASK_RESULTS and not source.exists()
        data.append(
            {
                "protocol_component": component,
                "value": jsonish(value),
                "source": builder.rel(source),
                "needs_manual_verification": needs,
                "verification_note": "" if not needs else "task_results source missing",
            }
        )
    if "summary" not in manifest:
        builder.warn(f"{table}: run_manifest.json has no top-level summary; protocol table used configs and split artifacts")
    builder.write_csv(table, pd.DataFrame(data), "table_iii_protocol.csv", sources)


def build_table_iv(builder: TableBuilder) -> None:
    table = "Table IV"
    sources = [METRICS_LONG, FAMILY_MAPPING]
    if not require_sources(builder, table, sources, "table_iv_direct_forecasting_NEEDS_SOURCE.csv"):
        return
    metrics = pd.read_parquet(METRICS_LONG)
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    rmse_max = pd.to_numeric(metrics["rmse"], errors="coerce").max()
    da_max = pd.to_numeric(metrics["directional_accuracy"], errors="coerce").max()
    rmse_scale = 100.0 if rmse_max <= 1.0 else 1.0
    da_scale = 100.0 if da_max <= 1.0 else 1.0
    if rmse_scale == 100.0:
        print("[INFO] Table IV: rmse values appear to be raw log-return units; multiplying by 100.")
    if da_scale == 100.0:
        print("[INFO] Table IV: directional_accuracy values appear to be fractions; multiplying by 100.")
    names = pd.read_csv(FAMILY_MAPPING)
    display_by_model = dict(zip(names["model_name"], names["paper_display_name"]))
    grouped = (
        metrics.groupby(["model_name", "horizon"], sort=False)
        .agg(rmse=("rmse", "mean"), directional_accuracy=("directional_accuracy", "mean"))
        .reset_index()
    )
    rows = []
    for model_name in MODEL_ORDERED:
        row: dict[str, Any] = {
            "model_name": model_name,
            "paper_display_name": display_by_model.get(model_name, model_name),
        }
        for horizon in [1, 5, 20]:
            sub = grouped[(grouped["model_name"] == model_name) & (grouped["horizon"] == horizon)]
            row[f"RMSE h={horizon}"] = fmt_metric(float(sub["rmse"].iloc[0]), rmse_scale) if not sub.empty else ""
            row[f"DA h={horizon}"] = fmt_metric(float(sub["directional_accuracy"].iloc[0]), da_scale) if not sub.empty else ""
        row["rmse_scale"] = "percentage log-return points" if rmse_scale == 100.0 else "as stored"
        row["da_scale"] = "percent" if da_scale == 100.0 else "as stored"
        row["needs_manual_verification"] = any(row[f"RMSE h={h}"] == "" or row[f"DA h={h}"] == "" for h in [1, 5, 20])
        row["verification_note"] = "" if not row["needs_manual_verification"] else "missing model/horizon metric rows"
        rows.append(row)
    builder.write_csv(table, pd.DataFrame(rows), "table_iv_direct_forecasting.csv", sources)


def build_table_v(builder: TableBuilder) -> None:
    table = "Table V"
    sources = [METRICS_LONG, FAMILY_MAPPING]
    if not require_sources(builder, table, sources, "table_v_winner_family_counts_NEEDS_SOURCE.csv"):
        return
    metrics = pd.read_parquet(METRICS_LONG)
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    family = pd.read_csv(FAMILY_MAPPING)
    fam_by_model = dict(zip(family["model_name"], family["family"]))
    series_level = (
        metrics.groupby(["series_id", "horizon", "model_name"], sort=False)
        .agg(rmse=("rmse", "mean"), directional_accuracy=("directional_accuracy", "mean"))
        .reset_index()
    )
    winners = []
    for (series_id, horizon), group in series_level.groupby(["series_id", "horizon"], sort=False):
        rmse_row = group.sort_values(["rmse", "model_name"], ascending=[True, True], kind="stable").iloc[0]
        da_row = group.sort_values(["directional_accuracy", "model_name"], ascending=[False, True], kind="stable").iloc[0]
        winners.append({"series_id": series_id, "horizon": horizon, "metric": "rmse", "model_name": rmse_row["model_name"]})
        winners.append({"series_id": series_id, "horizon": horizon, "metric": "directional_accuracy", "model_name": da_row["model_name"]})
    winners_df = pd.DataFrame(winners)
    winners_df["family"] = winners_df["model_name"].map(fam_by_model)
    if winners_df["family"].isna().any():
        builder.warn(f"{table}: model_family_mapping.csv missing families for {sorted(winners_df.loc[winners_df['family'].isna(), 'model_name'].unique())}")
    total_tasks = int(series_level[["series_id", "horizon"]].drop_duplicates().shape[0])
    rows = []
    for family_name in ["Zero/mean baselines", "Non-chaotic models", "Chaos-inspired models"]:
        rmse_wins = int(((winners_df["metric"] == "rmse") & (winners_df["family"] == family_name)).sum())
        da_wins = int(((winners_df["metric"] == "directional_accuracy") & (winners_df["family"] == family_name)).sum())
        rows.append(
            {
                "Model family": family_name,
                "RMSE wins": rmse_wins,
                "DA wins": da_wins,
                "RMSE share": rmse_wins / total_tasks if total_tasks else float("nan"),
                "DA share": da_wins / total_tasks if total_tasks else float("nan"),
                "task_count": total_tasks,
                "aggregation": "mean fold scores by series_id+horizon+model before selecting winners",
                "needs_manual_verification": total_tasks != 1254,
                "verification_note": "" if total_tasks == 1254 else f"expected 1254 tasks, found {total_tasks}",
            }
        )
    builder.write_csv(table, pd.DataFrame(rows), "table_v_winner_family_counts.csv", sources)


def parse_best_single_model(builder: TableBuilder, excel_path: Path, best_row: pd.Series) -> str:
    try:
        best_single = pd.read_excel(excel_path, sheet_name="best_single_repeat")
    except Exception:
        return ""
    filters = (
        (best_single["horizon"] == best_row["horizon"])
        & (best_single["target_metric"].astype(str) == str(best_row["target_metric"]))
        & (best_single["feature_set"].astype(str) == str(best_row["feature_set"]))
        & (best_single["candidate_set"].astype(str) == str(best_row["candidate_set"]))
        & (best_single["balancing_mode"].astype(str) == str(best_row["balancing_mode"]))
        & (best_single["decision_rule"].astype(str) == str(best_row["decision_rule"]))
    )
    threshold = best_row.get("confidence_threshold")
    if pd.isna(threshold):
        filters &= best_single["confidence_threshold"].isna()
    else:
        filters &= pd.to_numeric(best_single["confidence_threshold"], errors="coerce") == float(threshold)
    candidates = best_single[filters]
    if candidates.empty:
        return ""
    models = candidates["best_single_model"].dropna().astype(str)
    if models.empty:
        return ""
    mode = models.mode()
    if len(mode) > 1:
        builder.warn("Table VI: best_single_repeat has tied modal best_single_model values for a selected row")
    return str(mode.iloc[0])


def build_table_vi(builder: TableBuilder) -> None:
    table = "Table VI"
    if not META_EXCEL.exists():
        builder.missing_source_csv(
            table,
            "table_vi_meta_selection_results_NEEDS_SOURCE.csv",
            [META_EXCEL],
            "Excel report source missing; not reconstructing Table VI from task_results_v2.parquet by guesswork",
        )
        return
    print(f"[INFO] Table VI Excel source: {builder.rel(META_EXCEL)} ({META_EXCEL.stat().st_size} bytes); tracked={git_ls_files(META_EXCEL)}")
    try:
        xls = pd.ExcelFile(META_EXCEL)
    except Exception as exc:
        builder.warn(f"{table}: could not open Excel source: {exc}")
        builder.missing_source_csv(table, "table_vi_meta_selection_results_NEEDS_SOURCE.csv", [META_EXCEL], "Excel source unreadable")
        return
    print(f"[INFO] Table VI Excel sheets: {', '.join(xls.sheet_names)}")
    required_cols = {
        "horizon",
        "target_metric",
        "best_single_mean",
        "best_single_std",
        "achieved_mean",
        "achieved_std",
        "oracle_mean",
        "oracle_std",
        "improvement_mean",
        "improvement_std",
        "gap_mean",
    }
    candidate_sheets = []
    for sheet in xls.sheet_names:
        try:
            head = pd.read_excel(META_EXCEL, sheet_name=sheet, nrows=1)
        except Exception:
            continue
        if {"horizon", "target_metric"}.issubset(set(head.columns)):
            candidate_sheets.append(sheet)
    if "summary" not in candidate_sheets:
        for sheet in candidate_sheets:
            df = pd.read_excel(META_EXCEL, sheet_name=sheet)
            out = TABLE_DIR / f"debug_table_vi_candidate_{sheet}.csv"
            TABLE_DIR.mkdir(parents=True, exist_ok=True)
            df.head(200).to_csv(out, index=False)
            builder.created.append(out)
        builder.warn(f"{table}: exact summary sheet not identified; wrote candidate debug CSVs")
        builder.missing_source_csv(table, "table_vi_meta_selection_results_NEEDS_SOURCE.csv", [META_EXCEL], "summary sheet unavailable")
        return
    summary = pd.read_excel(META_EXCEL, sheet_name="summary")
    missing_cols = sorted(required_cols - set(summary.columns))
    if missing_cols:
        for sheet in candidate_sheets:
            df = pd.read_excel(META_EXCEL, sheet_name=sheet)
            out = TABLE_DIR / f"debug_table_vi_candidate_{sheet}.csv"
            TABLE_DIR.mkdir(parents=True, exist_ok=True)
            df.head(200).to_csv(out, index=False)
            builder.created.append(out)
        builder.warn(f"{table}: summary sheet missing required columns {missing_cols}; wrote candidate debug CSVs")
        builder.missing_source_csv(table, "table_vi_meta_selection_results_NEEDS_SOURCE.csv", [META_EXCEL], "summary sheet lacks required columns")
        return
    success = summary.copy()
    selected = (
        success.sort_values(["horizon", "target_metric", "improvement_mean", "gap_mean"], ascending=[True, True, False, True], kind="stable")
        .groupby(["horizon", "target_metric"], sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    scale_by_metric = {"rmse": "raw log-return units", "directional_accuracy": "fraction"}
    rows = []
    for _, row in selected.iterrows():
        metric = str(row["target_metric"])
        best_model = parse_best_single_model(builder, META_EXCEL, row)
        rows.append(
            {
                "h": int(row["horizon"]),
                "Metric": metric,
                "Best fixed model": best_model,
                "Best observed": str(row["model"]),
                "Delta": float(row["improvement_mean"]),
                "Oracle": float(row["oracle_mean"]),
                "best_fixed_mean": float(row["best_single_mean"]),
                "best_fixed_std": float(row["best_single_std"]) if not pd.isna(row["best_single_std"]) else "",
                "best_observed_mean": float(row["achieved_mean"]),
                "best_observed_std": float(row["achieved_std"]) if not pd.isna(row["achieved_std"]) else "",
                "delta_mean": float(row["improvement_mean"]),
                "delta_std": float(row["improvement_std"]) if not pd.isna(row["improvement_std"]) else "",
                "oracle_mean": float(row["oracle_mean"]),
                "oracle_std": float(row["oracle_std"]) if not pd.isna(row["oracle_std"]) else "",
                "gap_mean": float(row["gap_mean"]),
                "candidate_set": str(row["candidate_set"]),
                "feature_set": str(row["feature_set"]),
                "balancing_mode": str(row["balancing_mode"]),
                "decision_rule": str(row["decision_rule"]),
                "confidence_threshold": "" if pd.isna(row["confidence_threshold"]) else float(row["confidence_threshold"]),
                "scale": scale_by_metric.get(metric, "as stored"),
                "source_sheet": "summary",
                "needs_manual_verification": best_model == "",
                "verification_note": "" if best_model else "best fixed model name not found in best_single_repeat sheet",
            }
        )
    sources = [META_EXCEL]
    for source in [TASK_RESULTS, SPLIT_ASSIGNMENTS, MODEL_ORDER, ROUTING_ROWS]:
        if source.exists():
            sources.append(source)
    builder.write_csv(table, pd.DataFrame(rows), "table_vi_meta_selection_results.csv", sources)


def print_summary(builder: TableBuilder) -> None:
    print("\nCreated files:")
    for path in builder.created:
        print(f"- {builder.rel(path)}")
    print("\nSource files used:")
    for table, sources in builder.sources.items():
        print(f"- {table}: {', '.join(sources)}")
    if builder.manual:
        print("\nNeeds manual verification:")
        for table, reasons in builder.manual.items():
            print(f"- {table}: {'; '.join(reasons)}")
    if builder.warnings:
        print("\nWarnings:")
        for warning in builder.warnings:
            print(f"- {warning}")
    else:
        print("\nWarnings: none")


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        print(f"[ERROR] Run from repository root: {REPO_ROOT}", file=sys.stderr)
        return 2
    builder = TableBuilder()
    build_table_i(builder)
    build_table_ii(builder)
    build_table_iii(builder)
    build_table_iv(builder)
    build_table_v(builder)
    build_table_vi(builder)
    print_summary(builder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
