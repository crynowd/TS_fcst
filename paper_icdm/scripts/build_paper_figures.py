"""Build ICDM paper figures from existing compact paper tables.

Run from the repository root:

    python paper_icdm/scripts/build_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "paper_icdm" / "tables"
FIGURE_DIR = REPO_ROOT / "paper_icdm" / "figures"

TABLE_VI = TABLE_DIR / "table_vi_meta_selection_results.csv"
TABLE_V = TABLE_DIR / "table_v_winner_family_counts.csv"
TABLE_V_BY_HORIZON = TABLE_DIR / "table_v_winner_family_counts_by_horizon.csv"
METRICS_LONG = REPO_ROOT / "artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet"
FAMILY_MAPPING = REPO_ROOT / "paper_icdm" / "model_family_mapping.csv"

HORIZONS = [1, 5, 20]
FAMILY_ORDER = [
    "Zero/mean baselines",
    "Non-chaotic models",
    "Chaos-inspired models",
]
SERIES_PER_HORIZON = 418


class FigureBuilder:
    def __init__(self) -> None:
        self.created: list[Path] = []
        self.warnings: list[str] = []

    def rel(self, path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix()

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARNING] {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")


def require_source(builder: FigureBuilder, path: Path, label: str) -> bool:
    if path.exists():
        return True
    builder.warn(f"{label}: required source missing: {builder.rel(path)}")
    return False


def metric_rows(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = table[table["Metric"].astype(str).str.lower() == metric.lower()].copy()
    if "h" in rows.columns:
        rows["h"] = pd.to_numeric(rows["h"], errors="coerce")
        rows = rows.sort_values("h", kind="stable")
    return rows


def explicit_scale_multiplier(rows: pd.DataFrame, metric: str) -> float | None:
    text = " ".join(
        str(value).lower()
        for column in ["scale", "display_scale", "raw_scale"]
        if column in rows.columns
        for value in rows[column].dropna().unique()
    )
    if metric == "rmse" and "raw log-return" in text:
        return 100.0
    if metric == "directional_accuracy" and ("fraction" in text or "percent" in text):
        if "fraction" in text:
            return 100.0
        return 1.0
    return None


def figure_series(
    builder: FigureBuilder,
    table: pd.DataFrame,
    metric: str,
    display_columns: dict[str, str],
    raw_columns: dict[str, str],
) -> pd.DataFrame | None:
    rows = metric_rows(table, metric)
    if rows.empty:
        builder.warn(f"{metric}: no rows in {builder.rel(TABLE_VI)}")
        return None
    if set(rows["h"].dropna().astype(int)) != set(HORIZONS):
        builder.warn(f"{metric}: expected horizons {HORIZONS}, found {sorted(rows['h'].dropna().astype(int).unique())}")

    if set(display_columns.values()).issubset(rows.columns):
        selected = {"h": rows["h"].astype(int)}
        for label, column in display_columns.items():
            selected[label] = pd.to_numeric(rows[column], errors="coerce")
        return pd.DataFrame(selected).sort_values("h", kind="stable")

    missing_display = sorted(set(display_columns.values()) - set(rows.columns))
    builder.warn(
        f"{metric}: display columns missing {missing_display}; available columns: {list(rows.columns)}"
    )
    if not set(raw_columns.values()).issubset(rows.columns):
        missing_raw = sorted(set(raw_columns.values()) - set(rows.columns))
        builder.warn(f"{metric}: raw fallback columns missing {missing_raw}; not generating figure")
        return None

    multiplier = explicit_scale_multiplier(rows, metric)
    if multiplier is None:
        builder.warn(f"{metric}: raw fallback scale is not explicit; not generating figure")
        return None

    selected = {"h": rows["h"].astype(int)}
    for label, column in raw_columns.items():
        selected[label] = pd.to_numeric(rows[column], errors="coerce") * multiplier
    return pd.DataFrame(selected).sort_values("h", kind="stable")


def save_current_figure(builder: FigureBuilder, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        path = FIGURE_DIR / f"{stem}{suffix}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        builder.created.append(path)
        builder.info(f"created {builder.rel(path)}")
    plt.close()


def build_line_figure(
    builder: FigureBuilder,
    table_vi: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    data = figure_series(
        builder,
        table_vi,
        metric,
        {
            "Best fixed model": "best_fixed_display_mean",
            "Best observed metamodel": "best_observed_display_mean",
            "Oracle": "oracle_display_mean",
        },
        {
            "Best fixed model": "best_fixed_mean",
            "Best observed metamodel": "best_observed_mean",
            "Oracle": "oracle_mean",
        },
    )
    if data is None:
        return
    if data[["Best fixed model", "Best observed metamodel", "Oracle"]].isna().any().any():
        builder.warn(f"{metric}: non-numeric plotting values found; not generating figure")
        return

    plt.figure(figsize=(6.2, 4.0))
    for label in ["Best fixed model", "Best observed metamodel", "Oracle"]:
        plt.plot(data["h"], data[label], marker="o", linewidth=1.8, label=label)
    plt.xticks(HORIZONS, [str(horizon) for horizon in HORIZONS])
    plt.xlabel("Forecast horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle=":", linewidth=0.8)
    plt.legend(frameon=False)
    save_current_figure(builder, stem)


def build_winner_counts_by_horizon(builder: FigureBuilder) -> pd.DataFrame | None:
    if TABLE_V_BY_HORIZON.exists():
        builder.info(f"using existing {builder.rel(TABLE_V_BY_HORIZON)}")
        return pd.read_csv(TABLE_V_BY_HORIZON)

    if not require_source(builder, METRICS_LONG, "Figure 4"):
        return None
    if not require_source(builder, FAMILY_MAPPING, "Figure 4"):
        return None

    metrics = pd.read_parquet(METRICS_LONG)
    required = {"series_id", "horizon", "model_name", "status", "rmse", "directional_accuracy"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        builder.warn(f"Figure 4: metrics columns missing {missing}; available columns: {list(metrics.columns)}")
        return None
    metrics = metrics[metrics["status"].astype(str) == "success"].copy()
    family = pd.read_csv(FAMILY_MAPPING)
    missing_family = sorted({"model_name", "family"} - set(family.columns))
    if missing_family:
        builder.warn(f"Figure 4: family mapping columns missing {missing_family}; available columns: {list(family.columns)}")
        return None
    family_by_model = dict(zip(family["model_name"], family["family"]))

    series_level = (
        metrics.groupby(["series_id", "horizon", "model_name"], sort=False)
        .agg(rmse=("rmse", "mean"), directional_accuracy=("directional_accuracy", "mean"))
        .reset_index()
    )
    winners: list[dict[str, Any]] = []
    for (series_id, horizon), group in series_level.groupby(["series_id", "horizon"], sort=False):
        rmse_row = group.sort_values(["rmse", "model_name"], ascending=[True, True], kind="stable").iloc[0]
        da_row = group.sort_values(["directional_accuracy", "model_name"], ascending=[False, True], kind="stable").iloc[0]
        winners.append({"series_id": series_id, "horizon": int(horizon), "metric": "rmse", "model_name": rmse_row["model_name"]})
        winners.append({"series_id": series_id, "horizon": int(horizon), "metric": "directional_accuracy", "model_name": da_row["model_name"]})

    winners_df = pd.DataFrame(winners)
    winners_df["family"] = winners_df["model_name"].map(family_by_model)
    if winners_df["family"].isna().any():
        builder.warn(
            "Figure 4: model_family_mapping.csv missing families for "
            f"{sorted(winners_df.loc[winners_df['family'].isna(), 'model_name'].unique())}"
        )

    grouped = (
        winners_df.groupby(["metric", "horizon", "family"], dropna=False, sort=False)
        .size()
        .reset_index(name="winner_count")
    )
    rows = []
    for metric in ["rmse", "directional_accuracy"]:
        for horizon in HORIZONS:
            total = int(grouped[(grouped["metric"] == metric) & (grouped["horizon"] == horizon)]["winner_count"].sum())
            for family_name in FAMILY_ORDER:
                sub = grouped[
                    (grouped["metric"] == metric)
                    & (grouped["horizon"] == horizon)
                    & (grouped["family"] == family_name)
                ]
                count = int(sub["winner_count"].iloc[0]) if not sub.empty else 0
                rows.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "Model family": family_name,
                        "winner_count": count,
                        "winner_share": count / total if total else np.nan,
                        "task_count": total,
                        "aggregation": "mean fold scores by series_id+horizon+model before selecting winners",
                    }
                )
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_V_BY_HORIZON, index=False)
    builder.created.append(TABLE_V_BY_HORIZON)
    builder.info(f"created {builder.rel(TABLE_V_BY_HORIZON)}")
    return out


def validate_winner_counts(builder: FigureBuilder, counts: pd.DataFrame) -> None:
    for metric in ["rmse", "directional_accuracy"]:
        for horizon in HORIZONS:
            total = int(
                counts[(counts["metric"] == metric) & (counts["horizon"] == horizon)]["winner_count"].sum()
            )
            if total != SERIES_PER_HORIZON:
                builder.warn(f"Figure 4: {metric} h={horizon} totals {total}, expected {SERIES_PER_HORIZON}")

    if not TABLE_V.exists():
        builder.warn(f"Figure 4: {builder.rel(TABLE_V)} missing; aggregated Table V consistency not checked")
        return
    table_v = pd.read_csv(TABLE_V)
    required = {"Model family", "RMSE wins", "DA wins"}
    missing = sorted(required - set(table_v.columns))
    if missing:
        builder.warn(f"Figure 4: Table V columns missing {missing}; aggregated consistency not checked")
        return
    aggregate = (
        counts.groupby(["metric", "Model family"], sort=False)["winner_count"]
        .sum()
        .reset_index()
    )
    for family_name in FAMILY_ORDER:
        source = table_v[table_v["Model family"] == family_name]
        if source.empty:
            builder.warn(f"Figure 4: family {family_name!r} missing from Table V")
            continue
        rmse_total = int(aggregate[(aggregate["metric"] == "rmse") & (aggregate["Model family"] == family_name)]["winner_count"].sum())
        da_total = int(aggregate[(aggregate["metric"] == "directional_accuracy") & (aggregate["Model family"] == family_name)]["winner_count"].sum())
        if rmse_total != int(source["RMSE wins"].iloc[0]):
            builder.warn(f"Figure 4: RMSE aggregate for {family_name} is {rmse_total}, Table V has {int(source['RMSE wins'].iloc[0])}")
        if da_total != int(source["DA wins"].iloc[0]):
            builder.warn(f"Figure 4: DA aggregate for {family_name} is {da_total}, Table V has {int(source['DA wins'].iloc[0])}")


def build_winner_distribution_figure(builder: FigureBuilder) -> None:
    counts = build_winner_counts_by_horizon(builder)
    if counts is None:
        return
    required = {"metric", "horizon", "Model family", "winner_count"}
    missing = sorted(required - set(counts.columns))
    if missing:
        builder.warn(f"Figure 4: support table columns missing {missing}; available columns: {list(counts.columns)}")
        return
    counts["horizon"] = pd.to_numeric(counts["horizon"], errors="coerce").astype("Int64")
    counts["winner_count"] = pd.to_numeric(counts["winner_count"], errors="coerce")
    validate_winner_counts(builder, counts)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    metric_titles = [("rmse", "RMSE"), ("directional_accuracy", "Directional accuracy")]
    x = np.arange(len(FAMILY_ORDER))
    width = 0.24
    for axis, (metric, title) in zip(axes, metric_titles):
        for idx, horizon in enumerate(HORIZONS):
            values = []
            for family_name in FAMILY_ORDER:
                sub = counts[
                    (counts["metric"].astype(str) == metric)
                    & (counts["horizon"].astype(int) == horizon)
                    & (counts["Model family"].astype(str) == family_name)
                ]
                values.append(float(sub["winner_count"].iloc[0]) if not sub.empty else 0.0)
            offset = (idx - 1) * width
            bars = axis.bar(x + offset, values, width, label=f"h = {horizon}")
            axis.bar_label(bars, labels=[str(int(value)) for value in values], padding=2, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(FAMILY_ORDER, rotation=20, ha="right")
        axis.grid(True, axis="y", linestyle=":", linewidth=0.8)
        axis.set_ylim(0, max(SERIES_PER_HORIZON * 1.12, axis.get_ylim()[1]))
    axes[0].set_ylabel("Winner count")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Winner distribution by model family and horizon")
    save_current_figure(builder, "figure_4_winner_distribution")


def print_summary(builder: FigureBuilder) -> None:
    print("\nCreated files:")
    if builder.created:
        for path in builder.created:
            print(f"- {builder.rel(path)}")
    else:
        print("- none")
    print("\nSource files used:")
    print(f"- Figure 2: {builder.rel(TABLE_VI)}")
    print(f"- Figure 3: {builder.rel(TABLE_VI)}")
    print(f"- Figure 4: {builder.rel(TABLE_V_BY_HORIZON)}")
    print("\nWarnings:")
    if builder.warnings:
        for warning in builder.warnings:
            print(f"- {warning}")
    else:
        print("- none")


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        print(f"[ERROR] Run from repository root: {REPO_ROOT}", file=sys.stderr)
        return 2

    builder = FigureBuilder()
    if require_source(builder, TABLE_VI, "Figures 2-3"):
        table_vi = pd.read_csv(TABLE_VI)
        if {"Metric", "h"}.issubset(table_vi.columns):
            build_line_figure(
                builder,
                table_vi,
                "rmse",
                "RMSE, percentage log-return points",
                "Fixed model, metamodel selection, and oracle RMSE",
                "figure_2_rmse_fixed_meta_oracle",
            )
            build_line_figure(
                builder,
                table_vi,
                "directional_accuracy",
                "Directional accuracy, percent",
                "Fixed model, metamodel selection, and oracle directional accuracy",
                "figure_3_da_fixed_meta_oracle",
            )
        else:
            builder.warn(f"Figures 2-3: Table VI columns unclear; available columns: {list(table_vi.columns)}")

    build_winner_distribution_figure(builder)
    print_summary(builder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
