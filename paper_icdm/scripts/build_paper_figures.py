"""Build the current data-generated ICDM figure from frozen clean test rows.

The current manuscript's Figure 2 is the pooled h=5 directional-accuracy
confusion matrix. Legacy line/winner plots formerly numbered Figures 2--4 are
removed by this builder so that they cannot be mistaken for current figures.

Run from the repository root:

    python paper_icdm/scripts/build_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "paper_icdm" / "figures"
ROUTING_ROWS = (
    REPO_ROOT
    / "artifacts/meta_modeling/clean_meta_learning_v1/routing_rows_v2.parquet"
)

FIGURE_STEM = "figure_2_pooled_confusion_h5_da"
SUPPORT_CSV = FIGURE_DIR / f"{FIGURE_STEM}_counts.csv"
STALE_OUTPUTS = [
    "figure_2_rmse_fixed_meta_oracle.png",
    "figure_2_rmse_fixed_meta_oracle.pdf",
    "figure_3_da_fixed_meta_oracle.png",
    "figure_3_da_fixed_meta_oracle.pdf",
    "figure_4_winner_distribution.png",
    "figure_4_winner_distribution.pdf",
]

# This is the predeclared candidate/class order used by the clean meta pipeline.
CANDIDATE_ORDER = [
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
SHORT_LABEL = {
    "chaotic_esn": "Chaotic ESN",
    "chaotic_logistic_net": "Logistic-map net",
    "chaotic_lstm_forecast": "Chaotic LSTM",
    "chaotic_mlp": "Chaotic MLP",
    "esn": "ESN",
    "lstm_forecast": "LSTM",
    "naive_mean": "Mean",
    "naive_zero": "Zero",
    "ridge_lag": "Ridge",
    "transient_chaotic_esn": "Transient ESN",
    "vanilla_mlp": "MLP",
}


def remove_stale_outputs() -> list[Path]:
    removed: list[Path] = []
    figure_root = FIGURE_DIR.resolve()
    for name in STALE_OUTPUTS:
        path = (FIGURE_DIR / name).resolve()
        if path.parent != figure_root:
            raise RuntimeError(f"refusing to remove path outside paper figure directory: {path}")
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def load_pooled_rows() -> pd.DataFrame:
    if not ROUTING_ROWS.is_file():
        raise FileNotFoundError(f"required clean routing artifact missing: {ROUTING_ROWS}")
    routing = pd.read_parquet(ROUTING_ROWS)
    required = {
        "repeat_id",
        "horizon",
        "target_metric",
        "evaluation_partition",
        "selected_model",
        "final_winner_label",
        "selected_by_validation",
    }
    missing = sorted(required - set(routing.columns))
    if missing:
        raise ValueError(f"clean routing artifact lacks columns: {missing}")
    pooled = routing[
        (routing["horizon"] == 5)
        & (routing["target_metric"].astype(str) == "directional_accuracy")
        & (routing["evaluation_partition"].astype(str) == "test")
        & (routing["selected_by_validation"].astype(int) == 1)
    ].copy()
    if len(pooled) != 1260:
        raise ValueError(f"Figure 2 requires N=1260 frozen test rows, found {len(pooled)}")
    if set(pooled["repeat_id"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Figure 2 does not contain all five clean instrument-level repeats")
    return pooled


def build_confusion_matrix(pooled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    observed = set(pooled["selected_model"].astype(str)) | set(pooled["final_winner_label"].astype(str))
    unknown = observed - set(CANDIDATE_ORDER)
    if unknown:
        raise ValueError(f"Figure 2 contains models outside the clean candidate order: {sorted(unknown)}")
    labels = [model for model in CANDIDATE_ORDER if model in observed]
    counts = pd.crosstab(pooled["final_winner_label"], pooled["selected_model"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    if int(counts.to_numpy().sum()) != 1260:
        raise ValueError("Figure 2 confusion-matrix total is not 1260")
    return counts, labels


def save_support_csv(counts: pd.DataFrame) -> None:
    output = counts.copy()
    output.index.name = "actual_best_model"
    output.columns = [f"selected__{column}" for column in output.columns]
    output["row_total"] = output.sum(axis=1)
    output["pooled_N"] = int(output["row_total"].sum())
    output.to_csv(SUPPORT_CSV)


def save_figure(counts: pd.DataFrame, labels: list[str]) -> list[Path]:
    values = counts.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals != 0)

    fig, axis = plt.subplots(figsize=(8.4, 6.8))
    image = axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=max(0.65, float(shares.max())))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            count = int(values[row, column])
            share = shares[row, column]
            color = "white" if share > 0.36 else "black"
            axis.text(column, row, f"{count}\n{share:.0%}", ha="center", va="center", fontsize=8, color=color)

    tick_labels = [SHORT_LABEL[model] for model in labels]
    axis.set_xticks(np.arange(len(labels)), labels=tick_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=tick_labels)
    axis.set_xlabel("Selected model (validation-selected frozen route)")
    axis.set_ylabel("Actual best model on the frozen test object")
    axis.set_title("Pooled confusion matrix for h=5 directional accuracy (N=1,260)")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Share within actual-best row")
    fig.tight_layout()

    created = []
    for suffix in [".png", ".pdf"]:
        path = FIGURE_DIR / f"{FIGURE_STEM}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        created.append(path)
    plt.close(fig)
    return created


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        print(f"[ERROR] Run from repository root: {REPO_ROOT}", file=sys.stderr)
        return 2
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        pooled = load_pooled_rows()
        counts, labels = build_confusion_matrix(pooled)
        save_support_csv(counts)
        created = save_figure(counts, labels)
        removed = remove_stale_outputs()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("Created files:")
    for path in [SUPPORT_CSV, *created]:
        print(f"- {path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Source: {ROUTING_ROWS.relative_to(REPO_ROOT).as_posix()}")
    print("Removed stale generated outputs:")
    if removed:
        for path in removed:
            print(f"- {path.relative_to(REPO_ROOT).as_posix()}")
    else:
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
