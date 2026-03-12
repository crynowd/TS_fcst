from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.clustering.grid_search import fit_configuration
from src.clustering.preprocessing import METADATA_COLUMNS


def build_top_config_assignments(
    top_configs_df: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    assignment_rows = []
    reference_labels: Dict[str, np.ndarray] = {}

    for _, cfg in top_configs_df.iterrows():
        feature_set = str(cfg["feature_set"])
        table = feature_tables[feature_set].reset_index(drop=True)
        fit = fit_configuration(
            feature_table=table,
            scaler=str(cfg["scaler"]),
            space_type=str(cfg["space_type"]),
            pca_n_components=None if pd.isna(cfg["pca_n_components"]) else int(cfg["pca_n_components"]),
            algorithm=str(cfg["algorithm"]),
            n_clusters_requested=int(cfg["n_clusters_requested"]),
            random_state=int(random_state),
        )
        labels = fit.labels
        reference_labels[str(cfg["config_id"])] = labels.copy()

        meta_cols = [c for c in METADATA_COLUMNS if c in table.columns]
        meta = table[meta_cols].copy()
        for i in range(len(table)):
            assignment_rows.append(
                {
                    "config_id": str(cfg["config_id"]),
                    "feature_set": feature_set,
                    "scaler": str(cfg["scaler"]),
                    "space_type": str(cfg["space_type"]),
                    "algorithm": str(cfg["algorithm"]),
                    "n_clusters": int(cfg["n_clusters_requested"]),
                    **{c: meta.iloc[i][c] for c in meta_cols},
                    "cluster_label": int(labels[i]),
                    "assignment_confidence": (
                        float(fit.assignment_confidence[i]) if fit.assignment_confidence is not None else np.nan
                    ),
                }
            )

    assignments_df = pd.DataFrame(assignment_rows)
    return assignments_df, reference_labels


def run_bootstrap_stability(
    top_configs_df: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    reference_labels: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> pd.DataFrame:
    n_bootstrap = int(cfg["stability"]["n_bootstrap"])
    sample_fraction = float(cfg["stability"]["sample_fraction"])
    random_state = int(cfg["stability"]["random_state"])
    rng = np.random.default_rng(random_state)

    rows = []
    for idx, row in top_configs_df.iterrows():
        if logger:
            logger.info("Stability config %d/%d: %s", idx + 1, len(top_configs_df), row["config_id"])
        config_id = str(row["config_id"])
        feature_set = str(row["feature_set"])
        table = feature_tables[feature_set].reset_index(drop=True)
        ref_labels = np.asarray(reference_labels[config_id], dtype=int)

        n = len(table)
        sample_n = max(2, int(np.floor(sample_fraction * n)))
        aris = []
        nmis = []
        failed = 0

        for _ in range(n_bootstrap):
            sample_idx = np.sort(rng.choice(n, size=sample_n, replace=False))
            sampled_table = table.iloc[sample_idx].reset_index(drop=True)
            try:
                fit = fit_configuration(
                    feature_table=sampled_table,
                    scaler=str(row["scaler"]),
                    space_type=str(row["space_type"]),
                    pca_n_components=None if pd.isna(row["pca_n_components"]) else int(row["pca_n_components"]),
                    algorithm=str(row["algorithm"]),
                    n_clusters_requested=int(row["n_clusters_requested"]),
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )
                ari = adjusted_rand_score(ref_labels[sample_idx], fit.labels)
                nmi = normalized_mutual_info_score(ref_labels[sample_idx], fit.labels)
                aris.append(float(ari))
                nmis.append(float(nmi))
            except Exception:
                failed += 1

        rows.append(
            {
                "config_id": config_id,
                "feature_set": feature_set,
                "scaler": str(row["scaler"]),
                "space_type": str(row["space_type"]),
                "pca_n_components": row["pca_n_components"],
                "algorithm": str(row["algorithm"]),
                "n_clusters": int(row["n_clusters_requested"]),
                "n_bootstrap": int(n_bootstrap),
                "ari_mean": float(np.mean(aris)) if aris else np.nan,
                "ari_std": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0 if len(aris) == 1 else np.nan,
                "nmi_mean": float(np.mean(nmis)) if nmis else np.nan,
                "nmi_std": float(np.std(nmis, ddof=1)) if len(nmis) > 1 else 0.0 if len(nmis) == 1 else np.nan,
                "profile_stability_mean": np.nan,
                "notes": f"failed_bootstrap_runs={failed}",
            }
        )
    return pd.DataFrame(rows)

