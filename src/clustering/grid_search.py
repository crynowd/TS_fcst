from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from src.clustering.metrics import compute_cluster_metrics, compute_gmm_confidence_metrics
from src.clustering.preprocessing import preprocess_feature_table, split_metadata_and_features


@dataclass(frozen=True)
class ClusteringFitResult:
    labels: np.ndarray
    assignment_confidence: np.ndarray | None
    assignment_probabilities: np.ndarray | None
    preprocessing_info: Dict[str, Any]
    matrix: np.ndarray


def build_grid_configurations(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    feature_sets = [str(v) for v in cfg["feature_sets"]]
    scalers = [str(v) for v in cfg["scalers"]]
    spaces = [str(v) for v in cfg["spaces"]]
    pca_components = [int(v) for v in cfg["pca_n_components"]]
    algorithms = [str(v) for v in cfg["algorithms"]]
    k_min = int(cfg["cluster_range"]["k_min"])
    k_max = int(cfg["cluster_range"]["k_max"])

    out: List[Dict[str, Any]] = []
    for feature_set in feature_sets:
        for scaler in scalers:
            for space in spaces:
                pca_list: List[int | None] = [None] if space == "original" else pca_components
                for pca_n in pca_list:
                    for algorithm in algorithms:
                        for k in range(k_min, k_max + 1):
                            out.append(
                                {
                                    "feature_set": feature_set,
                                    "scaler": scaler,
                                    "space_type": space,
                                    "pca_n_components": pca_n,
                                    "algorithm": algorithm,
                                    "n_clusters_requested": k,
                                }
                            )
    return out


def _fit_cluster_model(
    matrix: np.ndarray,
    algorithm: str,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    algo = str(algorithm).lower()
    if algo == "gmm":
        model = GaussianMixture(
            n_components=int(n_clusters),
            covariance_type="full",
            random_state=int(random_state),
            n_init=5,
        )
        labels = model.fit_predict(matrix)
        probs = model.predict_proba(matrix)
        conf = np.max(probs, axis=1)
        return labels.astype(int), conf.astype(float), probs.astype(float)
    if algo == "agglomerative":
        model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage="ward")
        labels = model.fit_predict(matrix)
        return labels.astype(int), None, None
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_configuration(
    feature_table: pd.DataFrame,
    scaler: str,
    space_type: str,
    pca_n_components: int | None,
    algorithm: str,
    n_clusters_requested: int,
    random_state: int,
) -> ClusteringFitResult:
    _, feature_df = split_metadata_and_features(feature_table)
    prep = preprocess_feature_table(
        feature_df=feature_df,
        scaler=scaler,
        space_type=space_type,
        pca_n_components=pca_n_components,
        random_state=random_state,
    )
    labels, confidence, probabilities = _fit_cluster_model(
        matrix=prep.matrix,
        algorithm=algorithm,
        n_clusters=n_clusters_requested,
        random_state=random_state,
    )
    return ClusteringFitResult(
        labels=labels,
        assignment_confidence=confidence,
        assignment_probabilities=probabilities,
        preprocessing_info={
            "n_features_input": prep.n_features_input,
            "n_features_used": prep.n_features_used,
            "n_missing_imputed": prep.n_missing_imputed,
            "explained_variance_ratio_sum": prep.explained_variance_ratio_sum,
        },
        matrix=prep.matrix,
    )


def run_grid_search(
    feature_tables: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> pd.DataFrame:
    configs = build_grid_configurations(cfg)
    total = len(configs)
    rows: List[Dict[str, Any]] = []
    small_cluster_cfg = cfg["small_cluster_threshold"]
    random_state = int(cfg["stability"].get("random_state", 42))

    for i, conf in enumerate(configs, start=1):
        feature_set = conf["feature_set"]
        table = feature_tables[feature_set]
        config_id = f"cfg_{i:04d}"
        if logger and (i == 1 or i % 20 == 0 or i == total):
            logger.info("Grid progress: %d/%d", i, total)

        base_row = {
            "config_id": config_id,
            **conf,
            "n_rows": int(len(table)),
            "ranking_score": np.nan,
            "notes": "",
        }
        try:
            fit = fit_configuration(
                feature_table=table,
                scaler=conf["scaler"],
                space_type=conf["space_type"],
                pca_n_components=conf["pca_n_components"],
                algorithm=conf["algorithm"],
                n_clusters_requested=conf["n_clusters_requested"],
                random_state=random_state,
            )
            metrics = compute_cluster_metrics(
                matrix=fit.matrix,
                labels=fit.labels,
                small_cluster_mode=str(small_cluster_cfg.get("mode", "relative")),
                small_cluster_value=float(small_cluster_cfg.get("value", 0.05)),
            )
            if conf["algorithm"] == "gmm":
                conf_metrics = compute_gmm_confidence_metrics(
                    fit.assignment_probabilities if fit.assignment_probabilities is not None else np.empty((0, 0)),
                )
                if fit.assignment_confidence is not None:
                    conf_metrics["avg_assignment_confidence"] = float(np.mean(fit.assignment_confidence))
                    conf_metrics["min_assignment_confidence"] = float(np.min(fit.assignment_confidence))
            else:
                conf_metrics = {
                    "avg_assignment_confidence": np.nan,
                    "min_assignment_confidence": np.nan,
                    "entropy_of_assignment": np.nan,
                }

            row = {
                **base_row,
                **metrics,
                **fit.preprocessing_info,
                **conf_metrics,
            }
            rows.append(row)
        except Exception as exc:
            row = {
                **base_row,
                "n_clusters_actual": np.nan,
                "silhouette_score": np.nan,
                "davies_bouldin_score": np.nan,
                "calinski_harabasz_score": np.nan,
                "cluster_size_min": np.nan,
                "cluster_size_max": np.nan,
                "cluster_size_ratio_max": np.nan,
                "n_singleton_clusters": np.nan,
                "n_small_clusters": np.nan,
                "n_features_input": np.nan,
                "n_features_used": np.nan,
                "n_missing_imputed": np.nan,
                "explained_variance_ratio_sum": np.nan,
                "avg_assignment_confidence": np.nan,
                "min_assignment_confidence": np.nan,
                "entropy_of_assignment": np.nan,
                "notes": f"error: {type(exc).__name__}: {exc}",
            }
            rows.append(row)
            if logger:
                logger.exception("Grid config failed: %s", config_id)

    out = pd.DataFrame(rows)
    cols = [
        "config_id",
        "feature_set",
        "scaler",
        "space_type",
        "pca_n_components",
        "algorithm",
        "n_clusters_requested",
        "n_clusters_actual",
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "cluster_size_min",
        "cluster_size_max",
        "cluster_size_ratio_max",
        "n_singleton_clusters",
        "n_small_clusters",
        "n_rows",
        "n_features_input",
        "n_features_used",
        "n_missing_imputed",
        "explained_variance_ratio_sum",
        "avg_assignment_confidence",
        "min_assignment_confidence",
        "entropy_of_assignment",
        "ranking_score",
        "notes",
    ]
    return out.reindex(columns=cols)
