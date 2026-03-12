from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def _small_cluster_threshold_count(n_rows: int, mode: str, value: float) -> int:
    if str(mode).lower() == "relative":
        return max(1, int(np.ceil(float(value) * n_rows)))
    return max(1, int(value))


def compute_cluster_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    small_cluster_mode: str,
    small_cluster_value: float,
) -> Dict[str, float]:
    n_rows = int(len(labels))
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters_actual = int(len(unique))

    cluster_size_min = int(np.min(counts)) if counts.size else 0
    cluster_size_max = int(np.max(counts)) if counts.size else 0
    cluster_size_ratio_max = float(cluster_size_max / n_rows) if n_rows > 0 else np.nan

    small_thr = _small_cluster_threshold_count(n_rows, small_cluster_mode, small_cluster_value)
    n_singleton_clusters = int(np.sum(counts == 1))
    n_small_clusters = int(np.sum(counts < small_thr))

    if n_clusters_actual <= 1 or n_clusters_actual >= n_rows:
        silhouette = np.nan
        dbi = np.nan
        ch = np.nan
    else:
        silhouette = float(silhouette_score(matrix, labels))
        dbi = float(davies_bouldin_score(matrix, labels))
        ch = float(calinski_harabasz_score(matrix, labels))

    return {
        "n_clusters_actual": n_clusters_actual,
        "silhouette_score": silhouette,
        "davies_bouldin_score": dbi,
        "calinski_harabasz_score": ch,
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "cluster_size_ratio_max": cluster_size_ratio_max,
        "n_singleton_clusters": n_singleton_clusters,
        "n_small_clusters": n_small_clusters,
    }


def compute_gmm_confidence_metrics(probabilities: np.ndarray) -> Dict[str, float]:
    if probabilities.size == 0:
        return {
            "avg_assignment_confidence": np.nan,
            "min_assignment_confidence": np.nan,
            "entropy_of_assignment": np.nan,
        }
    max_prob = np.max(probabilities, axis=1)
    safe = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)
    return {
        "avg_assignment_confidence": float(np.mean(max_prob)),
        "min_assignment_confidence": float(np.min(max_prob)),
        "entropy_of_assignment": float(np.mean(entropy)),
    }

