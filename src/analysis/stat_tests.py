from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal


def run_kruskal_tests(
    series_joined_df: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    rows = []
    key_cols = ["clustering_config", "model_name", "horizon"]

    for (cfg, model_name, horizon), gdf in series_joined_df.groupby(key_cols, sort=True):
        for metric in metrics:
            metric_col = "directional_accuracy_mean" if str(metric) == "directional_accuracy" else f"{metric}_mean"
            if metric_col not in gdf.columns:
                continue

            samples = []
            for _, cdf in gdf.groupby("cluster_label", sort=True):
                vals = pd.to_numeric(cdf[metric_col], errors="coerce").dropna().values
                if len(vals) > 0:
                    samples.append(vals)

            if len(samples) < 2:
                rows.append(
                    {
                        "clustering_config": cfg,
                        "model_name": model_name,
                        "horizon": int(horizon),
                        "metric_name": str(metric),
                        "test_name": "kruskal",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "notes": "insufficient_clusters_for_test",
                    }
                )
                continue

            try:
                stat, p_value = kruskal(*samples, nan_policy="omit")
                rows.append(
                    {
                        "clustering_config": cfg,
                        "model_name": model_name,
                        "horizon": int(horizon),
                        "metric_name": str(metric),
                        "test_name": "kruskal",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "notes": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "clustering_config": cfg,
                        "model_name": model_name,
                        "horizon": int(horizon),
                        "metric_name": str(metric),
                        "test_name": "kruskal",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "notes": f"kruskal_failed:{exc}",
                    }
                )

    return pd.DataFrame(rows).sort_values(["clustering_config", "horizon", "model_name", "metric_name"], kind="stable").reset_index(drop=True)
