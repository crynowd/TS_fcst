from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SeriesFeatureResult:
    """Unified container for one series feature computation output."""

    series_id: str
    ticker: str
    market: str
    dataset_profile: str
    feature_block: str
    n_obs_used: int
    metrics: Dict[str, float] = field(default_factory=dict)
    warning_flags: List[str] = field(default_factory=list)
    status: str = "success"

    def add_warning(self, flag: str) -> None:
        if flag and flag not in self.warning_flags:
            self.warning_flags.append(flag)
        if self.status == "success":
            self.status = "warning"

    def as_row(self, metric_columns: List[str]) -> Dict[str, object]:
        row: Dict[str, object] = {
            "series_id": self.series_id,
            "ticker": self.ticker,
            "market": self.market,
            "dataset_profile": self.dataset_profile,
            "n_obs_used": int(self.n_obs_used),
            "feature_block": self.feature_block,
            "feature_status": self.status,
            "feature_warning_flags": ";".join(self.warning_flags),
        }
        for col in metric_columns:
            value = self.metrics.get(col, np.nan)
            row[col] = value
        return row

