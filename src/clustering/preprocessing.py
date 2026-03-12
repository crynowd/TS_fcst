from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, RobustScaler


METADATA_COLUMNS = ["series_id", "ticker", "market", "dataset_profile"]


@dataclass(frozen=True)
class PreprocessingResult:
    matrix: np.ndarray
    feature_names: List[str]
    n_features_input: int
    n_features_used: int
    n_missing_imputed: int
    explained_variance_ratio_sum: float


def split_metadata_and_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    metadata_df = df[metadata_cols].copy()
    feature_df = df.drop(columns=metadata_cols, errors="ignore").copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    return metadata_df, feature_df


def apply_median_imputation(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    out = feature_df.copy()
    n_missing = int(out.isna().sum().sum())
    if n_missing == 0:
        return out, 0
    medians = out.median(axis=0, numeric_only=True)
    out = out.fillna(medians)
    return out, n_missing


def apply_scaler(matrix: np.ndarray, scaler: str, random_state: int = 42) -> np.ndarray:
    name = str(scaler).strip().lower()
    if name == "identity":
        return matrix
    if name == "robust":
        transformer = RobustScaler()
        return transformer.fit_transform(matrix)
    if name == "quantile":
        n_quantiles = max(10, min(1000, matrix.shape[0]))
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=int(random_state),
        )
        return transformer.fit_transform(matrix)
    raise ValueError(f"Unsupported scaler: {scaler}")


def apply_feature_space(
    matrix: np.ndarray,
    space_type: str,
    pca_n_components: int | None,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    name = str(space_type).strip().lower()
    if name == "original":
        return matrix, np.nan
    if name == "pca":
        if pca_n_components is None:
            raise ValueError("pca_n_components must be provided for PCA space.")
        pca = PCA(n_components=int(pca_n_components), random_state=int(random_state))
        transformed = pca.fit_transform(matrix)
        return transformed, float(np.sum(pca.explained_variance_ratio_))
    raise ValueError(f"Unsupported space_type: {space_type}")


def preprocess_feature_table(
    feature_df: pd.DataFrame,
    scaler: str,
    space_type: str,
    pca_n_components: int | None,
    random_state: int = 42,
) -> PreprocessingResult:
    feature_cols = feature_df.columns.tolist()
    n_features_input = len(feature_cols)
    imputed_df, n_missing = apply_median_imputation(feature_df)
    matrix = imputed_df.to_numpy(dtype=float, copy=True)
    matrix = apply_scaler(matrix, scaler=scaler, random_state=random_state)
    matrix, explained_var = apply_feature_space(
        matrix,
        space_type=space_type,
        pca_n_components=pca_n_components,
        random_state=random_state,
    )
    n_features_used = int(matrix.shape[1])
    return PreprocessingResult(
        matrix=matrix,
        feature_names=feature_cols,
        n_features_input=n_features_input,
        n_features_used=n_features_used,
        n_missing_imputed=n_missing,
        explained_variance_ratio_sum=explained_var,
    )

