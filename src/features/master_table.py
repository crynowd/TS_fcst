from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd


KEY_COLUMNS = ["series_id", "ticker", "market", "dataset_profile"]
BLOCKS = ("A", "B", "C", "D")
BLOCK_STATUS_COL = "feature_status"
BLOCK_WARNINGS_COL = "feature_warning_flags"
BLOCK_NOBS_COL = "n_obs_used"


@dataclass(frozen=True)
class MasterTableBuildResult:
    master_df: pd.DataFrame
    feature_to_block: Dict[str, str]
    block_feature_columns: Dict[str, List[str]]
    input_rows_by_block: Dict[str, int]


def _warning_count(value: object) -> int:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return 0
    return len([token for token in raw.split(";") if token.strip()])


def _prepare_block_frame(
    block: str,
    df: pd.DataFrame,
    occupied_columns: set[str],
) -> tuple[pd.DataFrame, List[str], Dict[str, str]]:
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing_keys:
        raise ValueError(f"Block {block} is missing key columns: {missing_keys}")

    block_df = df.copy()
    rename_map: Dict[str, str] = {}
    block_features: List[str] = []

    for source_col, target_tpl in (
        (BLOCK_STATUS_COL, f"block_{block}_status"),
        (BLOCK_WARNINGS_COL, f"block_{block}_warning_flags"),
        (BLOCK_NOBS_COL, f"block_{block}_n_obs_used"),
    ):
        if source_col in block_df.columns:
            rename_map[source_col] = target_tpl
    if "feature_block" in block_df.columns:
        rename_map["feature_block"] = f"block_{block}_label"

    for col in block_df.columns:
        if col in KEY_COLUMNS:
            continue
        if col in rename_map:
            continue
        target = col
        if target in occupied_columns:
            target = f"{col}__block_{block}"
            rename_map[col] = target
        block_features.append(target)

    block_df = block_df.rename(columns=rename_map)
    occupied_columns.update(c for c in block_df.columns if c not in KEY_COLUMNS)
    return block_df, block_features, rename_map


def build_master_feature_table(
    block_tables: Mapping[str, pd.DataFrame],
    dataset_profile: str | None = None,
) -> MasterTableBuildResult:
    ordered_blocks = [b for b in BLOCKS if b in block_tables]
    if not ordered_blocks:
        raise ValueError("No feature block tables provided.")

    frames: Dict[str, pd.DataFrame] = {}
    input_rows_by_block: Dict[str, int] = {}
    for block in ordered_blocks:
        frame = block_tables[block].copy()
        if dataset_profile is not None and "dataset_profile" in frame.columns:
            frame = frame[frame["dataset_profile"] == dataset_profile].copy()
        input_rows_by_block[block] = int(len(frame))
        frames[block] = frame

    occupied_columns: set[str] = set()
    feature_to_block: Dict[str, str] = {}
    block_feature_columns: Dict[str, List[str]] = {}

    prepared: Dict[str, pd.DataFrame] = {}
    for block in ordered_blocks:
        prepared_df, feature_cols, _ = _prepare_block_frame(block, frames[block], occupied_columns)
        prepared[block] = prepared_df
        block_feature_columns[block] = feature_cols
        for col in feature_cols:
            feature_to_block[col] = block

    master_df = prepared[ordered_blocks[0]]
    for block in ordered_blocks[1:]:
        master_df = master_df.merge(prepared[block], on=KEY_COLUMNS, how="outer", validate="one_to_one")

    for block in ordered_blocks:
        status_col = f"block_{block}_status"
        warnings_col = f"block_{block}_warning_flags"
        if status_col not in master_df.columns:
            master_df[status_col] = ""
        if warnings_col not in master_df.columns:
            master_df[warnings_col] = ""
        master_df[warnings_col] = master_df[warnings_col].fillna("").astype(str)
        master_df[f"block_{block}_warning_count"] = master_df[warnings_col].map(_warning_count).astype(int)

    warning_count_cols = [f"block_{b}_warning_count" for b in ordered_blocks]
    master_df["overall_warning_count"] = master_df[warning_count_cols].sum(axis=1).astype(int)
    master_df["warning_blocks_count"] = (
        master_df[[f"block_{b}_warning_flags" for b in ordered_blocks]]
        .apply(lambda row: sum(bool(str(v).strip()) for v in row), axis=1)
        .astype(int)
    )

    sort_cols = [c for c in ["dataset_profile", "market", "ticker", "series_id"] if c in master_df.columns]
    if sort_cols:
        master_df = master_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return MasterTableBuildResult(
        master_df=master_df,
        feature_to_block=feature_to_block,
        block_feature_columns=block_feature_columns,
        input_rows_by_block=input_rows_by_block,
    )


def load_feature_block_tables(
    input_paths: Mapping[str, str],
    dataset_profile: str | None = None,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for block in BLOCKS:
        key = f"features_block_{block}_parquet"
        if key not in input_paths:
            continue
        path = Path(str(input_paths[key])).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing feature block input: {path}")
        frame = pd.read_parquet(path)
        if dataset_profile is not None and "dataset_profile" in frame.columns:
            frame = frame[frame["dataset_profile"] == dataset_profile].copy()
        tables[block] = frame
    return tables
