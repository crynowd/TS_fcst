from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.final_feature_sets import (
    BASE_FEATURES_BY_BLOCK,
    CHAOS_FEATURES_BY_BLOCK,
    METADATA_COLUMNS,
    build_final_feature_tables,
    run_final_feature_sets_pipeline,
)


def _all_feature_names() -> list[str]:
    feats = []
    for v in BASE_FEATURES_BY_BLOCK.values():
        feats.extend(v)
    for v in CHAOS_FEATURES_BY_BLOCK.values():
        feats.extend(v)
    return feats


def _synthetic_master() -> pd.DataFrame:
    rows = []
    all_feats = _all_feature_names()
    for i in range(3):
        row = {
            "series_id": f"S{i+1}",
            "ticker": f"T{i+1}",
            "market": "RU" if i == 0 else "US",
            "dataset_profile": "core_balanced",
        }
        for j, f in enumerate(all_feats):
            row[f] = float(i + j + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_final_base_set_columns_exist() -> None:
    master = _synthetic_master()
    base_df, _ = build_final_feature_tables(master)
    base_features = [f for feats in BASE_FEATURES_BY_BLOCK.values() for f in feats]
    for col in base_features:
        assert col in base_df.columns


def test_final_with_chaos_set_columns_exist() -> None:
    master = _synthetic_master()
    _, chaos_df = build_final_feature_tables(master)
    all_features = _all_feature_names()
    for col in all_features:
        assert col in chaos_df.columns


def test_metadata_columns_preserved() -> None:
    master = _synthetic_master()
    base_df, chaos_df = build_final_feature_tables(master)
    for col in METADATA_COLUMNS:
        assert col in base_df.columns
        assert col in chaos_df.columns


def test_final_feature_sets_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    (root / "artifacts" / "features").mkdir(parents=True)
    (root / "artifacts" / "reports").mkdir(parents=True)
    (root / "artifacts" / "logs").mkdir(parents=True)
    (root / "artifacts" / "manifests").mkdir(parents=True)

    master_path = root / "artifacts" / "features" / "features_master_v1.parquet"
    _synthetic_master().to_parquet(master_path, index=False)

    result = run_final_feature_sets_pipeline(master_path=str(master_path), project_root=str(root))
    assert Path(result.base_parquet_path).exists()
    assert Path(result.with_chaos_parquet_path).exists()
    assert Path(result.base_list_csv_path).exists()
    assert Path(result.with_chaos_list_csv_path).exists()
    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()
