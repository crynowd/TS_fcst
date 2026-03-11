from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.reporting.excel_export import export_final_feature_sets_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


METADATA_COLUMNS = ["series_id", "ticker", "market", "dataset_profile"]

BASE_FEATURES_BY_BLOCK: Dict[str, List[str]] = {
    "A": [
        "hurst_rs",
        "hurst_dfa",
        "acf_lag_2",
        "acf_lag_5",
        "acf_lag_10",
        "acf_lag_25",
        "acf_lag_50",
        "acf_lag_100",
        "abs_acf_lag_2",
        "abs_acf_lag_5",
        "abs_acf_lag_10",
        "abs_acf_lag_25",
        "abs_acf_lag_50",
        "vr_q10",
        "lb_ret_stat_50",
    ],
    "B": [
        "lz_complexity",
        "permutation_entropy",
        "spectral_flatness",
    ],
    "C": [
        "kurtosis",
        "robust_kurtosis",
        "tail_ratio_upper",
        "hill_tail_index",
    ],
}

CHAOS_FEATURES_BY_BLOCK: Dict[str, List[str]] = {
    "D": [
        "correlation_dimension",
        "embedding_dimension",
        "selected_delay_tau",
    ],
}


@dataclass(frozen=True)
class FinalFeatureSetsOutputs:
    run_id: str
    base_parquet_path: str
    with_chaos_parquet_path: str
    base_list_csv_path: str
    with_chaos_list_csv_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    rows: int
    base_feature_count: int
    with_chaos_feature_count: int
    breakdown_base: Dict[str, int]
    breakdown_with_chaos: Dict[str, int]


def _flatten_features(feature_map: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for feats in feature_map.values():
        out.extend(feats)
    return out


def _feature_list_table(feature_map: Dict[str, List[str]], set_name: str) -> pd.DataFrame:
    rows = []
    for block, features in feature_map.items():
        for feature in features:
            rows.append({"feature_name": feature, "source_block": block, "set_name": set_name})
    return pd.DataFrame(rows)


def _validate_columns_exist(df: pd.DataFrame, required: Sequence[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def build_final_feature_tables(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_features = _flatten_features(BASE_FEATURES_BY_BLOCK)
    chaos_features = _flatten_features(CHAOS_FEATURES_BY_BLOCK)

    required_columns = METADATA_COLUMNS + base_features + chaos_features
    missing = _validate_columns_exist(master_df, required_columns)
    if missing:
        raise ValueError(f"Missing required features/columns in master table: {missing}")

    base_columns = METADATA_COLUMNS + base_features
    with_chaos_columns = METADATA_COLUMNS + base_features + chaos_features
    return master_df[base_columns].copy(), master_df[with_chaos_columns].copy()


def run_final_feature_sets_pipeline(
    master_path: str = "artifacts/features/features_master_v1.parquet",
    project_root: str | None = None,
) -> FinalFeatureSetsOutputs:
    project_root_path = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    artifacts_dir = project_root_path / "artifacts"
    features_dir = artifacts_dir / "features"
    reports_dir = artifacts_dir / "reports"
    logs_dir = artifacts_dir / "logs"
    manifests_dir = artifacts_dir / "manifests"
    for d in [features_dir, reports_dir, logs_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_id = f"final_feature_sets_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)
    logger, log_path = setup_logger(run_id=run_id, logs_dir=logs_dir)

    master_abs_path = Path(master_path).resolve() if Path(master_path).is_absolute() else (project_root_path / master_path).resolve()
    logger.info("Final feature sets stage started")
    logger.info("Master table path: %s", master_abs_path)

    master_df = pd.read_parquet(master_abs_path)
    logger.info("Master table rows: %d", len(master_df))

    base_features = _flatten_features(BASE_FEATURES_BY_BLOCK)
    with_chaos_features = base_features + _flatten_features(CHAOS_FEATURES_BY_BLOCK)
    missing = _validate_columns_exist(master_df, METADATA_COLUMNS + with_chaos_features)
    if missing:
        logger.error("Missing required features: %s", missing)
        raise ValueError(f"Missing required features: {missing}")

    base_df, with_chaos_df = build_final_feature_tables(master_df)

    base_list_df = _feature_list_table(BASE_FEATURES_BY_BLOCK, "final_clustering_features_base_v1")
    with_chaos_map = dict(BASE_FEATURES_BY_BLOCK)
    with_chaos_map.update(CHAOS_FEATURES_BY_BLOCK)
    with_chaos_list_df = _feature_list_table(with_chaos_map, "final_clustering_features_with_chaos_v1")

    base_parquet_path = features_dir / "final_clustering_features_base_v1.parquet"
    with_chaos_parquet_path = features_dir / "final_clustering_features_with_chaos_v1.parquet"
    base_list_csv_path = reports_dir / "final_clustering_features_base_list_v1.csv"
    with_chaos_list_csv_path = reports_dir / "final_clustering_features_with_chaos_list_v1.csv"
    excel_path = reports_dir / "final_clustering_feature_sets_v1.xlsx"

    base_df.to_parquet(base_parquet_path, index=False)
    with_chaos_df.to_parquet(with_chaos_parquet_path, index=False)
    base_list_df.to_csv(base_list_csv_path, index=False)
    with_chaos_list_df.to_csv(with_chaos_list_csv_path, index=False)

    breakdown_base = {k: len(v) for k, v in BASE_FEATURES_BY_BLOCK.items()}
    breakdown_with_chaos = dict(breakdown_base)
    for block, feats in CHAOS_FEATURES_BY_BLOCK.items():
        breakdown_with_chaos[block] = len(feats)

    summary = {
        "rows": int(len(master_df)),
        "base_feature_count": int(len(base_features)),
        "with_chaos_feature_count": int(len(with_chaos_features)),
        "breakdown_base": breakdown_base,
        "breakdown_with_chaos": breakdown_with_chaos,
    }
    excel_out = export_final_feature_sets_excel(
        excel_path=excel_path,
        run_id=run_id,
        summary=summary,
        base_feature_set_df=base_list_df,
        with_chaos_feature_set_df=with_chaos_list_df,
        input_master_path=str(master_abs_path),
        output_paths={
            "base_parquet": str(base_parquet_path),
            "with_chaos_parquet": str(with_chaos_parquet_path),
            "base_list_csv": str(base_list_csv_path),
            "with_chaos_list_csv": str(with_chaos_list_csv_path),
        },
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Base set features: %d", len(base_features))
    logger.info("With-chaos set features: %d", len(with_chaos_features))
    logger.info("Base parquet path: %s", base_parquet_path)
    logger.info("With-chaos parquet path: %s", with_chaos_parquet_path)
    logger.info("Excel report path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "final_feature_sets",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(project_root_path),
        "input_sources": {"features_master_parquet": str(master_abs_path)},
        "outputs": {
            "base_parquet": str(base_parquet_path),
            "with_chaos_parquet": str(with_chaos_parquet_path),
            "base_list_csv": str(base_list_csv_path),
            "with_chaos_list_csv": str(with_chaos_list_csv_path),
            "excel_report": str(excel_out),
            "log": str(log_path),
        },
        "summary": summary,
    }
    manifest_path = write_manifest(manifest, manifests_dir=manifests_dir, run_id=run_id)

    return FinalFeatureSetsOutputs(
        run_id=run_id,
        base_parquet_path=str(base_parquet_path),
        with_chaos_parquet_path=str(with_chaos_parquet_path),
        base_list_csv_path=str(base_list_csv_path),
        with_chaos_list_csv_path=str(with_chaos_list_csv_path),
        excel_path=str(excel_out),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        rows=int(len(master_df)),
        base_feature_count=len(base_features),
        with_chaos_feature_count=len(with_chaos_features),
        breakdown_base=breakdown_base,
        breakdown_with_chaos=breakdown_with_chaos,
    )
