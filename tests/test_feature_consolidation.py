from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.features.consolidation import (
    build_candidate_pool,
    build_correlation_groups,
    run_feature_consolidation_pipeline,
    select_group_representatives,
)


def _shortlist_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_name": "feat_a_keep",
                "source_block": "A",
                "screening_status": "keep_candidate",
                "missing_rate": 0.0,
                "warning_coverage": 0.0,
                "std": 1.0,
            },
            {
                "feature_name": "feat_a_corr",
                "source_block": "A",
                "screening_status": "review_highly_correlated",
                "missing_rate": 0.05,
                "warning_coverage": 0.0,
                "std": 0.8,
            },
            {
                "feature_name": "feat_b_keep",
                "source_block": "B",
                "screening_status": "keep_candidate",
                "missing_rate": 0.0,
                "warning_coverage": 0.0,
                "std": 1.1,
            },
            {
                "feature_name": "feat_d_reserve",
                "source_block": "D",
                "screening_status": "reserve_candidate",
                "missing_rate": 0.0,
                "warning_coverage": 0.35,
                "std": 0.4,
            },
            {
                "feature_name": "feat_low_var",
                "source_block": "B",
                "screening_status": "review_low_variance",
                "missing_rate": 0.0,
                "warning_coverage": 0.0,
                "std": 0.0,
            },
        ]
    )


def _corr_pairs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_1": "feat_a_keep",
                "feature_2": "feat_a_corr",
                "pearson_corr": 0.95,
                "spearman_corr": 0.94,
                "suggested_relation": "strongly_related",
            },
            {
                "feature_1": "feat_a_corr",
                "feature_2": "feat_b_keep",
                "pearson_corr": 0.91,
                "spearman_corr": 0.90,
                "suggested_relation": "strongly_related",
            },
        ]
    )


def test_candidate_pool_selection() -> None:
    shortlist = _shortlist_df()
    pool = build_candidate_pool(
        shortlist_df=shortlist,
        include_statuses=["keep_candidate", "review_highly_correlated", "reserve_candidate"],
        excluded_statuses=["review_low_variance", "review_high_missing", "drop_candidate"],
    )
    included = set(pool.loc[pool["included_in_pool"], "feature_name"])
    assert included == {"feat_a_keep", "feat_a_corr", "feat_b_keep", "feat_d_reserve"}
    assert "feat_low_var" not in included


def test_correlation_group_building() -> None:
    shortlist = _shortlist_df()
    candidate_features = {"feat_a_keep", "feat_a_corr", "feat_b_keep", "feat_d_reserve"}
    groups_df, group_items = build_correlation_groups(
        corr_pairs_df=_corr_pairs_df(),
        candidate_features=candidate_features,
        shortlist_df=shortlist,
        threshold=0.90,
    )
    assert len(groups_df) == 1
    members = set(group_items[0]["members"])
    assert members == {"feat_a_keep", "feat_a_corr", "feat_b_keep"}


def test_representative_selection_prefers_keep_candidate() -> None:
    shortlist = _shortlist_df()
    group_items = [{"correlation_group_id": "G001", "members": ["feat_a_keep", "feat_a_corr"]}]
    rep_df, selected = select_group_representatives(
        group_items=group_items,
        shortlist_df=shortlist,
        status_priority=["keep_candidate", "reserve_candidate", "review_highly_correlated"],
    )
    assert "feat_a_keep" in selected
    assert rep_df.iloc[0]["chosen_representative"] == "feat_a_keep"


def test_base_set_excludes_chaos_features(tmp_path: Path) -> None:
    root = tmp_path
    (root / "artifacts" / "features").mkdir(parents=True)
    (root / "artifacts" / "reports").mkdir(parents=True)
    (root / "artifacts" / "logs").mkdir(parents=True)
    (root / "artifacts" / "manifests").mkdir(parents=True)
    (root / "artifacts" / "processed").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)

    master = pd.DataFrame(
        [
            {"series_id": "S1", "ticker": "A", "market": "RU", "dataset_profile": "core_balanced", "feat_a_keep": 1.0, "feat_d_reserve": 0.1},
            {"series_id": "S2", "ticker": "B", "market": "US", "dataset_profile": "core_balanced", "feat_a_keep": 2.0, "feat_d_reserve": 0.2},
        ]
    )
    shortlist = _shortlist_df()[["feature_name", "source_block", "screening_status", "missing_rate", "warning_coverage", "std"]]
    corr = pd.DataFrame(columns=["feature_1", "feature_2", "pearson_corr", "spearman_corr", "suggested_relation"])
    master.to_parquet(root / "artifacts" / "features" / "features_master_v1.parquet", index=False)
    shortlist.to_csv(root / "artifacts" / "reports" / "screening_shortlist_v1.csv", index=False)
    corr.to_csv(root / "artifacts" / "reports" / "high_correlation_pairs_v1.csv", index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "interim": str(root / "artifacts" / "interim"),
            "processed": str(root / "artifacts" / "processed"),
            "features": str(root / "artifacts" / "features"),
            "reports": str(root / "artifacts" / "reports"),
            "logs": str(root / "artifacts" / "logs"),
            "manifests": str(root / "artifacts" / "manifests"),
        },
    }
    (root / "configs" / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    cfg = {
        "run_name": "feature_consolidation_v1",
        "stage": "feature_consolidation",
        "input": {
            "features_master_parquet": "artifacts/features/features_master_v1.parquet",
            "high_correlation_pairs_csv": "artifacts/reports/high_correlation_pairs_v1.csv",
            "screening_shortlist_csv": "artifacts/reports/screening_shortlist_v1.csv",
        },
        "dataset_profile": "core_balanced",
        "include_screening_statuses": ["keep_candidate", "review_highly_correlated", "reserve_candidate"],
        "excluded_screening_statuses": ["review_low_variance", "review_high_missing", "drop_candidate"],
        "high_correlation_threshold": 0.9,
        "representative_selection_priority": ["keep_candidate", "reserve_candidate", "review_highly_correlated"],
        "include_chaos_in_extended_set": True,
        "output": {
            "base_parquet_name": "clustering_features_base_v1.parquet",
            "with_chaos_parquet_name": "clustering_features_with_chaos_v1.parquet",
            "correlation_groups_csv_name": "correlation_groups_v1.csv",
            "base_feature_list_csv_name": "clustering_features_base_list_v1.csv",
            "with_chaos_feature_list_csv_name": "clustering_features_with_chaos_list_v1.csv",
            "scaling_plan_csv_name": "clustering_scaling_plan_v1.csv",
            "excel_name": "feature_consolidation_v1.xlsx",
        },
    }
    cfg_path = root / "configs" / "feature_consolidation_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_consolidation_pipeline(str(cfg_path))
    base_df = pd.read_parquet(result.base_parquet_path)
    assert "feat_d_reserve" not in base_df.columns


def test_with_chaos_set_includes_selected_chaos_features(tmp_path: Path) -> None:
    root = tmp_path
    (root / "artifacts" / "features").mkdir(parents=True)
    (root / "artifacts" / "reports").mkdir(parents=True)
    (root / "artifacts" / "logs").mkdir(parents=True)
    (root / "artifacts" / "manifests").mkdir(parents=True)
    (root / "artifacts" / "processed").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)

    master = pd.DataFrame(
        [
            {"series_id": "S1", "ticker": "A", "market": "RU", "dataset_profile": "core_balanced", "feat_a_keep": 1.0, "feat_d_reserve": 0.1},
            {"series_id": "S2", "ticker": "B", "market": "US", "dataset_profile": "core_balanced", "feat_a_keep": 2.0, "feat_d_reserve": 0.2},
        ]
    )
    shortlist = _shortlist_df()[["feature_name", "source_block", "screening_status", "missing_rate", "warning_coverage", "std"]]
    corr = pd.DataFrame(columns=["feature_1", "feature_2", "pearson_corr", "spearman_corr", "suggested_relation"])
    master.to_parquet(root / "artifacts" / "features" / "features_master_v1.parquet", index=False)
    shortlist.to_csv(root / "artifacts" / "reports" / "screening_shortlist_v1.csv", index=False)
    corr.to_csv(root / "artifacts" / "reports" / "high_correlation_pairs_v1.csv", index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "interim": str(root / "artifacts" / "interim"),
            "processed": str(root / "artifacts" / "processed"),
            "features": str(root / "artifacts" / "features"),
            "reports": str(root / "artifacts" / "reports"),
            "logs": str(root / "artifacts" / "logs"),
            "manifests": str(root / "artifacts" / "manifests"),
        },
    }
    (root / "configs" / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    cfg = {
        "run_name": "feature_consolidation_v1",
        "stage": "feature_consolidation",
        "input": {
            "features_master_parquet": "artifacts/features/features_master_v1.parquet",
            "high_correlation_pairs_csv": "artifacts/reports/high_correlation_pairs_v1.csv",
            "screening_shortlist_csv": "artifacts/reports/screening_shortlist_v1.csv",
        },
        "dataset_profile": "core_balanced",
        "include_screening_statuses": ["keep_candidate", "review_highly_correlated", "reserve_candidate"],
        "excluded_screening_statuses": ["review_low_variance", "review_high_missing", "drop_candidate"],
        "high_correlation_threshold": 0.9,
        "representative_selection_priority": ["keep_candidate", "reserve_candidate", "review_highly_correlated"],
        "include_chaos_in_extended_set": True,
        "output": {
            "base_parquet_name": "clustering_features_base_v1.parquet",
            "with_chaos_parquet_name": "clustering_features_with_chaos_v1.parquet",
            "correlation_groups_csv_name": "correlation_groups_v1.csv",
            "base_feature_list_csv_name": "clustering_features_base_list_v1.csv",
            "with_chaos_feature_list_csv_name": "clustering_features_with_chaos_list_v1.csv",
            "scaling_plan_csv_name": "clustering_scaling_plan_v1.csv",
            "excel_name": "feature_consolidation_v1.xlsx",
        },
    }
    cfg_path = root / "configs" / "feature_consolidation_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_consolidation_pipeline(str(cfg_path))
    chaos_df = pd.read_parquet(result.with_chaos_parquet_path)
    assert "feat_d_reserve" in chaos_df.columns


def test_feature_consolidation_stage_small_e2e(tmp_path: Path) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    artifacts_dir = root / "artifacts"
    features_dir = artifacts_dir / "features"
    reports_dir = artifacts_dir / "reports"
    logs_dir = artifacts_dir / "logs"
    manifests_dir = artifacts_dir / "manifests"
    processed_dir = artifacts_dir / "processed"
    for d in [configs_dir, features_dir, reports_dir, logs_dir, manifests_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        [
            {
                "series_id": "S1",
                "ticker": "AAA",
                "market": "RU",
                "dataset_profile": "core_balanced",
                "feat_a_keep": 1.0,
                "feat_a_corr": 1.1,
                "feat_b_keep": 5.0,
                "feat_d_reserve": 0.1,
            },
            {
                "series_id": "S2",
                "ticker": "BBB",
                "market": "US",
                "dataset_profile": "core_balanced",
                "feat_a_keep": 2.0,
                "feat_a_corr": 2.1,
                "feat_b_keep": 5.5,
                "feat_d_reserve": 0.2,
            },
            {
                "series_id": "S3",
                "ticker": "CCC",
                "market": "US",
                "dataset_profile": "core_balanced",
                "feat_a_keep": 3.0,
                "feat_a_corr": 3.1,
                "feat_b_keep": 6.0,
                "feat_d_reserve": 0.3,
            },
        ]
    )
    master.to_parquet(features_dir / "features_master_v1.parquet", index=False)

    shortlist = _shortlist_df()[["feature_name", "source_block", "screening_status", "missing_rate", "warning_coverage", "std"]]
    shortlist.to_csv(reports_dir / "screening_shortlist_v1.csv", index=False)
    _corr_pairs_df().to_csv(reports_dir / "high_correlation_pairs_v1.csv", index=False)

    paths_cfg = {
        "project_root": str(root),
        "artifacts": {
            "interim": str(artifacts_dir / "interim"),
            "processed": str(processed_dir),
            "features": str(features_dir),
            "reports": str(reports_dir),
            "logs": str(logs_dir),
            "manifests": str(manifests_dir),
        },
    }
    (configs_dir / "paths.local.yaml").write_text(yaml.safe_dump(paths_cfg, sort_keys=False), encoding="utf-8")

    stage_cfg = {
        "run_name": "feature_consolidation_v1",
        "stage": "feature_consolidation",
        "input": {
            "features_master_parquet": "artifacts/features/features_master_v1.parquet",
            "high_correlation_pairs_csv": "artifacts/reports/high_correlation_pairs_v1.csv",
            "screening_shortlist_csv": "artifacts/reports/screening_shortlist_v1.csv",
        },
        "dataset_profile": "core_balanced",
        "include_screening_statuses": ["keep_candidate", "review_highly_correlated", "reserve_candidate"],
        "excluded_screening_statuses": ["review_low_variance", "review_high_missing", "drop_candidate"],
        "high_correlation_threshold": 0.9,
        "representative_selection_priority": ["keep_candidate", "reserve_candidate", "review_highly_correlated"],
        "include_chaos_in_extended_set": True,
        "pre_scaling": {
            "robust_outlier_threshold": 0.05,
            "rank_outlier_threshold": 0.15,
            "robust_abs_skew_threshold": 1.0,
            "rank_abs_skew_threshold": 2.0,
        },
        "output": {
            "base_parquet_name": "clustering_features_base_v1.parquet",
            "with_chaos_parquet_name": "clustering_features_with_chaos_v1.parquet",
            "correlation_groups_csv_name": "correlation_groups_v1.csv",
            "base_feature_list_csv_name": "clustering_features_base_list_v1.csv",
            "with_chaos_feature_list_csv_name": "clustering_features_with_chaos_list_v1.csv",
            "scaling_plan_csv_name": "clustering_scaling_plan_v1.csv",
            "excel_name": "feature_consolidation_v1.xlsx",
        },
    }
    cfg_path = configs_dir / "feature_consolidation_v1.yaml"
    cfg_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False), encoding="utf-8")

    result = run_feature_consolidation_pipeline(str(cfg_path))
    assert Path(result.base_parquet_path).exists()
    assert Path(result.with_chaos_parquet_path).exists()
    assert Path(result.correlation_groups_csv_path).exists()
    assert Path(result.base_list_csv_path).exists()
    assert Path(result.with_chaos_list_csv_path).exists()
    assert Path(result.scaling_plan_csv_path).exists()
    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()
