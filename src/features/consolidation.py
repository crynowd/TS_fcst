from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.config.loader import load_feature_consolidation_config
from src.features.clustering_prep import build_clustering_table, build_scaling_plan
from src.reporting.excel_export import export_feature_consolidation_excel
from src.utils.logging_utils import setup_logger
from src.utils.manifest import get_git_commit, write_manifest


METADATA_COLUMNS = ["series_id", "ticker", "market", "dataset_profile"]


@dataclass(frozen=True)
class FeatureConsolidationOutputs:
    run_id: str
    base_parquet_path: str
    with_chaos_parquet_path: str
    correlation_groups_csv_path: str
    base_list_csv_path: str
    with_chaos_list_csv_path: str
    scaling_plan_csv_path: str
    excel_path: str
    log_path: str
    manifest_path: str
    candidate_pool_size: int
    correlation_groups_count: int
    base_feature_count: int
    with_chaos_feature_count: int
    selected_by_block: Dict[str, int]


def _safe_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _interpretability_score(feature_name: str) -> float:
    name = str(feature_name)
    score = 0.0
    if name.startswith("iacf_"):
        score += 2.5
    if name.startswith("acf_lag_") or name.startswith("abs_acf_lag_"):
        score -= 2.0
    if name.startswith("lb_"):
        score += 0.5
    if "spectral_" in name or "entropy" in name or "kurtosis" in name or "tail_ratio" in name:
        score += 1.0
    if "largest_lyapunov_exponent" in name or "correlation_dimension" in name:
        score += 1.0
    if "_lag_" in name:
        suffix = name.split("_lag_")[-1]
        if suffix.isdigit():
            score -= min(int(suffix) / 100.0, 2.0)
    return score


def build_candidate_pool(
    shortlist_df: pd.DataFrame,
    include_statuses: Sequence[str],
    excluded_statuses: Sequence[str],
) -> pd.DataFrame:
    include = set(include_statuses)
    excluded = set(excluded_statuses)
    pool = shortlist_df.copy()

    def _reason(status: str) -> str:
        if status in include:
            return "included_by_screening_status"
        if status in excluded:
            return "excluded_by_screening_status"
        return "excluded_not_whitelisted"

    pool["included_in_pool"] = pool["screening_status"].astype(str).isin(include)
    pool["inclusion_reason"] = pool["screening_status"].astype(str).map(_reason)
    columns = ["feature_name", "source_block", "screening_status", "included_in_pool", "inclusion_reason"]
    return pool[columns].sort_values(["included_in_pool", "source_block", "feature_name"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def _build_connected_components(edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: Dict[str, Set[str]] = {}
    for node in list(parent):
        root = find(node)
        groups.setdefault(root, set()).add(node)
    return [g for g in groups.values() if g]


def build_correlation_groups(
    corr_pairs_df: pd.DataFrame,
    candidate_features: Set[str],
    shortlist_df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    scoped = corr_pairs_df.copy()
    if scoped.empty:
        cols = ["correlation_group_id", "feature_names", "group_size", "source_blocks_present"]
        return pd.DataFrame(columns=cols), []

    def _pair_strength(row: pd.Series) -> float:
        p = _safe_float(row.get("pearson_corr"), np.nan)
        s = _safe_float(row.get("spearman_corr"), np.nan)
        return max(abs(p) if np.isfinite(p) else 0.0, abs(s) if np.isfinite(s) else 0.0)

    scoped = scoped[scoped.apply(_pair_strength, axis=1) >= float(threshold)].copy()
    scoped = scoped[
        scoped["feature_1"].astype(str).isin(candidate_features) & scoped["feature_2"].astype(str).isin(candidate_features)
    ].copy()
    if scoped.empty:
        cols = ["correlation_group_id", "feature_names", "group_size", "source_blocks_present"]
        return pd.DataFrame(columns=cols), []

    edges = [(str(r["feature_1"]), str(r["feature_2"])) for _, r in scoped.iterrows()]
    components = _build_connected_components(edges)

    short_idx = shortlist_df.set_index("feature_name")
    rows: List[Dict[str, object]] = []
    group_items: List[Dict[str, object]] = []
    for i, members in enumerate(sorted(components, key=lambda g: (-len(g), sorted(g)[0])), start=1):
        gid = f"G{i:03d}"
        names_sorted = sorted(members)
        blocks = sorted({str(short_idx.loc[n, "source_block"]) for n in names_sorted if n in short_idx.index})
        rows.append(
            {
                "correlation_group_id": gid,
                "feature_names": ";".join(names_sorted),
                "group_size": int(len(names_sorted)),
                "source_blocks_present": ";".join(blocks),
            }
        )
        group_items.append({"correlation_group_id": gid, "members": names_sorted})

    table = pd.DataFrame(rows).sort_values(["group_size", "correlation_group_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return table, group_items


def select_group_representatives(
    group_items: List[Dict[str, object]],
    shortlist_df: pd.DataFrame,
    status_priority: Sequence[str],
) -> Tuple[pd.DataFrame, Set[str]]:
    short = shortlist_df.set_index("feature_name")
    status_rank = {status: idx for idx, status in enumerate(status_priority)}
    default_rank = len(status_priority) + 1

    selected: Set[str] = set()
    rows: List[Dict[str, object]] = []
    for group in group_items:
        gid = str(group["correlation_group_id"])
        members = [m for m in group["members"] if m in short.index]
        if not members:
            continue

        candidates: List[Tuple[Tuple[float, ...], str]] = []
        for feat in members:
            row = short.loc[feat]
            rank = float(status_rank.get(str(row["screening_status"]), default_rank))
            missing = _safe_float(row.get("missing_rate"), 1.0)
            warn = _safe_float(row.get("warning_coverage"), 1.0)
            std = _safe_float(row.get("std"), 0.0)
            interp = _interpretability_score(feat)
            score = (rank, missing, warn, -std, -interp, len(feat))
            candidates.append((score, feat))

        candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = candidates[0][1]
        selected.add(chosen)
        excluded = [f for f in members if f != chosen]
        chosen_row = short.loc[chosen]
        reason = (
            f"status={chosen_row['screening_status']}; missing={_safe_float(chosen_row['missing_rate'], np.nan):.4f}; "
            f"warning={_safe_float(chosen_row['warning_coverage'], np.nan):.4f}; std={_safe_float(chosen_row['std'], np.nan):.6f}; "
            f"interpretability_score={_interpretability_score(chosen):.2f}"
        )
        rows.append(
            {
                "correlation_group_id": gid,
                "chosen_representative": chosen,
                "excluded_features": ";".join(excluded),
                "selection_reason": reason,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=["correlation_group_id", "chosen_representative", "excluded_features", "selection_reason"],
        )
    else:
        table = table.sort_values("correlation_group_id", kind="stable").reset_index(drop=True)
    return table, selected


def _feature_list_from_set(
    features: Sequence[str],
    shortlist_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    short = shortlist_df.set_index("feature_name")
    rows: List[Dict[str, object]] = []
    for feature in features:
        if feature not in short.index:
            continue
        row = short.loc[feature]
        rows.append(
            {
                "feature_name": feature,
                "source_block": row["source_block"],
                "screening_status": row["screening_status"],
                "set_name": label,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_block", "feature_name"], kind="stable").reset_index(drop=True)


def run_feature_consolidation_pipeline(
    config_path: str = "configs/feature_consolidation_v1.yaml",
) -> FeatureConsolidationOutputs:
    cfg = load_feature_consolidation_config(config_path)
    run_name = str(cfg.get("run_name", "feature_consolidation_v1"))
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_ts = datetime.now(timezone.utc)

    artifacts_cfg = cfg["artifacts"]
    logger, log_path = setup_logger(run_id=run_id, logs_dir=artifacts_cfg["logs"])

    dataset_profile = str(cfg["dataset_profile"])
    logger.info("Feature consolidation started")
    logger.info("Dataset profile: %s", dataset_profile)
    logger.info("Config path: %s", cfg["meta"]["config_path"])

    master_path = Path(cfg["input"]["features_master_parquet"]).resolve()
    corr_path = Path(cfg["input"]["high_correlation_pairs_csv"]).resolve()
    shortlist_path = Path(cfg["input"]["screening_shortlist_csv"]).resolve()

    master_df = pd.read_parquet(master_path)
    if "dataset_profile" in master_df.columns:
        master_df = master_df[master_df["dataset_profile"] == dataset_profile].copy()
    corr_pairs_df = pd.read_csv(corr_path)
    shortlist_df = pd.read_csv(shortlist_path)

    logger.info("Master table size: rows=%d cols=%d", len(master_df), len(master_df.columns))

    candidate_pool_df = build_candidate_pool(
        shortlist_df=shortlist_df,
        include_statuses=[str(x) for x in cfg.get("include_screening_statuses", [])],
        excluded_statuses=[str(x) for x in cfg.get("excluded_screening_statuses", [])],
    )
    candidate_features = set(candidate_pool_df[candidate_pool_df["included_in_pool"]]["feature_name"].astype(str).tolist())
    available_features = set(master_df.columns) - set(METADATA_COLUMNS)
    unavailable = sorted(candidate_features - available_features)
    if unavailable:
        logger.info("Candidate features unavailable in master table: %d", len(unavailable))
    candidate_features = candidate_features & available_features
    logger.info("Candidate pool size: %d", len(candidate_features))

    groups_df, group_items = build_correlation_groups(
        corr_pairs_df=corr_pairs_df,
        candidate_features=candidate_features,
        shortlist_df=shortlist_df,
        threshold=float(cfg.get("high_correlation_threshold", 0.90)),
    )
    logger.info("Correlation groups: %d", len(groups_df))

    representative_df, selected_group_features = select_group_representatives(
        group_items=group_items,
        shortlist_df=shortlist_df,
        status_priority=[str(x) for x in cfg.get("representative_selection_priority", [])],
    )

    grouped_members = set()
    for item in group_items:
        grouped_members.update(item["members"])
    singleton_features = sorted(candidate_features - grouped_members)

    selected_features = set(selected_group_features) | set(singleton_features)
    shortlist_idx = shortlist_df.set_index("feature_name")

    base_features = sorted([f for f in selected_features if f in shortlist_idx.index and str(shortlist_idx.loc[f, "source_block"]) != "D"])
    chaos_features = sorted([f for f in selected_features if f in shortlist_idx.index and str(shortlist_idx.loc[f, "source_block"]) == "D"])
    with_chaos_features = sorted(base_features + chaos_features) if bool(cfg.get("include_chaos_in_extended_set", True)) else list(base_features)

    logger.info("Representative features selected: %d", len(selected_features))
    logger.info("Features in base set: %d", len(base_features))
    logger.info("Features in with-chaos set: %d", len(with_chaos_features))

    base_table_df = build_clustering_table(master_df, base_features)
    with_chaos_table_df = build_clustering_table(master_df, with_chaos_features)

    with_chaos_source_map = {f: str(shortlist_idx.loc[f, "source_block"]) for f in with_chaos_features if f in shortlist_idx.index}
    scaling_plan_df = build_scaling_plan(
        with_chaos_table_df,
        feature_source_map=with_chaos_source_map,
        cfg=cfg.get("pre_scaling", {}),
    )

    base_list_df = _feature_list_from_set(base_features, shortlist_df, "clustering_features_base_v1")
    chaos_list_df = _feature_list_from_set(with_chaos_features, shortlist_df, "clustering_features_with_chaos_v1")

    selected_by_block = {
        str(block): int(count)
        for block, count in chaos_list_df["source_block"].value_counts().sort_index().to_dict().items()
    }

    features_dir = Path(artifacts_cfg["features"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = cfg["output"]
    base_parquet_path = features_dir / str(output_cfg["base_parquet_name"])
    with_chaos_parquet_path = features_dir / str(output_cfg["with_chaos_parquet_name"])
    groups_csv_path = reports_dir / str(output_cfg["correlation_groups_csv_name"])
    base_list_csv_path = reports_dir / str(output_cfg["base_feature_list_csv_name"])
    chaos_list_csv_path = reports_dir / str(output_cfg["with_chaos_feature_list_csv_name"])
    scaling_csv_path = reports_dir / str(output_cfg["scaling_plan_csv_name"])
    excel_path = reports_dir / str(output_cfg["excel_name"])

    base_table_df.to_parquet(base_parquet_path, index=False)
    with_chaos_table_df.to_parquet(with_chaos_parquet_path, index=False)
    groups_df.to_csv(groups_csv_path, index=False)
    base_list_df.to_csv(base_list_csv_path, index=False)
    chaos_list_df.to_csv(chaos_list_csv_path, index=False)
    scaling_plan_df.to_csv(scaling_csv_path, index=False)

    summary = {
        "master_features_total": int(len(shortlist_df)),
        "candidate_pool_size": int(len(candidate_features)),
        "correlation_groups_count": int(len(groups_df)),
        "representative_features_count": int(len(selected_features)),
        "base_feature_count": int(len(base_features)),
        "with_chaos_feature_count": int(len(with_chaos_features)),
        "selected_by_block": selected_by_block,
    }
    excel_out = export_feature_consolidation_excel(
        excel_path=excel_path,
        run_id=run_id,
        dataset_profile=dataset_profile,
        summary=summary,
        candidate_pool_df=candidate_pool_df,
        correlation_groups_df=groups_df,
        representative_df=representative_df,
        base_feature_set_df=base_list_df,
        chaos_feature_set_df=chaos_list_df,
        scaling_plan_df=scaling_plan_df,
        input_paths={k: str(v) for k, v in cfg["input"].items()},
        output_paths={
            "base_parquet": str(base_parquet_path),
            "with_chaos_parquet": str(with_chaos_parquet_path),
            "correlation_groups_csv": str(groups_csv_path),
            "base_feature_list_csv": str(base_list_csv_path),
            "with_chaos_feature_list_csv": str(chaos_list_csv_path),
            "scaling_plan_csv": str(scaling_csv_path),
        },
        config_params={
            "include_screening_statuses": cfg.get("include_screening_statuses", []),
            "excluded_screening_statuses": cfg.get("excluded_screening_statuses", []),
            "high_correlation_threshold": cfg.get("high_correlation_threshold", 0.90),
            "representative_selection_priority": cfg.get("representative_selection_priority", []),
            "include_chaos_in_extended_set": cfg.get("include_chaos_in_extended_set", True),
            "pre_scaling": cfg.get("pre_scaling", {}),
        },
    )

    end_ts = datetime.now(timezone.utc)
    logger.info("Base parquet path: %s", base_parquet_path)
    logger.info("With-chaos parquet path: %s", with_chaos_parquet_path)
    logger.info("Correlation groups CSV path: %s", groups_csv_path)
    logger.info("Scaling plan CSV path: %s", scaling_csv_path)
    logger.info("Excel report path: %s", excel_out)
    logger.info("Execution time: %.2f sec", (end_ts - start_ts).total_seconds())

    manifest = {
        "run_id": run_id,
        "stage": "feature_consolidation",
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["config_path"]).resolve().parents[1]),
        "config_path": cfg["meta"]["config_path"],
        "config_hash": cfg["meta"].get("config_hash", ""),
        "input_sources": {
            **{k: str(v) for k, v in cfg["input"].items()},
            "dataset_profile": dataset_profile,
        },
        "outputs": {
            "base_parquet": str(base_parquet_path),
            "with_chaos_parquet": str(with_chaos_parquet_path),
            "correlation_groups_csv": str(groups_csv_path),
            "base_feature_list_csv": str(base_list_csv_path),
            "with_chaos_feature_list_csv": str(chaos_list_csv_path),
            "scaling_plan_csv": str(scaling_csv_path),
            "excel_report": str(excel_out),
            "log": str(log_path),
        },
        "summary": summary,
    }
    manifest_path = write_manifest(manifest, artifacts_cfg["manifests"], run_id)

    return FeatureConsolidationOutputs(
        run_id=run_id,
        base_parquet_path=str(base_parquet_path),
        with_chaos_parquet_path=str(with_chaos_parquet_path),
        correlation_groups_csv_path=str(groups_csv_path),
        base_list_csv_path=str(base_list_csv_path),
        with_chaos_list_csv_path=str(chaos_list_csv_path),
        scaling_plan_csv_path=str(scaling_csv_path),
        excel_path=str(excel_out),
        log_path=str(log_path),
        manifest_path=str(manifest_path),
        candidate_pool_size=int(len(candidate_features)),
        correlation_groups_count=int(len(groups_df)),
        base_feature_count=int(len(base_features)),
        with_chaos_feature_count=int(len(with_chaos_features)),
        selected_by_block=selected_by_block,
    )
