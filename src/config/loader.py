from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when inventory configuration is invalid."""


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML config and return mapping content.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML mapping.

    Raises:
        ConfigError: If file is missing or YAML root is not a mapping.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f)

    if not isinstance(parsed, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    return parsed


def _compute_hash(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash for a config dictionary."""
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def load_inventory_config(config_path: str) -> Dict[str, Any]:
    """Load and merge inventory config with local path config.

    The function reads stage config (for inventory logic) and `paths.local.yaml`
    located in the same config directory. Returned dictionary contains all values
    required by stage-1 pipeline execution.

    Args:
        config_path: Path to stage config, e.g. `configs/data_inventory_v1.yaml`.

    Returns:
        Normalized merged config with absolute paths and `meta.config_hash`.

    Raises:
        ConfigError: If required sections are missing.
    """
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "data_inventory_v1"),
        "input": stage_cfg.get("input", {}),
        "schema": stage_cfg.get("schema", {}),
        "standardization": stage_cfg.get("standardization", {}),
        "series_policy": stage_cfg.get("series_policy", {}),
        "quality_checks": stage_cfg.get("quality_checks", {}),
        "data_sources": paths_cfg.get("data_sources", {}),
        "artifacts": paths_cfg.get("artifacts", {}),
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["data_sources", "artifacts", "input", "schema"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )

    return merged


def load_feature_block_config(config_path: str) -> Dict[str, Any]:
    """Load feature-block stage config merged with local artifact paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)

    input_cfg = dict(stage_cfg.get("input", {}))
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))
    for key in ["log_returns_parquet", "dataset_profiles_parquet"]:
        value = input_cfg.get(key, "")
        if value:
            p = Path(value)
            input_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "feature_block_A_v1"),
        "stage": stage_cfg.get("stage", "feature_block_A"),
        "input": input_cfg,
        "dataset_profile": stage_cfg.get("dataset_profile", "core_balanced"),
        "metrics": stage_cfg.get("metrics", {}),
        "output": stage_cfg.get("output", {}),
        "artifacts": paths_cfg.get("artifacts", {}),
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["input", "metrics", "artifacts", "output"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged


def load_feature_screening_config(config_path: str) -> Dict[str, Any]:
    """Load feature-screening config merged with local artifact paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    input_cfg = dict(stage_cfg.get("input", {}))
    for key, value in list(input_cfg.items()):
        if not value:
            continue
        p = Path(str(value))
        input_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "feature_screening_v1"),
        "stage": stage_cfg.get("stage", "feature_screening"),
        "input": input_cfg,
        "dataset_profile": stage_cfg.get("dataset_profile", "core_balanced"),
        "screening": stage_cfg.get("screening", {}),
        "output": stage_cfg.get("output", {}),
        "artifacts": paths_cfg.get("artifacts", {}),
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["input", "screening", "output", "artifacts"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged


def load_feature_consolidation_config(config_path: str) -> Dict[str, Any]:
    """Load feature-consolidation config merged with local artifact paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    input_cfg = dict(stage_cfg.get("input", {}))
    for key, value in list(input_cfg.items()):
        if not value:
            continue
        p = Path(str(value))
        input_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "feature_consolidation_v1"),
        "stage": stage_cfg.get("stage", "feature_consolidation"),
        "input": input_cfg,
        "dataset_profile": stage_cfg.get("dataset_profile", "core_balanced"),
        "include_screening_statuses": stage_cfg.get(
            "include_screening_statuses",
            ["keep_candidate", "review_highly_correlated", "reserve_candidate"],
        ),
        "excluded_screening_statuses": stage_cfg.get(
            "excluded_screening_statuses",
            ["review_high_missing", "review_low_variance", "drop_candidate"],
        ),
        "high_correlation_threshold": stage_cfg.get("high_correlation_threshold", 0.90),
        "representative_selection_priority": stage_cfg.get(
            "representative_selection_priority",
            ["keep_candidate", "reserve_candidate", "review_highly_correlated"],
        ),
        "include_chaos_in_extended_set": stage_cfg.get("include_chaos_in_extended_set", True),
        "pre_scaling": stage_cfg.get("pre_scaling", {}),
        "output": stage_cfg.get("output", {}),
        "artifacts": paths_cfg.get("artifacts", {}),
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["input", "output", "artifacts"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged


def load_clustering_experiments_config(config_path: str) -> Dict[str, Any]:
    """Load clustering-experiments config merged with local artifact paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    input_cfg = dict(stage_cfg.get("input", {}))
    for key, value in list(input_cfg.items()):
        if not value:
            continue
        p = Path(str(value))
        input_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    if "clustering" not in artifacts_cfg:
        artifacts_cfg["clustering"] = str((project_root / "artifacts" / "clustering").resolve())

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "clustering_experiments_v1"),
        "stage": stage_cfg.get("stage", "clustering_experiments"),
        "input": input_cfg,
        "feature_sets": stage_cfg.get("feature_sets", ["base", "with_chaos"]),
        "scalers": stage_cfg.get("scalers", ["identity", "robust", "quantile"]),
        "spaces": stage_cfg.get("spaces", ["original", "pca"]),
        "pca_n_components": stage_cfg.get("pca_n_components", [2, 3, 4, 5, 6]),
        "algorithms": stage_cfg.get("algorithms", ["gmm", "agglomerative"]),
        "cluster_range": stage_cfg.get("cluster_range", {"k_min": 2, "k_max": 8}),
        "imputation": stage_cfg.get("imputation", {"strategy": "median"}),
        "small_cluster_threshold": stage_cfg.get("small_cluster_threshold", {"mode": "relative", "value": 0.05}),
        "selection": stage_cfg.get("selection", {"top_n_per_algorithm_per_feature_set": 3}),
        "stability": stage_cfg.get("stability", {"n_bootstrap": 30, "sample_fraction": 0.8, "random_state": 42}),
        "output": stage_cfg.get("output", {}),
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["input", "output", "artifacts"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged


def load_cluster_profiling_config(config_path: str) -> Dict[str, Any]:
    """Load cluster-profiling config merged with local artifact paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    input_cfg = dict(stage_cfg.get("input", {}))
    for key, value in list(input_cfg.items()):
        if not value:
            continue
        p = Path(str(value))
        input_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    if "clustering" not in artifacts_cfg:
        artifacts_cfg["clustering"] = str((project_root / "artifacts" / "clustering").resolve())
    if "figures" not in artifacts_cfg:
        artifacts_cfg["figures"] = str((project_root / "artifacts" / "figures").resolve())

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "cluster_profiling_v1"),
        "stage": stage_cfg.get("stage", "cluster_profiling"),
        "input": input_cfg,
        "selection": stage_cfg.get("selection", {}),
        "visualization": stage_cfg.get("visualization", {}),
        "output": stage_cfg.get("output", {}),
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        },
    }

    required_root_keys = ["input", "output", "artifacts"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged
