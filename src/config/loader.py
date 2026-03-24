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


def load_forecasting_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load forecasting benchmark config merged with local paths config."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    if "forecasting" not in artifacts_cfg:
        artifacts_cfg["forecasting"] = str((project_root / "artifacts" / "forecasting").resolve())

    data_cfg = dict(stage_cfg.get("data", {}))
    source_path = Path(str(data_cfg.get("source_path", "artifacts/processed/log_returns_v1.parquet")))
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()
    data_cfg["source_path"] = str(source_path)

    output_cfg = dict(stage_cfg.get("outputs", {}))
    forecasting_dir = Path(artifacts_cfg["forecasting"]).resolve()
    reports_dir = Path(artifacts_cfg.get("reports", project_root / "artifacts" / "reports")).resolve()
    output_cfg.setdefault("run_name", "forecasting_benchmark_smoke_v1")
    output_cfg.setdefault("raw_predictions_path", str((forecasting_dir / "raw_predictions_smoke_v1.parquet").resolve()))
    output_cfg.setdefault("fold_metrics_path", str((forecasting_dir / "fold_metrics_smoke_v1.parquet").resolve()))
    output_cfg.setdefault("series_metrics_path", str((forecasting_dir / "series_metrics_smoke_v1.parquet").resolve()))
    output_cfg.setdefault("task_audit_path", str((forecasting_dir / "task_audit_smoke_v1.parquet").resolve()))
    output_cfg.setdefault("excel_report_path", str((reports_dir / "forecasting_smoke_summary_v1.xlsx").resolve()))
    for key in [
        "raw_predictions_path",
        "fold_metrics_path",
        "series_metrics_path",
        "task_audit_path",
        "excel_report_path",
    ]:
        value = output_cfg.get(key)
        if not value:
            continue
        p = Path(str(value))
        output_cfg[key] = str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    merged: Dict[str, Any] = {
        "stage": stage_cfg.get("stage", "forecasting_benchmark_smoke"),
        "data": data_cfg,
        "models": stage_cfg.get("models", {}),
        "horizons": stage_cfg.get("horizons", [1, 5, 20]),
        "window_sizes": stage_cfg.get("window_sizes", {"1": 64, "5": 32, "20": 16}),
        "validation": stage_cfg.get("validation", {"method": "rolling_origin", "n_folds": 3}),
        "timeouts": stage_cfg.get(
            "timeouts",
            {"max_train_seconds_per_task": 60, "max_predict_seconds_per_task": 15},
        ),
        "training": stage_cfg.get(
            "training",
            {"max_epochs": 20, "early_stopping_patience": 5, "batch_size": 64, "learning_rate": 1e-3},
        ),
        "model_overrides": stage_cfg.get("model_overrides", {}),
        "filters": stage_cfg.get(
            "filters",
            {
                "active_models": [],
                "horizons": [],
                "series_ids": [],
                "resume_failed_only": False,
            },
        ),
        "outputs": output_cfg,
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
            "project_root": str(project_root),
        },
    }

    required_root_keys = ["data", "models", "outputs", "artifacts"]
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
