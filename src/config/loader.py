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


def _resolve_from_project_root(project_root: Path, value: str | Path) -> Path:
    p = Path(str(value))
    return (project_root / p).resolve() if not p.is_absolute() else p.resolve()


def _load_selected_architectures_for_forecasting(
    stage_cfg: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    selected_cfg = stage_cfg.get("selected_architectures", {})
    if not isinstance(selected_cfg, dict) or not selected_cfg:
        return {
            "applied": False,
            "source_config_path": "",
            "requested_candidate_ids": [],
            "resolved_candidate_ids": [],
            "missing_candidate_ids": [],
            "model_overrides": {},
            "active_models": [],
            "model_metadata": {},
        }

    source_config = str(selected_cfg.get("source_config", "")).strip()
    if not source_config:
        raise ConfigError("selected_architectures.source_config must be provided when selected_architectures is set")

    source_path = _resolve_from_project_root(project_root, source_config)
    selected_payload = _read_yaml(source_path)
    selected_models = selected_payload.get("selected_models", [])
    if not isinstance(selected_models, list):
        raise ConfigError("selected_architectures source YAML must contain list field 'selected_models'")

    by_candidate_id: Dict[str, Dict[str, Any]] = {}
    for item in selected_models:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip()
        if cid:
            by_candidate_id[cid] = item

    requested = [str(x) for x in selected_cfg.get("candidate_ids", []) if str(x).strip()]
    if not requested:
        requested = [
            str(item.get("candidate_id", "")).strip()
            for item in selected_models
            if isinstance(item, dict) and bool(item.get("selected_for_main_run", True))
        ]
        requested = [cid for cid in requested if cid]

    missing_requested = [cid for cid in requested if cid not in by_candidate_id]
    missing_from_payload = [str(x) for x in selected_payload.get("missing_candidates", []) if str(x).strip()]
    missing_candidate_ids = sorted(set(missing_requested + missing_from_payload))

    baselines_keep = [str(x) for x in selected_cfg.get("baselines_keep", []) if str(x).strip()]
    model_overrides: Dict[str, Dict[str, Any]] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}
    shortlisted_model_names: list[str] = []
    resolved_candidate_ids: list[str] = []

    for cid in requested:
        item = by_candidate_id.get(cid)
        if item is None:
            continue
        model_name = str(item.get("model_name", "")).strip()
        if not model_name:
            continue
        model_params = item.get("model_params", {})
        runtime_params = item.get("runtime_params", {})
        merged_params: Dict[str, Any] = {}
        if isinstance(model_params, dict):
            merged_params.update(model_params)
        if isinstance(runtime_params, dict):
            merged_params.update(runtime_params)
        model_overrides[model_name] = merged_params
        model_metadata[model_name] = {
            "candidate_id": cid,
            "selection_role": str(item.get("selection_role", "")),
            "family": str(item.get("family", "")),
            "source_run_name": str(item.get("source_run_name", "")),
        }
        shortlisted_model_names.append(model_name)
        resolved_candidate_ids.append(cid)

    active_models = list(dict.fromkeys(baselines_keep + shortlisted_model_names))

    return {
        "applied": True,
        "source_config_path": str(source_path),
        "requested_candidate_ids": requested,
        "resolved_candidate_ids": resolved_candidate_ids,
        "missing_candidate_ids": missing_candidate_ids,
        "model_overrides": model_overrides,
        "active_models": active_models,
        "model_metadata": model_metadata,
    }


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

    models_cfg = dict(stage_cfg.get("models", {}))
    active_models_cfg = [str(x) for x in models_cfg.get("active", [])]
    inactive_models_cfg = [str(x) for x in models_cfg.get("inactive_but_supported", [])]

    model_overrides_cfg = dict(stage_cfg.get("model_overrides", {}))
    selected_arch = _load_selected_architectures_for_forecasting(stage_cfg=stage_cfg, project_root=project_root)
    if selected_arch.get("applied"):
        for model_name, params in selected_arch.get("model_overrides", {}).items():
            model_overrides_cfg[model_name] = params

        selected_active = [str(x) for x in selected_arch.get("active_models", [])]
        if selected_active:
            active_models_cfg = list(dict.fromkeys(selected_active + active_models_cfg))

        selected_model_names = set(selected_arch.get("model_metadata", {}).keys())
        inactive_models_cfg = [m for m in inactive_models_cfg if m not in selected_model_names]

    models_cfg["active"] = active_models_cfg
    models_cfg["inactive_but_supported"] = inactive_models_cfg

    merged: Dict[str, Any] = {
        "stage": stage_cfg.get("stage", "forecasting_benchmark_smoke"),
        "data": data_cfg,
        "models": models_cfg,
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
        "model_overrides": model_overrides_cfg,
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
            "selected_architectures": selected_arch,
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


def load_cluster_forecasting_analysis_config(config_path: str) -> Dict[str, Any]:
    """Load cluster-conditioned forecasting analysis config merged with local paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    artifacts_cfg.setdefault("reports", str((project_root / "artifacts" / "reports").resolve()))
    artifacts_cfg.setdefault("logs", str((project_root / "artifacts" / "logs").resolve()))
    artifacts_cfg.setdefault("manifests", str((project_root / "artifacts" / "manifests").resolve()))
    artifacts_cfg.setdefault("analysis", str((project_root / "artifacts" / "analysis").resolve()))

    def _resolve_path(value: str) -> str:
        p = Path(str(value))
        return str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    forecasting_inputs = dict(stage_cfg.get("forecasting_inputs", {}))
    clustering_inputs = dict(stage_cfg.get("clustering_inputs", {}))
    for key, value in list(forecasting_inputs.items()):
        if value:
            forecasting_inputs[key] = _resolve_path(value)
    for key, value in list(clustering_inputs.items()):
        if value:
            clustering_inputs[key] = _resolve_path(value)

    output_cfg = dict(stage_cfg.get("outputs", {}))
    run_name = str(output_cfg.get("run_name", "cluster_forecasting_analysis_smoke_v1"))
    analysis_dir = Path(artifacts_cfg["analysis"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    output_cfg.setdefault("cluster_model_performance_path", str((analysis_dir / "cluster_model_performance_v1.parquet").resolve()))
    output_cfg.setdefault("cluster_model_performance_csv_path", str((analysis_dir / "cluster_model_performance_v1.csv").resolve()))
    output_cfg.setdefault("best_model_by_cluster_path", str((analysis_dir / "best_model_by_cluster_v1.csv").resolve()))
    output_cfg.setdefault("clustering_utility_path", str((analysis_dir / "clustering_utility_v1.csv").resolve()))
    output_cfg.setdefault("cluster_metric_tests_path", str((analysis_dir / "cluster_metric_tests_v1.csv").resolve()))
    output_cfg.setdefault("tidy_cluster_model_metrics_path", str((analysis_dir / "tidy_cluster_model_metrics_v1.parquet").resolve()))
    output_cfg.setdefault("tidy_series_cluster_metrics_path", str((analysis_dir / "tidy_series_cluster_metrics_v1.parquet").resolve()))
    output_cfg.setdefault("excel_report_path", str((reports_dir / "cluster_forecasting_analysis_v1.xlsx").resolve()))
    for key, value in list(output_cfg.items()):
        if key.endswith("_path") and value:
            output_cfg[key] = _resolve_path(str(value))

    merged: Dict[str, Any] = {
        "run_name": run_name,
        "stage": stage_cfg.get("stage", "cluster_forecasting_analysis_smoke"),
        "forecasting_inputs": forecasting_inputs,
        "clustering_inputs": clustering_inputs,
        "shortlist_configs": stage_cfg.get("shortlist_configs", []),
        "baselines": stage_cfg.get("baselines", {"primary": "naive_zero", "secondary": "naive_mean"}),
        "metrics_for_comparison": stage_cfg.get("metrics_for_comparison", ["mae", "rmse", "mase", "directional_accuracy"]),
        "stat_tests": stage_cfg.get("stat_tests", {"use_kruskal": True, "use_pairwise_posthoc": False}),
        "outputs": output_cfg,
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
            "project_root": str(project_root),
        },
    }

    required_root_keys = ["forecasting_inputs", "clustering_inputs", "shortlist_configs", "outputs", "artifacts"]
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


def load_architecture_tuning_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load architecture tuning benchmark config merged with local paths config."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    artifacts_cfg.setdefault("reports", str((project_root / "artifacts" / "reports").resolve()))
    artifacts_cfg.setdefault("logs", str((project_root / "artifacts" / "logs").resolve()))
    artifacts_cfg.setdefault("manifests", str((project_root / "artifacts" / "manifests").resolve()))
    artifacts_cfg.setdefault("architecture_tuning", str((project_root / "artifacts" / "architecture_tuning").resolve()))

    benchmark_cfg = dict(stage_cfg.get("benchmark", {}))
    data_transforms_cfg = dict(stage_cfg.get("data_transforms", {}))
    evaluation_cfg = dict(stage_cfg.get("evaluation", {}))
    models_cfg = dict(stage_cfg.get("models", {}))
    reporting_cfg = dict(stage_cfg.get("reporting", {}))
    outputs_cfg = dict(stage_cfg.get("outputs", {}))

    def _resolve_path(value: str) -> str:
        p = Path(str(value))
        return str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    external_data_dir = str(benchmark_cfg.get("external_data_dir", ""))
    if external_data_dir:
        external_data_dir = _resolve_path(external_data_dir)

    output_dir = str(outputs_cfg.get("output_dir", artifacts_cfg["architecture_tuning"]))
    output_dir = _resolve_path(output_dir)
    run_name = str(stage_cfg.get("run_name", "architecture_tuning_benchmark_v1"))

    outputs_cfg.setdefault(
        "selected_series_csv",
        str((Path(output_dir) / "selected_series_architecture_tuning_v1.csv").resolve()),
    )
    outputs_cfg.setdefault(
        "candidate_level_results_parquet",
        str((Path(output_dir) / "candidate_level_results_v1.parquet").resolve()),
    )
    outputs_cfg.setdefault(
        "candidate_level_results_csv",
        str((Path(output_dir) / "candidate_level_results_v1.csv").resolve()),
    )
    outputs_cfg.setdefault(
        "series_level_results_parquet",
        str((Path(output_dir) / "series_level_results_v1.parquet").resolve()),
    )
    outputs_cfg.setdefault(
        "pair_comparison_csv",
        str((Path(output_dir) / "pair_comparison_summary_v1.csv").resolve()),
    )
    outputs_cfg.setdefault(
        "best_candidate_summary_csv",
        str((Path(output_dir) / "best_candidate_summary_v1.csv").resolve()),
    )
    outputs_cfg.setdefault(
        "excel_report_path",
        str((Path(artifacts_cfg["reports"]).resolve() / "architecture_tuning_benchmark_v1.xlsx").resolve()),
    )

    for key, value in list(outputs_cfg.items()):
        if key.endswith("_path") or key.endswith("_csv") or key.endswith("_parquet") or key == "output_dir":
            outputs_cfg[key] = _resolve_path(str(value))

    window_sizes_cfg = dict(evaluation_cfg.get("window_sizes", {"1": 64, "5": 32, "20": 16}))
    normalized_window_sizes = {str(int(k)): int(v) for k, v in window_sizes_cfg.items()}

    merged: Dict[str, Any] = {
        "run_name": run_name,
        "stage": stage_cfg.get("stage", "architecture_tuning_benchmark"),
        "external_data_dir": external_data_dir,
        "file_pattern": benchmark_cfg.get("file_pattern", "*.csv"),
        "sample_size": int(benchmark_cfg.get("sample_size", 25)),
        "random_seed": int(benchmark_cfg.get("random_seed", 42)),
        "selected_models": [str(m) for m in models_cfg.get("selected_models", [])],
        "candidates": models_cfg.get("candidates", {}),
        "horizons": [int(h) for h in stage_cfg.get("horizons", [1, 5, 20])],
        "data_transforms": {
            "date_column": data_transforms_cfg.get("date_column", "Date"),
            "close_column": data_transforms_cfg.get("close_column", "Close"),
            "min_history": int(data_transforms_cfg.get("min_history", 120)),
        },
        "device": stage_cfg.get("device", "cpu"),
        "validation": evaluation_cfg.get("validation", {"method": "rolling_origin", "n_folds": 3}),
        "window_sizes": normalized_window_sizes,
        "training": evaluation_cfg.get(
            "training",
            {
                "max_epochs": 10,
                "early_stopping_patience": 3,
                "batch_size": 128,
                "learning_rate": 1e-3,
            },
        ),
        "timeouts": evaluation_cfg.get(
            "timeouts",
            {
                "max_train_seconds_per_task": 30,
                "max_predict_seconds_per_task": 10,
            },
        ),
        "reporting": reporting_cfg,
        "outputs": outputs_cfg,
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
            "project_root": str(project_root),
        },
    }

    required_root_keys = ["external_data_dir", "selected_models", "horizons", "outputs", "artifacts"]
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


def load_architecture_tuning_final_shortlist_config(config_path: str) -> Dict[str, Any]:
    """Load architecture tuning final shortlist config merged with local paths."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    artifacts_cfg.setdefault("reports", str((project_root / "artifacts" / "reports").resolve()))
    artifacts_cfg.setdefault("logs", str((project_root / "artifacts" / "logs").resolve()))
    artifacts_cfg.setdefault("manifests", str((project_root / "artifacts" / "manifests").resolve()))
    artifacts_cfg.setdefault("architecture_tuning", str((project_root / "artifacts" / "architecture_tuning").resolve()))

    def _resolve_path(value: str) -> str:
        p = Path(str(value))
        return str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    input_families = []
    for item in stage_cfg.get("input_families", []):
        row = dict(item)
        for key in ["candidate_level_csv", "best_candidate_summary_csv", "pair_comparison_csv"]:
            if row.get(key):
                row[key] = _resolve_path(str(row[key]))
        input_families.append(row)

    outputs_cfg = dict(stage_cfg.get("outputs", {}))
    defaults = {
        "unified_summary_csv": "artifacts/architecture_tuning/final_shortlist/unified_tuning_summary_v1.csv",
        "unified_summary_parquet": "artifacts/architecture_tuning/final_shortlist/unified_tuning_summary_v1.parquet",
        "family_summary_csv": "artifacts/architecture_tuning/final_shortlist/family_summary_v1.csv",
        "selected_architectures_yaml": "configs/forecasting_selected_architectures_v1.yaml",
        "excel_report_path": "artifacts/reports/architecture_tuning_final_shortlist_v1.xlsx",
    }
    for key, default_value in defaults.items():
        outputs_cfg.setdefault(key, default_value)
        outputs_cfg[key] = _resolve_path(str(outputs_cfg[key]))

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "architecture_tuning_final_shortlist_v1"),
        "stage": stage_cfg.get("stage", "architecture_tuning_final_shortlist"),
        "shortlist_candidate_ids": stage_cfg.get("shortlist_candidate_ids", []),
        "shortlist_metadata": stage_cfg.get("shortlist_metadata", {}),
        "input_families": input_families,
        "outputs": outputs_cfg,
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
            "project_root": str(project_root),
        },
    }

    required_root_keys = ["input_families", "outputs", "artifacts"]
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


def load_meta_modeling_config(config_path: str) -> Dict[str, Any]:
    """Load meta-modeling config merged with local paths config."""
    stage_config_path = Path(config_path).resolve()
    stage_cfg = _read_yaml(stage_config_path)

    paths_cfg_path = stage_config_path.parent / "paths.local.yaml"
    paths_cfg = _read_yaml(paths_cfg_path)
    project_root = Path(paths_cfg.get("project_root", stage_config_path.parents[1]))

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    artifacts_cfg.setdefault("reports", str((project_root / "artifacts" / "reports").resolve()))
    artifacts_cfg.setdefault("logs", str((project_root / "artifacts" / "logs").resolve()))
    artifacts_cfg.setdefault("manifests", str((project_root / "artifacts" / "manifests").resolve()))
    artifacts_cfg.setdefault("meta_modeling", str((project_root / "artifacts" / "meta_modeling").resolve()))

    def _resolve_path(value: str) -> str:
        p = Path(str(value))
        return str((project_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

    inputs_cfg = dict(stage_cfg.get("inputs", {}))
    for key, value in list(inputs_cfg.items()):
        if key.endswith("_path") and value:
            inputs_cfg[key] = _resolve_path(str(value))

    outputs_cfg = dict(stage_cfg.get("outputs", {}))
    meta_dir = Path(artifacts_cfg["meta_modeling"]).resolve()
    reports_dir = Path(artifacts_cfg["reports"]).resolve()
    defaults = {
        "meta_dataset_summary_csv_path": meta_dir / "meta_dataset_summary_v1.csv",
        "meta_dataset_summary_parquet_path": meta_dir / "meta_dataset_summary_v1.parquet",
        "model_order_mapping_csv_path": meta_dir / "model_order_mapping_v1.csv",
        "routing_rows_parquet_path": meta_dir / "routing_rows_v1.parquet",
        "routing_rows_csv_path": meta_dir / "routing_rows_v1.csv",
        "task_results_parquet_path": meta_dir / "task_results_v1.parquet",
        "task_results_csv_path": meta_dir / "task_results_v1.csv",
        "split_assignments_csv_path": meta_dir / "split_assignments_v1.csv",
        "repeat_aggregated_results_csv_path": meta_dir / "repeat_aggregated_results_v1.csv",
        "best_single_baseline_by_repeat_csv_path": meta_dir / "best_single_baseline_by_repeat_v1.csv",
        "forecasting_mean_by_model_csv_path": meta_dir / "forecasting_mean_by_model_v1.csv",
        "forecasting_model_ranking_csv_path": meta_dir / "forecasting_model_ranking_v1.csv",
        "best_single_global_baseline_csv_path": meta_dir / "best_single_global_baseline_v1.csv",
        "forecasting_model_wins_csv_path": meta_dir / "forecasting_model_wins_v1.csv",
        "feature_list_csv_path": meta_dir / "feature_list_v1.csv",
        "feature_manifest_json_path": meta_dir / "feature_manifest_v1.json",
        "excel_report_path": reports_dir / "meta_modeling_v1.xlsx",
    }
    for key, default_value in defaults.items():
        outputs_cfg.setdefault(key, str(default_value))
        outputs_cfg[key] = _resolve_path(str(outputs_cfg[key]))

    merged: Dict[str, Any] = {
        "run_name": stage_cfg.get("run_name", "meta_modeling_v1"),
        "stage": stage_cfg.get("stage", "meta_modeling"),
        "inputs": inputs_cfg,
        "dataset_filter": stage_cfg.get("dataset_filter", ""),
        "join_keys": stage_cfg.get("join_keys", ["series_id", "ticker"]),
        "target_metrics": stage_cfg.get("target_metrics", ["rmse", "directional_accuracy"]),
        "metric_columns": stage_cfg.get("metric_columns", {}),
        "meta_models": stage_cfg.get("meta_models", ["ridge", "random_forest", "catboost"]),
        "model_overrides": stage_cfg.get("model_overrides", {}),
        "split": stage_cfg.get(
            "split",
            {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_seed": 42,
                "n_repeats": 3,
                "random_seeds": [],
            },
        ),
        "outputs": outputs_cfg,
        "artifacts": artifacts_cfg,
        "meta": {
            "config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
            "project_root": str(project_root),
        },
    }

    required_root_keys = ["inputs", "outputs", "artifacts"]
    missing = [k for k in required_root_keys if not merged.get(k)]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    required_input_keys = ["features_path", "forecasting_series_metrics_path"]
    missing_inputs = [k for k in required_input_keys if not str(merged["inputs"].get(k, "")).strip()]
    if missing_inputs:
        raise ConfigError(f"Missing meta-modeling inputs: {', '.join(missing_inputs)}")

    merged["meta"]["config_hash"] = _compute_hash(
        {
            "stage_config": stage_cfg,
            "paths_config": paths_cfg,
            "stage_config_path": str(stage_config_path),
            "paths_config_path": str(paths_cfg_path),
        }
    )
    return merged
