from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.reporting.excel_export import export_architecture_tuning_final_shortlist_excel
from src.utils.manifest import get_git_commit, write_manifest

PARAMETER_COLUMNS = [
    "hidden_dims",
    "depth",
    "hidden_size",
    "num_layers",
    "dropout",
    "n_reservoir",
    "spectral_radius",
    "input_scale",
    "leak_rate",
    "ridge_alpha",
    "train_r",
    "beta",
    "r_min",
    "r_max",
]

UNIFIED_COLUMNS = [
    "family",
    "model_name",
    "candidate_id",
    "source_run_name",
    "compare_group_id",
    "horizon_aggregate_type",
    "mae_mean_over_horizons",
    "rmse_mean_over_horizons",
    "mase_mean_over_horizons",
    "directional_accuracy_mean_over_horizons",
    "fit_time_mean_over_horizons",
    "predict_time_mean_over_horizons",
    "total_runtime_mean_over_horizons",
    "status",
    "notes",
    "selected_for_main_run",
    "selection_role",
    "rationale",
    *PARAMETER_COLUMNS,
]

FINAL_CONFIG_REQUIRED_KEYS = [
    "candidate_id",
    "model_name",
    "family",
    "source_run_name",
    "compare_group_id",
    "model_params",
    "runtime_params",
    "selection_role",
    "selected_for_main_run",
]

DEFAULT_SHORTLIST_METADATA: dict[str, dict[str, str]] = {
    "esn_g004": {
        "family": "esn",
        "model_name": "esn",
        "selection_role": "practical_main",
        "rationale": "Лучший нехаотический ESN по среднему MAE между горизонтами при устойчивом времени обучения.",
    },
    "chaotic_esn_g003": {
        "family": "esn",
        "model_name": "chaotic_esn",
        "selection_role": "chaotic_counterpart",
        "rationale": "Хаотический ESN из парного сравнения; сохраняет исследовательскую ценность на нелинейных сериях при умеренном runtime.",
    },
    "transient_chaotic_esn_g002": {
        "family": "esn",
        "model_name": "transient_chaotic_esn",
        "selection_role": "exploratory_chaotic",
        "rationale": "Транзиентный хаотический ESN оставлен как exploratory-вариант для проверки gain-schedule гипотезы.",
    },
    "vanilla_mlp_g004": {
        "family": "mlp",
        "model_name": "vanilla_mlp",
        "selection_role": "practical_main",
        "rationale": "Лучший practical MLP-кандидат по агрегированным ошибкам с минимальным runtime в семействе.",
    },
    "chaotic_mlp_g003": {
        "family": "mlp",
        "model_name": "chaotic_mlp",
        "selection_role": "chaotic_counterpart",
        "rationale": "Хаотический MLP сохраняется как парный контркандидат для проверки выигрыша на подмножествах рядов.",
    },
    "logistic_g006": {
        "family": "logistic",
        "model_name": "chaotic_logistic_net",
        "selection_role": "nonlinear_specialist",
        "rationale": "Лучший logistic reservoir-кандидат; специализирован на выраженной нелинейной динамике при приемлемом времени.",
    },
    "lstm_g001": {
        "family": "lstm",
        "model_name": "lstm_forecast",
        "selection_role": "practical_main",
        "rationale": "Практический LSTM baseline с лучшим балансом точности и скорости в LSTM family.",
    },
    "chaotic_lstm_g003": {
        "family": "lstm",
        "model_name": "chaotic_lstm_forecast",
        "selection_role": "chaotic_counterpart",
        "rationale": "Хаотический LSTM сохранен как counterpart для проверки эффекта хаотической инициализации.",
    },
}


def check_shortlist_candidate_presence(
    all_candidates_df: pd.DataFrame,
    shortlist_candidate_ids: list[str],
) -> tuple[list[str], list[str]]:
    existing_ids = set(all_candidates_df.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist())
    found = [cid for cid in shortlist_candidate_ids if cid in existing_ids]
    missing = [cid for cid in shortlist_candidate_ids if cid not in existing_ids]
    return found, missing


def assign_selection_role(candidate_id: str, shortlist_metadata: dict[str, dict[str, str]]) -> str:
    return str(shortlist_metadata.get(candidate_id, {}).get("selection_role", ""))


def _safe_num_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def _build_status(candidate_df: pd.DataFrame) -> str:
    statuses = candidate_df.get("status", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    statuses = statuses[statuses != ""]
    if statuses.empty:
        return "unknown"
    if (statuses == "success").all():
        return "success"
    if (statuses == "success").any():
        return "partial_success"
    return "failed"


def _extract_notes(candidate_df: pd.DataFrame) -> str:
    if "notes" not in candidate_df.columns:
        return ""
    notes = [str(v).strip() for v in candidate_df["notes"].tolist() if pd.notna(v) and str(v).strip()]
    uniq = []
    for value in notes:
        if value not in uniq:
            uniq.append(value)
    return "; ".join(uniq[:3])


def _parse_params_json(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}, {}
    if not isinstance(value, str) or not value.strip():
        return {}, {}
    try:
        payload = json.loads(value)
    except Exception:
        return {}, {}
    model_params = payload.get("model_params", {}) if isinstance(payload, dict) else {}
    runtime_params = payload.get("runtime_params", {}) if isinstance(payload, dict) else {}
    if not isinstance(model_params, dict):
        model_params = {}
    if not isinstance(runtime_params, dict):
        runtime_params = {}
    return model_params, runtime_params


def _candidate_params_snapshot(candidate_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    for row in candidate_df.itertuples(index=False):
        raw = getattr(row, "params_json", "")
        model_params, runtime_params = _parse_params_json(raw)
        if model_params or runtime_params:
            return model_params, runtime_params
    return {}, {}


def _param_value(
    param_name: str,
    candidate_df: pd.DataFrame,
    model_params: dict[str, Any],
) -> Any:
    if param_name in model_params:
        value = model_params[param_name]
        if param_name == "hidden_dims" and isinstance(value, (list, tuple)):
            return str([int(v) for v in value])
        return value

    if param_name in candidate_df.columns:
        non_null = candidate_df[param_name].dropna()
        if not non_null.empty:
            return non_null.iloc[0]

    if param_name == "depth":
        hidden_dims = model_params.get("hidden_dims")
        if isinstance(hidden_dims, (list, tuple)):
            return int(len(hidden_dims))
    return np.nan


def build_unified_tuning_summary_table(
    all_candidates_df: pd.DataFrame,
    shortlist_candidate_ids: list[str],
    shortlist_metadata: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    selected_models_cfg: list[dict[str, Any]] = []
    missing_candidate_ids: list[str] = []

    for candidate_id in shortlist_candidate_ids:
        meta = shortlist_metadata.get(candidate_id, {})
        candidate_df = all_candidates_df[all_candidates_df["candidate_id"].astype(str) == str(candidate_id)].copy()
        found = not candidate_df.empty
        selection_role = assign_selection_role(candidate_id, shortlist_metadata)
        rationale = str(meta.get("rationale", ""))

        if not found:
            missing_candidate_ids.append(candidate_id)
            row = {
                "family": str(meta.get("family", "")),
                "model_name": str(meta.get("model_name", "")),
                "candidate_id": candidate_id,
                "source_run_name": "",
                "compare_group_id": "",
                "horizon_aggregate_type": "mean_over_horizons",
                "mae_mean_over_horizons": np.nan,
                "rmse_mean_over_horizons": np.nan,
                "mase_mean_over_horizons": np.nan,
                "directional_accuracy_mean_over_horizons": np.nan,
                "fit_time_mean_over_horizons": np.nan,
                "predict_time_mean_over_horizons": np.nan,
                "total_runtime_mean_over_horizons": np.nan,
                "status": "missing_candidate",
                "notes": f"candidate_id={candidate_id} not found in input artifacts",
                "selected_for_main_run": False,
                "selection_role": selection_role,
                "rationale": rationale,
            }
            for col in PARAMETER_COLUMNS:
                row[col] = np.nan
            rows.append(row)
            continue

        model_params, runtime_params = _candidate_params_snapshot(candidate_df)
        model_name = str(candidate_df["model_name"].iloc[0]) if "model_name" in candidate_df.columns else str(meta.get("model_name", ""))
        family = str(candidate_df["family"].iloc[0]) if "family" in candidate_df.columns else str(meta.get("family", ""))
        source_run_name = (
            str(candidate_df["source_run_name"].iloc[0]) if "source_run_name" in candidate_df.columns else ""
        )
        compare_group_id = ""
        if "compare_group_id" in candidate_df.columns:
            non_null_groups = candidate_df["compare_group_id"].dropna().astype(str)
            compare_group_id = non_null_groups.iloc[0] if not non_null_groups.empty else ""

        fit_col = "fit_time_sec_mean" if "fit_time_sec_mean" in candidate_df.columns else "fit_time"
        pred_col = "predict_time_sec_mean" if "predict_time_sec_mean" in candidate_df.columns else "predict_time"
        total_runtime_col = "total_runtime_sec" if "total_runtime_sec" in candidate_df.columns else ""

        row = {
            "family": family,
            "model_name": model_name,
            "candidate_id": candidate_id,
            "source_run_name": source_run_name,
            "compare_group_id": compare_group_id,
            "horizon_aggregate_type": "mean_over_horizons",
            "mae_mean_over_horizons": _safe_num_mean(candidate_df.get("mae_mean", pd.Series(dtype=float))),
            "rmse_mean_over_horizons": _safe_num_mean(candidate_df.get("rmse_mean", pd.Series(dtype=float))),
            "mase_mean_over_horizons": _safe_num_mean(candidate_df.get("mase_mean", pd.Series(dtype=float))),
            "directional_accuracy_mean_over_horizons": _safe_num_mean(
                candidate_df.get("directional_accuracy_mean", pd.Series(dtype=float))
            ),
            "fit_time_mean_over_horizons": _safe_num_mean(candidate_df.get(fit_col, pd.Series(dtype=float))),
            "predict_time_mean_over_horizons": _safe_num_mean(candidate_df.get(pred_col, pd.Series(dtype=float))),
            "total_runtime_mean_over_horizons": (
                _safe_num_mean(candidate_df.get(total_runtime_col, pd.Series(dtype=float)))
                if total_runtime_col
                else _safe_num_mean(candidate_df.get(fit_col, pd.Series(dtype=float)))
                + _safe_num_mean(candidate_df.get(pred_col, pd.Series(dtype=float)))
            ),
            "status": _build_status(candidate_df),
            "notes": _extract_notes(candidate_df),
            "selected_for_main_run": True,
            "selection_role": selection_role,
            "rationale": rationale,
        }
        for col in PARAMETER_COLUMNS:
            row[col] = _param_value(param_name=col, candidate_df=candidate_df, model_params=model_params)

        rows.append(row)
        selected_models_cfg.append(
            {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "family": family,
                "source_run_name": source_run_name,
                "compare_group_id": compare_group_id,
                "model_params": model_params,
                "runtime_params": runtime_params,
                "selection_role": selection_role,
                "selected_for_main_run": True,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=UNIFIED_COLUMNS)
    else:
        for col in UNIFIED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[UNIFIED_COLUMNS]
    return df, selected_models_cfg, missing_candidate_ids


def _runtime_comment(family_df: pd.DataFrame) -> str:
    runtime = pd.to_numeric(family_df["total_runtime_mean_over_horizons"], errors="coerce")
    if runtime.notna().any():
        return f"Средний runtime shortlisted-кандидатов: {runtime.mean():.2f} sec."
    return "Недостаточно данных для оценки runtime."


def _chaos_takeaway(family_df: pd.DataFrame) -> str:
    practical = family_df[family_df["selection_role"] == "practical_main"]
    chaotic = family_df[family_df["selection_role"].isin(["chaotic_counterpart", "exploratory_chaotic", "nonlinear_specialist"])]
    if practical.empty or chaotic.empty:
        return "Хаотическая ветка сохранена для targeted-анализа подмножеств рядов."
    p_mae = _safe_num_mean(practical["mae_mean_over_horizons"])
    c_mae = _safe_num_mean(chaotic["mae_mean_over_horizons"])
    if np.isfinite(p_mae) and np.isfinite(c_mae):
        delta = c_mae - p_mae
        return f"Delta MAE (chaotic - practical): {delta:.4f}; хаотические кандидаты сохранены как исследовательский контур."
    return "Сравнение chaotic vs practical неполное из-за пропусков."


def build_family_summary(unified_df: pd.DataFrame, family_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in family_order:
        family_df = unified_df[unified_df["family"].astype(str) == family].copy()
        best_practical = family_df.loc[family_df["selection_role"] == "practical_main", "candidate_id"]
        best_chaotic = family_df.loc[family_df["selection_role"] == "chaotic_counterpart", "candidate_id"]
        best_exploratory = family_df.loc[
            family_df["selection_role"].isin(["exploratory_chaotic", "nonlinear_specialist"]),
            "candidate_id",
        ]
        rows.append(
            {
                "family": family,
                "best_practical_model": str(best_practical.iloc[0]) if not best_practical.empty else "",
                "best_chaotic_model": str(best_chaotic.iloc[0]) if not best_chaotic.empty else "",
                "best_exploratory_model": str(best_exploratory.iloc[0]) if not best_exploratory.empty else "",
                "main_takeaway": (
                    "Семейство покрыто shortlist-кандидатами."
                    if not family_df.empty
                    else "В shortlist нет доступных кандидатов по семейству."
                ),
                "average_runtime_comment": _runtime_comment(family_df) if not family_df.empty else "Нет данных.",
                "chaos_takeaway": _chaos_takeaway(family_df) if not family_df.empty else "Нет данных.",
            }
        )
    return pd.DataFrame(rows)


def export_final_shortlist_config_yaml(
    output_path: str | Path,
    run_id: str,
    selected_models_cfg: list[dict[str, Any]],
    missing_candidate_ids: list[str],
) -> Path:
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "forecasting_selected_architectures",
        "version": "v1",
        "generated_by_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_models": selected_models_cfg,
        "missing_candidates": missing_candidate_ids,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return out_path


def run_architecture_tuning_final_shortlist(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    start_ts = datetime.now(timezone.utc)
    run_name = str(cfg.get("run_name", "architecture_tuning_final_shortlist_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{start_ts.strftime('%Y%m%dT%H%M%SZ')}"))
    stage_name = "architecture_tuning_final_shortlist"

    shortlist_candidate_ids = [str(x) for x in cfg.get("shortlist_candidate_ids", list(DEFAULT_SHORTLIST_METADATA.keys()))]
    shortlist_metadata = dict(DEFAULT_SHORTLIST_METADATA)
    shortlist_metadata.update(cfg.get("shortlist_metadata", {}))

    input_families = list(cfg.get("input_families", []))
    input_sources: list[str] = []
    all_candidate_frames: list[pd.DataFrame] = []
    read_dirs: list[str] = []

    for family_cfg in input_families:
        family = str(family_cfg.get("family", ""))
        source_run_name = str(family_cfg.get("source_run_name", ""))
        candidate_csv = Path(str(family_cfg.get("candidate_level_csv", ""))).resolve()
        input_sources.append(str(candidate_csv))
        read_dirs.append(str(candidate_csv.parent))
        logger.info("reading_family_artifacts family=%s source_run=%s dir=%s", family, source_run_name, candidate_csv.parent)

        if not candidate_csv.exists():
            logger.warning("candidate_level_csv_missing family=%s path=%s", family, candidate_csv)
            continue

        cdf = pd.read_csv(candidate_csv)
        cdf["family"] = family
        cdf["source_run_name"] = source_run_name
        all_candidate_frames.append(cdf)

    all_candidates_df = pd.concat(all_candidate_frames, ignore_index=True) if all_candidate_frames else pd.DataFrame()
    found_ids, missing_ids = check_shortlist_candidate_presence(
        all_candidates_df=all_candidates_df,
        shortlist_candidate_ids=shortlist_candidate_ids,
    )
    logger.info("shortlist_candidates_found=%s", found_ids)
    if missing_ids:
        logger.warning("shortlist_candidates_missing=%s", missing_ids)

    unified_df, selected_models_cfg, missing_from_unified = build_unified_tuning_summary_table(
        all_candidates_df=all_candidates_df,
        shortlist_candidate_ids=shortlist_candidate_ids,
        shortlist_metadata=shortlist_metadata,
    )
    family_summary_df = build_family_summary(unified_df=unified_df, family_order=["esn", "mlp", "logistic", "lstm"])

    outputs_cfg = dict(cfg.get("outputs", {}))
    unified_csv_path = Path(outputs_cfg["unified_summary_csv"]).resolve()
    unified_parquet_path = Path(outputs_cfg["unified_summary_parquet"]).resolve()
    family_summary_csv_path = Path(outputs_cfg["family_summary_csv"]).resolve()
    selected_arch_yaml_path = Path(outputs_cfg["selected_architectures_yaml"]).resolve()
    excel_path = Path(outputs_cfg["excel_report_path"]).resolve()
    for path in [unified_csv_path, unified_parquet_path, family_summary_csv_path, selected_arch_yaml_path, excel_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    unified_df.to_csv(unified_csv_path, index=False)
    unified_df.to_parquet(unified_parquet_path, index=False)
    family_summary_df.to_csv(family_summary_csv_path, index=False)

    selected_arch_path = export_final_shortlist_config_yaml(
        output_path=selected_arch_yaml_path,
        run_id=run_id,
        selected_models_cfg=selected_models_cfg,
        missing_candidate_ids=missing_from_unified,
    )

    roles = sorted(set(unified_df["selection_role"].dropna().astype(str).tolist()))
    summary_df = pd.DataFrame(
        [
            {
                "covered_families": ", ".join(sorted(set(unified_df["family"].dropna().astype(str).tolist()))),
                "shortlisted_models_total": int(len(unified_df)),
                "selected_for_main_run_total": int(unified_df["selected_for_main_run"].fillna(False).astype(bool).sum()),
                "roles_assigned": ", ".join(roles),
                "chaotic_vs_nonchaotic_takeaway": "Хаотические и нехаотические версии сохранены совместно; отбор не сводится к глобальному среднему MAE.",
                "runtime_takeaway": "Runtime учитывается как вторичный критерий при сохранении исследовательски значимых chaotic-кандидатов.",
                "missing_candidates": ", ".join(missing_from_unified),
            }
        ]
    )
    parameter_snapshot_df = unified_df[["candidate_id", "model_name", *PARAMETER_COLUMNS]].copy()
    readme_df = pd.DataFrame(
        [
            {"key": "run_id", "value": run_id},
            {"key": "stage", "value": stage_name},
            {"key": "config_path", "value": str(cfg.get("meta", {}).get("config_path", ""))},
            {"key": "input_sources", "value": "; ".join(input_sources)},
            {"key": "outputs", "value": f"{unified_csv_path}; {unified_parquet_path}; {family_summary_csv_path}; {selected_arch_path}; {excel_path}"},
            {"key": "missing_candidates", "value": ", ".join(missing_from_unified)},
        ]
    )
    excel_out = export_architecture_tuning_final_shortlist_excel(
        excel_path=excel_path,
        summary_df=summary_df,
        final_shortlist_df=unified_df,
        family_summary_df=family_summary_df,
        parameter_snapshot_df=parameter_snapshot_df,
        readme_df=readme_df,
    )

    end_ts = datetime.now(timezone.utc)
    missing_set = set(missing_from_unified)
    shortlist_effective = [cid for cid in shortlist_candidate_ids if cid not in missing_set]
    role_counts = unified_df["selection_role"].astype(str).value_counts().to_dict() if not unified_df.empty else {}

    manifest = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp_start": start_ts.isoformat(),
        "timestamp_end": end_ts.isoformat(),
        "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])),
        "config_path": cfg.get("meta", {}).get("config_path", ""),
        "input_sources": input_sources,
        "outputs": {
            "unified_summary_csv": str(unified_csv_path),
            "unified_summary_parquet": str(unified_parquet_path),
            "family_summary_csv": str(family_summary_csv_path),
            "selected_architectures_yaml": str(selected_arch_path),
            "excel_report": str(excel_out),
            "log": cfg.get("meta", {}).get("log_path", ""),
        },
        "summary": {
            "number_of_families": int(unified_df["family"].nunique()) if not unified_df.empty else 0,
            "number_of_shortlisted_models": int(len(shortlist_effective)),
            "number_of_practical_models": int(role_counts.get("practical_main", 0)),
            "number_of_chaotic_models": int(role_counts.get("chaotic_counterpart", 0)),
            "number_of_exploratory_models": int(
                role_counts.get("exploratory_chaotic", 0) + role_counts.get("nonlinear_specialist", 0)
            ),
            "shortlist_candidate_ids": shortlist_effective,
            "missing_candidate_ids": sorted(missing_set),
        },
    }
    manifest_path = write_manifest(
        manifest=manifest,
        manifests_dir=cfg["artifacts"]["manifests"],
        run_id=run_id,
    )

    logger.info("artifact_dirs_read=%s", sorted(set(read_dirs)))
    logger.info("shortlist_found=%s", found_ids)
    logger.info("shortlist_missing=%s", sorted(missing_set))
    logger.info("saved_unified_csv=%s", unified_csv_path)
    logger.info("saved_unified_parquet=%s", unified_parquet_path)
    logger.info("saved_family_summary_csv=%s", family_summary_csv_path)
    logger.info("saved_selected_architectures_yaml=%s", selected_arch_path)
    logger.info("saved_excel_report=%s", excel_out)
    logger.info("saved_manifest=%s", manifest_path)
    logger.info("final_shortlist_count=%d", int(len(shortlist_effective)))

    return {
        "run_id": run_id,
        "unified_summary_csv": str(unified_csv_path),
        "unified_summary_parquet": str(unified_parquet_path),
        "family_summary_csv": str(family_summary_csv_path),
        "selected_architectures_yaml": str(selected_arch_path),
        "excel_report_path": str(excel_out),
        "manifest_path": str(manifest_path),
        "shortlist_candidate_ids": shortlist_effective,
        "missing_candidate_ids": sorted(missing_set),
    }
