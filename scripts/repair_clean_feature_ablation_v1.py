"""Repair only failed CatBoost validation configurations from ablation v1.

The original successful validation rows are immutable inputs.  This script
fits only the exact rows that failed because CatBoost could not create its
overlong Windows train directory, then re-selects validation winners and
performs one frozen test evaluation per completed task.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.meta_modeling.classification_pipeline import (
    _build_task_arrays_v2,
    _candidate_set_from_train_v2,
    _evaluate_v2,
    _load_meta_dataset_v2,
    _metric_direction,
    _pick_best_index,
    _predict_aligned_probabilities,
    _select_validation_configuration,
    _split_object_indices_by_series,
)
from src.meta_modeling.experimental_pipeline import repeat_seeds
from src.meta_modeling.models import build_meta_classifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/meta_modeling/clean_meta_learning_feature_ablation_v1"
OUT = ROOT / "artifacts/meta_modeling/clean_meta_learning_feature_ablation_v1_repaired"
CONFIG = ROOT / "configs/meta_modeling_clean_feature_ablation_v1.yaml"
RUN_ID = "clean_meta_learning_feature_ablation_v1_20260819T144537Z"


def _rule_specs(cfg: dict) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = [("top_1", None)] if "top_1" in cfg["decision_rules"] else []
    for threshold in cfg["confidence_thresholds"]:
        specs.append((f"confidence_fallback_{float(threshold):.2f}", float(threshold)))
    return specs


def _feature_sets(cfg: dict) -> dict[str, list[str]]:
    names = cfg["feature_sets"]
    definitions = cfg["feature_set_definitions"]
    return {name: list(definitions[name]) for name in names}


def _task_context(cfg: dict, meta: pd.DataFrame, *, feature_set: str, repeat_id: int, horizon: int, metric: str, seed: int):
    features = _feature_sets(cfg)[feature_set]
    horizon_df = meta[meta["horizon"].astype(int) == int(horizon)].copy()
    object_base = horizon_df[["object_id", "series_id", "horizon", "fold_id", *features]].drop_duplicates("object_id").reset_index(drop=True)
    train_base, _, _, _ = _split_object_indices_by_series(object_base, cfg, seed)
    train_ids = set(object_base.iloc[train_base]["object_id"].astype(str))
    train_long = horizon_df[horizon_df["object_id"].astype(str).isin(train_ids)].copy()
    return features, train_long


def _arrays(cfg: dict, meta: pd.DataFrame, *, feature_set: str, repeat_id: int, horizon: int, metric: str, seed: int, top_k: int):
    features, train_long = _task_context(cfg, meta, feature_set=feature_set, repeat_id=repeat_id, horizon=horizon, metric=metric, seed=seed)
    candidate = _candidate_set_from_train_v2(train_long, horizon=horizon, metric=metric, top_k=top_k)
    names = candidate.sort_values("candidate_rank", kind="stable")["model_name"].astype(str).tolist()
    object_df, X, y, model_order = _build_task_arrays_v2(meta, horizon=horizon, metric=metric, feature_cols=features, candidate_models=names)
    train_idx, val_idx, test_idx, split_meta = _split_object_indices_by_series(object_df, cfg, seed)
    direction = _metric_direction(metric)
    y_cls = np.array([_pick_best_index(row, direction) for row in y[train_idx]], dtype=int)
    fixed_idx = _pick_best_index(np.nanmean(y[train_idx], axis=0), direction)
    return candidate, object_df, X, y, model_order, train_idx, val_idx, test_idx, y_cls, fixed_idx, split_meta


def _fit_probs(cfg: dict, *, classifier: str, balancing: str, X_train: np.ndarray, y_cls: np.ndarray, X_eval: np.ndarray, n_models: int, train_dir: Path):
    if len(np.unique(y_cls)) < 2:
        return None, int(y_cls[0]), _predict_aligned_probabilities(None, int(y_cls[0]), X_eval, n_models)
    model = build_meta_classifier(classifier, cfg, balancing_mode=balancing, catboost_train_dir=str(train_dir) if classifier == "catboost_classifier" else None)
    model.fit(X_train, y_cls)
    return model, None, _predict_aligned_probabilities(model, None, X_eval, n_models)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CONFIG.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["meta"] = {"project_root": str(ROOT), "config_path": str(CONFIG)}
    cfg["artifacts"] = {"manifests": str(ROOT / "artifacts/manifests")}
    cfg["output_dir"] = str(OUT)
    cfg["report_dir"] = str(ROOT / "artifacts/reports/clean_meta_learning_feature_ablation_v1_repaired")

    original = pd.read_csv(SOURCE / "task_results_v2.csv")
    key_cols = ["feature_set", "repeat_id", "horizon", "target_metric", "config_id"]
    failed = original[original["status"].eq("failed")].copy()
    success = original[original["status"].eq("success")].copy()
    failed.to_csv(OUT / "failed_validation_keys_v1.csv", index=False)
    if len(failed) != 2750 or len(success) != 11650:
        raise RuntimeError(f"Unexpected source coverage: success={len(success)} failed={len(failed)}")
    if failed.duplicated(key_cols).any() or success.duplicated(key_cols).any():
        raise RuntimeError("Source validation keys are not unique")
    if not failed["model"].eq("catboost_classifier").all():
        raise RuntimeError("Repair scope must contain only CatBoost failures")

    meta, ds = _load_meta_dataset_v2(cfg)
    seeds = repeat_seeds({"random_seed": cfg["random_seed"], "n_repeats": cfg["n_repeats"], "random_seeds": cfg["split"]["random_seeds"]})
    repaired: list[dict] = []
    cb_root = OUT / "_catboost"
    cb_root.mkdir(exist_ok=True)

    fit_keys = ["feature_set", "repeat_id", "horizon", "target_metric", "candidate_set", "balancing_mode"]
    for fit_key, group in failed.groupby(fit_keys, sort=True):
        feature_set, repeat_id, horizon, metric, candidate_set, balancing = fit_key
        repeat_id, horizon = int(repeat_id), int(horizon)
        seed = int(seeds[repeat_id - 1])
        top_k = int(str(candidate_set).split("_")[1])
        candidate, object_df, X, y, model_order, train_idx, val_idx, _, y_cls, fixed_idx, _ = _arrays(
            cfg, meta, feature_set=feature_set, repeat_id=repeat_id, horizon=horizon, metric=metric, seed=seed, top_k=top_k
        )
        model, constant, probs = _fit_probs(
            cfg, classifier="catboost_classifier", balancing=balancing,
            X_train=X[train_idx], y_cls=y_cls, X_eval=X[val_idx], n_models=len(model_order),
            train_dir=cb_root / f"fs-{feature_set}_r{repeat_id}_h{horizon}_{metric}_k{top_k}_{balancing}",
        )
        del model, constant
        top1 = np.argmax(probs, axis=1).astype(int)
        top2 = np.argsort(probs, axis=1)[:, -2] if probs.shape[1] >= 2 else top1
        conf = np.max(probs, axis=1)
        for _, old in group.iterrows():
            threshold = None if old["decision_rule"] == "top_1" else float(old["confidence_threshold"])
            selected = top1 if threshold is None else np.where(conf >= threshold, top1, fixed_idx).astype(int)
            _, summary, _ = _evaluate_v2(
                run_id=RUN_ID, repeat_id=repeat_id, seed=seed, model_name="catboost_classifier", horizon=horizon, metric=metric,
                feature_set=feature_set, candidate_set=candidate_set, object_df=object_df, y=y, model_order=model_order,
                train_idx=train_idx, eval_idx=val_idx, evaluation_partition="validation", fixed_idx=fixed_idx,
                selected_idx=selected, class_probs=probs, top2_idx=top2, confidence=conf, balancing=balancing,
                decision_rule=str(old["decision_rule"]), threshold=threshold, selected_feature_count=len(_feature_sets(cfg)[feature_set]),
                config_id=str(old["config_id"]), config_order=int(old["config_order"]),
            )
            summary.update({"selection_metric": "achieved_metric", "selection_direction": _metric_direction(metric), "selected_by_validation": 0, "status": "success", "notes": "repaired_catboost_train_dir"})
            repaired.append(summary)

    repaired_df = pd.DataFrame(repaired)
    repaired_df.to_csv(OUT / "repaired_validation_rows.csv", index=False)
    combined = pd.concat([success, repaired_df], ignore_index=True, sort=False)
    combined["selected_by_validation"] = 0
    if len(combined) != 14400 or combined.duplicated(key_cols).any() or combined["status"].ne("success").any():
        raise RuntimeError("Validation repair did not produce a complete, unique successful grid")

    selections: list[dict] = []
    tests: list[dict] = []
    routing: list[pd.DataFrame] = []
    for (feature_set, repeat_id, horizon, metric), task in combined.groupby(["feature_set", "repeat_id", "horizon", "target_metric"], sort=True):
        direction = _metric_direction(metric)
        chosen = _select_validation_configuration(task, direction)
        mask = (combined["feature_set"] == feature_set) & (combined["repeat_id"] == repeat_id) & (combined["horizon"] == horizon) & (combined["target_metric"] == metric) & (combined["config_id"] == chosen["config_id"])
        combined.loc[mask, "selected_by_validation"] = 1
        selection = chosen.to_dict()
        selection.pop("_selection_metric", None)
        selection.update({"selected_by_validation": 1, "validation_metric": float(chosen["achieved_metric"]), "selection_scope": "feature_set_x_repeat_x_horizon_x_metric", "test_used_for_selection": False})
        selections.append(selection)

        seed = int(seeds[int(repeat_id) - 1])
        top_k = int(str(chosen["candidate_set"]).split("_")[1])
        candidate, object_df, X, y, model_order, train_idx, _, test_idx, y_cls, fixed_idx, split_meta = _arrays(
            cfg, meta, feature_set=feature_set, repeat_id=int(repeat_id), horizon=int(horizon), metric=metric, seed=seed, top_k=top_k
        )
        classifier, balancing = str(chosen["model"]), str(chosen["balancing_mode"])
        _, _, probs = _fit_probs(
            cfg, classifier=classifier, balancing=balancing, X_train=X[train_idx], y_cls=y_cls, X_eval=X[test_idx], n_models=len(model_order),
            train_dir=cb_root / f"test_fs-{feature_set}_r{repeat_id}_h{horizon}_{metric}_k{top_k}_{balancing}",
        )
        top1 = np.argmax(probs, axis=1).astype(int)
        top2 = np.argsort(probs, axis=1)[:, -2] if probs.shape[1] >= 2 else top1
        conf = np.max(probs, axis=1)
        threshold = None if chosen["decision_rule"] == "top_1" else float(chosen["confidence_threshold"])
        selected = top1 if threshold is None else np.where(conf >= threshold, top1, fixed_idx).astype(int)
        rdf, summary, _ = _evaluate_v2(
            run_id=RUN_ID, repeat_id=int(repeat_id), seed=seed, model_name=classifier, horizon=int(horizon), metric=metric,
            feature_set=feature_set, candidate_set=str(chosen["candidate_set"]), object_df=object_df, y=y, model_order=model_order,
            train_idx=train_idx, eval_idx=test_idx, evaluation_partition="test", fixed_idx=fixed_idx, selected_idx=selected,
            class_probs=probs, top2_idx=top2, confidence=conf, balancing=balancing, decision_rule=str(chosen["decision_rule"]),
            threshold=threshold, selected_feature_count=len(_feature_sets(cfg)[feature_set]), config_id=str(chosen["config_id"]), config_order=int(chosen["config_order"]),
        )
        rdf["selected_by_validation"] = 1
        rdf["validation_metric"] = float(chosen["achieved_metric"])
        routing.append(rdf)
        summary.update({"selected_by_validation": 1, "validation_metric": float(chosen["achieved_metric"]), "test_evaluations_for_task": 1, "split_overlap_series": int(split_meta["overlap_series"])})
        tests.append(summary)

    selection_df, test_df = pd.DataFrame(selections), pd.DataFrame(tests)
    routing_df = pd.concat(routing, ignore_index=True)
    if len(selection_df) != 120 or len(test_df) != 120 or test_df["split_overlap_series"].ne(0).any():
        raise RuntimeError("Selection/test coverage or split isolation failed")
    combined.to_csv(OUT / "task_results_v2_combined.csv", index=False)
    combined.to_parquet(OUT / "task_results_v2_combined.parquet", index=False)
    selection_df.to_csv(OUT / "best_config_per_task_reselected.csv", index=False)
    test_df.to_csv(OUT / "selected_test_results_recomputed.csv", index=False)
    routing_df.to_csv(OUT / "selector_decisions_recomputed.csv", index=False)

    summary = test_df.groupby(["feature_set", "horizon", "target_metric"], sort=True).agg(
        Fixed=("best_single_metric", "mean"), Selected=("achieved_metric", "mean"), Oracle=("oracle_metric", "mean"),
        Fixed_SD=("best_single_metric", "std"), Selected_SD=("achieved_metric", "std"), Oracle_SD=("oracle_metric", "std"),
    ).reset_index()
    summary["gain_selected_vs_fixed"] = np.where(summary["target_metric"].eq("rmse"), summary["Fixed"] - summary["Selected"], summary["Selected"] - summary["Fixed"])
    full = summary[summary["feature_set"].eq("full_25")][["horizon", "target_metric", "gain_selected_vs_fixed"]].rename(columns={"gain_selected_vs_fixed": "full_25_gain"})
    summary = summary.merge(full, on=["horizon", "target_metric"], how="left")
    summary["gain_change_vs_full_25"] = summary["gain_selected_vs_fixed"] - summary["full_25_gain"]
    summary.to_csv(OUT / "feature_ablation_summary.csv", index=False)
    h5 = test_df[(test_df["target_metric"].eq("directional_accuracy")) & (test_df["horizon"] == 5)].copy()
    h5["gain"] = h5["achieved_metric"] - h5["best_single_metric"]
    h5[["feature_set", "repeat_id", "gain"]].to_csv(OUT / "da_h5_repeat_gains.csv", index=False)
    checks = {
        "source_success_rows_preserved": int(len(success)), "failed_rows_recomputed": int(len(repaired_df)),
        "validation_rows": int(len(combined)), "validation_duplicates": int(combined.duplicated(key_cols).sum()),
        "validation_missing": int(14400 - len(combined)), "validation_failures": int(combined["status"].ne("success").sum()),
        "final_selections": int(len(selection_df)), "frozen_tests_recomputed": int(len(test_df)),
        "test_series_overlap_max": int(test_df["split_overlap_series"].max()),
        "clean_inputs": ds["coverage_checks"],
    }
    with (OUT / "repair_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, ensure_ascii=False, indent=2)
    for name in ["split_assignments_v2.csv", "candidate_models_v2.csv", "meta_dataset_summary_v2.csv", "model_order_mapping_v2.csv", "feature_list_v2.csv"]:
        shutil.copy2(SOURCE / name, OUT / name)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
