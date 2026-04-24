from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.meta_modeling.experimental_pipeline import (
    METRIC_DIRECTIONS,
    _evaluate_predictions,
    _normalize_metric_alias,
    _pick_best_index,
    _train_only_feature_selection,
    build_candidate_sets,
    build_meta_dataset,
    build_task,
    compute_best_single_baseline,
    repeat_seeds,
    split_by_series_ids,
)
from src.meta_modeling.models import build_meta_classifier
from src.reporting.excel_export import export_meta_modeling_excel
from src.utils.manifest import get_git_commit, write_manifest


def run_meta_modeling_experiments(cfg: dict[str, Any], logger: Any) -> dict[str, Any]:
    run_name = str(cfg.get("run_name", "meta_modeling_experiments_v1"))
    run_id = str(cfg.get("meta", {}).get("run_id", f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    target_metrics = [_normalize_metric_alias(x) for x in cfg.get("target_metrics", ["rmse", "directional_accuracy"])]
    classifiers = [str(x) for x in cfg.get("classification_models", ["logistic_regression", "random_forest_classifier", "catboost_classifier"])]
    balancing_modes = [str(x) for x in cfg.get("balancing_modes", ["default", "balanced"])]
    thresholds = [float(x) for x in cfg.get("confidence_thresholds", [0.50, 0.60, 0.70, 0.80])]

    meta_long_df, ds = build_meta_dataset(cfg)
    join_key = ds["join_key"]
    seeds = repeat_seeds(cfg.get("split", {}))

    full_features = list(ds["feature_cols"])
    basic_features = list(ds.get("basic_feature_cols", []))
    full_plus_basic = list(dict.fromkeys(full_features + basic_features))
    feature_sets = {
        "full_features": full_features,
        "full_plus_basic_features": full_plus_basic,
        "selected_features": full_plus_basic,
    }

    candidate_df = build_candidate_sets(
        meta_long_df,
        join_key=join_key,
        target_metrics=target_metrics,
        top_k_values=[int(x) for x in cfg.get("candidate_selection", {}).get("top_k_values", [3, 4, 5, 6])],
        closeness_tolerance=float(cfg.get("candidate_selection", {}).get("closeness_tolerance", 0.05)),
    )

    routing_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_single_rows: list[dict[str, Any]] = []
    probs_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    class_dist_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    catboost_root = Path(str(cfg.get("artifacts", {}).get("meta_modeling", "artifacts/meta_modeling"))).resolve() / "catboost_info" / run_id
    catboost_root.mkdir(parents=True, exist_ok=True)

    for feature_set_name, feature_cols in feature_sets.items():
        for horizon in sorted(int(x) for x in ds["horizons"]):
            for metric in target_metrics:
                csub = candidate_df[(candidate_df["horizon"] == int(horizon)) & (candidate_df["target_metric"] == metric)].copy()
                for cset, g in csub.groupby("candidate_set", sort=True):
                    models = g.sort_values("candidate_rank", kind="stable")["model_name"].astype(str).tolist()
                    task = build_task(
                        meta_long_df,
                        join_key=join_key,
                        feature_cols=feature_cols,
                        horizon=horizon,
                        target_metric=metric,
                        feature_set=feature_set_name,
                        candidate_set=str(cset),
                        candidate_models=models,
                    )
                    dataset_rows.append(
                        {
                            "run_id": run_id,
                            "horizon": int(task.horizon),
                            "target_metric": task.target_metric,
                            "feature_set": task.feature_set,
                            "candidate_set": task.candidate_set,
                            "candidate_size": int(len(task.model_order)),
                            "n_series": int(task.X.shape[0]),
                            "n_features": int(task.X.shape[1]),
                            "n_models": int(task.y.shape[1]),
                            "feature_columns_json": json.dumps(task.feature_cols),
                            "model_order_json": json.dumps(task.model_order),
                        }
                    )
                    for i, m in enumerate(task.model_order):
                        mapping_rows.append({"horizon": task.horizon, "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "class_idx": i, "model_name": m})

                    for repeat_id, seed in enumerate(seeds, start=1):
                        split = split_by_series_ids(task.row_ids, cfg=cfg, random_seed=int(seed))
                        marker = np.full(task.row_ids.shape[0], "test", dtype=object)
                        marker[split.train] = "train"
                        marker[split.validation] = "validation"
                        for i, sid in enumerate(task.row_ids):
                            split_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "series_id": str(sid), "split": str(marker[i])})

                        X_train = task.X[split.train]
                        y_train = task.y[split.train]
                        X_test = task.X[split.test]
                        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
                            continue
                        direction = METRIC_DIRECTIONS[task.target_metric]
                        y_train_cls = np.array([_pick_best_index(row, direction) for row in y_train], dtype=int)
                        feat_idx = np.arange(len(task.feature_cols), dtype=int)
                        selected_names = list(task.feature_cols)
                        meta_sel = {"method": "none", "top_n": len(selected_names), "min_features": 0}
                        if feature_set_name == "selected_features":
                            feat_idx, selected_names, meta_sel = _train_only_feature_selection(X_train, y_train_cls, task.feature_cols, cfg)
                        Xtr = X_train[:, feat_idx]
                        Xte = X_test[:, feat_idx]
                        selected_feature_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "selection_method": str(meta_sel.get("method", "")), "selection_top_n": int(meta_sel.get("top_n", len(selected_names))), "selection_min_features": int(meta_sel.get("min_features", 0)), "selected_feature_count": int(len(selected_names)), "selected_features_json": json.dumps(selected_names)})
                        baseline_idx = int(compute_best_single_baseline(y_train=y_train, y_test=task.y[split.test], model_order=task.model_order, direction=direction)["baseline_idx"])

                        for clf in classifiers:
                            for balancing in balancing_modes:
                                model = build_meta_classifier(model_name=clf, cfg=cfg, balancing_mode=balancing, catboost_train_dir=str(catboost_root / f"{clf}_{balancing}_h{task.horizon}_{task.target_metric}"))
                                if len(np.unique(y_train_cls)) < 2:
                                    class_probs = np.zeros((Xte.shape[0], len(task.model_order)), dtype=np.float64)
                                    class_probs[:, int(y_train_cls[0])] = 1.0
                                else:
                                    model.fit(Xtr, y_train_cls)
                                    class_probs = np.asarray(model.predict_proba(Xte), dtype=np.float64)
                                    classes_ = np.asarray(getattr(model, "classes_", np.arange(class_probs.shape[1]))).astype(int)
                                    if class_probs.shape[1] != len(task.model_order):
                                        aligned = np.zeros((class_probs.shape[0], len(task.model_order)), dtype=np.float64)
                                        for src, cls_idx in enumerate(classes_):
                                            if 0 <= int(cls_idx) < len(task.model_order):
                                                aligned[:, int(cls_idx)] = class_probs[:, src]
                                        class_probs = aligned
                                top1 = np.argmax(class_probs, axis=1).astype(int)
                                top2 = np.argsort(class_probs, axis=1)[:, -2] if class_probs.shape[1] >= 2 else top1
                                conf = np.max(class_probs, axis=1)
                                prob_json = [json.dumps({m: float(v) for m, v in zip(task.model_order, row)}) for row in class_probs]
                                for i, sid in enumerate(task.row_ids[split.test]):
                                    probs_rows.append({"run_id": run_id, "repeat_id": int(repeat_id), "split_seed": int(seed), "method": "classification", "model": clf, "horizon": int(task.horizon), "target_metric": task.target_metric, "feature_set": task.feature_set, "candidate_set": task.candidate_set, "candidate_size": int(len(task.model_order)), "balancing_mode": balancing, "series_id": str(sid), "predicted_top1_model": task.model_order[int(top1[i])], "predicted_top2_model": task.model_order[int(top2[i])], "confidence": float(conf[i]), "class_probability_json": prob_json[i]})
                                for rule, thr, sel in [("top_1", None, top1)] + [(f"confidence_fallback_{t:.2f}", float(t), np.where(conf >= float(t), top1, baseline_idx).astype(int)) for t in thresholds]:
                                    rdf, summ, brow, pclass, cmat, cdist, cconf = _evaluate_predictions(run_id=run_id, repeat_id=repeat_id, split_seed=int(seed), method="classification", model_name=clf, task=task, split=split, selected_idx=sel, predicted_scores=class_probs if direction == "max" else -class_probs, top2_idx=top2, confidence=conf, class_probability_json=prob_json, balancing_mode=balancing, decision_rule=rule, confidence_threshold=thr, selected_feature_count=int(len(selected_names)))
                                    routing_rows.append(rdf)
                                    summary_rows.append(summ)
                                    best_single_rows.append(brow)
                                    per_class_rows.extend(pclass)
                                    confusion_rows.extend(cmat)
                                    class_dist_rows.extend(cdist)
                                    confidence_rows.append(cconf)

    task_results_df = pd.DataFrame(summary_rows)
    routing_df = pd.concat(routing_rows, ignore_index=True) if routing_rows else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    best_single_df = pd.DataFrame(best_single_rows).drop_duplicates(subset=["repeat_id", "split_seed", "horizon", "target_metric", "feature_set", "candidate_set", "balancing_mode", "decision_rule", "best_single_model"], keep="first")
    probs_df = pd.DataFrame(probs_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    class_dist_df = pd.DataFrame(class_dist_rows)
    confidence_df = pd.DataFrame(confidence_rows)
    selected_features_df = pd.DataFrame(selected_feature_rows)
    success_df = task_results_df[task_results_df["status"].astype(str) == "success"].copy() if not task_results_df.empty else pd.DataFrame()
    repeat_agg_df = success_df.groupby(["model", "horizon", "target_metric", "candidate_set", "feature_set", "balancing_mode", "decision_rule", "confidence_threshold"], dropna=False, sort=True).agg(n_repeats=("repeat_id", "nunique"), achieved_metric_mean=("achieved_metric", "mean"), achieved_metric_std=("achieved_metric", "std"), best_single_metric_mean=("best_single_metric", "mean"), oracle_metric_mean=("oracle_metric", "mean"), improvement_mean=("improvement_vs_best_single", "mean"), gap_mean=("gap_to_oracle", "mean"), routing_hit_oracle_rate_mean=("routing_hit_oracle_rate", "mean"), accuracy_mean=("classification_accuracy", "mean"), balanced_accuracy_mean=("balanced_accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"), weighted_f1_mean=("weighted_f1", "mean"), top2_hit_rate_mean=("top2_hit_rate", "mean"), fallback_rate_mean=("fallback_rate", "mean")).reset_index() if not success_df.empty else pd.DataFrame()
    comparison_df = repeat_agg_df.copy()
    best_cfg_df = repeat_agg_df.sort_values(["horizon", "target_metric", "improvement_mean", "gap_mean"], ascending=[True, True, False, True], kind="stable").groupby(["horizon", "target_metric"], sort=False).head(1).reset_index(drop=True) if not repeat_agg_df.empty else pd.DataFrame()

    output_cfg = cfg["outputs"]
    for k, v in output_cfg.items():
        if k.endswith("_path"):
            Path(str(v)).resolve().parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_cfg["meta_dataset_summary_csv_path"], index=False)
    dataset_df.to_parquet(output_cfg["meta_dataset_summary_parquet_path"], index=False)
    mapping_df.to_csv(output_cfg["model_order_mapping_csv_path"], index=False)
    routing_df.to_csv(output_cfg["routing_rows_csv_path"], index=False)
    routing_df.to_parquet(output_cfg["routing_rows_parquet_path"], index=False)
    task_results_df.to_csv(output_cfg["task_results_csv_path"], index=False)
    task_results_df.to_parquet(output_cfg["task_results_parquet_path"], index=False)
    split_df.to_csv(output_cfg["split_assignments_csv_path"], index=False)
    repeat_agg_df.to_csv(output_cfg["repeat_aggregated_results_csv_path"], index=False)
    best_single_df.to_csv(output_cfg["best_single_baseline_by_repeat_csv_path"], index=False)
    comparison_df.to_csv(output_cfg["comparison_table_csv_path"], index=False)
    candidate_df.to_csv(output_cfg["candidate_models_csv_path"], index=False)
    probs_df.to_csv(output_cfg["classification_probabilities_csv_path"], index=False)
    pd.DataFrame({"feature_name": full_features, "feature_set": "full_features"}).to_csv(output_cfg["feature_list_csv_path"], index=False)
    pd.DataFrame({"feature_name": full_plus_basic, "feature_set": "full_plus_basic_features"}).to_csv(output_cfg["pruned_feature_list_csv_path"], index=False)
    per_class_df.to_csv(output_cfg["per_class_metrics_csv_path"], index=False)
    confusion_df.to_csv(output_cfg["confusion_matrix_csv_path"], index=False)
    class_dist_df.to_csv(output_cfg["class_distribution_csv_path"], index=False)
    confidence_df.to_csv(output_cfg["confidence_summary_csv_path"], index=False)
    selected_features_df.to_csv(output_cfg["selected_features_csv_path"], index=False)
    candidate_df.groupby(["horizon", "target_metric", "candidate_set"], dropna=False).size().reset_index(name="n_models").to_csv(output_cfg["candidate_filtering_summary_csv_path"], index=False)
    dataset_df.groupby(["horizon", "target_metric", "feature_set"], dropna=False).agg(n_series=("n_series", "max"), n_features=("n_features", "max")).reset_index().to_csv(output_cfg["feature_regime_summary_csv_path"], index=False)
    best_cfg_df.to_csv(output_cfg["best_config_per_task_csv_path"], index=False)
    confident_examples_df = pd.concat([routing_df[(routing_df["routing_hit_oracle"] == 1) & (pd.to_numeric(routing_df["confidence"], errors="coerce") >= 0.7)].head(50), routing_df[(routing_df["routing_hit_oracle"] == 0) & (pd.to_numeric(routing_df["confidence"], errors="coerce") >= 0.7)].head(50)], ignore_index=True) if not routing_df.empty else pd.DataFrame()
    confident_examples_df.to_csv(output_cfg["confident_examples_csv_path"], index=False)
    with Path(output_cfg["feature_manifest_json_path"]).open("w", encoding="utf-8") as f:
        json.dump({"full_features": full_features, "basic_features": basic_features, "full_plus_basic_features": full_plus_basic, "selection_method": str(cfg.get("feature_selection", {}).get("method", "mutual_info"))}, f, ensure_ascii=False, indent=2)

    def _preview(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty or len(df) <= n:
            return df
        return df.head(n).copy()

    excel_path = export_meta_modeling_excel(
        excel_path=output_cfg["excel_report_path"],
        summary_df=_preview(repeat_agg_df, 50000),
        task_results_df=_preview(task_results_df, 200000),
        routing_df=_preview(routing_df, 200000),
        model_order_df=mapping_df,
        dataset_summary_df=dataset_df,
        split_df=_preview(split_df, 200000),
        repeat_agg_df=_preview(repeat_agg_df, 50000),
        feature_list_df=pd.read_csv(output_cfg["feature_list_csv_path"]) if Path(output_cfg["feature_list_csv_path"]).exists() else None,
        best_single_repeat_df=_preview(best_single_df, 200000),
        comparison_df=_preview(comparison_df, 50000),
        candidates_df=candidate_df,
        prediction_examples_df=confident_examples_df,
        classification_probabilities_df=_preview(probs_df, 200000),
        pruned_feature_list_df=pd.read_csv(output_cfg["pruned_feature_list_csv_path"]) if Path(output_cfg["pruned_feature_list_csv_path"]).exists() else None,
    )

    manifest = {"run_id": run_id, "stage": str(cfg.get("stage", "meta_modeling_experiments")), "timestamp_start": datetime.now(timezone.utc).isoformat(), "timestamp_end": datetime.now(timezone.utc).isoformat(), "git_commit": get_git_commit(Path(cfg["meta"]["project_root"])), "config_path": cfg["meta"]["config_path"], "inputs_used": {"features_path": ds["feature_source_path"], "basic_feature_source_path": ds.get("basic_feature_source_path", ""), "forecasting_series_metrics_path": ds.get("forecasting_series_metrics_path", ""), "forecasting_manifest_path": ds.get("forecasting_manifest_path", "")}, "outputs": output_cfg, "summary": {"n_rows_meta_long": int(ds["n_rows"]), "n_series": int(ds["n_series"]), "horizons": ds["horizons"], "target_metrics": target_metrics, "candidate_set_sizes": sorted(candidate_df["candidate_set"].astype(str).unique().tolist()) if not candidate_df.empty else [], "classifiers": classifiers, "balancing_modes": balancing_modes, "feature_regimes": list(feature_sets.keys()), "confidence_thresholds": thresholds, "repeat_splits": len(seeds), "repeat_seeds": seeds, "tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "tasks_failed": int((task_results_df.get("status", pd.Series(dtype=str)) == "failed").sum())}}
    manifest_path = write_manifest(manifest=manifest, manifests_dir=cfg["artifacts"]["manifests"], run_id=run_id)
    logger.info("classification meta-modeling run_id=%s tasks_total=%d success=%d", run_id, len(task_results_df), int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()))
    return {"run_id": run_id, "manifest_path": str(manifest_path), "comparison_table_path": output_cfg["comparison_table_csv_path"], "candidate_models_path": output_cfg["candidate_models_csv_path"], "classification_probabilities_path": output_cfg["classification_probabilities_csv_path"], "excel_report_path": str(excel_path), "tasks_total": int(len(task_results_df)), "tasks_success": int((task_results_df.get("status", pd.Series(dtype=str)) == "success").sum()), "best_config_path": output_cfg["best_config_per_task_csv_path"], "dataset_summary": ds}
