# Feature-Based Meta-Learning for Forecasting Model Selection in Financial Time Series

This directory is the paper-facing layer for the current final, clean/leakage-safe experiments. The scripts here only aggregate completed artifacts. They do not rerun the forecasting benchmark, architecture tuning, meta-learning, or any other expensive experiment.

## Final experimental protocol

The benchmark contains 418 log-return series: 209 Russian and 209 U.S. instruments. Eleven forecasting models are evaluated at horizons 1, 5, and 20 with horizon-specific input windows 64, 32, and 16 and three rolling-origin folds.

Forecast samples use the `target_end_lte_right_origin_v1` separation policy. A sample is retained on the left side of a temporal boundary only when its complete target window ends no later than the first forecast origin on the right side. Consequently, the outer-train and fit boundaries purge 0, 4, and 19 samples for horizons 1, 5, and 20. The final clean `split_metadata.parquet` has zero outer-train/test, fit/validation, or validation/test target-window violations.

Architecture tuning is independent of the 418-series benchmark. It used the same fixed panel of 12 U.S. ETFs (`afk.us`, `dwm.us`, `effe.us`, `fdis.us`, `iyg.us`, `jkf.us`, `jkh.us`, `mlpg.us`, `oil.us`, `qdyn.us`, `rth.us`, and `silj.us`), with zero instrument overlap with the benchmark. Final candidates are recorded in `configs/forecasting_selected_architectures_v1.yaml`; the per-family selected-series files are under `artifacts/architecture_tuning/*_v1/`.

The meta-learning objects are the 418 instruments times three forecasting folds for each horizon. Features are rebuilt from the corresponding fold's training segment only. Five repeated instrument-level partitions keep all folds of an instrument together: 292 instruments (876 objects) for meta-train, 42 instruments (126 objects) for validation, and 84 instruments (252 objects) for test in every repeat, horizon, and metric.

The final selection protocol is strictly:

1. rank candidate forecasting models and fit preprocessing/classifiers on meta-train only;
2. select one complete configuration independently for every `repeat x horizon x metric` using validation only, with metric direction and deterministic configuration order resolving ties;
3. freeze that configuration and evaluate it once on test.

Test scores never select candidate sets, fixed baselines, features, classifiers, balancing, decision rules, thresholds, or any other configuration. The clean outputs contain 30 selections and 30 one-time frozen tests (five repeats times three horizons times two metrics).

Uncertainty for the central selected-versus-fixed comparison uses a 10,000-draw cluster bootstrap over `series_id`, so the three folds and repeat appearances of the same instrument are not treated as independent observations. The corresponding artifact is `artifacts/meta_modeling/clean_meta_learning_v1/paired_uncertainty_clustered_by_series_v1.csv`.

## Numbered tables and their sources

`paper_icdm/scripts/build_paper_tables.py` regenerates Tables I-VI from these sources:

| Paper item | Final source artifacts |
| --- | --- |
| Table I | `artifacts/meta_modeling/clean_meta_learning_v1/feature_list_v2.csv`; `artifacts/features/fold_aware_features_v2_clean_batched/final_train_only_features_by_fold.parquet` |
| Table II | `configs/forecasting_selected_architectures_v1.yaml`; the four `configs/forecasting_benchmark_v2_clean_batched_batch_*.yaml` manifests |
| Table III | clean batch configs; `configs/meta_modeling_clean_v1.yaml`; clean `split_metadata.parquet` and `task_audit.parquet`; clean split/selection/bootstrap artifacts; independent tuning selected-series files |
| Table IV | `artifacts/forecasting/forecasting_benchmark_v2_clean_batched/metrics_long.parquet` |
| Table V | the same clean `metrics_long.parquet` plus `paper_icdm/model_family_mapping.csv` |
| Table VI | `selector_decisions_by_repeat_v1.csv`, `best_config_per_task_v2.csv`, and `selected_test_results_v2.csv` under `artifacts/meta_modeling/clean_meta_learning_v1/` |

Table V first averages the three fold scores for each `series_id x horizon x model`. Exact metric ties use the predeclared clean candidate order stored in the builder; they are not resolved by dynamically sorting model names.

Table VI reports the mean of the five frozen test evaluations. Directional-accuracy gain is `selected - fixed` in percentage points. RMSE is displayed in percentage log-return points and its gain is `fixed - selected`, so a negative gain means the selected route has higher RMSE.

## Current paper figure

`paper_icdm/scripts/build_paper_figures.py` generates the current Figure 2: the pooled confusion matrix for horizon 5 directional accuracy. It compares the validation-selected frozen route with the actual best candidate on each frozen test object and pools 1,260 rows (252 test objects times five repeats). Counts and row-normalized shares are shown in the cells; the supporting counts are saved beside the PNG/PDF.

The former RMSE line chart, directional-accuracy line chart, and winner-distribution chart are from the previous paper layout. They are deliberately removed and are no longer presented as Figures 2-4. The current builder does not invent replacements for manuscript figures whose current specification is not present in the repository.

## Supporting analyses and scope

- **Market heterogeneity (RU/US).** Both clean benchmark rows and frozen routing rows retain instrument/market identity. RU and US slices should be reported separately before interpreting pooled averages; the balanced 209/209 benchmark design does not imply homogeneous effects across markets.

- **Feature-family ablation.** The canonical completed artifact is `artifacts/meta_modeling/clean_meta_learning_feature_ablation_v1_repaired/feature_ablation_summary.csv`. It compares `full_25`, `standard`, `without_phase_space`, and `nonlinear_only` under the same meta-train/validation/frozen-test protocol. The repaired run preserved successful validation rows, completed the missing technical failures, reselected on the complete validation grid, and recomputed all 120 frozen tests; the abandoned partial v2 directory is not a final source.

- **Price-scale-transition sensitivity.** The post-hoc check transforms stored log-return targets and predictions with `exp(x) - 1`, without retraining. Directional accuracy is sign-invariant under this monotone zero-preserving transform, while RMSE/MAE change scale and may change local rankings. `artifacts/review_simple_returns_analysis/` is an older sensitivity snapshot based on the pre-clean benchmark path; it supports the qualitative diagnostic only and is not a numerical source for current Tables IV-VI.

- **Feature stability.** Stability is assessed across the three rolling-origin training folds using rank/linear association and distribution-shift summaries. For current paper claims, the source feature matrix is `artifacts/features/fold_aware_features_v2_clean_batched/final_train_only_features_by_fold.parquet`. The older `artifacts/review_statistical_analysis/feature_stability_*` files predate the clean rebuild and are not final numbered-table sources.

- **Chaos-model train/validation/test diagnostic.** `artifacts/review_overfitting_diagnostics_full/` is a family-level sensitivity diagnostic of train/validation/test gaps, not proof for or against overfitting of any single final architecture. It predates the final independent retuning, and some architectures were subsequently retuned (including the final ESN, chaotic ESN, and chaotic LSTM settings). Therefore its model-specific parameter values must not be presented as the final Table II architectures; only the broad family-level sensitivity interpretation is retained.

## Cheap paper-only rebuild and validation

Run from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe paper_icdm\scripts\build_paper_tables.py
.\.venv\Scripts\python.exe paper_icdm\scripts\build_paper_figures.py
.\.venv\Scripts\python.exe paper_icdm\scripts\check_artifacts.py
```

These commands read existing Parquet/CSV/YAML artifacts and write only under `paper_icdm/tables` and `paper_icdm/figures`. They do not execute forecasting, tuning, feature-ablation training, or meta-learning.

Raw market data, if a full reconstruction is intentionally required outside this paper-only workflow, comes from the U.S. Stocks/ETFs and Russian equities datasets documented in the repository-level configuration and data pipeline.
