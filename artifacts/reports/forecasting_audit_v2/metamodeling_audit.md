# Meta-modeling audit for forecasting_benchmark_v2

Date: 2026-04-27

## Scope

Full meta-modeling experiments were not run. The audit inspected the current implementation and ran only a lightweight diagnostic that reads forecasting metrics and series features.

Checked implementation files:

- `src/cli/run_meta_modeling.py`
- `src/cli/run_meta_modeling_experiments.py`
- `src/config/loader.py`
- `src/meta_modeling/__init__.py`
- `src/meta_modeling/pipeline.py`
- `src/meta_modeling/experimental_pipeline.py`
- `src/meta_modeling/classification_pipeline.py`
- `src/meta_modeling/models.py`
- `src/reporting/excel_export.py`
- `tests/test_meta_modeling_pipeline.py`
- `tests/test_meta_modeling_experimental_pipeline.py`
- `configs/meta_modeling_v1.yaml`
- `configs/meta_modeling_experiments_v1.yaml`
- `.gitignore`

New diagnostic files added:

- `src/meta_modeling/diagnostics.py`
- `src/cli/run_metamodeling_diagnostic.py`
- `artifacts/reports/forecasting_audit_v2/metamodeling_diagnostic.json`

## Current pipeline structure

There are two user-facing CLI entry points:

- `python -m src.cli.run_meta_modeling --config configs/meta_modeling_v1.yaml`
- `python -m src.cli.run_meta_modeling_experiments --config configs/meta_modeling_experiments_v1.yaml`

The base pipeline in `src/meta_modeling/pipeline.py` is regression-style routing. It builds a multi-output target matrix where each row is a series and each output dimension is one forecasting model's metric value. A meta-regressor predicts the metric vector on held-out series, then routing selects the best predicted model.

The experiments CLI imports `run_meta_modeling_experiments` from `src.meta_modeling.experimental_pipeline`. However, the bottom of `experimental_pipeline.py` reassigns that symbol to `src.meta_modeling.classification_pipeline.run_meta_modeling_experiments`. Therefore the active experiments implementation is currently `classification_pipeline.py`; the large `run_meta_modeling_experiments` body inside `experimental_pipeline.py` is shadowed/dead for that import path. Shared helper functions such as `build_meta_dataset`, `build_candidate_sets`, `build_task`, `split_by_series_ids`, `_evaluate_predictions`, and `_train_only_feature_selection` still come from `experimental_pipeline.py`.

Inputs:

- Series features: `inputs.features_path`, currently `artifacts/features/final_clustering_features_with_chaos_v1.parquet`.
- Forecasting metrics: `inputs.forecasting_series_metrics_path`.
- Optional latest forecasting manifest auto-selection in experiments: `auto_select_latest_forecasting: true`.
- Optional basic features from `inputs.log_returns_path`.

Outputs:

- CSV/parquet routing/task/split summaries in `artifacts/meta_modeling/`.
- Excel reports in `artifacts/reports/`.
- Manifests in `artifacts/manifests/`.
- Feature manifests/lists in `artifacts/meta_modeling/`.

## Target construction

Base regression pipeline:

- `build_task_dataset` filters by horizon and metric.
- It pivots to `series_id x model_name`.
- Fold or duplicate rows are aggregated with `pivot_table(..., aggfunc="mean")`.
- Winner/oracle is computed at evaluation time from the actual metric vector on test rows.
- RMSE uses `min`; directional accuracy uses `max`.

Active classification experiments:

- `build_task` also filters by horizon, metric, and candidate models, then pivots to `series_id x model_name` using mean aggregation.
- Training labels are the winner model index from `y_train` only.
- Test oracle labels are computed from `y_test`.
- Confidence fallback can replace low-confidence classifier top-1 with the train-only best-single baseline.

Important limitation: candidate set construction is global. `build_candidate_sets` ranks models using mean metric and win counts over all series before the meta train/test split. This uses future/test-series performance to decide the candidate shortlist. That is leakage if candidate selection is considered part of the learned meta-model protocol. It is acceptable only if candidate sets are declared as a fixed benchmark design selected on a separate development run.

## Fold aggregation

The implementation can technically consume `metrics_long.parquet` from v2 because it accepts raw `rmse` and `directional_accuracy` columns. Both `build_task_dataset` and `build_task` then average duplicate rows in the pivot. For v2, duplicate rows are folds, so this becomes mean-over-folds by `(series_id, model_name, horizon)`.

However:

- The current configs still point to old `series_metrics_smoke_v1.parquet`.
- `auto_select_latest_forecasting` searches only `artifacts/manifests/forecasting_benchmark*.json`; v2 stores `run_manifest.json` inside `artifacts/forecasting/forecasting_benchmark_v2/`, so auto-selection may not find v2.
- There is no explicit assertion that all expected folds are present before averaging.

Diagnostic result for v2:

- input rows: 10,890
- aggregated rows: 3,630
- series: 110
- models: 11
- horizons: 1, 5, 20
- `fold_id` present: yes
- `n_folds_values`: `[3]`
- all metrics rows successful

## Evaluation protocol

The meta split is series-level random holdout with repeated seeds:

- base config: `n_repeats=3`, `random_seed=42`
- experiments config: `n_repeats=5`, `random_seed=42`
- generated seeds use `base_seed + i * 9973`

There is validation split allocation, but no validation data is used for model selection or early stopping in the active classification path. The reported evaluation is on held-out test series, not train rows.

Repeat aggregation exists:

- base pipeline stores means and stds for achieved/oracle/best_single/improvement/gap/routing rates.
- active classification path stores means for most fields and std only for `achieved_metric`; it should also store std for best_single/oracle/improvement/gap/routing/classification metrics.

Metric formulas:

- RMSE improvement = `best_single - achieved`, gap/oracle distance = `achieved - oracle`.
- Directional accuracy improvement = `achieved - best_single`, gap/oracle distance = `oracle - achieved`.
- These are directionally correct.

## Reproducibility

Strengths:

- Split seeds are explicit and deterministic.
- RandomForest, LogisticRegression, CatBoost wrappers receive `random_state`/`random_seed`.
- Manifests are written with git commit and run metadata.
- Base and active classification paths save split assignments.

Risks:

- Config snapshot is not saved as a standalone snapshot for meta-modeling. Manifests include `config_path`, not the full resolved config payload.
- CatBoost writes `catboost_info` under meta-modeling artifacts; this is ignored by `.gitignore`.
- Experimental import shadowing makes the active implementation non-obvious and easy to audit incorrectly.
- `auto_select_latest_forecasting` is not compatible with v2's manifest location.

## Class imbalance

V2 winner class distribution after fold aggregation:

| horizon | metric | majority class | majority share |
|---:|---|---|---:|
| 1 | rmse | naive_zero | 0.4545 |
| 1 | directional_accuracy | esn | 0.2545 |
| 5 | rmse | naive_zero | 0.5273 |
| 5 | directional_accuracy | chaotic_esn | 0.1909 |
| 20 | rmse | naive_zero | 0.5091 |
| 20 | directional_accuracy | chaotic_logistic_net | 0.1545 |

RMSE labels are materially imbalanced toward `naive_zero`. Directional-accuracy labels are more diffuse.

Balancing support exists:

- LogisticRegression and RandomForestClassifier use `class_weight="balanced"` for `balancing_mode=balanced`.
- CatBoostClassifier uses `auto_class_weights="Balanced"`.
- Balancing is applied only when fitting on train labels.

Fallback support exists:

- `top_1`
- `confidence_fallback_<threshold>` to best-single train baseline.

## Candidate sets

Current experiments use top-k values `[3, 4, 5, 6]`. Candidate sets are built from global mean performance and global win counts.

V2 diagnostic found no incomplete models in top-k candidate sets. All 11 models have coverage for all 110 series, all 3 horizons, and all 3 folds.

Risk: the implementation does not explicitly require complete coverage before candidate ranking. If a future run has missing metrics, models can still be ranked on fewer series and later rows may be dropped in `build_task`, silently changing the effective task dataset.

## Feature loading and leakage

Feature source:

- `artifacts/features/final_clustering_features_with_chaos_v1.parquet`
- 418 feature rows
- 25 numeric features
- no NaN/inf cells
- no constant numeric features
- all 110 v2 series are present
- 308 extra feature series are present and harmless after inner join

Leakage risk:

- The final feature table appears to be computed once per full series, not per forecasting fold's train window. If the meta-model is meant to choose a forecasting model before seeing the future/test period, features computed on the full available history can leak future/test information relative to the forecasting folds.
- Active experiments can add `basic_ts_*` features from the full `log_returns_v1.parquet`. These are also full-series features and have the same leakage risk unless the meta task is explicitly "choose after observing the full historical series".

The current code has train-only feature selection for `selected_features`, which avoids test-label leakage during feature selection. The upstream feature values themselves remain the main leakage concern.

## Support for forecasting_benchmark_v2

Can the code consume v2 metrics? Mostly yes, with explicit path override and fold aggregation by mean.

Not ready to run as-is from current config:

- `configs/meta_modeling_experiments_v1.yaml` still points to smoke v1 metrics.
- `auto_select_latest_forecasting` likely will not discover v2 because v2 manifest is not under `artifacts/manifests`.
- There is no explicit v2 config for meta-modeling that points to `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`.
- Candidate selection has global test-series leakage.
- Feature leakage policy is unresolved.

## Diagnostic command

Executed:

```powershell
python -m src.cli.run_metamodeling_diagnostic --metrics artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet --features artifacts/features/final_clustering_features_with_chaos_v1.parquet --output artifacts/reports/forecasting_audit_v2/metamodeling_diagnostic.json
```

Output:

- `artifacts/reports/forecasting_audit_v2/metamodeling_diagnostic.json`

## Test commands

Executed:

```powershell
pytest tests/test_meta_modeling_pipeline.py tests/test_meta_modeling_experimental_pipeline.py
```

This failed before running tests because the external `langsmith` pytest plugin imports missing dependency `httpx`.

Executed:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='.'; pytest tests/test_meta_modeling_pipeline.py tests/test_meta_modeling_experimental_pipeline.py
```

Result: 11 passed, 864 sklearn warnings about single-label confusion matrices in tiny synthetic tests.

## .gitignore

Heavy/generated outputs are covered:

- `artifacts/forecasting/**/*.parquet`, `*.csv`, `*.xlsx`, `*.json`, `*.yaml`
- global `*.parquet`, `*.xlsx`, `*.log`
- `artifacts/reports/**/*.json`, `*.csv`, `*.parquet`, `*.xlsx`
- `artifacts/meta_modeling/*.csv`
- `artifacts/meta_modeling/catboost_info/`

Audit markdown reports are not ignored, which is appropriate for this task.

## Required fixes before a new meta-modeling run

1. Create or update a meta-modeling config that explicitly points to `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`.
2. Disable or fix `auto_select_latest_forecasting` so it does not accidentally select old smoke v1 artifacts.
3. Add explicit fold-count/coverage assertions before fold aggregation: expected `n_folds=3` per `(series_id, model_name, horizon)`.
4. Decide and document leakage policy for feature computation. If model choice must be made before each forecasting test window, rebuild features from train-only history per split/fold or use a separate meta train/test period.
5. Make candidate set construction train-only per meta split, or freeze candidate sets from a separate development artifact. Do not rank candidate sets using test-series forecasting outcomes.
6. Save full resolved config snapshots for meta-modeling runs, not only `config_path`.
7. Remove or rename the shadowed experimental implementation so the active CLI path is unambiguous.
8. Add std fields in active classification repeat aggregation for best_single, oracle, improvement, gap, routing, and classification metrics.

## Go/no-go

Do not launch the full meta-modeling experiment on forecasting_benchmark_v2 yet.

V2 metrics are structurally usable and diagnostics are clean, but the pipeline still has unresolved leakage risks from candidate selection and potentially from full-series features. A constrained dry run is reasonable only after an explicit v2 config is created and the candidate/feature leakage policy is fixed or documented as an intentional benchmark assumption.
