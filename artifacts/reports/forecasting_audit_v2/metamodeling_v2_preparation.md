# Meta-Modeling V2 Preparation

Timestamp: 2026-04-27 17:12 Europe/Moscow

## Changed Files

- `.gitignore`
- `configs/meta_modeling_experiments_v2.yaml`
- `configs/meta_modeling_experiments_v2_dry_run.yaml`
- `src/config/loader.py`
- `src/meta_modeling/classification_pipeline.py`
- `src/meta_modeling/models.py`
- `artifacts/reports/forecasting_audit_v2/metamodeling_v2_preparation.md`

## Config

Created full-run config:

- `configs/meta_modeling_experiments_v2.yaml`
- `metrics_path: artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`
- `features_path: artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet`
- `feature_scope: fold_aware_train_only`
- `join_keys: [series_id, horizon, fold_id]`
- `auto_select_latest_forecasting: false`
- `expected_n_folds: 3`
- `horizons: [1, 5, 20]`
- `metrics: [rmse, directional_accuracy]`
- `output_dir: artifacts/meta_modeling`
- `report_dir: artifacts/reports/forecasting_audit_v2`

Dry-run config:

- `configs/meta_modeling_experiments_v2_dry_run.yaml`
- `horizons: [1]`
- `metrics: [directional_accuracy]`
- `n_repeats: 1`
- `classification_models: [logistic_regression, random_forest_classifier, catboost_classifier]`
- `candidate_selection.top_k_values: [3]`
- `feature_sets: [full]`
- `balancing_modes: [default]`
- `decision_rules: [top_1]`

## CatBoost Update

CatBoost support already existed through `catboost_classifier`, backed by `CatBoostClassifierWrapper` in `src/meta_modeling/models.py`.

The wrapper and model factory were extended to pass the v2 config parameters:

- `iterations: 300`
- `depth: 5`
- `learning_rate: 0.05`
- `l2_leaf_reg: 3.0`
- `random_seed: 42`
- `verbose: false`

`random_forest_classifier` also now honors `min_samples_leaf: 3`.

CatBoost balancing uses `auto_class_weights="Balanced"` when `balancing_mode=balanced`. The dry-run used `balancing_mode=default`; compatibility with the balanced code path is covered by the existing factory path and does not change the pipeline architecture.

Dependency check:

- `catboost` is installed.
- Version: `1.2.10`
- CatBoost trained successfully in dry-run.

If CatBoost is missing in another environment, the v2 runner logs `CatBoost not installed, skipping catboost_classifier` and continues with the remaining classifiers.

## Fold-Level Target

One meta-modeling object is `series_id x horizon x fold_id`.

For each horizon, metric, repeat, and train/test split:

1. The fold-aware feature row is joined to `metrics_long` on `series_id, horizon, fold_id`.
2. Candidate model metric vectors are pivoted per `object_id`.
3. The target class is the winning `model_name` for the same `series_id x horizon x fold_id` and metric.
4. Folds are not averaged before target construction.

## Leakage Controls

- Meta split is by `series_id`; all folds of a series stay in the same split.
- Candidate `top_k` ranking is computed only from meta-train rows for the current repeat/split.
- `best_single` fallback/baseline is computed from meta-train only, then evaluated on meta-test.
- Candidate output contains `ranking_scope=meta_train_only`.
- Split assignments include `series_id`, `fold_id`, and `object_id` for inspection.

## Coverage Checks

Coverage checks passed for the dry-run scope:

- Metrics rows: 3630
- Feature rows: 330
- Metrics objects: 330
- Feature objects: 330
- Missing feature matches: 0
- Feature-only rows after filtering: 0
- Expected folds: 3
- Bad `series_id x horizon` fold counts: 0
- Numeric features: 25
- Non-finite feature values: 0
- Duplicate metric rows by `series_id x horizon x fold_id x model_name`: 0
- Expected models per object: 11
- Incomplete metric objects: 0
- NaN `directional_accuracy`: 0

Coverage JSON:

- `artifacts/reports/forecasting_audit_v2/metamodeling_v2_coverage_checks.json`

## Dry-Run Results

Command:

```powershell
python -m src.cli.run_meta_modeling_experiments --config configs/meta_modeling_experiments_v2_dry_run.yaml
```

Latest result after adding CatBoost:

- `run_id=meta_modeling_experiments_v2_dry_run_20260427T142557Z`
- `tasks_total=3`
- `tasks_success=3`
- Dry-run elapsed in runner log: 2.7s
- End-to-end command wall time: about 4.7s

Task results:

| classifier | achieved | best_single | oracle | improvement | gap | accuracy | balanced_accuracy | elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.492885 | 0.494088 | 0.506015 | -0.001204 | 0.013131 | 0.303030 | 0.293870 | 0.225 |
| random_forest_classifier | 0.490662 | 0.494088 | 0.506015 | -0.003426 | 0.015353 | 0.333333 | 0.326934 | 0.289 |
| catboost_classifier | 0.488408 | 0.494088 | 0.506015 | -0.005680 | 0.017607 | 0.272727 | 0.266987 | 1.214 |

The logistic regression dry-run emitted a convergence warning at `max_iter=1000`. The run still completed successfully.

Mean time by model:

- `logistic_regression`: 0.225s
- `random_forest_classifier`: 0.289s
- `catboost_classifier`: 1.214s

Dry-run outputs were written under:

- `artifacts/meta_modeling/*_v2.csv`
- `artifacts/meta_modeling/*_v2.parquet`
- `artifacts/meta_modeling/feature_manifest_v2.json`
- `artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2.xlsx`
- `artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2_dry_run_resolved_config_snapshot.yaml`
- `artifacts/manifests/meta_modeling_experiments_v2_dry_run_20260427T142557Z.json`

## Test Run

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_meta_modeling_experimental_pipeline.py -q
```

Result:

- 6 passed
- 864 sklearn warnings from existing tests

Plain pytest without `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` fails before collection because an external `langsmith` pytest plugin imports missing `httpx`.

## Full Run Command

```powershell
python -m src.cli.run_meta_modeling_experiments --config configs/meta_modeling_experiments_v2.yaml
```

Full config size:

- 3 horizons
- 2 target metrics
- 5 repeats
- 4 candidate `top_k` values
- 3 classifiers
- 2 balancing modes
- 5 decision-rule evaluations (`top_1` plus 4 confidence fallback thresholds)
- 3600 evaluated configurations

Estimated full runtime after adding CatBoost: roughly 15-30 minutes on this machine. The estimate is based on the dry-run model timings, 720 train/probability setups in the full config, and extra overhead for output writing and Excel export.

## Gitignore

`.gitignore` now excludes heavy generated meta-modeling outputs:

- `artifacts/meta_modeling/*.csv`
- `artifacts/meta_modeling/*.json`
- `artifacts/meta_modeling/*.parquet`
- `artifacts/meta_modeling/*.xlsx`
- `artifacts/meta_modeling/**/*.csv`
- `artifacts/meta_modeling/**/*.json`
- `artifacts/meta_modeling/**/*.parquet`
- `artifacts/meta_modeling/**/*.xlsx`

Report JSON/CSV/parquet/xlsx outputs under `artifacts/reports/**` are also ignored by existing rules. Markdown audit reports are not ignored.

## Git Status

At report time, tracked code/config changes are:

- `.gitignore`
- `src/config/loader.py`
- `src/meta_modeling/classification_pipeline.py`
- `src/meta_modeling/models.py`

New files relevant to this task:

- `configs/meta_modeling_experiments_v2.yaml`
- `configs/meta_modeling_experiments_v2_dry_run.yaml`
- `artifacts/reports/forecasting_audit_v2/metamodeling_v2_preparation.md`
- `artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2_dry_run_resolved_config_snapshot.yaml`

There are pre-existing untracked files from earlier audit work that were not modified intentionally:

- `artifacts/reports/forecasting_audit_v2/fold_aware_feature_rebuild.md`
- `artifacts/reports/forecasting_audit_v2/metamodeling_audit.md`
- `src/cli/run_fold_aware_feature_rebuild.py`
- `src/cli/run_metamodeling_diagnostic.py`
- `src/features/fold_aware_rebuild.py`
- `src/meta_modeling/diagnostics.py`
