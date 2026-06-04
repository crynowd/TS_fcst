# ICDM Paper Reproducibility Plan

This document describes the current reproducibility environment and the planned validation checks for the ICDM paper route.

## Environment

Observed dependency files:

- `requirements.txt` exists.

Not observed at the repository root:

- `environment.yml`;
- `pyproject.toml`;
- `setup.py`;
- lock files.

The current `requirements.txt` lists dependencies but does not pin versions. A pinned environment file is recommended before final archival.

Current listed dependencies:

```text
pandas
numpy
scipy
statsmodels
openpyxl
pyarrow
PyYAML
pytest
scikit-learn
matplotlib
seaborn
torch
catboost
```

TODO: record the exact Python version, operating system, accelerator/CUDA details if applicable, and runtime infrastructure used for the final reported experiments. These values are not specified by the checked repository files.

## Reproduction Modes

### Full Recomputation

Full recomputation covers:

- preprocessing and data inventory;
- log-return construction;
- feature computation;
- forecasting benchmark execution;
- meta-learning experiments.

The relevant configs are:

- `configs/data_inventory_v1.yaml`;
- `configs/features_block_A_v1.yaml`;
- `configs/features_block_B_v1.yaml`;
- `configs/features_block_C_v1.yaml`;
- `configs/features_block_D_v1.yaml`;
- `configs/feature_screening_v1.yaml`;
- `configs/feature_consolidation_v1.yaml`;
- `configs/forecasting_benchmark_v2.yaml`;
- `configs/forecasting_selected_architectures_v1.yaml`;
- `configs/meta_modeling_experiments_v2.yaml`.

Exact commands for full recomputation should be verified in a later stage before archival. This document intentionally does not provide unverified full-run commands.

### Artifact-Based Validation

Artifact-based validation checks already computed results without rerunning heavy training. It should validate:

- existence of key artifacts;
- dataset counts;
- split structure;
- feature dimensions;
- model list;
- forecasting metrics and statuses;
- meta-learning task results;
- explicit broad-family model mapping;
- source coverage for planned paper tables and figures.

Run the lightweight artifact checker from the repository root:

```bash
python paper_icdm/scripts/check_artifacts.py
```

The checker reads existing files only. It validates processed-data counts and length policy, feature-list and feature-matrix dimensions, forecasting benchmark models/metrics/folds/horizons, meta-learning splits and task outputs, the explicit model-family mapping, paper-output source coverage, and artifact tracking/upload recommendations.

Build compact paper table CSV files from existing artifacts:

```bash
python paper_icdm/scripts/build_paper_tables.py
```

The table builder reads existing configs, parquet/csv artifacts, and the meta-modeling Excel report only. It does not rerun forecasting or meta-learning. It writes `paper_icdm/tables/table_i_features.csv`, `table_ii_candidates.csv`, `table_iii_protocol.csv`, `table_iv_direct_forecasting.csv`, `table_v_winner_family_counts.csv`, and `table_vi_meta_selection_results.csv`. If a required source is unavailable, it writes a `*_NEEDS_SOURCE.csv` or debug candidate CSV and prints a warning instead of fabricating values.

## Expected Validation Checks

`paper_icdm/scripts/check_artifacts.py` verifies:

- all key files listed in `RESULTS_MANIFEST.md` are present or explicitly marked unavailable;
- `artifacts/processed/log_returns_v1.parquet` contains 418 series for `dataset_profile = core_balanced`;
- markets in the `core_balanced` profile contain 209 RU and 209 US series;
- series lengths follow the policy: minimum length at least 1500, target length 2000, longer histories truncated;
- current tracked artifacts show 380 series with 2000 returns, 38 shorter series, and minimum observed selected length 1501;
- horizons are `{1, 5, 20}`;
- folds are `{1, 2, 3}`;
- window sizes are `{64, 32, 16}` for horizons `{1, 5, 20}`;
- forecasting metrics contain 11 models;
- expected model names are present: `naive_zero`, `naive_mean`, `ridge_lag`, `esn`, `chaotic_esn`, `transient_chaotic_esn`, `vanilla_mlp`, `chaotic_mlp`, `chaotic_logistic_net`, `lstm_forecast`, `chaotic_lstm_forecast`;
- failed forecasting tasks are absent or explicitly documented;
- the feature matrix has 25 feature columns;
- feature rows correspond to `series_id x horizon x fold`;
- split assignments have repeats `{1, 2, 3, 4, 5}`;
- split counts are 876 train, 126 validation, and 252 test meta-observations per repeat/horizon/metric where this structure is present;
- there is no leakage by key: the same instrument must not appear simultaneously in train, validation, and test within one repeat/horizon/metric;
- `artifacts/meta_modeling/task_results_v2.parquet` contains RMSE and directional accuracy targets;
- top-k values are `{3, 4, 5, 6}`;
- classification models include logistic regression, random forest, and CatBoost;
- `paper_icdm/model_family_mapping.csv` contains exactly the 11 expected forecasting models and exactly the families `Zero/mean baselines`, `Non-chaotic models`, and `Chaos-inspired models`;
- `artifacts/meta_modeling/routing_rows_v2.parquet`, when available externally or locally, contains fields for selected, best single, and oracle routes or enough data to reconstruct them;
- Table I, II, IV, V, and VI can be built from deterministic source artifacts;
- Figure 2, 3, and 4 can be built from deterministic source artifacts;
- Figure 1 is a schematic/paper figure and can be either stored or manually recreated;
- every generated paper output has a deterministic source artifact or a clearly documented manual source.

## Known Limitations and TODO

- Kaggle raw data download is external.
- Dependency versions are not pinned yet.
- Table builder output should be regenerated after any source artifact changes with `python paper_icdm/scripts/build_paper_tables.py`.
- Figure builder scripts are not added yet.
- Exact runtime and infrastructure values need manual filling.
- Exact artifact upload strategy needs verification for large optional artifacts. `predictions.parquet`, `routing_rows_v2.parquet`, and `artifacts/reports.zip` are intentionally not committed directly to Git; use Git LFS, GitHub Release assets, or an external archive if they are needed.

## Table VI Lineage

Table VI is built from `artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2.xlsx`, sheet `summary`. That Excel report is written by `src/reporting/excel_export.py` from the v2 meta-modeling pipeline and includes the lineage sheets `task_results`, `routing_rows`, `model_order_mapping`, `meta_dataset_summary`, `split_assignments`, `repeat_aggregates`, `best_single_repeat`, `comparison`, and `candidates`.

The compact source files tied to that report are:

- `artifacts/meta_modeling/task_results_v2.parquet`;
- `artifacts/meta_modeling/split_assignments_v2.csv`;
- `artifacts/meta_modeling/model_order_mapping_v2.csv`;
- `artifacts/meta_modeling/routing_rows_v2.parquet`, if available locally for route-level inspection.

`routing_rows_v2.parquet` remains an optional large local artifact and is not committed directly.
