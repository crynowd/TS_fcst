# ICDM Paper Data

This document records the data provenance and processed-data artifacts for the ICDM paper reproducibility layer.

## Source Datasets

U.S. equities:

```text
https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
```

Russian equities:

```text
https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv
```

## Raw Data Policy

Raw Kaggle files do not have to be stored in Git if they are large or subject to external distribution terms. Reproducibility should instead be supported through:

- Kaggle source links;
- raw schema documentation;
- preprocessing config and code;
- processed artifacts needed for validation.

The current preprocessing config is `configs/data_inventory_v1.yaml`.

## Expected Unified Format

`configs/data_inventory_v1.yaml` standardizes raw market files to these output columns:

```text
series_id
ticker
market
date
open
high
low
close
volume
```

The common paper-level price format is:

```text
series_id
ticker
market
date
close
```

Log returns are then built from the standardized close prices.

## Processed Data Artifacts

### `artifacts/processed/log_returns_v1.parquet`

Purpose: canonical log-return panel used by the benchmark.

Observed key columns:

```text
series_id
ticker
market
date
log_return
dataset_profile
```

Role: source for selecting the `core_balanced` benchmark profile and for building temporal train/test windows.

### `artifacts/processed/series_catalog_v1.parquet`

Purpose: inventory of standardized raw series and eligibility decisions.

Observed key columns include:

```text
series_id
ticker
market
source_path
file_name
n_rows_raw
n_rows_after_standardization
min_date
max_date
eligible_target_2000
eligible_min_1500
proposed_slice_mode
status
status_reason
```

Role: documents data quality and eligibility before the final benchmark profile is selected.

### `artifacts/processed/dataset_profiles_v1.parquet`

Purpose: profile-level series selection metadata.

Observed key columns:

```text
series_id
ticker
market
dataset_profile
original_length
returns_length
selected_length
short_series
status
```

Role: confirms the composition and selected lengths for the `core_balanced` profile used in the paper benchmark.

## Length Policy

The benchmark uses a minimum length requirement of 1500 returns and a target length of 2000 returns. Longer histories are truncated to the most recent 2000-return fragment. Therefore, 2000 is a target length, not a claim that every retained series has exactly 2000 returns.

Verified from tracked processed artifacts for `dataset_profile = core_balanced`:

- 418 series are present;
- 209 series have `market = RU`;
- 209 series have `market = US`;
- 380 series have 2000 returns;
- 38 series are shorter;
- minimum observed selected length is 1501.

## Files Required for Paper Reproducibility

Raw files required for the full pipeline:

- U.S. equities raw files from the Kaggle U.S. equities dataset;
- Russian equities raw files from the Kaggle Russian equities dataset.

Processed files required for validation without full recomputation:

- `artifacts/processed/log_returns_v1.parquet`;
- `artifacts/processed/series_catalog_v1.parquet`;
- `artifacts/processed/dataset_profiles_v1.parquet`;
- `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet`;
- `artifacts/meta_modeling/feature_list_v2.csv`;
- `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`;
- `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json`;
- `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet`;
- `artifacts/meta_modeling/split_assignments_v2.csv`;
- `artifacts/meta_modeling/task_results_v2.parquet`;
- `artifacts/meta_modeling/model_order_mapping_v2.csv`.

Large optional files for full route/prediction archival:

- `artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet`;
- `artifacts/meta_modeling/routing_rows_v2.parquet`;
- `artifacts/reports.zip`.

Tracked in Git in the current environment:

- `artifacts/processed/log_returns_v1.parquet`;
- `artifacts/processed/series_catalog_v1.parquet`;
- `artifacts/processed/dataset_profiles_v1.parquet`.
- `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet`;
- `artifacts/meta_modeling/feature_list_v2.csv`;
- `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`;
- `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json`;
- `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet`;
- `artifacts/meta_modeling/split_assignments_v2.csv`;
- `artifacts/meta_modeling/task_results_v2.parquet`;
- `artifacts/meta_modeling/model_order_mapping_v2.csv`;
- `paper_icdm/model_family_mapping.csv`.

Existing locally but intentionally not tracked directly in Git:

- `artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet`;
- `artifacts/meta_modeling/routing_rows_v2.parquet`;
- `artifacts/reports.zip`.

Recommended next-step upload strategy: large local artifacts such as `predictions.parquet`, `routing_rows_v2.parquet`, and `artifacts/reports.zip` should be evaluated for Git LFS, GitHub Release assets, or external archival storage rather than committed directly.

