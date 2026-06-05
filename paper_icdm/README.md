# Feature-Based Meta-Learning for Forecasting Model Selection in Financial Time Series

This directory contains the reproducibility notes for the ICDM paper "Feature-Based Meta-Learning for Forecasting Model Selection in Financial Time Series". 

## Brief Experiment Description

The experiment builds a balanced financial time-series panel from Russian and U.S. equity data and converts adjusted price histories into log returns. The benchmark profile contains 418 series, with 209 Russian and 209 U.S. instruments, using a minimum length of 1500 returns and a target length of 2000 returns. Time-series features are computed from training segments and cover long memory, linear dependence, volatility dependence, complexity and spectrum, distribution and tails, and phase-space structure. The final fold-aware feature matrix contains 25 feature columns for each series, horizon, and rolling-origin fold. The forecasting benchmark evaluates 11 candidate models across horizons 1, 5, and 20 with three rolling-origin folds and horizon-specific input windows 64, 32, and 16. Forecasting quality is measured by RMSE and directional accuracy. Meta-learning experiments train logistic regression, random forest, and CatBoost classifiers to select forecasting models from top-k candidate sets, using five repeated instrument splits and confidence fallback rules. Paper tables I-VI and figures 2-4 are generated from the resulting processed data, benchmark outputs, meta-learning outputs, and paper-specific scripts.

## Data Sources

Raw market data must be downloaded externally from Kaggle:

- U.S. equities: <https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs>
- Russian equities: <https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv>

## Large Artifacts

For full reconstruction download files from: https://drive.google.com/drive/folders/1crDB4n5BZN9IxuYYTVE-sfyRUKmIdmjS?usp=sharing

- `forecasting_benchmark_forecasting_benchmark_v2.xlsx` - Excel report summarizing the final v2 forecasting benchmark run, including model registry, fold metrics, and benchmark diagnostics.

- `log_returns_v1.parquet` - canonical processed log-return panel used as the input data for feature computation, forecasting windows, and benchmark evaluation.

- `meta_modeling_experiments_v2.xlsx` - final v2 meta-learning report used for Table VI lineage, including summary results, task results, routing rows, model mappings, split assignments, comparisons, and candidate details.

- `metamodeling_v2_improvement_counts_best_config.csv` - compact summary of improvement counts for the best meta-modeling configuration.

- `metamodeling_v2_improvement_counts_best_config_by_series.csv` - per-series breakdown of improvement counts for the best meta-modeling configuration.

- `neural_training_params.csv` - audit table of neural model training parameters used in the forecasting benchmark.

- `predictions.parquet` - full prediction-level output from the v2 forecasting benchmark, with true and predicted values by model, series, horizon, fold, and timestamp.

- `routing_rows_v2.parquet` - route-level meta-learning output showing selected models, oracle models, best single models, achieved metrics, confidence values, and fallback decisions.

- `series_catalog_v1.parquet` - processed data inventory documenting standardized raw series, eligibility decisions, lengths, date ranges, and quality/status metadata.

## Repository Structure

- `src/` - source code and CLI entry points.
- `configs/` - pipeline and experiment configurations.
- `artifacts/` - processed data, feature matrices, benchmark outputs, meta-learning outputs, and reports.
- `paper_icdm/scripts/` - paper table and figure builders.
- `paper_icdm/tables/` - generated paper table CSV files.
- `paper_icdm/figures/` - generated paper figures.

## Complete Experimental Steps and Commands

Run commands from the repository root.

### 1. Data Inventory and Log Returns

```bash
python -m src.cli.run_data_inventory --config configs/data_inventory_v1.yaml
python -m src.cli.run_log_returns_pipeline --config configs/data_inventory_v1.yaml
```

This stage standardizes raw RU/US market files according to `configs/data_inventory_v1.yaml` and creates processed data artifacts such as `artifacts/processed/log_returns_v1.parquet`, `series_catalog_v1.parquet`, and `dataset_profiles_v1.parquet`.

### 2. Feature Computation: Blocks A-D

```bash
python -m src.cli.run_feature_block --block A --config configs/features_block_A_v1.yaml
python -m src.cli.run_feature_block --block B --config configs/features_block_B_v1.yaml
python -m src.cli.run_feature_block --block C --config configs/features_block_C_v1.yaml
python -m src.cli.run_feature_block --block D --config configs/features_block_D_v1.yaml
```

Block A covers dependence and long-memory features. Block B covers spectrum and complexity features. Block C covers distribution and tail features. Block D covers phase-space and chaos-related features.

### 3. Feature Screening

```bash
python -m src.cli.run_feature_screening --config configs/feature_screening_v1.yaml
```

This stage merges feature blocks and applies missingness, variance, and correlation screening.

### 4. Feature Consolidation

```bash
python -m src.cli.run_feature_consolidation --config configs/feature_consolidation_v1.yaml
python -m src.cli.run_final_feature_sets --master-path artifacts/features/features_master_v1.parquet
```

These commands create consolidated feature sets and final feature-list artifacts from the screened master table.

### 5. Forecasting Benchmark

```bash
python -m src.cli.run_forecasting_benchmark --config configs/forecasting_benchmark_v2.yaml
```

The final benchmark route is `configs/forecasting_benchmark_v2.yaml`, which uses `artifacts/processed/log_returns_v1.parquet`, 11 models, horizons 1/5/20, rolling-origin folds, and the selected architectures listed in `configs/forecasting_selected_architectures_v1.yaml`.

### 6. Fold-Aware Feature Rebuild

```bash
python -m src.cli.run_fold_aware_feature_rebuild --log-returns artifacts/processed/log_returns_v1.parquet --split-metadata artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet --old-features artifacts/features/final_clustering_features_with_chaos_v1.parquet --output-dir artifacts/features/fold_aware_features_v2 --report-dir artifacts/reports/forecasting_audit_v2 --overwrite true
```

This command rebuilds the final time-series features by forecasting fold using train-only data. It creates `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet`.

### 7. Meta-Learning Experiments

```bash
python -m src.cli.run_meta_modeling_experiments --config configs/meta_modeling_experiments_v2.yaml
```

This is the final meta-learning route recorded by the repository files. It reads benchmark metrics and fold-aware features, evaluates logistic regression, random forest, and CatBoost classifiers, and writes v2 meta-modeling artifacts and the Excel report under `artifacts/reports/forecasting_audit_v2/`.

### 8. Paper Tables and Figures

```bash
python paper_icdm/scripts/build_paper_tables.py
python paper_icdm/scripts/build_paper_figures.py
```

The table builder generates tables I-VI under `paper_icdm/tables/`. The figure builder generates figures 2-4 under `paper_icdm/figures/`. Figure 1 is recorded in the existing documents as manual/schematic, and no data-generation script for it exists in the checked files.

## Dependencies

Dependencies are listed with pinned versions in `requirements.txt`.
