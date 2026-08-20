# Fold-aware Train-only Feature Rebuild

Generated: 2026-08-19T13:43:21.195530+00:00

## Feature engineering files found

- `src/features/block_a_dependence.py`
- `src/features/block_b_spectrum.py`
- `src/features/block_c_tails.py`
- `src/features/block_d_chaos.py`
- `src/features/consolidation.py`
- `src/features/final_feature_sets.py`
- `src/cli/run_feature_block.py`
- `src/cli/run_feature_consolidation.py`
- `src/cli/run_final_feature_sets.py`

## Recomputed features

- Block A: hurst_rs, hurst_dfa, acf_lag_2, acf_lag_5, acf_lag_10, acf_lag_25, acf_lag_50, acf_lag_100, abs_acf_lag_2, abs_acf_lag_5, abs_acf_lag_10, abs_acf_lag_25, abs_acf_lag_50, vr_q10, lb_ret_stat_50
- Block B: lz_complexity, permutation_entropy, spectral_flatness
- Block C: kurtosis, robust_kurtosis, tail_ratio_upper, hill_tail_index
- Block D: correlation_dimension, embedding_dimension, selected_delay_tau

## Coverage

- unique series_id: 418
- horizons: [1, 5, 20]
- folds: [1, 2, 3]
- expected product: 418 x 3 x 3 = 3762
- expected rows: 3762
- output rows: 3762
- train segments below 300: 0

## NaN/Inf summary

| feature | nan_count | nan_rate | inf_count |
| --- | --- | --- | --- |
| abs_acf_lag_10 | 0 | 0 | 0 |
| abs_acf_lag_2 | 0 | 0 | 0 |
| abs_acf_lag_25 | 0 | 0 | 0 |
| abs_acf_lag_5 | 0 | 0 | 0 |
| abs_acf_lag_50 | 0 | 0 | 0 |
| acf_lag_10 | 0 | 0 | 0 |
| acf_lag_100 | 0 | 0 | 0 |
| acf_lag_2 | 0 | 0 | 0 |
| acf_lag_25 | 0 | 0 | 0 |
| acf_lag_5 | 0 | 0 | 0 |
| acf_lag_50 | 0 | 0 | 0 |
| correlation_dimension | 0 | 0 | 0 |
| embedding_dimension | 0 | 0 | 0 |
| hill_tail_index | 0 | 0 | 0 |
| hurst_dfa | 0 | 0 | 0 |
| hurst_rs | 0 | 0 | 0 |
| kurtosis | 0 | 0 | 0 |
| lb_ret_stat_50 | 0 | 0 | 0 |
| lz_complexity | 0 | 0 | 0 |
| permutation_entropy | 0 | 0 | 0 |
| robust_kurtosis | 0 | 0 | 0 |
| selected_delay_tau | 0 | 0 | 0 |
| spectral_flatness | 0 | 0 | 0 |
| tail_ratio_upper | 0 | 0 | 0 |
| vr_q10 | 0 | 0 | 0 |

## Errors and warnings

- warning/error rows: 1578
| severity | block | rows |
| --- | --- | --- |
| warning | D | 1578 |

Warning/error flag breakdown:
| severity | block | feature_warning_flags | rows |
| --- | --- | --- | --- |
| warning | D | lyapunov_time_nonpositive_lle | 1392 |
| warning | D | lyapunov_time_near_zero_lle | 186 |

## Old vs new artifact comparison

- old artifact exists: True
- old shape: [3762, 33]
- new shape: [3762, 33]
- common features: 25
- old-only features: ['feature_status', 'feature_warning_flags', 'fold_id', 'horizon', 'n_train', 'train_end', 'train_start']
- new-only features: []
- full vs train-only comparison rows: 3762 (fold_id=1,horizon=1)

Largest full-series vs train-only mean absolute deltas:
| feature | paired_rows | mean_abs_delta | median_abs_delta |
| --- | --- | --- | --- |
| lb_ret_stat_50 | 3762 | 19.5518 | 9.86511 |
| kurtosis | 3762 | 9.35874 | 1.40765 |
| embedding_dimension | 3762 | 1.6563 | 1 |
| tail_ratio_upper | 3762 | 1.33553 | 0.464383 |
| selected_delay_tau | 3762 | 0.895268 | 0 |
| correlation_dimension | 3762 | 0.770529 | 0.578873 |
| hill_tail_index | 3762 | 0.453689 | 0.290903 |
| spectral_flatness | 3762 | 0.113304 | 0.13304 |
| robust_kurtosis | 3762 | 0.0927752 | 0.05731 |
| vr_q10 | 3762 | 0.0911067 | 0.0525149 |
| abs_acf_lag_2 | 3762 | 0.053942 | 0.0298145 |
| hurst_dfa | 3762 | 0.0413135 | 0.0301401 |
| abs_acf_lag_5 | 3762 | 0.0386565 | 0.0219706 |
| abs_acf_lag_10 | 3762 | 0.0351042 | 0.0212736 |
| acf_lag_2 | 3762 | 0.0316736 | 0.0186065 |

## Outputs

- features_parquet: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2_clean_batched\final_train_only_features_by_fold.parquet`
- feature_manifest_json: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2_clean_batched\feature_manifest.json`
- feature_summary_csv: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2_clean_batched\feature_summary.csv`
- feature_errors_csv: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2_clean_batched\feature_errors.csv`
- config_snapshot_yaml: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2_clean_batched\config_snapshot.yaml`
- audit_report_md: `D:\Projects\TS_fcst\artifacts\reports\forecasting_audit_v2_clean_batched\fold_aware_feature_rebuild.md`

## Full rebuild command

```bash
python -m src.cli.run_fold_aware_feature_rebuild --log-returns artifacts/processed/log_returns_v1.parquet --split-metadata artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet --old-features artifacts/features/final_clustering_features_with_chaos_v1.parquet --output-dir artifacts/features/fold_aware_features_v2 --report-dir artifacts/reports/forecasting_audit_v2 --overwrite
```

## Git status

```text
M artifacts/architecture_tuning/esn_v1/best_candidate_summary_esn_v1.csv
 M artifacts/architecture_tuning/esn_v1/candidate_level_results_esn_v1.csv
 M artifacts/architecture_tuning/esn_v1/candidate_level_results_esn_v1.parquet
 M artifacts/architecture_tuning/esn_v1/pair_comparison_summary_esn_v1.csv
 M artifacts/architecture_tuning/esn_v1/series_level_results_esn_v1.parquet
 M artifacts/architecture_tuning/final_shortlist/family_summary_v1.csv
 M artifacts/architecture_tuning/final_shortlist/unified_tuning_summary_v1.csv
 M artifacts/architecture_tuning/final_shortlist/unified_tuning_summary_v1.parquet
 M artifacts/architecture_tuning/logistic_v1/best_candidate_summary_logistic_v1.csv
 M artifacts/architecture_tuning/logistic_v1/candidate_level_results_logistic_v1.csv
 M artifacts/architecture_tuning/logistic_v1/candidate_level_results_logistic_v1.parquet
 M artifacts/architecture_tuning/logistic_v1/pair_comparison_summary_logistic_v1.csv
 M artifacts/architecture_tuning/logistic_v1/series_level_results_logistic_v1.parquet
 M artifacts/architecture_tuning/lstm_v1/best_candidate_summary_lstm_v1.csv
 M artifacts/architecture_tuning/lstm_v1/candidate_level_results_lstm_v1.csv
 M artifacts/architecture_tuning/lstm_v1/candidate_level_results_lstm_v1.parquet
 M artifacts/architecture_tuning/lstm_v1/pair_comparison_summary_lstm_v1.csv
 M artifacts/architecture_tuning/lstm_v1/series_level_results_lstm_v1.parquet
 M artifacts/architecture_tuning/mlp_v1/best_candidate_summary_mlp_v1.csv
 M artifacts/architecture_tuning/mlp_v1/candidate_level_results_mlp_v1.csv
 M artifacts/architecture_tuning/mlp_v1/candidate_level_results_mlp_v1.parquet
 M artifacts/architecture_tuning/mlp_v1/pair_comparison_summary_mlp_v1.csv
 M artifacts/architecture_tuning/mlp_v1/series_level_results_mlp_v1.parquet
 M artifacts/reports/architecture_tuning_esn_v1.xlsx
 M artifacts/reports/architecture_tuning_final_shortlist_v1.xlsx
 M artifacts/reports/architecture_tuning_logistic_v1.xlsx
 M artifacts/reports/architecture_tuning_lstm_v1.xlsx
 M artifacts/reports/architecture_tuning_mlp_v1.xlsx
 M configs/architecture_tuning_final_shortlist_v1.yaml
 M configs/forecasting_benchmark_smoke_v1.yaml
 M configs/forecasting_benchmark_v2.yaml
 M configs/forecasting_selected_architectures_v1.yaml
 M src/architecture_tuning/__pycache__/__init__.cpython-313.pyc
 M src/architecture_tuning/__pycache__/benchmark.cpython-313.pyc
 M src/architecture_tuning/__pycache__/dataset.cpython-313.pyc
 M src/architecture_tuning/__pycache__/final_shortlist.cpython-313.pyc
 M src/architecture_tuning/benchmark.py
 M src/cli/__pycache__/run_feature_block.cpython-313.pyc
 M src/cli/__pycache__/run_forecasting_benchmark.cpython-313.pyc
 M src/config/__pycache__/loader.cpython-313.pyc
 M src/features/__pycache__/registry.cpython-313.pyc
 M src/forecasting/__pycache__/data.cpython-313.pyc
 M src/forecasting/__pycache__/io.cpython-313.pyc
 M src/forecasting/__pycache__/registry.cpython-313.pyc
 M src/forecasting/__pycache__/runners.cpython-313.pyc
 M src/forecasting/__pycache__/windowing.cpython-313.pyc
 M src/forecasting/adapters/__pycache__/base.cpython-313.pyc
 M src/forecasting/adapters/__pycache__/torch_models.cpython-313.pyc
 M src/forecasting/io.py
 M src/forecasting/runners.py
 M src/forecasting/windowing.py
 M src/meta_modeling/__pycache__/models.cpython-313.pyc
 M tests/__pycache__/test_architecture_tuning_benchmark.cpython-313-pytest-9.0.2.pyc
 M tests/__pycache__/test_architecture_tuning_final_shortlist.cpython-313-pytest-9.0.2.pyc
 M tests/__pycache__/test_forecasting_benchmark.cpython-313-pytest-9.0.2.pyc
 M tests/__pycache__/test_meta_modeling_experimental_pipeline.cpython-313-pytest-9.0.2.pyc
 M tests/test_forecasting_benchmark.py
?? artifacts/ICDM/
?? artifacts/architecture_tuning/legacy_target_overlap_v1/
?? artifacts/review_overfitting_diagnostics/
?? artifacts/review_overfitting_diagnostics_full/
?? artifacts/review_simple_returns_analysis/
?? artifacts/review_statistical_analysis/
?? artifacts/tmp/
?? configs/forecasting_benchmark_v2_clean_batched_batch_01.yaml
?? configs/forecasting_benchmark_v2_clean_batched_batch_02.yaml
?? configs/forecasting_benchmark_v2_clean_batched_batch_03.yaml
?? configs/forecasting_benchmark_v2_clean_batched_batch_04.yaml
?? scripts/
?? src/cli/run_review_overfitting_diagnostics.py
```

## .gitignore check

Relevant ignore rules:
- `artifacts/features/`
- `artifacts/reports/**/*.json`
- `artifacts/reports/**/*.csv`
- `artifacts/reports/**/*.parquet`
- `artifacts/reports/**/*.xlsx`
- `*.parquet`
- `*.xlsx`
- `*.xls`
- `*.log`
- New parquet/csv/json/yaml outputs under `artifacts/features/` or report heavy formats are ignored.
- The markdown audit report is not ignored by the current `.gitignore` rules.
