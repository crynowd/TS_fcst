# Fold-aware Train-only Feature Rebuild

Generated: 2026-04-27T19:37:44.751116+00:00

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

- warning/error rows: 1569
| severity | block | rows |
| --- | --- | --- |
| warning | D | 1569 |

Warning/error flag breakdown:
| severity | block | feature_warning_flags | rows |
| --- | --- | --- | --- |
| warning | D | lyapunov_time_nonpositive_lle | 1376 |
| warning | D | lyapunov_time_near_zero_lle | 193 |

## Old vs new artifact comparison

- old artifact exists: True
- old shape: [418, 29]
- new shape: [3762, 33]
- common features: 25
- old-only features: []
- new-only features: []
- full vs train-only comparison rows: 418 (fold_id=1,horizon=1)

Largest full-series vs train-only mean absolute deltas:
| feature | paired_rows | mean_abs_delta | median_abs_delta |
| --- | --- | --- | --- |
| lb_ret_stat_50 | 418 | 39.8822 | 24.1196 |
| kurtosis | 418 | 31.3845 | 8.86461 |
| tail_ratio_upper | 418 | 2.42613 | 1.27063 |
| embedding_dimension | 418 | 2.28947 | 2 |
| selected_delay_tau | 418 | 1.45933 | 1 |
| correlation_dimension | 418 | 0.996638 | 0.818378 |
| hill_tail_index | 418 | 0.7287 | 0.59405 |
| spectral_flatness | 418 | 0.201648 | 0.199852 |
| vr_q10 | 418 | 0.159203 | 0.12729 |
| robust_kurtosis | 418 | 0.134115 | 0.10377 |
| abs_acf_lag_2 | 418 | 0.0919429 | 0.0724278 |
| hurst_dfa | 418 | 0.0652005 | 0.0520704 |
| abs_acf_lag_5 | 418 | 0.0630015 | 0.051051 |
| abs_acf_lag_10 | 418 | 0.0571386 | 0.046215 |
| abs_acf_lag_25 | 418 | 0.0516732 | 0.0414093 |

## Outputs

- features_parquet: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2\final_train_only_features_by_fold.parquet`
- feature_manifest_json: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2\feature_manifest.json`
- feature_summary_csv: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2\feature_summary.csv`
- feature_errors_csv: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2\feature_errors.csv`
- config_snapshot_yaml: `D:\Projects\TS_fcst\artifacts\features\fold_aware_features_v2\config_snapshot.yaml`
- audit_report_md: `D:\Projects\TS_fcst\artifacts\reports\forecasting_audit_v2\fold_aware_feature_rebuild.md`

## Full rebuild command

```bash
python -m src.cli.run_fold_aware_feature_rebuild --log-returns artifacts/processed/log_returns_v1.parquet --split-metadata artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet --old-features artifacts/features/final_clustering_features_with_chaos_v1.parquet --output-dir artifacts/features/fold_aware_features_v2 --report-dir artifacts/reports/forecasting_audit_v2 --overwrite
```

## Git status

```text
M configs/forecasting_benchmark_v2.yaml
?? artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2_dry_run_resolved_config_snapshot.yaml
?? artifacts/reports/forecasting_audit_v2/meta_modeling_experiments_v2_resolved_config_snapshot.yaml
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
