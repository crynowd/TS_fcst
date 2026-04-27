# Fold-aware Train-only Feature Rebuild

Generated: 2026-04-27T13:53:28.924896+00:00

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

- unique series_id: 110
- horizons: [1, 5, 20]
- folds: [1, 2, 3]
- expected product: 110 x 3 x 3 = 990
- expected rows: 990
- output rows: 990
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

- warning/error rows: 381
| severity | block | rows |
| --- | --- | --- |
| warning | D | 381 |

Warning/error flag breakdown:
| severity | block | feature_warning_flags | rows |
| --- | --- | --- | --- |
| warning | D | lyapunov_time_nonpositive_lle | 338 |
| warning | D | lyapunov_time_near_zero_lle | 43 |

## Old vs new artifact comparison

- old artifact exists: True
- old shape: [418, 29]
- new shape: [990, 33]
- common features: 25
- old-only features: []
- new-only features: []
- full vs train-only comparison rows: 110 (fold_id=1,horizon=1)

Largest full-series vs train-only mean absolute deltas:
| feature | paired_rows | mean_abs_delta | median_abs_delta |
| --- | --- | --- | --- |
| lb_ret_stat_50 | 110 | 60.0552 | 34.2562 |
| kurtosis | 110 | 54.317 | 22.1885 |
| tail_ratio_upper | 110 | 3.82028 | 2.56673 |
| embedding_dimension | 110 | 2.24545 | 2 |
| selected_delay_tau | 110 | 1.52727 | 1 |
| correlation_dimension | 110 | 1.1813 | 1.07678 |
| hill_tail_index | 110 | 0.787866 | 0.626744 |
| spectral_flatness | 110 | 0.1969 | 0.190681 |
| vr_q10 | 110 | 0.176034 | 0.144134 |
| robust_kurtosis | 110 | 0.141459 | 0.108636 |
| abs_acf_lag_2 | 110 | 0.125834 | 0.118076 |
| abs_acf_lag_5 | 110 | 0.0770022 | 0.0584949 |
| hurst_dfa | 110 | 0.0710694 | 0.0536881 |
| acf_lag_2 | 110 | 0.0686751 | 0.0582495 |
| abs_acf_lag_10 | 110 | 0.0680729 | 0.0585219 |

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
?? artifacts/reports/forecasting_audit_v2/fold_aware_feature_rebuild.md
?? artifacts/reports/forecasting_audit_v2/metamodeling_audit.md
?? src/cli/run_fold_aware_feature_rebuild.py
?? src/cli/run_metamodeling_diagnostic.py
?? src/features/fold_aware_rebuild.py
?? src/meta_modeling/diagnostics.py
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
