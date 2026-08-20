# Overfitting diagnostics summary

Full train/validation/test metric artifacts for every forecasting task were not found. The direct train-test gap can therefore not be computed without rerunning the forecasting models.

Available evidence used here:
- test-fold metrics from `forecasting_benchmark_v2/metrics_long.parquet`;
- one lightweight neural diagnostic from `neural_training_diagnostic.json`, containing final train and validation losses for one series/horizon/fold;
- neural training parameter audit from `neural_training_params.csv`.

Family-level test summary:
| family | rmse_mean | rmse_cv_mean | da_mean | n_models |
| --- | --- | --- | --- | --- |
| baseline | 0.0671525 | 0.996186 | 0.263929 | 2 |
| chaos_inspired | 0.0710676 | 0.969959 | 0.497791 | 5 |
| linear | 0.0686082 | 0.975577 | 0.503127 | 1 |
| neural_or_reservoir | 0.0684454 | 0.985361 | 0.500088 | 3 |

Interpretation: the saved artifacts support only indirect overfitting diagnostics across all tasks. Chaos-inspired models can be compared by test performance dispersion, but systematic train-test overfitting cannot be proven from the saved benchmark outputs alone.