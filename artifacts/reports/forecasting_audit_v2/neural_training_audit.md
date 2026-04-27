# Neural training audit v2

Generated at UTC: 2026-04-27T09:19:53.980113+00:00

## Checked files
- `src/forecasting/registry.py`
- `src/forecasting/runners.py`
- `src/forecasting/windowing.py`
- `src/forecasting/adapters/base.py`
- `src/forecasting/adapters/torch_models.py`
- `src/forecasting/adapters/sklearn_models.py`
- `src/forecasting/adapters/naive.py`
- `src/forecasting/architectures/mlp.py`
- `src/forecasting/architectures/lstm.py`
- `src/forecasting/architectures/chaotic_logistic.py`
- `configs/forecasting_benchmark_smoke_v1.yaml`
- `configs/forecasting_selected_architectures_v1.yaml`
- `configs/architecture_tuning_mlp_v1.yaml`
- `configs/architecture_tuning_lstm_v1.yaml`
- `configs/architecture_tuning_logistic_v1.yaml`

## Registered non-ESN models
- `naive_zero`
- `naive_mean`
- `ridge_lag`
- `vanilla_mlp`
- `chaotic_mlp`
- `chaotic_logistic_net`
- `lstm_forecast`
- `chaotic_lstm_forecast`

## Actual training parameters

| model | epochs | early stopping | patience | lr | batch | hidden | layers | dropout | weight decay | validation | shuffle | seed | device |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|
| naive_zero |  | False |  |  |  |  |  |  |  | unused | False |  |  |
| naive_mean |  | False |  |  |  |  |  |  |  | unused | False |  |  |
| ridge_lag |  | False |  |  |  |  |  |  |  | unused | False | 42 |  |
| vanilla_mlp | 2 | True | 1 | 0.001 | 128 | [128, 64] |  |  | 0.0 | last_20pct_of_train_no_shuffle | True | 2026 | cpu |
| chaotic_mlp | 2 | True | 1 | 0.001 | 128 | [64, 32] |  |  | 0.0 | last_20pct_of_train_no_shuffle | True | 2026 | cpu |
| chaotic_logistic_net | 2 | True | 1 | 0.001 | 128 | 64 |  |  | 0.0 | last_20pct_of_train_no_shuffle | True | 2026 | cpu |
| lstm_forecast | 2 | True | 1 | 0.001 | 128 | 32 | 1 | 0.0 | 0.0 | last_20pct_of_train_no_shuffle | True | 2026 | cpu |
| chaotic_lstm_forecast | 2 | True | 1 | 0.001 | 128 | 64 | 2 | 0.1 | 0.0 | last_20pct_of_train_no_shuffle | True | 2026 | cpu |

## Findings
- The benchmark loader merges `configs/forecasting_selected_architectures_v1.yaml` into `model_overrides`, so selected YAML architecture params do reach the adapters.
- Runtime params in the selected architecture YAML are empty for the neural models; therefore the smoke benchmark global `training` block controls epochs, patience, batch size and learning rate.
- Torch models use `torch.nn.MSELoss`; early stopping monitors validation MSE when the runner provides a validation split.
- Validation is the final 20% of each rolling-origin train fold. It is time-ordered and does not use future test observations, so the split itself is not a leakage source.
- The torch `DataLoader` uses `shuffle=True` inside the fit subset. This does not leak labels or future test data, but it breaks chronological mini-batch ordering; for MLP it is usually acceptable, for LSTM sequence samples remain internally ordered but batches are shuffled.
- `build_model` is called inside each model/series/horizon/fold task, so model objects are reinitialized across series, folds and horizons. The adapter would reuse state if the same instance were fitted twice, but the benchmark runner does not do that.
- `weight_decay` is now represented in `FitContext`; current configs do not set it, so the effective value is 0.0.

## Underfitting risk
- High: the main smoke config gives neural models only 2 max epochs and patience 1. Diagnostics below show most torch tasks stopped at the max epoch, and one stopped as soon as validation loss worsened after a single non-improving epoch.
- LSTM selected architectures are intentionally small (`lstm_forecast` 32x1 and `chaotic_lstm_forecast` 64x2). The bigger risk for the next benchmark is the epoch cap, not architecture size.
- Dropout is 0.0 for `lstm_forecast` and 0.1 for `chaotic_lstm_forecast`; MLPs and logistic net have no dropout. There is no evidence of excessive regularization.

## Diagnostic run
- Series: 1
- Horizon: 1
- Window size: 64

| model | status | epochs | final train loss | final val loss | early stopping | seconds | n_fit | n_val | n_test | device |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| naive_zero | success | None | None | None |  | 0.0000 | 1575 | 393 | 1968 |  |
| naive_mean | success | None | None | None |  | 0.0001 | 1575 | 393 | 1968 |  |
| ridge_lag | success | None | None | None |  | 0.0021 | 1575 | 393 | 1968 |  |
| vanilla_mlp | success | 2 | 0.001140164334386114 | 0.004099708981812 | max_epochs_reached | 1.5195 | 1575 | 393 | 1968 | cpu |
| chaotic_mlp | success | 2 | 0.005040374196075376 | 0.0042648278176784515 | max_epochs_reached | 0.0756 | 1575 | 393 | 1968 | cpu |
| chaotic_logistic_net | success | 2 | 0.009504042911151098 | 0.022380134090781212 | patience_exhausted | 0.3102 | 1575 | 393 | 1968 | cpu |
| lstm_forecast | success | 2 | 0.0009419316810644453 | 0.003808077424764633 | max_epochs_reached | 0.2548 | 1575 | 393 | 1968 | cpu |
| chaotic_lstm_forecast | success | 2 | 0.0009588930465369707 | 0.00394896324723959 | max_epochs_reached | 0.9912 | 1575 | 393 | 1968 | cpu |

## Safe recommendations
- For the next non-smoke benchmark, increase neural `max_epochs` before changing architectures. A conservative starting point is 30-50 epochs for MLP/logistic and 40-80 for LSTM.
- Keep early stopping enabled and use patience around 5-10 so ESN remains fast while neural models are not capped at 2 epochs.
- Keep the last-block validation split for time-series safety; consider making the 20% ratio explicit in config later.
- Consider `shuffle=False` specifically for LSTM if preserving chronological batch order is desired; sample windows themselves are not shuffled internally.
- Do not change the main smoke config for this audit. Treat the higher epoch settings as recommendations for a future full run config.
