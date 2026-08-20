# Simple-Returns Post-Hoc Analysis

Input artifacts: `D:/Projects/TS_fcst/artifacts/forecasting/forecasting_benchmark_v2`.

No models were retrained. Existing forecast-level `y_true` and `y_pred` were transformed as:

```text
simple_true = exp(y_true) - 1
simple_pred = exp(y_pred) - 1
```

## Forecast Artifact Schema

Full schema was saved to `schema_predictions.csv`; `metrics_long.parquet` schema was saved to `schema_metrics_long.csv`.

## Mean Metrics

Overall averages across `horizon x model_name`:

| return_space | rmse | mae | directional_accuracy |
| --- | --- | --- | --- |
| log | 0.069417 | 0.0468415 | 0.456382 |
| simple | 0.190995 | 0.0630758 | 0.456382 |

By horizon:

| return_space | horizon | rmse | mae | directional_accuracy |
| --- | --- | --- | --- | --- |
| log | 1 | 0.0295158 | 0.0177444 | 0.445662 |
| log | 5 | 0.0615532 | 0.0395819 | 0.455799 |
| log | 20 | 0.117182 | 0.0831982 | 0.467686 |
| simple | 1 | 0.056877 | 0.0216101 | 0.445662 |
| simple | 5 | 0.103015 | 0.0470122 | 0.455799 |
| simple | 20 | 0.413093 | 0.120605 | 0.467686 |

## Best Models

| return_space | horizon | metric | best_model | best_value |
| --- | --- | --- | --- | --- |
| log | 1 | rmse | naive_zero | 0.0284285 |
| log | 1 | mae | naive_zero | 0.0166137 |
| log | 1 | directional_accuracy | esn | 0.492506 |
| log | 5 | rmse | naive_zero | 0.0595196 |
| log | 5 | mae | naive_zero | 0.0375815 |
| log | 5 | directional_accuracy | naive_mean | 0.506563 |
| log | 20 | rmse | naive_zero | 0.112945 |
| log | 20 | mae | naive_zero | 0.0793293 |
| log | 20 | directional_accuracy | naive_mean | 0.525371 |
| simple | 1 | rmse | naive_zero | 0.0557622 |
| simple | 1 | mae | naive_zero | 0.0204699 |
| simple | 1 | directional_accuracy | esn | 0.492506 |
| simple | 5 | rmse | naive_zero | 0.101058 |
| simple | 5 | mae | naive_zero | 0.0449785 |
| simple | 5 | directional_accuracy | naive_mean | 0.506563 |
| simple | 20 | rmse | naive_zero | 0.186335 |
| simple | 20 | mae | naive_zero | 0.0994256 |
| simple | 20 | directional_accuracy | naive_mean | 0.525371 |

## Rank Stability

Spearman rank correlations:

| horizon | metric | spearman_rho | p_value | n_models |
| --- | --- | --- | --- | --- |
| 1 | rmse | 0.990909 | 3.76257e-09 | 11 |
| 1 | mae | 1 | 0 | 11 |
| 1 | directional_accuracy | 1 | 0 | 11 |
| 5 | rmse | 0.972727 | 5.14218e-07 | 11 |
| 5 | mae | 1 | 0 | 11 |
| 5 | directional_accuracy | 1 | 0 | 11 |
| 20 | rmse | 0.9 | 0.000159971 | 11 |
| 20 | mae | 0.9 | 0.000159971 | 11 |
| 20 | directional_accuracy | 1 | 0 | 11 |

Maximum rank shifts:

| horizon | metric | model_name | rank_log | rank_simple | rank_shift_abs |
| --- | --- | --- | --- | --- | --- |
| 1 | rmse | transient_chaotic_esn | 7 | 6 | 1 |
| 1 | mae | chaotic_esn | 11 | 11 | 0 |
| 1 | directional_accuracy | chaotic_esn | 7 | 7 | 0 |
| 5 | rmse | vanilla_mlp | 5 | 7 | 2 |
| 5 | mae | chaotic_esn | 11 | 11 | 0 |
| 5 | directional_accuracy | chaotic_esn | 6 | 6 | 0 |
| 20 | rmse | vanilla_mlp | 6 | 10 | 4 |
| 20 | mae | vanilla_mlp | 5 | 9 | 4 |
| 20 | directional_accuracy | chaotic_esn | 5 | 5 | 0 |

## Selected Forecast Cases

| series_id | ticker | market | horizon | fold_id | selection_metric | best_model | best_value | case_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| US:bfo.us | bfo.us | US | 1 | 3 | rmse | transient_chaotic_esn | 0.00238676 | best |
| RU:GLTR | GLTR | RU | 20 | 2 | rmse | ridge_lag | 58.5225 | bad |
| RU:IRAO | IRAO | RU | 5 | 1 | rmse | naive_mean | 0.0494344 | median |
| US:amp.us | amp.us | US | 20 | 1 | directional_accuracy | chaotic_logistic_net | 0.8107 | best |
| RU:TGKN | TGKN | RU | 20 | 1 | directional_accuracy | chaotic_mlp | 0.288066 | bad |
| US:iht.us | iht.us | US | 1 | 2 | directional_accuracy | esn | 0.538636 | median |

## Figures

- `figures\case_rmse_best_US_bfo.us_h1_f3.png`
- `figures\case_rmse_bad_RU_GLTR_h20_f2.png`
- `figures\case_rmse_median_RU_IRAO_h5_f1.png`
- `figures\case_directional_accuracy_best_US_amp.us_h20_f1.png`
- `figures\case_directional_accuracy_bad_RU_TGKN_h20_f1.png`
- `figures\case_directional_accuracy_median_US_iht.us_h1_f2.png`
- `figures\two_model_rmse_vs_da_1_RU_GLTR_h20_f2.png`
- `figures\two_model_rmse_vs_da_2_RU_IRAO_h5_f1.png`

## Conclusion

Переход к обычным накопленным доходностям частично меняет локальные выводы: есть изменения победителей или заметные сдвиги рангов, поэтому для защиты стоит явно оговорить пространство доходностей.

For the defense, the usable point is that RMSE/MAE change scale after the nonlinear `exp(.) - 1` transform, but the broad model ordering is stable. DA is effectively invariant because the transform preserves the sign around zero.
