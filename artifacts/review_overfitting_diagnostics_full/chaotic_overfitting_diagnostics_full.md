# Диагностика переобучения хаотических моделей

Generated at UTC: 2026-06-15T06:13:54Z

## 1. Цель диагностики
Проверить, наблюдается ли у хаотических forecasting-моделей выраженный систематический train-test или validation-test gap, который мог бы указывать на переобучение. Диагностика не доказывает полное отсутствие переобучения; она проверяет наличие выраженного разрыва в воспроизведенной benchmark-схеме.

## 2. Источники истины
| key | value |
| --- | --- |
| benchmark_config | D:\Projects\TS_fcst\configs\forecasting_benchmark_v2.yaml |
| selected_architectures_config | D:\Projects\TS_fcst\configs\forecasting_selected_architectures_v1.yaml |
| dataset_source | D:\Projects\TS_fcst\artifacts\processed\log_returns_v1.parquet |
| dataset_profile | core_balanced |
| dataset_max_series | 418 |
| actual_series_count | 418 |
| horizons | [1, 5, 20] |
| window_sizes | {"1": 64, "5": 32, "20": 16} |
| validation | {"method": "rolling_origin", "n_folds": 3} |
| training | {"max_epochs": 20, "early_stopping_patience": 5, "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.0, "by_model": {"vanilla_mlp": {"max_epochs": 30, "early_stopping_patience": 5, "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0}, "chaotic_mlp": {"max_epochs": 30, "early_stopping_patience": 5, "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0}, "chaotic_logistic_net": {"max_epochs": 30, "early_stopping_patience": 5, "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0}, "lstm_forecast": {"max_epochs": 40, "early_stopping_patience": 6, "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0}, "chaotic_lstm_forecast": {"max_epochs": 40, "early_stopping_patience": 6, "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0}}} |
| models | ["chaotic_esn", "transient_chaotic_esn", "chaotic_mlp", "chaotic_logistic_net", "chaotic_lstm_forecast"] |
| model_overrides | {"chaotic_esn": {"chaotic_spectral_radius": 1.3, "input_scale": 0.08, "leak_rate": 0.7, "n_reservoir": 128, "ridge_alpha": 0.0005, "seed": 2026, "spectral_radius": 0.98}, "transient_chaotic_esn": {"g_end": 0.95, "g_start": 1.3, "input_scale": 0.1, "leak_rate": 0.8, "n_reservoir": 96, "ridge_alpha": 0.0001, "seed": 2026, "spectral_radius": 0.95}, "chaotic_mlp": {"hidden_dims": [64, 32], "seed": 2026}, "chaotic_logistic_net": {"beta": 0.08, "hidden_size": 64, "r_max": 3.8, "r_min": 3.55, "seed": 2026, "train_r": false}, "chaotic_lstm_forecast": {"dropout": 0.1, "hidden_size": 64, "num_layers": 2, "r": 3.9, "seed": 2026}} |
| diagnostic_output_dir | D:\Projects\TS_fcst\artifacts\review_overfitting_diagnostics_full |

## 3. Воспроизведенная экспериментальная схема
- Dataset/profile: `D:\Projects\TS_fcst\artifacts\processed\log_returns_v1.parquet`, `core_balanced`.
- Число series: 418.
- Horizons: `[1, 5, 20]`.
- Rolling-origin folds: `3`.
- Window sizes: `{'1': 64, '5': 32, '20': 16}`.
- Validation: последний 20% блок train-fold, как в benchmark runner; для torch-моделей используется для early stopping.
- Models: `['chaotic_esn', 'transient_chaotic_esn', 'chaotic_mlp', 'chaotic_logistic_net', 'chaotic_lstm_forecast']`.
- Seeds/repeats: seed из benchmark config и model overrides; отдельных repeats в forecasting benchmark не найдено.
- Metric definitions: RMSE/MAE/Directional Accuracy из `src.forecasting.metrics.compute_regression_metrics`; gaps рассчитаны как test-train/test-validation для ошибок и train-test/validation-test для DA.

## 4. Объем расчета
- Ожидалось комбинаций: 18810.
- Успешно рассчитано: 18810.
- Failed/timeout: 0.
- Запуск начат: 2026-06-15T05:46:52Z.
- Checkpoint files: `D:\Projects\TS_fcst\artifacts\review_overfitting_diagnostics_full\raw_diagnostics_checkpoint.csv`, `D:\Projects\TS_fcst\artifacts\review_overfitting_diagnostics_full\raw_diagnostics_checkpoint.parquet`.

## 5. Главная таблица model x horizon
| presentation_model_name | horizon | n_observations | train_RMSE | validation_RMSE | test_RMSE | RMSE_test_train_gap | RMSE_test_val_gap | train_MAE | validation_MAE | test_MAE | MAE_test_train_gap | MAE_test_val_gap | train_DA | validation_DA | test_DA | DA_train_test_gap | DA_val_test_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chaotic_esn | 1 | 1254 | 0.026004 | 0.035175 | 0.034506 | 0.008502 | -0.000669 | 0.017235 | 0.022820 | 0.021234 | 0.003999 | -0.001586 | 0.578715 | 0.486832 | 0.484830 | 0.093885 | 0.002002 |
| transient_chaotic_esn | 1 | 1254 | 0.028302 | 0.029676 | 0.028910 | 0.000608 | -0.000766 | 0.018241 | 0.018857 | 0.017104 | -0.001137 | -0.001753 | 0.518455 | 0.494299 | 0.491815 | 0.026641 | 0.002484 |
| chaoticMLP / chaotic_mlp | 1 | 1254 | 0.029944 | 0.030017 | 0.029492 | -0.000453 | -0.000525 | 0.019526 | 0.019464 | 0.018070 | -0.001455 | -0.001393 | 0.488700 | 0.491213 | 0.480856 | 0.007844 | 0.010357 |
| chaotic_logistic_net | 1 | 1254 | 0.032245 | 0.031960 | 0.031370 | -0.000875 | -0.000590 | 0.022005 | 0.021879 | 0.020509 | -0.001496 | -0.001370 | 0.475597 | 0.492913 | 0.480195 | -0.004598 | 0.012718 |
| chaotic_lstm_forecast | 1 | 1254 | 0.029262 | 0.028980 | 0.028494 | -0.000768 | -0.000486 | 0.018482 | 0.018292 | 0.016719 | -0.001762 | -0.001573 | 0.484893 | 0.505831 | 0.486655 | -0.001762 | 0.019176 |
| chaotic_esn | 5 | 1254 | 0.050024 | 0.071872 | 0.070622 | 0.020598 | -0.001250 | 0.034809 | 0.049700 | 0.046218 | 0.011409 | -0.003482 | 0.611179 | 0.503875 | 0.499328 | 0.111851 | 0.004547 |
| transient_chaotic_esn | 5 | 1254 | 0.055931 | 0.061772 | 0.061467 | 0.005536 | -0.000305 | 0.038113 | 0.041929 | 0.039703 | 0.001590 | -0.002226 | 0.558594 | 0.506220 | 0.501997 | 0.056597 | 0.004223 |
| chaoticMLP / chaotic_mlp | 5 | 1254 | 0.058404 | 0.059233 | 0.060784 | 0.002381 | 0.001551 | 0.039569 | 0.040084 | 0.039244 | -0.000325 | -0.000840 | 0.520676 | 0.535481 | 0.494740 | 0.025936 | 0.040741 |
| chaotic_logistic_net | 5 | 1254 | 0.061415 | 0.060818 | 0.062342 | 0.000928 | 0.001525 | 0.042137 | 0.041586 | 0.041170 | -0.000967 | -0.000416 | 0.500124 | 0.535202 | 0.495099 | 0.005026 | 0.040103 |
| chaotic_lstm_forecast | 5 | 1254 | 0.058686 | 0.058990 | 0.059981 | 0.001295 | 0.000991 | 0.039189 | 0.039462 | 0.038045 | -0.001144 | -0.001417 | 0.514041 | 0.537326 | 0.499872 | 0.014170 | 0.037455 |
| chaotic_esn | 20 | 1254 | 0.101713 | 0.119066 | 0.125896 | 0.024184 | 0.006830 | 0.073566 | 0.088175 | 0.087704 | 0.014138 | -0.000471 | 0.595181 | 0.520193 | 0.516753 | 0.078428 | 0.003439 |
| transient_chaotic_esn | 20 | 1254 | 0.102926 | 0.113825 | 0.119251 | 0.016325 | 0.005426 | 0.074429 | 0.085160 | 0.084776 | 0.010347 | -0.000384 | 0.589062 | 0.520490 | 0.517327 | 0.071735 | 0.003162 |
| chaoticMLP / chaotic_mlp | 20 | 1254 | 0.110317 | 0.103418 | 0.118274 | 0.007957 | 0.014856 | 0.080115 | 0.077098 | 0.085155 | 0.005039 | 0.008056 | 0.534449 | 0.600780 | 0.499073 | 0.035376 | 0.101707 |
| chaotic_logistic_net | 20 | 1254 | 0.114979 | 0.104861 | 0.119614 | 0.004635 | 0.014753 | 0.083751 | 0.078206 | 0.086519 | 0.002768 | 0.008313 | 0.519257 | 0.596133 | 0.503987 | 0.015270 | 0.092147 |
| chaotic_lstm_forecast | 20 | 1254 | 0.106783 | 0.105668 | 0.115011 | 0.008228 | 0.009343 | 0.076548 | 0.078303 | 0.081385 | 0.004836 | 0.003081 | 0.562782 | 0.565570 | 0.514341 | 0.048441 | 0.051230 |

## 6. Интерпретация
Диагностика выявила train-test gap у части хаотических моделей; выводы о них следует формулировать ограниченно.

Overall gap distribution:
- RMSE test-train median: 0.000713; p75: 0.015853.
- DA train-test median: 0.033304; p75: 0.085390.

- Largest single RMSE test-train gap: `chaotic_esn`, series `RU:GLTR`, h=20, fold=2: 1.972193.
- Largest single DA train-test gap: `chaotic_lstm_forecast`, series `RU:OGKB`, h=20, fold=1: 0.404971.


Validation-test gaps are interpreted separately from train-test gaps. A large time-series gap can reflect overfitting, but can also reflect a regime shift; therefore the conclusion is diagnostic and deliberately limited.

## 7. Ограничения
- Диагностика не доказывает невозможность переобучения.
- Train/test gap во временных рядах может отражать не только overfitting, но и изменение рыночного режима.
- Метрики train рассчитаны на fit-подвыборке после отделения validation-блока; validation-метрики рассчитаны на последнем 20% блоке train-fold.
- Для reservoir-моделей validation-блок не управляет обучением, но сохранен как дополнительный хронологический holdout.

## 8. Готовый текст для ответа рецензенту
Дополнительная диагностика выявила train-test gap у части хаотических моделей, наиболее выраженный для отдельных моделей/горизонтов. Это подтверждает чувствительность хаотических архитектур к параметрам и необходимость контроля переобучения. Поэтому в работе не утверждается универсальное преимущество cNN: вывод ограничен тем, что хаотические модели могут быть конкурентоспособны в отдельных постановках, особенно по Directional Accuracy, но требуют аккуратной validation/test-проверки.

## Backup slide
- Expected combinations: 18810; successful: 18810; failed: 0.
- Series: 418; horizons: 3; folds: 3.
- RMSE test-train median/p75: 0.000713/0.015853.
- DA train-test median/p75: 0.033304/0.085390.
- Main conclusion: Диагностика выявила train-test gap у части хаотических моделей; выводы о них следует формулировать ограниченно.
