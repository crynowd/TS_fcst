# ESN / Baseline Leakage Audit

Дата проверки: 2026-04-27

Рабочая директория: `D:\Projects\TS_fcst`

Фактическая ветка при переносе и повторной проверке: `experiment/forecasting-audit-v2`

## Проверенные файлы

- `src/forecasting/windowing.py`
- `src/forecasting/targets.py`
- `src/forecasting/runners.py`
- `src/forecasting/registry.py`
- `src/forecasting/adapters/naive.py`
- `src/forecasting/adapters/sklearn_models.py`
- `src/forecasting/adapters/esn_models.py`
- `src/forecasting/architectures/esn.py`
- `src/forecasting/architectures/transient_esn.py`
- `src/forecasting/metrics.py`
- `src/config/loader.py`
- `configs/forecasting_benchmark_smoke_v1.yaml`
- `configs/forecasting_selected_architectures_v1.yaml`

## Проверенные модели

Baseline:

- `naive_zero`
- `naive_mean`
- `ridge_lag`

ESN-like модели, зарегистрированные в `registry.py`:

- `esn`
- `chaotic_esn`
- `transient_chaotic_esn`

## Подтвержденные безопасные места

- Train/test split строится через `build_rolling_origin_folds`: train индексы идут строго раньше test индексов, пересечения нет, shuffle не используется.
- `build_supervised_windows` формирует direct-horizon target как сумму `r[t+1]..r[t+h]`; feature window заканчивается в `t`, target начинается в `t+1`.
- Для `h=1`, `h=5`, `h=20` диагностически подтверждено: `feature_end_idx < target_start_idx`, длина target window равна horizon.
- Общая benchmark-логика передает всем моделям одни и те же `fold.train_idx` / `fold.test_idx`, поэтому split у baseline и ESN совпадает.
- В проверенных ESN-адаптерах нет scaler и нет `fit_transform` на полном ряду.
- ESN readout обучается только в `model.fit(X_fit, y_fit)`; test arrays в `fit` не передаются.
- `naive_zero` предсказывает только `0.0`.
- `naive_mean` считает среднее по переданному train/fitting target, не по полному ряду.
- `ridge_lag` получает только уже построенные прошлые lag/window features и обучается через `Ridge.fit(X_train, y_train)` на train slice.
- `compute_regression_metrics` общий для всех моделей; RMSE и directional accuracy считаются одинаковым кодом.
- ESN, chaotic ESN и transient chaotic ESN сбрасывают reservoir state внутри каждой строки окна; diagnostic order-invariance check не выявил state carry-over между test rows.
- Transient chaotic ESN строит gain schedule только по длине одного окна `T`, не по train/test участку целиком.

## Найденные риски

- `run_forecasting_benchmark` строит все supervised windows до fold split. По текущей реализации это безопасно, потому что windowing не обучает scaler/model и использует только арифметику по локальным прошлым/будущим индексам. Но такой порядок хрупкий: любая будущая нормализация внутри window builder может стать leakage. Для защиты добавлены metadata/assertions.
- В benchmark есть validation split внутри train: `X_fit = X_train[:-val_size]`, `X_val = X_train[-val_size:]`. ESN/baseline сейчас validation не используют. Для torch-моделей это отдельный вопрос, но test туда не попадает.
- `target permutation check` на реальном малом ряду показывает ожидаемое ухудшение для `h=1`, но не для `h=5` и `h=20`. Это не доказывает leakage: финансовые ряды имеют слабый сигнал, а permutation check на одном ряду и одном fold может быть нестабилен. Но для большого прогона стоит трактовать "слишком хорошие" ESN-результаты на длинных горизонтах осторожно и повторить sanity-check на 3+ рядах.
- `configs/paths.local.yaml` указывает `project_root: D:/Projects/TS_fcst`, что соответствует рабочей директории повторной проверки. Diagnostic script пишет JSON в `artifacts/reports/forecasting_audit_v2/`.
- В репозитории уже отслеживаются тяжелые артефакты в `artifacts/forecasting/` и `artifacts/reports/`. `.gitignore` предотвращает добавление новых matching файлов, но не удаляет из индекса уже tracked файлы.

## Найденные ошибки

- Прямой leakage в ESN/baseline коде не найден.
- Найдена поврежденная `.gitignore` с NUL-строками и неполным покрытием generated forecasting/report outputs. Файл очищен и расширен.

## Внесенные исправления

- В `src/forecasting/windowing.py` добавлены metadata для каждой supervised-строки:
  - `feature_start_idx`
  - `feature_end_idx`
  - `target_start_idx`
  - `target_end_idx`
- В `build_supervised_windows` добавлены assertions против overlap feature/target и выхода target за пределы ряда.
- В `build_rolling_origin_folds` добавлены assertions против пересечения train/test и нарушения временного порядка.
- Добавлен diagnostic CLI:
  - `src/cli/run_esn_baseline_leakage_diagnostics.py`
- В `tests/test_forecasting_benchmark.py` добавлена проверка window metadata.
- `.gitignore` очищен от поврежденных строк и расширен для generated parquet/csv/xlsx/json outputs.

## Команда diagnostic check

```powershell
python -m src.cli.run_esn_baseline_leakage_diagnostics --config configs/forecasting_benchmark_smoke_v1.yaml --max-series 1 --horizons 1,5,20 --output artifacts/reports/forecasting_audit_v2/esn_baseline_leakage_diagnostic.json
```

Результат:

```text
diagnostic_ok tasks=18 series=1 models=6 permutation_warnings=6 output=D:\Projects\TS_fcst\artifacts\reports\forecasting_audit_v2\esn_baseline_leakage_diagnostic.json
```

Проверенный ряд: `RU:ABIO`

Модели: `naive_zero`, `naive_mean`, `ridge_lag`, `esn`, `chaotic_esn`, `transient_chaotic_esn`

Horizon results:

| Horizon | Structural checks | ESN order-invariance | Target permutation sanity |
| --- | --- | --- | --- |
| 1 | pass | pass | pass для всех ESN-like |
| 5 | pass | pass | warning для всех ESN-like |
| 20 | pass | pass | warning для всех ESN-like |

Target permutation RMSE ratios:

| Model | h=1 | h=5 | h=20 |
| --- | ---: | ---: | ---: |
| `esn` | 1.287 | 1.004 | 0.986 |
| `chaotic_esn` | 1.189 | 0.843 | 0.947 |
| `transient_chaotic_esn` | 1.274 | 1.001 | 0.977 |

Интерпретация: на `h=1` разрушение связи X/y ухудшает качество ESN, что поддерживает отсутствие очевидной leakage. На `h=5` и `h=20` sanity-check неинформативен или слабый на одном финансовом ряду; это residual risk, а не найденная ошибка split/windowing.

## Дополнительная проверка тестами

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_forecasting_benchmark.py -q
```

Результат:

```text
9 passed in 4.91s
```

Без `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` тестовый запуск в данном окружении падает до выполнения тестов из-за стороннего pytest plugin `langsmith` и отсутствующего пакета `httpx`.

## Вывод

Текущей реализации ESN/baseline можно доверять в части проверенных structural leakage инвариантов: split общий и rolling-origin, shuffle нет, scaler leakage нет, readout ESN не обучается на test, horizon alignment корректен, baseline реализованы train-only.

Перед большим прогоном рекомендуется:

- повторить diagnostic на `--max-series 3`;
- отдельно просмотреть результаты target permutation sanity для `h=5` и `h=20`;
- не интерпретировать улучшение ESN на длинных горизонтах как надежный signal без дополнительного permutation/shift sanity на нескольких рядах.
