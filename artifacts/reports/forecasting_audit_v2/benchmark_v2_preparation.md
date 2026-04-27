# Forecasting Benchmark V2 Preparation

Date: 2026-04-27

## Changed files

- `configs/forecasting_benchmark_v2.yaml` - new v2 benchmark config.
- `src/config/loader.py` - supports v2 output layout, explicit `run_id`, `device`, `resume`, `save_predictions`, `progress_every_n`, and `training.by_model`.
- `src/cli/run_forecasting_benchmark.py` - respects configured `outputs.run_id` instead of always creating a timestamped id.
- `src/forecasting/runners.py` - adds v2 resume/cache, progress logging, split metadata, device reporting, detailed outputs, v2 Excel sheets, and config snapshot.
- `src/forecasting/registry.py` - adds `device: auto` resolution for torch models.
- `src/forecasting/io.py` - adds v2 split metadata schema and summary-only Excel exporter.
- `src/forecasting/data.py` - `build_series_lookup` can filter by `dataset_profile`; this fixed duplicated dates in split metadata.
- `src/forecasting/adapters/base.py`, `src/forecasting/adapters/torch_models.py` - existing neural training diagnostics and `weight_decay` support are used by v2.
- `src/forecasting/windowing.py` - existing leakage checks and fold ordering checks are retained.
- `tests/test_forecasting_benchmark.py` - adds regression coverage for profile-filtered lookup.
- `.gitignore` - adds recursive generated forecasting output ignores.

## New config

Created `configs/forecasting_benchmark_v2.yaml`.

Differences from `configs/forecasting_benchmark_smoke_v1.yaml`:

- Stage is `forecasting_benchmark_v2`.
- `n_folds: 3`.
- `horizons: [1, 5, 20]`.
- `device: auto`.
- `resume: true`.
- `save_metrics: true`.
- `save_predictions: false`.
- Explicit `random_seed: 2026`.
- Explicit `progress_every_n`.
- Stable `outputs.run_id: forecasting_benchmark_v2`.
- Heavy outputs go to `artifacts/forecasting/forecasting_benchmark_v2/`.
- Report outputs go to `artifacts/reports/forecasting_audit_v2/`.
- `data.max_series`, `data.dataset_limit`, and `data.series_ids` are configurable.

## Neural training params

MLP, chaotic MLP, chaotic logistic:

- `max_epochs: 30`
- `early_stopping_patience: 5`
- `learning_rate: 0.001`
- `batch_size: 128`
- `weight_decay: 0.0`

LSTM, chaotic LSTM:

- `max_epochs: 40`
- `early_stopping_patience: 6`
- `learning_rate: 0.001`
- `batch_size: 128`
- `weight_decay: 0.0`

The config uses `training.by_model`, while the runner still supports previous global `training` and runtime keys in `model_overrides`.

## Device selection

`device: auto` resolves through `resolve_torch_device`:

- CUDA available -> `cuda`
- CUDA unavailable -> `cpu`
- requested CUDA without CUDA -> falls back to `cpu`

The run logs and manifest include `requested_device`, `resolved_device`, `cuda_available`, and `gpu_name`. Torch task audit rows include the resolved device per torch model.

Smoke environment:

- `requested_device: auto`
- `resolved_device: cpu`
- `torch.cuda.is_available(): false`
- `gpu_name: ""`

## Folds

`n_folds=3` uses the existing rolling-origin splitter. Smoke verification after fixing `dataset_profile` lookup:

| series_id | fold_id | train_end | test_start | test_end | n_train | n_test |
|---|---:|---|---|---|---:|---:|
| RU:ABIO | 1 | 2018-10-29 | 2018-10-30 | 2020-10-02 | 484 | 484 |
| RU:ABIO | 2 | 2020-10-02 | 2020-10-05 | 2022-09-29 | 968 | 484 |
| RU:ABIO | 3 | 2022-09-29 | 2022-09-30 | 2024-08-26 | 1452 | 484 |
| RU:ABRD | 1 | 2018-10-10 | 2018-10-11 | 2020-10-02 | 484 | 484 |
| RU:ABRD | 2 | 2020-10-02 | 2020-10-05 | 2022-09-29 | 968 | 484 |
| RU:ABRD | 3 | 2022-09-29 | 2022-09-30 | 2024-08-26 | 1452 | 484 |

Checks:

- 3 distinct `train_end`, `test_start`, and `test_end` values per `series_id x horizon`.
- `train_end < test_start` for all rows.
- Split metadata is shared across models because it is generated once per `series_id x horizon x fold`.

## Resume/cache

When `resume=true`, the runner skips successful tasks by:

`series_id x model_name x horizon x fold x config_hash`

The task config hash includes:

- model params,
- neural training params,
- horizon,
- fold id,
- n_folds and validation settings,
- window size,
- random seed,
- dataset profile,
- dataset version,
- source path.

Smoke resume check:

- First `forecasting_benchmark_v2_smoke2`: 66 tasks, 66 success, 0 skipped.
- Second same run: 66 tasks, 66 success retained, 66 skipped, elapsed ~0.86 s.

## Outputs

V2 outputs created under `artifacts/forecasting/<run_id>/`:

- `metrics_long.parquet`
- `metrics_long.csv`
- `split_metadata.parquet`
- `errors.csv`
- `run_manifest.json`
- `config_snapshot.yaml`
- `task_audit.parquet`
- `predictions.parquet` only when `save_predictions=true`

Smoke `save_predictions=false`, so `predictions.parquet` was not created.

Excel summary:

- `artifacts/reports/forecasting_audit_v2/forecasting_benchmark_forecasting_benchmark_v2_smoke2.xlsx`

Sheets:

- `run_info`
- `dataset_summary`
- `model_summary_by_horizon`
- `model_series_summary`
- `fold_stability`
- `winners_by_horizon_metric`
- `error_summary`
- `resume_summary`
- `speed_summary`
- `device_summary`
- `config_summary`
- `model_registry`

Excel does not include predictions.

## Smoke run

Command shape used:

```powershell
$env:PYTHONPATH='.'
# Load configs/forecasting_benchmark_v2.yaml, override in-memory:
# max_series=2, dataset_limit=2, horizons=[1], n_folds=3, save_predictions=false,
# run_id=forecasting_benchmark_v2_smoke2.
```

Results:

- Series: 2 (`RU:ABIO`, `RU:ABRD`)
- Horizons: `[1]`
- Folds: 3
- Models: 11
- Total tasks: 66
- Success: 66
- Timeout: 0
- Error: 0
- First-run wall time: ~49.13 s
- Resume run wall time: ~0.86 s
- Mean measured task time from metrics: ~0.73 s

Mean smoke metrics by model:

| model_name | horizon | mean RMSE | mean directional_accuracy | count |
|---|---:|---:|---:|---:|
| naive_zero | 1 | 0.034798 | 0.073347 | 6 |
| naive_mean | 1 | 0.034817 | 0.454545 | 6 |
| chaotic_lstm_forecast | 1 | 0.034900 | 0.457989 | 6 |
| lstm_forecast | 1 | 0.035047 | 0.458678 | 6 |
| chaotic_mlp | 1 | 0.035052 | 0.440083 | 6 |
| ridge_lag | 1 | 0.035370 | 0.474174 | 6 |
| vanilla_mlp | 1 | 0.035510 | 0.448003 | 6 |
| esn | 1 | 0.036512 | 0.481061 | 6 |
| transient_chaotic_esn | 1 | 0.036592 | 0.481405 | 6 |
| chaotic_logistic_net | 1 | 0.036733 | 0.444215 | 6 |
| chaotic_esn | 1 | 0.041311 | 0.463843 | 6 |

Mean smoke metrics by series and model:

| series_id | model_name | horizon | mean RMSE | mean directional_accuracy | count |
|---|---|---:|---:|---:|---:|
| RU:ABIO | naive_zero | 1 | 0.042154 | 0.010331 | 3 |
| RU:ABIO | naive_mean | 1 | 0.042195 | 0.518595 | 3 |
| RU:ABIO | chaotic_lstm_forecast | 1 | 0.042258 | 0.475895 | 3 |
| RU:ABIO | lstm_forecast | 1 | 0.042395 | 0.479339 | 3 |
| RU:ABIO | chaotic_mlp | 1 | 0.042407 | 0.462810 | 3 |
| RU:ABIO | ridge_lag | 1 | 0.043058 | 0.504132 | 3 |
| RU:ABIO | vanilla_mlp | 1 | 0.043065 | 0.474518 | 3 |
| RU:ABIO | chaotic_logistic_net | 1 | 0.043789 | 0.481405 | 3 |
| RU:ABIO | esn | 1 | 0.044719 | 0.495868 | 3 |
| RU:ABIO | transient_chaotic_esn | 1 | 0.044937 | 0.498623 | 3 |
| RU:ABIO | chaotic_esn | 1 | 0.048116 | 0.465565 | 3 |
| RU:ABRD | naive_mean | 1 | 0.027439 | 0.390496 | 3 |
| RU:ABRD | naive_zero | 1 | 0.027443 | 0.136364 | 3 |
| RU:ABRD | chaotic_lstm_forecast | 1 | 0.027542 | 0.440083 | 3 |
| RU:ABRD | ridge_lag | 1 | 0.027682 | 0.444215 | 3 |
| RU:ABRD | chaotic_mlp | 1 | 0.027698 | 0.417355 | 3 |
| RU:ABRD | lstm_forecast | 1 | 0.027699 | 0.438017 | 3 |
| RU:ABRD | vanilla_mlp | 1 | 0.027955 | 0.421488 | 3 |
| RU:ABRD | transient_chaotic_esn | 1 | 0.028247 | 0.464187 | 3 |
| RU:ABRD | esn | 1 | 0.028305 | 0.466253 | 3 |
| RU:ABRD | chaotic_logistic_net | 1 | 0.029677 | 0.407025 | 3 |
| RU:ABRD | chaotic_esn | 1 | 0.034506 | 0.462121 | 3 |

Time by series from first-run metrics:

- `RU:ABIO`: ~23.97 s
- `RU:ABRD`: ~24.11 s

Time by model from first-run metrics:

- `naive_zero`: ~0.00004 s
- `naive_mean`: ~0.00027 s
- `ridge_lag`: ~0.006 s
- `esn`: ~6.01 s
- `chaotic_esn`: ~5.23 s
- `transient_chaotic_esn`: ~5.30 s
- `vanilla_mlp`: ~1.93 s
- `chaotic_mlp`: ~1.19 s
- `chaotic_logistic_net`: ~4.49 s
- `lstm_forecast`: ~5.90 s
- `chaotic_lstm_forecast`: ~18.01 s

Rough full-run estimate for `300 series x 3 horizons x 3 folds x 11 models = 29,700 tasks`:

- Using smoke average ~0.73 s/task: ~21,600 s, about 6.0 hours on this CPU environment.
- This is approximate; horizons 5 and 20 have different window sizes and effective sample counts, and GPU availability can materially change torch model time.

## Commands run

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='.'; pytest tests/test_forecasting_benchmark.py -q
```

Result: `10 passed in 3.32s`.

Initial plain `pytest` failed before test collection because a user-level pytest plugin `langsmith` required missing `httpx`. The repository tests passed with plugin autoload disabled.

## Gitignore

Verified generated heavy outputs are ignored:

- `artifacts/forecasting/**/*.parquet`
- `artifacts/forecasting/**/*.csv`
- `artifacts/forecasting/**/*.xlsx`
- `artifacts/forecasting/**/*.json`
- `artifacts/forecasting/**/*.yaml`
- `artifacts/forecasting/**/predictions*`
- `artifacts/forecasting/**/temp/`
- `artifacts/reports/**/*.xlsx`
- `artifacts/reports/**/*.json`
- `artifacts/reports/**/*.csv`
- `artifacts/reports/**/*.parquet`

Markdown reports under `artifacts/reports/forecasting_audit_v2/` remain trackable.

## Git status

Relevant source/config changes:

```text
 M .gitignore
 M src/cli/run_forecasting_benchmark.py
 M src/config/loader.py
 M src/forecasting/adapters/base.py
 M src/forecasting/adapters/torch_models.py
 M src/forecasting/data.py
 M src/forecasting/io.py
 M src/forecasting/registry.py
 M src/forecasting/runners.py
 M src/forecasting/windowing.py
 M tests/test_forecasting_benchmark.py
?? configs/forecasting_benchmark_v2.yaml
?? artifacts/reports/forecasting_audit_v2/
```

The working tree also shows tracked `__pycache__/*.pyc` modifications after pytest; they are generated bytecode changes, not source changes. Existing unrelated untracked audit CLI files are still present:

```text
?? src/cli/run_esn_baseline_leakage_diagnostics.py
?? src/cli/run_neural_training_audit.py
```

No commit or push was made.
