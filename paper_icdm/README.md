# ICDM Paper Reproducibility

`paper_icdm/` is a thin reproducibility layer for the ICDM paper on feature-based meta-learning for financial time-series forecasting model selection.

The main research code remains in `src/`. Existing `configs/` files describe the data, feature, forecasting, and meta-learning pipeline. Existing `artifacts/` files provide the processed data and computed results used to validate and later rebuild the paper tables and figures. This directory does not replace the full research repository; it identifies the exact paper route through the repository.

## Directory Structure

```text
paper_icdm/
  README.md              # entry point
  DATA.md                # data provenance and preprocessing
  REPRODUCIBILITY.md     # environment and validation plan
  RESULTS_MANIFEST.md    # mapping from paper outputs to artifacts
```

## Main Paper Path

Data inventory:

```text
configs/data_inventory_v1.yaml
```

Feature computation:

```text
configs/features_block_A_v1.yaml
configs/features_block_B_v1.yaml
configs/features_block_C_v1.yaml
configs/features_block_D_v1.yaml
configs/feature_screening_v1.yaml
configs/feature_consolidation_v1.yaml
```

Forecasting benchmark:

```text
configs/forecasting_benchmark_v2.yaml
configs/forecasting_selected_architectures_v1.yaml
```

Meta-learning:

```text
configs/meta_modeling_experiments_v2.yaml
```

## Experiment Scope

The paper studies 418 financial return series in the `core_balanced` dataset profile: 209 Russian equities and 209 U.S. equities. The forecasting benchmark uses horizons `1`, `5`, and `20`, input windows `64`, `32`, and `16`, and three rolling-origin folds.

The forecasting benchmark covers 11 candidate models:

1. zero-return predictor (`naive_zero`)
2. historical-mean predictor (`naive_mean`)
3. ridge autoregression (`ridge_lag`)
4. echo state network (`esn`)
5. chaotic spectral-radius ESN (`chaotic_esn`)
6. transient chaotic ESN (`transient_chaotic_esn`)
7. feed-forward MLP (`vanilla_mlp`)
8. logistic-map activated MLP (`chaotic_mlp`)
9. input-driven logistic-map reservoir (`chaotic_logistic_net`)
10. LSTM regressor (`lstm_forecast`)
11. chaos-initialized LSTM (`chaotic_lstm_forecast`)

The feature-based meta-learning stage uses 25 time-series features, grouped conceptually into long memory, linear dependence, volatility dependence, complexity and spectrum, distribution and tails, and phase-space structure. The v2 meta-learning configuration evaluates logistic regression, random forest, and CatBoost classifiers with candidate top-k sets `3`, `4`, `5`, and `6` across five repeated instrument splits.

## Smoke and Legacy Configs

`configs/forecasting_benchmark_smoke_v1.yaml` is a smoke/demo configuration and is not the main paper benchmark.

Older `meta_modeling_v1.yaml` and `meta_modeling_experiments_v1.yaml` paths are not the final paper route when the v2 artifacts are used. The final paper route documented here uses `configs/meta_modeling_experiments_v2.yaml` and the corresponding `*_v2` artifacts.

Clustering, architecture tuning, notebooks, and smoke configs are useful background or exploratory materials. They are not required to reproduce the final paper tables.

## Reproducible from Existing Artifacts

The current repository state can support artifact-based validation of:

- dataset counts for the `core_balanced` profile;
- the 25-feature list;
- forecasting benchmark summaries;
- winner counts by model or broad model family after a documented family mapping is added;
- meta-learning task results;
- paper tables and figures after future builder scripts are added.

Future scripts are intentionally not created in this stage:

- `check_artifacts.py`: to be added in the next stage;
- `build_paper_tables.py`: to be added later;
- `build_paper_figures.py`: to be added later.

No commands for those scripts are documented yet because the scripts do not exist in this stage.

