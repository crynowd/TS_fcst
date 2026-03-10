# TS_fcst

Research project for classification of financial time series and construction of a meta-model for forecasting model selection.

The project analyzes structural properties of financial time series and groups them into regimes based on statistical, spectral, distributional and nonlinear dynamics metrics.

The resulting classification will be used to build a meta-model that selects the most appropriate forecasting architecture depending on the detected regime.

---

## Main Goals

1. Extract structural metrics of time series:
   - memory and dependence
   - spectral structure and noise
   - tail properties of distributions
   - nonlinear / chaos-like dynamics

2. Cluster time series into structural regimes.

3. Evaluate forecasting models across regimes.

4. Build a meta-model that selects the best forecasting model depending on regime type.

---

## Data

Raw data is stored **outside the repository**.

Example locations:

RU market: D:\Projects\Data\Stocks\ru\D1
US market: D:\Projects\Data\Stocks\us\Stocks


Each file contains daily OHLCV data for one ticker.

---

## Core Pipeline

1. Data inventory and validation
2. Log-return transformation
3. Feature extraction (multiple blocks)
4. Feature validation
5. Clustering
6. Forecasting experiments
7. Meta-model construction

---

## Project Structure

See:

- `PROJECT_STRUCTURE.md`
- `DATA_LINEAGE.md`

---

## Configuration

All parameters are stored in YAML configs: configs/


Examples:
- paths.local.yaml
- data_inventory_v1.yaml

---

## Artifacts

Generated artifacts are stored in: artifacts/

but **not tracked by git**.

---

## Status

Early research phase.

