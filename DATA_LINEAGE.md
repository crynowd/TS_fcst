# Data Lineage

This document describes how data flows through the pipeline.

---

## Raw Data (external)

External sources:

RU:
D:\Projects\Data\Stocks\ru\D1

US:
D:\Projects\Data\Stocks\us\Stocks

Each file contains OHLCV data for a single ticker.

---

## Stage 1: Data Inventory

Script:
src/cli/run_data_inventory.py

Output:

artifacts/processed/series_catalog_v1.parquet

Report:

artifacts/reports/series_catalog_v1.xlsx

Purpose:

- scan data sources
- validate files
- build catalog of time series
- detect data quality issues

---

## Stage 2: Log Returns

Input:

series_catalog_v1.parquet

Output:

log_returns_v1.parquet

---

## Stage 3: Feature Extraction

Multiple blocks:

A: Dependence / Memory  
B: Spectrum / Noise / Complexity  
C: Tails / Distribution  
D: Nonlinear / Chaos

Outputs:

features_block_A_v1.parquet  
features_block_B_v1.parquet  
features_block_C_v1.parquet  
features_block_D_v1.parquet

---

## Stage 4: Feature Validation

Output:

feature_quality_report.xlsx

---

## Stage 5: Clustering

Output:

regime_labels.parquet

---

## Stage 6: Forecast Experiments

Output:

model_performance_by_regime.xlsx

---

## Stage 7: Meta-model

Output:

model_selection_pipeline.pkl