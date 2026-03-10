# Metrics Specification v1

This document defines structural metrics used to characterize time series.

---

## Target Variable

Log-returns.

Definition:

r_t = log(P_t) − log(P_{t-1})

---

## Time Series Length

Target length:
2000 observations

Minimum acceptable length:
1500 observations

Series shorter than 1500 are excluded from the main dataset.

Series are constructed using the **last N observations**.

---

## Unit of Analysis

One time series corresponds to one asset.

Metrics are computed over the entire series.

---

## Feature Blocks

### Block A: Dependence / Memory

- Hurst exponent (R/S)
- Hurst exponent (DFA)
- Integrated ACF of returns
- Integrated ACF of absolute returns
- Ljung–Box statistics
- Variance Ratio

---

### Block B: Spectrum / Noise

- Spectral slope
- Spectral entropy
- Spectral flatness
- NoiseFN

Reserve metrics:

- Permutation entropy
- Lempel-Ziv complexity
- Sample entropy

---

### Block C: Distribution / Tails

- Kurtosis
- Robust kurtosis
- Symmetric quantile tail ratio
- Upper tail ratio
- Lower tail ratio

Reserve:

- Hill tail index

---

### Block D: Nonlinear Dynamics

- Largest Lyapunov exponent
- Lyapunov time
- Embedding dimension
- Correlation dimension

Reserve:

- 0–1 chaos test
- Recurrence Quantification Analysis metrics

---

## Feature Validation

After computation, features will be evaluated using:

- missingness analysis
- stability checks
- correlation analysis
- synthetic benchmark tests