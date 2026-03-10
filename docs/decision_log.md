# Decisions Log

Record of major methodological decisions.

---

## 2026-03-10

Initial project design decisions.

- Target variable chosen as log-returns.
- Forecast horizons defined as 1, 5, and 20.
- Unit of classification: entire time series.
- Target time series length: 2000 observations.
- Minimum acceptable length: 1500 observations.
- Time slice policy: use most recent observations.
- Feature extraction divided into four blocks.
- Data stored outside repository.