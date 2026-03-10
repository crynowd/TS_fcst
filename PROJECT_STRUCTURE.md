# Project Structure

This document describes the architecture of the repository.

---

## Root
TS_fcst - Main research repository.

---

## configs/

Configuration files for all pipeline stages.

Examples:
- data paths
- feature parameters
- pipeline settings

Configs are versioned to guarantee reproducibility.

---

## docs/

Project documentation.

Important files:

- metrics_spec_v1.md
- decisions_log.md
- test_plan.md

---

## src/

Source code.

Main modules:

### src/data
Data ingestion and validation.

### src/features
Feature extraction modules.

### src/reporting
Excel and summary report generation.

### src/utils
Shared utilities:
- logging
- manifests
- validation helpers

### src/cli
Command line entry points for pipeline stages.

---

## tests/

Unit tests for critical parts of the pipeline.

---

## artifacts/

Generated files.

Subfolders:

- interim
- processed
- features
- reports
- logs
- manifests

Artifacts are **not version controlled**.

---

## notebooks/

Exploratory notebooks.

Not required for pipeline execution.

---

## Design Philosophy

The project is organized as a **research pipeline** with reproducible stages.

Each stage should:

- read config
- produce artifacts
- log execution
- write a manifest