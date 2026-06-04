# ICDM Paper Results Manifest

This manifest maps each paper result to its source files in the repository and records what still needs to be generated or uploaded.

## Paper Output Mapping

Coverage can be checked from the repository root with:

```bash
python paper_icdm/scripts/check_artifacts.py
```

| Paper item | Description | Primary source artifact/config | Derived output planned path | Status | Notes |
|---|---|---|---|---|---|
| Table I | 25 time-series features | `artifacts/meta_modeling/feature_list_v2.csv`; `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet` | `paper_icdm/tables/table_i_features.csv` | Compact sources tracked; output not created | `feature_list_v2.csv` has 25 rows. |
| Table II | 11 forecasting candidates and protocol parameters | `configs/forecasting_selected_architectures_v1.yaml`; `configs/forecasting_benchmark_v2.yaml`; model registry/code for baselines | `paper_icdm/tables/table_ii_candidates.csv` | Config sources tracked; output not created | Baselines are listed in `forecasting_benchmark_v2.yaml`; tuned architectures are listed in selected architectures config. |
| Table III | Experimental protocol | `configs/forecasting_benchmark_v2.yaml`; `configs/meta_modeling_experiments_v2.yaml`; `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json`; `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet` | `paper_icdm/tables/table_iii_protocol.csv` or documentation table | Compact sources tracked; output not created | Exact table format to be defined later. |
| Table IV | Direct forecasting benchmark | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet` | `paper_icdm/tables/table_iv_direct_forecasting.csv` | Compact source tracked; output not created | Fold-level metrics are available without `predictions.parquet`. |
| Table V | Winner counts by broad model family | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`; `paper_icdm/model_family_mapping.csv` | `paper_icdm/tables/table_v_winner_family_counts.csv` | Compact sources tracked; output not created | The family mapping is an explicit paper source, not inferred from data. |
| Table VI | Best fixed model, best observed metamodel, and oracle | `artifacts/meta_modeling/task_results_v2.parquet`; `artifacts/meta_modeling/split_assignments_v2.csv` | `paper_icdm/tables/table_vi_meta_selection_results.csv` | Compact sources tracked; output not created | `routing_rows_v2.parquet` remains an optional external source for route-level reconstruction. |
| Figure 1 | Experimental scheme | Paper diagram or future manual recreation | `paper_icdm/figures/figure_1_experimental_scheme.png` | Schematic; not generated from data in this stage | Do not claim data-generated status unless code is added later. |
| Figure 2 | RMSE fixed/metamodel/oracle comparison | `artifacts/meta_modeling/task_results_v2.parquet` | `paper_icdm/figures/figure_2_rmse_fixed_meta_oracle.png` | Compact source tracked; output not created | Plot script to be added later. |
| Figure 3 | Directional accuracy fixed/metamodel/oracle comparison | `artifacts/meta_modeling/task_results_v2.parquet` | `paper_icdm/figures/figure_3_da_fixed_meta_oracle.png` | Compact source tracked; output not created | Plot script to be added later. |
| Figure 4 | Winner distribution by family, metric, and horizon | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`; `paper_icdm/model_family_mapping.csv` | `paper_icdm/figures/figure_4_winner_distribution.png` | Compact sources tracked; output not created | Uses the explicit broad-family mapping. |

## Artifact Availability Checklist

| Required artifact | Exists locally | Tracked in Git | Recommended action |
|---|---:|---:|---|
| `artifacts/processed/log_returns_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/processed/series_catalog_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/processed/dataset_profiles_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet` | yes | yes | Compact tracked source for Table I validation. |
| `artifacts/meta_modeling/feature_list_v2.csv` | yes | yes | Compact tracked source for Table I. |
| `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet` | yes | yes | Compact tracked source for Tables IV-V and Figure 4. |
| `artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet` | yes | no | Large optional file; prefer Git LFS, GitHub Release, or external archival storage. |
| `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json` | yes | yes | Compact tracked metadata for Table III. |
| `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet` | yes | yes | Compact tracked metadata for protocol validation. |
| `artifacts/meta_modeling/split_assignments_v2.csv` | yes | yes | Compact tracked split source for Table VI validation. |
| `artifacts/meta_modeling/routing_rows_v2.parquet` | yes | no | Large optional file; prefer Git LFS, GitHub Release, or external archival storage. |
| `artifacts/meta_modeling/task_results_v2.parquet` | yes | yes | Compact tracked source for Table VI and Figures 2-3. |
| `artifacts/meta_modeling/model_order_mapping_v2.csv` | yes | yes | Compact tracked source for model-name validation. |
| `paper_icdm/model_family_mapping.csv` | yes | yes | Explicit paper source for Table V and Figure 4 broad-family grouping. |
| `artifacts/reports.zip` | yes | no | Large optional report bundle; prefer Git LFS, GitHub Release, or external archival storage if needed. |

## Large and Generated Reports

The compact v2 sources listed above are tracked directly because each is below 10 MB. The large optional files are intentionally not committed directly:

- `artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet` (152.4 MiB);
- `artifacts/meta_modeling/routing_rows_v2.parquet` (98.7 MiB);
- `artifacts/reports.zip` (271.9 MiB).

Recommended archival strategies for those files are Git LFS, GitHub Release assets, or an external archive with a checksum manifest. `routing_rows_v2.parquet` is useful for route-level reconstruction, but `task_results_v2.parquet` is the preferred compact source for Table VI and Figures 2-3.

Excel reports and generated paper report files: needs verification. No paper-specific `paper_icdm/tables/` or `paper_icdm/figures/` outputs are created in this stage. Stage 3B will add `build_paper_tables.py`; figure/table builder scripts are intentionally out of scope for Stage 3A.
