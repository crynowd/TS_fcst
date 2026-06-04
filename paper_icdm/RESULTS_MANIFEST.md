# ICDM Paper Results Manifest

This manifest maps each paper result to its source files in the repository and records what still needs to be generated or uploaded.

## Paper Output Mapping

| Paper item | Description | Primary source artifact/config | Derived output planned path | Status | Notes |
|---|---|---|---|---|---|
| Table I | 25 time-series features | `artifacts/meta_modeling/feature_list_v2.csv`; `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet` | `paper_icdm/tables/table_i_features.csv` | Source exists locally; output not created | `feature_list_v2.csv` has 25 rows locally but is not tracked. |
| Table II | 11 forecasting candidates and protocol parameters | `configs/forecasting_selected_architectures_v1.yaml`; `configs/forecasting_benchmark_v2.yaml`; model registry/code for baselines | `paper_icdm/tables/table_ii_candidates.csv` | Config sources tracked; output not created | Baselines are listed in `forecasting_benchmark_v2.yaml`; tuned architectures are listed in selected architectures config. |
| Table III | Experimental protocol | `configs/forecasting_benchmark_v2.yaml`; `configs/meta_modeling_experiments_v2.yaml`; `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json`; `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet` | `paper_icdm/tables/table_iii_protocol.csv` or documentation table | Configs tracked; artifacts exist locally but are not tracked | Exact table format to be defined later. |
| Table IV | Direct forecasting benchmark | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet` | `paper_icdm/tables/table_iv_direct_forecasting.csv` | Source exists locally; output not created | Source is not tracked and should be archived/uploaded. |
| Table V | Winner counts by broad model family | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`; model-family mapping to be documented | `paper_icdm/tables/table_v_winner_family_counts.csv` | Source exists locally; output not created | Family mapping must be documented before final generation. |
| Table VI | Best fixed model, best observed metamodel, and oracle | `artifacts/meta_modeling/task_results_v2.parquet`; `artifacts/meta_modeling/routing_rows_v2.parquet`; `artifacts/meta_modeling/split_assignments_v2.csv` | `paper_icdm/tables/table_vi_meta_selection_results.csv` | Sources exist locally; output not created | Sources are not tracked and should be archived/uploaded. |
| Figure 1 | Experimental scheme | Paper diagram or future manual recreation | `paper_icdm/figures/figure_1_experimental_scheme.png` | Schematic; not generated from data in this stage | Do not claim data-generated status unless code is added later. |
| Figure 2 | RMSE fixed/metamodel/oracle comparison | `artifacts/meta_modeling/task_results_v2.parquet`; `artifacts/meta_modeling/routing_rows_v2.parquet` | `paper_icdm/figures/figure_2_rmse_fixed_meta_oracle.png` | Sources exist locally; output not created | Plot script to be added later. |
| Figure 3 | Directional accuracy fixed/metamodel/oracle comparison | `artifacts/meta_modeling/task_results_v2.parquet`; `artifacts/meta_modeling/routing_rows_v2.parquet` | `paper_icdm/figures/figure_3_da_fixed_meta_oracle.png` | Sources exist locally; output not created | Plot script to be added later. |
| Figure 4 | Winner distribution by family, metric, and horizon | `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet`; model-family mapping | `paper_icdm/figures/figure_4_winner_distribution.png` | Source exists locally; output not created | Requires documented family mapping. |

## Artifact Availability Checklist

| Required artifact | Exists locally | Tracked in Git | Recommended action |
|---|---:|---:|---|
| `artifacts/processed/log_returns_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/processed/series_catalog_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/processed/dataset_profiles_v1.parquet` | yes | yes | Keep tracked unless repository size policy changes. |
| `artifacts/features/fold_aware_features_v2/final_train_only_features_by_fold.parquet` | yes | no | Upload or track in Git LFS/release storage; needed for Table I and meta-learning validation. |
| `artifacts/meta_modeling/feature_list_v2.csv` | yes | no | Small file; consider tracking in Git in a later stage. |
| `artifacts/forecasting/forecasting_benchmark_v2/metrics_long.parquet` | yes | no | Upload or track via Git LFS/release storage; needed for Tables IV and V. |
| `artifacts/forecasting/forecasting_benchmark_v2/predictions.parquet` | yes | no | Large file; prefer Git LFS, GitHub Release, or external archival storage. |
| `artifacts/forecasting/forecasting_benchmark_v2/run_manifest.json` | yes | no | Small/medium metadata; consider tracking in Git in a later stage. |
| `artifacts/forecasting/forecasting_benchmark_v2/split_metadata.parquet` | yes | no | Upload or track; needed for protocol validation. |
| `artifacts/meta_modeling/split_assignments_v2.csv` | yes | no | Upload or track; needed for split validation and Table VI. |
| `artifacts/meta_modeling/routing_rows_v2.parquet` | yes | no | Large file; prefer Git LFS, GitHub Release, or external archival storage. |
| `artifacts/meta_modeling/task_results_v2.parquet` | yes | no | Upload or track; needed for Table VI and Figures 2-3. |
| `artifacts/meta_modeling/model_order_mapping_v2.csv` | yes | no | Upload or track; useful for model-name validation. |
| `artifacts/reports.zip` | yes | no | Large untracked report bundle; decide whether it is required for paper archival before upload. |

## Large and Generated Reports

Large `.parquet` files and `artifacts/reports.zip` are present locally but not tracked. No large artifacts are added in this documentation stage. The next stage should decide whether each large file belongs in Git LFS, GitHub Release assets, or external storage.

Excel reports and generated paper report files: needs verification. No paper-specific `paper_icdm/tables/` or `paper_icdm/figures/` outputs are created in this stage.

