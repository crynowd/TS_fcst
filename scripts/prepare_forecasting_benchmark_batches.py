from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config.loader import load_forecasting_benchmark_config
from src.forecasting.data import load_log_returns_table, select_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic, resumable forecasting benchmark batches.")
    parser.add_argument("--config", default="configs/forecasting_benchmark_v2.yaml")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--run-id", default="forecasting_benchmark_v2_clean_batched")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    stage_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_forecasting_benchmark_config(str(config_path))
    selected = select_series(
        log_returns_df=load_log_returns_table(cfg["data"]["source_path"]),
        dataset_profile=str(cfg["data"]["dataset_profile"]),
        max_series=int(cfg["data"]["max_series"]),
        series_selection_mode=str(cfg["data"]["series_selection_mode"]),
        series_ids=None,
    )
    series_ids = [str(row.series_id) for row in selected]
    if len(series_ids) != len(set(series_ids)):
        raise ValueError("Selected series_ids are not unique")

    batch_size = int(args.batch_size)
    # Keep the first three batches near the requested size and put the complete
    # deterministic remainder into Batch 4.
    batches = [series_ids[i * batch_size : (i + 1) * batch_size] for i in range(3)]
    batches.append(series_ids[3 * batch_size :])
    batches = [batch for batch in batches if batch]
    if sorted(item for batch in batches for item in batch) != sorted(series_ids):
        raise AssertionError("Batch union does not equal selected series")

    clean_dir = Path("artifacts/forecasting") / args.run_id
    batch_manifest_dir = clean_dir / "batch_manifests"
    manifest = {
        "batch_plan_version": "forecasting_benchmark_v2_clean_batched_v1",
        "base_config": str(config_path),
        "run_id": args.run_id,
        "temporal_split_policy": "target_end_lte_right_origin_v1",
        "selection": {
            "dataset_profile": cfg["data"]["dataset_profile"],
            "series_selection_mode": cfg["data"]["series_selection_mode"],
            "max_series": cfg["data"]["max_series"],
            "ordered_series_count": len(series_ids),
        },
        "batches": [
            {"batch_id": index, "n_series": len(batch), "series_ids": batch}
            for index, batch in enumerate(batches, start=1)
        ],
        "validation": {
            "total_unique_series": len(set(series_ids)),
            "overlap_count": 0,
            "missing_count": 0,
        },
    }
    manifest_path = Path("artifacts/manifests") / f"{args.run_id}_series_batches.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    for index, batch in enumerate(batches, start=1):
        batch_cfg = yaml.safe_load(yaml.safe_dump(stage_cfg, sort_keys=False))
        batch_cfg.setdefault("filters", {})["series_ids"] = batch
        batch_cfg["resume"] = True
        batch_cfg["outputs"] = {
            **dict(batch_cfg.get("outputs", {})),
            "run_name": args.run_id,
            "run_id": args.run_id,
            "output_dir": str(clean_dir),
            "report_dir": "artifacts/reports/forecasting_audit_v2_clean_batched",
            "run_manifest_path": str(batch_manifest_dir / f"batch_{index:02d}_run_manifest.json"),
            "config_snapshot_path": str(batch_manifest_dir / f"batch_{index:02d}_config_snapshot.yaml"),
        }
        batch_cfg["batch_metadata"] = {
            "batch_plan_path": str(manifest_path),
            "batch_id": index,
            "n_series": len(batch),
        }
        batch_path = config_path.parent / f"{config_path.stem}_clean_batched_batch_{index:02d}.yaml"
        batch_path.write_text(yaml.safe_dump(batch_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"batch_{index:02d} series={len(batch)} config={batch_path}")
    print(f"manifest={manifest_path} total_series={len(series_ids)} batches={len(batches)}")


if __name__ == "__main__":
    main()
