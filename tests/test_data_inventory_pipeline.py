from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.config.loader import load_inventory_config
from src.data.inventory import (
    build_series_summary,
    run_data_inventory,
    standardize_series_dataframe,
)
from src.data.readers import read_ru_daily_file, read_us_daily_file


def _base_stage_config() -> dict:
    return {
        "run_name": "data_inventory_v1",
        "input": {"ru_pattern": "*_D1.csv", "us_pattern": "*.txt"},
        "schema": {
            "ru": {
                "date_col": "datetime",
                "open_col": "open",
                "high_col": "high",
                "low_col": "low",
                "close_col": "close",
                "volume_col": "volume",
            },
            "us": {
                "date_col": "Date",
                "open_col": "Open",
                "high_col": "High",
                "low_col": "Low",
                "close_col": "Close",
                "volume_col": "Volume",
            },
        },
        "standardization": {
            "output_columns": [
                "series_id",
                "ticker",
                "market",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            "market_values": {"ru": "RU", "us": "US"},
        },
        "series_policy": {"target_length": 2000, "min_length": 1500, "slice_mode": "last_n"},
        "quality_checks": {"require_positive_close": True, "sort_by_date": True},
    }


def _write_test_configs(configs_dir: Path, ru_dir: Path, us_dir: Path, art_dir: Path) -> Path:
    stage_cfg = _base_stage_config()
    (configs_dir / "data_inventory_v1.yaml").write_text(
        yaml.safe_dump(stage_cfg, sort_keys=False),
        encoding="utf-8",
    )

    paths_cfg = {
        "data_sources": {"ru_daily": str(ru_dir), "us_daily": str(us_dir)},
        "artifacts": {
            "interim": str(art_dir / "interim"),
            "processed": str(art_dir / "processed"),
            "features": str(art_dir / "features"),
            "reports": str(art_dir / "reports"),
            "logs": str(art_dir / "logs"),
            "manifests": str(art_dir / "manifests"),
        },
    }
    (configs_dir / "paths.local.yaml").write_text(
        yaml.safe_dump(paths_cfg, sort_keys=False),
        encoding="utf-8",
    )
    return configs_dir / "data_inventory_v1.yaml"


def test_read_ru_daily_file(tmp_path: Path) -> None:
    file_path = tmp_path / "GAZP_D1.csv"
    file_path.write_text(
        "datetime,open,high,low,close,volume\n2020-01-01,1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )

    df = read_ru_daily_file(file_path)

    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(df) == 1


def test_read_us_daily_file(tmp_path: Path) -> None:
    file_path = tmp_path / "AAPL.txt"
    file_path.write_text(
        "Date,Open,High,Low,Close,Volume,OpenInt\n2020-01-01,1,2,0.5,1.5,100,0\n",
        encoding="utf-8",
    )

    df = read_us_daily_file(file_path)

    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    assert len(df) == 1


def test_standardize_series_dataframe_columns() -> None:
    config = _base_stage_config()
    raw_df = pd.DataFrame(
        {
            "datetime": ["2020-01-02", "2020-01-01"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [100, 200],
        }
    )

    out = standardize_series_dataframe(raw_df, ticker="GAZP", market="ru", config=config)

    assert list(out.columns) == config["standardization"]["output_columns"]
    assert out.iloc[0]["date"] <= out.iloc[1]["date"]
    assert out.iloc[0]["series_id"] == "RU:GAZP"


def test_build_series_summary() -> None:
    config = _base_stage_config()
    df = pd.DataFrame(
        {
            "series_id": ["RU:GAZP", "RU:GAZP"],
            "ticker": ["GAZP", "GAZP"],
            "market": ["RU", "RU"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 150],
        }
    )
    df.attrs["n_rows_raw"] = 2

    summary = build_series_summary(df, "D:/data/GAZP_D1.csv", "GAZP", "RU", config)

    assert summary["n_rows_raw"] == 2
    assert summary["can_compute_log_returns"] is True
    assert summary["eligible_min_1500"] is False
    assert summary["status"] == "invalid"
    assert summary["status_reason"] == "too_short"


def test_load_inventory_config_reads_yaml(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    ru_dir = tmp_path / "data" / "ru"
    us_dir = tmp_path / "data" / "us"
    art_dir = tmp_path / "artifacts"

    configs_dir.mkdir(parents=True)
    ru_dir.mkdir(parents=True)
    us_dir.mkdir(parents=True)

    config_path = _write_test_configs(configs_dir, ru_dir, us_dir, art_dir)

    cfg = load_inventory_config(str(config_path))

    assert cfg["run_name"] == "data_inventory_v1"
    assert cfg["series_policy"]["slice_mode"] == "last_n"
    assert cfg["quality_checks"]["require_positive_close"] is True
    assert "config_hash" in cfg["meta"]


def test_run_data_inventory_small_dataset(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    configs_dir = root / "configs"
    ru_dir = root / "data" / "ru"
    us_dir = root / "data" / "us"
    art_dir = root / "artifacts"

    configs_dir.mkdir(parents=True)
    ru_dir.mkdir(parents=True)
    us_dir.mkdir(parents=True)

    (ru_dir / "GAZP_D1.csv").write_text(
        "datetime,open,high,low,close,volume\n2020-01-01,1,2,0.5,1.5,100\n"
        "2020-01-02,1.5,2.2,1.1,2.0,120\n",
        encoding="utf-8",
    )
    (us_dir / "AAPL.txt").write_text(
        "Date,Open,High,Low,Close,Volume,OpenInt\n2020-01-01,10,11,9,10.5,1000,0\n"
        "2020-01-02,10.5,11.5,10,11,900,0\n",
        encoding="utf-8",
    )

    config_path = _write_test_configs(configs_dir, ru_dir, us_dir, art_dir)

    def _fake_to_parquet(self, path, index=False):  # noqa: ANN001
        Path(path).write_text("parquet_stub", encoding="utf-8")

    def _fake_excel_export(catalog_df, excel_path, run_metadata, source_paths):  # noqa: ANN001
        p = Path(excel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("excel_stub", encoding="utf-8")
        return p

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)
    monkeypatch.setattr("src.data.inventory.export_series_catalog_excel", _fake_excel_export, raising=True)

    result = run_data_inventory(str(config_path))

    assert result.total_files == 2
    assert Path(result.parquet_path).exists()
    assert Path(result.excel_path).exists()
    assert Path(result.log_path).exists()
    assert Path(result.manifest_path).exists()
