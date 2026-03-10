from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


CANONICAL_COLUMNS: List[str] = [
    "series_id",
    "ticker",
    "market",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass(frozen=True)
class SourceSchema:
    """Column mapping for one source market."""

    date_col: str
    open_col: str
    high_col: str
    low_col: str
    close_col: str
    volume_col: str



def get_source_schema(config: dict, market: str) -> SourceSchema:
    """Build source schema object from config for the given market.

    Args:
        config: Loaded inventory config.
        market: `ru` or `us` (case-insensitive).

    Returns:
        Parsed source schema.
    """
    schema_cfg = config.get("schema", {})
    market_key = market.lower()
    market_schema = schema_cfg.get(market_key, {})

    return SourceSchema(
        date_col=market_schema.get("date_col", "date"),
        open_col=market_schema.get("open_col", "open"),
        high_col=market_schema.get("high_col", "high"),
        low_col=market_schema.get("low_col", "low"),
        close_col=market_schema.get("close_col", "close"),
        volume_col=market_schema.get("volume_col", "volume"),
    )


def get_market_label(config: dict, market: str) -> str:
    """Return canonical market label from config market_values mapping."""
    values: Dict[str, str] = config.get("standardization", {}).get("market_values", {})
    return values.get(market.lower(), market.upper())
