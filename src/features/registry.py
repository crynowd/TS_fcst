from __future__ import annotations

from typing import Callable, Dict

from src.features.block_a_dependence import FeatureBlockAOutputs, run_feature_block_a_pipeline


FeatureRunner = Callable[[str], FeatureBlockAOutputs]

FEATURE_BLOCK_REGISTRY: Dict[str, FeatureRunner] = {
    "A": run_feature_block_a_pipeline,
}


def run_feature_block(block: str, config_path: str) -> FeatureBlockAOutputs:
    normalized = block.strip().upper()
    if normalized not in FEATURE_BLOCK_REGISTRY:
        supported = ", ".join(sorted(FEATURE_BLOCK_REGISTRY))
        raise ValueError(f"Unsupported feature block: {block}. Supported blocks: {supported}")
    return FEATURE_BLOCK_REGISTRY[normalized](config_path)

