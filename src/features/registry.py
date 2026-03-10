from __future__ import annotations

from typing import Any, Callable, Dict

from src.features.block_a_dependence import FeatureBlockAOutputs, run_feature_block_a_pipeline
from src.features.block_b_spectrum import FeatureBlockBOutputs, run_feature_block_b_pipeline
from src.features.block_c_tails import FeatureBlockCOutputs, run_feature_block_c_pipeline


FeatureRunner = Callable[[str], Any]

FEATURE_BLOCK_REGISTRY: Dict[str, FeatureRunner] = {
    "A": run_feature_block_a_pipeline,
    "B": run_feature_block_b_pipeline,
    "C": run_feature_block_c_pipeline,
}


def run_feature_block(block: str, config_path: str) -> FeatureBlockAOutputs | FeatureBlockBOutputs | FeatureBlockCOutputs:
    normalized = block.strip().upper()
    if normalized not in FEATURE_BLOCK_REGISTRY:
        supported = ", ".join(sorted(FEATURE_BLOCK_REGISTRY))
        raise ValueError(f"Unsupported feature block: {block}. Supported blocks: {supported}")
    return FEATURE_BLOCK_REGISTRY[normalized](config_path)
