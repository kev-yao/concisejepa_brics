"""Supported imports for the BRICS fragment model family.

The implementations remain shared with the original spike while the research
pipeline is promoted into the standard Hydra runner. Keeping a single
implementation lets parity tests compare construction paths without allowing
the two copies to drift.
"""

from spikes.phase1.fragment_encoder import ConciseFragment, ConciseJEPAFragment
from spikes.phase1.fragment_pooling import (
    POOLING_NEEDS_CONTEXT,
    CrossAttentionPool,
    F2RPool,
    LatentQueryPool,
    MaxPool,
    MeanPool,
    MLPWeightedSumPool,
    MultiHeadWeightedSumPool,
    PreAttnLatentQueryPool,
    WeightedSumPool,
    build_pooling,
)
from spikes.phase1.fragment_xattn import ConciseJEPAFragmentXAttn, CrossAttnJEPAPredictor

__all__ = [
    "POOLING_NEEDS_CONTEXT",
    "ConciseFragment",
    "ConciseJEPAFragment",
    "ConciseJEPAFragmentXAttn",
    "CrossAttentionPool",
    "CrossAttnJEPAPredictor",
    "F2RPool",
    "LatentQueryPool",
    "MLPWeightedSumPool",
    "MaxPool",
    "MeanPool",
    "MultiHeadWeightedSumPool",
    "PreAttnLatentQueryPool",
    "WeightedSumPool",
    "build_pooling",
]
