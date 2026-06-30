# phase1_fragment_pooling.py — 5 pooling strategies for fragment-level FSQ comparison study.
#
# All poolers have the same interface:
#   forward(frag_embs [B, F, D], mask [B, F] bool, context=None) -> [B, D]
# `mask` is True for valid (non-padding) fragments.
# `context` is used only by CrossAttentionPool (protein embedding [B, 50, residue_dim]).

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    """Masked mean over valid fragment embeddings."""

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # [B, F, 1]
        return (frag_embs * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)  # [B, D]


class MaxPool(nn.Module):
    """Masked max per channel over valid fragment embeddings."""

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        masked = frag_embs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        result = masked.max(dim=1).values  # [B, D]
        # If a whole row is invalid (mask all False), replace -inf with 0
        all_invalid = ~mask.any(dim=1, keepdim=True)  # [B, 1]
        return result.masked_fill(all_invalid, 0.0)


class WeightedSumPool(nn.Module):
    """
    Learned scalar attention: score(frag_emb) → scalar weight per fragment.
    Equivalent to single-head attention with a learned query but no separate Q matrix.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        weights = self.score(frag_embs).squeeze(-1)  # [B, F]
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # [B, F]
        weights = torch.nan_to_num(weights, nan=0.0)  # guard against all-invalid rows
        return (frag_embs * weights.unsqueeze(-1)).sum(1)  # [B, D]


class CrossAttentionPool(nn.Module):
    """
    Protein-conditioned pooling: protein embedding (mean-pooled, projected) as query,
    fragment embeddings as keys/values.  The protein "selects" which fragments matter.

    context: protein embedding [B, 50, residue_dim] (raw Raygun output).
    """

    def __init__(self, dim: int, residue_dim: int = 1280, nheads: int = 4):
        super().__init__()
        self.protein_proj = nn.Linear(residue_dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, nheads, batch_first=True)

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # context: [B, 50, residue_dim]
        q = self.protein_proj(context).mean(dim=1, keepdim=True)  # [B, 1, dim]
        key_padding_mask = ~mask  # [B, F]; True = ignore (PyTorch convention)
        out, _ = self.cross_attn(q, frag_embs, frag_embs, key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # [B, dim]


class LatentQueryPool(nn.Module):
    """
    DeepSeek MLA-inspired: N learned query vectors cross-attend over fragment embeddings,
    then outputs are concatenated and projected to dim.

    With n_queries=1 this degenerates to standard attention pooling with a learned query.
    With n_queries>1 the model can extract multiple "aspects" from the fragment set.
    """

    def __init__(self, dim: int, n_queries: int = 4, nheads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.out_proj = nn.Linear(n_queries * dim, dim)

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        B = frag_embs.shape[0]
        q = self.queries.expand(B, -1, -1)  # [B, n_queries, dim]
        key_padding_mask = ~mask  # [B, F]
        out, _ = self.cross_attn(q, frag_embs, frag_embs, key_padding_mask=key_padding_mask)
        return self.out_proj(out.reshape(B, -1))  # [B, dim]


def build_pooling(name: str, dim: int, residue_dim: int = 1280) -> nn.Module:
    registry = {
        "mean": lambda: MeanPool(),
        "max": lambda: MaxPool(),
        "weighted_sum": lambda: WeightedSumPool(dim),
        "cross_attention": lambda: CrossAttentionPool(dim, residue_dim=residue_dim),
        "latent_query": lambda: LatentQueryPool(dim),
    }
    if name not in registry:
        raise ValueError(f"Unknown pooling '{name}'. Available: {sorted(registry)}")
    return registry[name]()


POOLING_NEEDS_CONTEXT = {"cross_attention"}
