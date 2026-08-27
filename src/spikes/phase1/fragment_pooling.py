"""fragment_pooling.py — every pooling strategy tested for combining per-fragment embeddings
into one molecule embedding. This is the main axis of comparison throughout the whole project
(see docs/PROJECT_HANDOFF.md §2, §6).

All poolers share one interface:
    forward(frag_embs [B, F, D], mask [B, F] bool, context=None) -> [B, D]
`mask` is True for valid (non-padding) fragments. `context` (protein residue embeddings,
[B, 50, residue_dim]) is only used by poolers in POOLING_NEEDS_CONTEXT below (CrossAttentionPool,
F2RPool) — everything else pools fragments without seeing the protein at all.

The two poolers that matter most for this project's headline findings:
    F2RPool — best for ATTRIBUTION (median ratio 0.802; correctly identifies which fragment
        drives binding for a given protein). Each fragment queries the protein's residues to get
        a protein-specific importance score, but the fragment CONTENT that gets pooled is never
        touched by the protein — only the mixing weight is.
    MaxPool — best for RECONSTRUCTION (0.389 absolute Tanimoto similarity, well ahead of every
        other pooler). Structurally cannot blend fragments together the way every attention-style
        pooler here can (each output channel comes from exactly one fragment's winning value) —
        this seems to matter more for generative fidelity than any other single choice tested.

Everything else (MeanPool, WeightedSumPool, CrossAttentionPool, LatentQueryPool and its variants,
MLPWeightedSumPool, MultiHeadWeightedSumPool) is retained from the earlier Phase 1/Phase 2
pooling-ablation sweep (see docs/history/plan-phase1.md, plan-phase2.md) — kept working and in
the registry below, but not the focus of the most recent work.

Constructed via build_pooling(name, dim, residue_dim) — called from fragment_encoder.py's
ConciseFragment.__init__ using whichever name is passed to --pooling in fragment_train.py.
last_weights (set by every pooler except MeanPool/CrossAttentionPool/LatentQueryPool) is read by
the attribution scripts (attribute_fragments.py, e3_f2r_attribution.py, e4_realprotein_attribution.py)
to measure how much weight lands on a specific fragment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    """Masked mean over valid fragment embeddings."""

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # [B, F, 1]
        return (frag_embs * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)  # [B, D]


class MaxPool(nn.Module):
    """Masked max per channel over valid fragment embeddings.

    Attribution proxy: max-pooling has no softmax weight, but each output channel is
    supplied by exactly one fragment (whichever has the largest value in that channel).
    last_weights = the fraction of channels each fragment "wins" via argmax (sums to 1
    over valid fragments) — the max-pool analog of an attention weight, read by the same
    attribution pipeline (attribute_molecule) that reads WeightedSumPool/F2RPool weights.
    """

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        masked = frag_embs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        result, argmax = masked.max(dim=1)  # result: [B, D], argmax: [B, D] winning fragment per channel
        all_invalid = ~mask.any(dim=1, keepdim=True)  # [B, 1]
        result = result.masked_fill(all_invalid, 0.0)

        B, Fn = mask.shape
        counts = torch.zeros(B, Fn, device=frag_embs.device, dtype=frag_embs.dtype)
        counts.scatter_add_(1, argmax, torch.ones_like(argmax, dtype=frag_embs.dtype))
        weights = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
        self.last_weights = weights.detach()

        return result


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
        self.last_weights = weights.detach()  # saved for attribution extraction
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


class PreAttnLatentQueryPool(nn.Module):
    """
    One round of fragment self-attention (fragments see each other) then latent query pooling.
    Lets fragment importance be context-dependent before the latent queries aggregate.
    """

    def __init__(self, dim: int, n_queries: int = 4, nheads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.latent = LatentQueryPool(dim, n_queries=n_queries, nheads=nheads)

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        attended, _ = self.self_attn(
            frag_embs, frag_embs, frag_embs, key_padding_mask=~mask
        )
        frag_embs = self.norm(frag_embs + attended)  # residual + norm
        return self.latent(frag_embs, mask, context)


class MLPWeightedSumPool(nn.Module):
    """
    2-layer MLP scorer for fragment importance — more expressive than a single linear layer.
    Can capture nonlinear importance signals (e.g. presence of a pharmacophore pattern).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        weights = self.scorer(frag_embs).squeeze(-1)  # [B, F]
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.last_weights = weights.detach()  # saved for attribution extraction
        return (frag_embs * weights.unsqueeze(-1)).sum(1)  # [B, D]


class MultiHeadWeightedSumPool(nn.Module):
    """
    k independent scalar scorers each produce a weighted sum; outputs are concatenated
    and projected back to dim.  Intermediate between WeightedSumPool and LatentQueryPool.
    """

    def __init__(self, dim: int, k: int = 2):
        super().__init__()
        self.scorers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(k)])
        self.out_proj = nn.Linear(k * dim, dim)

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context=None) -> torch.Tensor:
        heads = []
        for scorer in self.scorers:
            w = scorer(frag_embs).squeeze(-1)  # [B, F]
            w = w.masked_fill(~mask, float("-inf"))
            w = F.softmax(w, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
            heads.append((frag_embs * w.unsqueeze(-1)).sum(1))  # [B, D]
        return self.out_proj(torch.cat(heads, dim=-1))  # [B, dim]


class F2RPool(nn.Module):
    """Fragment-to-Residue attention pooler.

    Each fragment queries protein residues to compute a protein-conditional importance
    score. Softmax of these scores becomes the pooling weights, making attribution
    protein-specific: the same fragment receives different weights for different proteins.

    Unlike WeightedSumPool (weights depend only on the fragment's own embedding),
    F2R weights capture "how much does this fragment interact with this protein?"

    context: protein residue embeddings [B, R, residue_dim] (raw ESM output).
    """

    def __init__(self, dim: int, residue_dim: int = 1280, nheads: int = 4):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(residue_dim, dim)
        self.last_weights = None

    def forward(self, frag_embs: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # frag_embs: [B, Nf, D], mask: [B, Nf], context: [B, R, residue_dim]
        B, Nf, D = frag_embs.shape
        R = context.shape[1]
        H, Dh = self.nheads, self.head_dim

        Q = self.q_proj(frag_embs).view(B, Nf, H, Dh)   # [B, Nf, H, Dh]
        K = self.k_proj(context).view(B, R, H, Dh)       # [B, R,  H, Dh]

        # [B, Nf, H, R]: per-fragment, per-head attention over residues
        scores = torch.einsum('bfhd,brhd->bfhr', Q, K) * self.scale
        # collapse residues then heads → per-fragment scalar [B, Nf]
        scores = scores.mean(dim=-1).mean(dim=-1)

        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.last_weights = weights.detach()

        return (frag_embs * weights.unsqueeze(-1)).sum(1)  # [B, D]


def build_pooling(name: str, dim: int, residue_dim: int = 1280) -> nn.Module:
    registry = {
        # Phase 1 (original)
        "mean":                       lambda: MeanPool(),
        "max":                        lambda: MaxPool(),
        "whole_mol":                  lambda: MeanPool(),  # single-fragment; pooler is trivial
        "weighted_sum":               lambda: WeightedSumPool(dim),
        "cross_attention":            lambda: CrossAttentionPool(dim, residue_dim=residue_dim),
        "latent_query":               lambda: LatentQueryPool(dim, n_queries=4),
        # Phase 2 — latent_query n_queries sweep
        "latent_query_q1":            lambda: LatentQueryPool(dim, n_queries=1),
        "latent_query_q2":            lambda: LatentQueryPool(dim, n_queries=2),
        "latent_query_q8":            lambda: LatentQueryPool(dim, n_queries=8),
        # Phase 2 — pre-attention variant
        "pre_attn_latent_query":      lambda: PreAttnLatentQueryPool(dim, n_queries=4),
        # Phase 2 — weighted_sum variants
        "mlp_weighted_sum":           lambda: MLPWeightedSumPool(dim),
        "multi_head_weighted_sum_k2": lambda: MultiHeadWeightedSumPool(dim, k=2),
        "multi_head_weighted_sum_k4": lambda: MultiHeadWeightedSumPool(dim, k=4),
        # Phase 2 — protein-conditional attribution
        "f2r":                        lambda: F2RPool(dim, residue_dim=residue_dim),
    }
    if name not in registry:
        raise ValueError(f"Unknown pooling '{name}'. Available: {sorted(registry)}")
    return registry[name]()


POOLING_NEEDS_CONTEXT = {"cross_attention", "f2r"}
