"""A typed Set Transformer readout for frozen fragment/protein pair features.

The four roles are drug, protein, elementwise product, and absolute difference.
These are derived feature tokens, not individual BRICS fragments or whole-molecule
inputs. This module owns neither backbone encoding nor secondary-score fusion.
"""

import math

import torch
from torch import nn


class _MultiheadAttentionBlock(nn.Module):
    """Residual multihead attention and row-wise feed-forward transformation."""

    def __init__(self, dim, heads, hidden_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, query, values, valid):
        attended, _ = self.attention(
            query, values, values, key_padding_mask=~valid, need_weights=False
        )
        mixed = self.attention_norm(query + self.attention_dropout(attended))
        return self.output_norm(mixed + self.feed_forward(mixed))


class SetTransformerBindingHead(nn.Module):
    """Typed SAB encoder and single-seed PMA producing primary binding logits.

    ``tokens`` has shape [B,F,feature_dim]. Type IDs 0..3 identify drug, protein,
    product and absolute-difference roles. With F=4, omitted types default to
    that order. Explicit types have shape [F] or [B,F]. ``mask`` is boolean
    [B,F], true for valid tokens. Jointly permuting tokens/types/mask preserves
    evaluation outputs; there are no positional encodings.

    Input mean/scale buffers [4,feature_dim] are supplied by the caller using
    TRAIN-only statistics. They are indexed by type, not by set position.
    This class never fits statistics, reads data, or encodes whole molecules.
    """

    def __init__(
        self,
        feature_dim=256,
        model_dim=64,
        num_heads=4,
        num_blocks=2,
        dropout=0.1,
        ff_multiplier=2,
    ):
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim), ("model_dim", model_dim),
            ("num_heads", num_heads), ("num_blocks", num_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        dropout = float(dropout)
        ff_multiplier = float(ff_multiplier)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        if not math.isfinite(ff_multiplier) or ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be finite and positive")
        hidden_width = model_dim * ff_multiplier
        if not math.isfinite(hidden_width) or hidden_width < 1:
            raise ValueError("ff_multiplier must produce a finite positive hidden width")
        hidden_dim = int(hidden_width)

        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.register_buffer("input_mean", torch.zeros(4, feature_dim))
        self.register_buffer("input_scale", torch.ones(4, feature_dim))
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.type_embedding = nn.Embedding(4, model_dim)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        self.self_attention_blocks = nn.ModuleList([
            _MultiheadAttentionBlock(model_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.pool_seed = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.pool_seed, std=0.02)
        self.pooling_attention = _MultiheadAttentionBlock(
            model_dim, num_heads, hidden_dim, dropout
        )
        self.readout = nn.Sequential(
            nn.Linear(model_dim, max(1, model_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, model_dim // 2), 1),
        )

    @torch.no_grad()
    def set_input_stats(self, mean, scale):
        """Copy caller-provided TRAIN statistics into checkpointed buffers."""
        mean = torch.as_tensor(mean, device=self.input_mean.device, dtype=self.input_mean.dtype).detach()
        scale = torch.as_tensor(scale, device=self.input_scale.device, dtype=self.input_scale.dtype).detach()
        if mean.shape != self.input_mean.shape or scale.shape != self.input_scale.shape:
            raise ValueError(f"Input statistics must have shape [4,{self.feature_dim}]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("Input statistics must be finite")
        if not bool((scale > 0).all()):
            raise ValueError("Input scales must be strictly positive")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale)

    def forward(self, tokens, token_types=None, mask=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim:
            raise ValueError(f"tokens must have shape [B,F,{self.feature_dim}]")
        batch_size, count, _ = tokens.shape
        if batch_size == 0 or count == 0:
            raise ValueError("tokens must contain nonempty batches and sets")
        if not tokens.is_floating_point():
            raise ValueError("tokens must be floating-point features")
        if token_types is None:
            if count != 4:
                raise ValueError("Explicit token_types are required unless F=4")
            token_types = torch.arange(4, device=tokens.device)
        if token_types.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("token_types must be integer IDs in [0,3]")
        if token_types.shape == (count,):
            token_types = token_types.unsqueeze(0).expand(batch_size, -1)
        elif token_types.shape != (batch_size, count):
            raise ValueError("token_types must have shape [F] or [B,F]")
        if token_types.device != tokens.device:
            raise ValueError("token_types and tokens must be on the same device")
        token_types = token_types.long()
        if not bool(((token_types >= 0) & (token_types < 4)).all()):
            raise ValueError("token_types must be in [0,3], including padded positions")
        if mask is None:
            mask = torch.ones(batch_size, count, dtype=torch.bool, device=tokens.device)
        if mask.shape != (batch_size, count) or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B,F]")
        if mask.device != tokens.device:
            raise ValueError("mask and tokens must be on the same device")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every example must contain at least one valid token")

        # Remove even NaN/Inf padding before arithmetic/projection. Masked tokens
        # stay excluded as keys in every SAB and in the learned-seed PMA.
        clean = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Valid token features must be finite")
        normalized = (clean - self.input_mean[token_types]) / self.input_scale[token_types]
        normalized = normalized.masked_fill(~mask.unsqueeze(-1), 0)
        encoded = self.input_projection(normalized) + self.type_embedding(token_types)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        for block in self.self_attention_blocks:
            encoded = block(encoded, encoded, mask)
            encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        seed = self.pool_seed.expand(batch_size, -1, -1)
        pooled = self.pooling_attention(seed, encoded, mask).squeeze(1)
        logits = self.readout(pooled).squeeze(-1)
        return {"binding_logits": logits, "binding": logits.sigmoid(), "pooled": pooled}


__all__ = ["SetTransformerBindingHead"]
