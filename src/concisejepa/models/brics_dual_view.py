"""Dual-view BRICS model with shared fragment/whole-molecule quantization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from esm.model.esm2 import TransformerLayer

from .concise import activation_choices
from .drug_decoder import DrugEncoder


class MaskedSetTransformer(nn.Module):
    """Permutation-invariant fragment aggregation with a learned summary token."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"Set-transformer dim={dim} must be divisible by heads={heads}.")
        self.summary_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.summary_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self, fragment_vectors: torch.Tensor, fragment_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fragment_vectors.ndim != 3:
            raise ValueError("fragment_vectors must have shape [batch, fragments, dim].")
        if fragment_mask.shape != fragment_vectors.shape[:2]:
            raise ValueError("fragment_mask must match fragment_vectors' first two dimensions.")
        if not bool(fragment_mask.any(dim=1).all()):
            raise ValueError("Every molecule must contain at least one valid fragment.")

        batch_size = fragment_vectors.shape[0]
        summary = self.summary_token.expand(batch_size, -1, -1)
        tokens = torch.cat([summary, fragment_vectors], dim=1)
        summary_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=fragment_mask.device)
        valid_tokens = torch.cat([summary_valid, fragment_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid_tokens)
        encoded = self.output_norm(encoded)
        return encoded[:, 0], encoded[:, 1:]


class BricsDualViewJEPA(nn.Module):
    """Learn fragment composition and DTI geometry with one shared DrugEncoder/FSQ."""

    def __init__(
        self,
        drug_layers,
        ligand_dim: int = 2048,
        residue_dim: int = 1280,
        drug_dim: int = 128,
        dti_dim: int = 256,
        protein_heads: int = 32,
        encoder_activation: str = "tanh",
        drug_quantizer=None,
        fragment_transformer_depth: int = 2,
        fragment_transformer_heads: int = 8,
        fragment_transformer_mlp_ratio: float = 4.0,
        fragment_transformer_dropout: float = 0.1,
        coati_target_dim: int = 256,
        coati_hidden_dim: int = 512,
        molecular_representation: str = "fsq",
    ) -> None:
        super().__init__()
        if encoder_activation not in activation_choices:
            raise ValueError(
                f"Unknown encoder activation {encoder_activation!r}; "
                f"expected one of {sorted(activation_choices)}."
            )
        if dti_dim % protein_heads != 0:
            raise ValueError(f"DTI dim={dti_dim} must be divisible by protein_heads={protein_heads}.")

        self.drug_dim = drug_dim
        self.dti_dim = dti_dim
        if molecular_representation not in ("fsq", "continuous"):
            raise ValueError("molecular_representation must be fsq or continuous")
        self.molecular_representation = molecular_representation
        self.drug_encoder = DrugEncoder(
            layers=drug_layers,
            dim=ligand_dim,
            latent_dim=drug_dim,
            activation=activation_choices[encoder_activation],
            quantizer=drug_quantizer,
        )
        self.fragment_transformer = MaskedSetTransformer(
            dim=drug_dim,
            depth=fragment_transformer_depth,
            heads=fragment_transformer_heads,
            mlp_ratio=fragment_transformer_mlp_ratio,
            dropout=fragment_transformer_dropout,
        )

        # Both molecular views use exactly the same projection into the DTI space.
        self.drug_dti_projector = nn.Sequential(
            nn.Linear(drug_dim, dti_dim),
            nn.ReLU(),
        )
        self.protein_project = nn.Linear(residue_dim, dti_dim)
        self.protein_self_attention = TransformerLayer(
            embed_dim=dti_dim,
            ffn_embed_dim=dti_dim,
            attention_heads=protein_heads,
            use_rotary_embeddings=True,
        )
        self.coati_predictor = nn.Sequential(
            nn.Linear(drug_dim, coati_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(coati_hidden_dim, coati_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(coati_hidden_dim, coati_target_dim),
        )

    @staticmethod
    def _positive_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Cosine of nonnegative vectors, interpreted directly as a probability."""
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        return (left * right).sum(dim=-1).clamp(min=0.0, max=1.0)

    @staticmethod
    def _pool_protein(residue_vectors: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(10 * residue_vectors, dim=1)
        return (residue_vectors * weights).sum(dim=1)

    def _encode_fragments(
        self, fragment_fingerprints: torch.Tensor, fragment_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, fragment_count, fingerprint_dim = fragment_fingerprints.shape
        flattened = fragment_fingerprints.reshape(batch_size * fragment_count, fingerprint_dim)
        encoded = self._encode_molecule(flattened)
        quantized = encoded["quantized"].squeeze(1).reshape(
            batch_size, fragment_count, self.drug_dim
        )
        codes = encoded["codes"].reshape(batch_size, fragment_count, -1)
        pooled, contextualized = self.fragment_transformer(quantized, fragment_mask)
        return pooled, contextualized, quantized, codes

    def _encode_molecule(self, fingerprints: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.molecular_representation == "fsq":
            return self.drug_encoder(fingerprints)
        # Ablate the entire FSQ bottleneck (including its 3-D projection).
        # Keep the shared encoder and bounded 128-D downstream interface.
        pre = self.drug_encoder.pre_transform(fingerprints)
        continuous = torch.tanh(pre)
        return {
            "quantized": continuous,
            "pre_quantized": pre,
            "codes": torch.full(
                (fingerprints.shape[0], 1), -1, dtype=torch.long, device=fingerprints.device
            ),
        }

    def _encode_protein(self, protein_embedding: torch.Tensor) -> torch.Tensor:
        residue_vectors = self.protein_project(protein_embedding)
        mixed, _ = self.protein_self_attention(rearrange(residue_vectors, "b r d -> r b d"))
        residue_vectors = residue_vectors + rearrange(mixed, "r b d -> b r d")
        return F.relu(self._pool_protein(residue_vectors))

    def forward(
        self,
        protein_embedding: torch.Tensor,
        fragment_fingerprints: torch.Tensor,
        fragment_mask: torch.Tensor,
        whole_molecule_fingerprint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fragment_set_vector, contextualized_fragments, fragment_quantized, fragment_codes = (
            self._encode_fragments(fragment_fingerprints, fragment_mask)
        )
        whole_encoded = self._encode_molecule(whole_molecule_fingerprint)
        whole_molecule_vector = whole_encoded["quantized"].squeeze(1)

        # ReLU immediately before cosine makes every coordinate nonnegative, so
        # each resulting cosine lies in [0, 1] and can be used as a BCE probability.
        fragment_dti_vector = self.drug_dti_projector(fragment_set_vector)
        whole_dti_vector = self.drug_dti_projector(whole_molecule_vector)
        receptor_dti_vector = self._encode_protein(protein_embedding)

        fragment_binding = self._positive_cosine(fragment_dti_vector, receptor_dti_vector)
        whole_binding = self._positive_cosine(whole_dti_vector, receptor_dti_vector)
        binding = 0.5 * (fragment_binding + whole_binding)
        fragment_molecule_similarity = self._positive_cosine(
            fragment_dti_vector, whole_dti_vector
        )

        return {
            "binding": binding,
            "fragment_binding": fragment_binding,
            "whole_binding": whole_binding,
            "fragment_molecule_similarity": fragment_molecule_similarity,
            "jepa_pred": self.coati_predictor(fragment_set_vector),
            "fragment_set_vector": fragment_set_vector,
            "whole_molecule_vector": whole_molecule_vector,
            "fragment_dti_vector": fragment_dti_vector,
            "whole_dti_vector": whole_dti_vector,
            "receptor_dti_vector": receptor_dti_vector,
            "contextualized_fragments": contextualized_fragments,
            "fragment_quantized": fragment_quantized,
            "fragment_codes": fragment_codes,
            "fragment_mask": fragment_mask,
            "whole_codes": whole_encoded["codes"],
            "whole_pre_quantized": whole_encoded["pre_quantized"],
            "whole_quantized": whole_encoded["quantized"],
        }


__all__ = ["BricsDualViewJEPA", "MaskedSetTransformer"]
