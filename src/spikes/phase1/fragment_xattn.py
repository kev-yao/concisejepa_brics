"""fragment_xattn.py — cross-attention JEPA predictor, adapted from a labmate's better-
performing DTI-JEPA script (see docs/history/solution3-flatcb-xattn-results.md for the full
comparison + results: 2-3x reconstruction lift over the MLP predictor).

The labmate's design: don't pool the drug down to one token before predicting; keep multiple
drug tokens and a real cross-modal transformer (drug tokens + protein tokens, bidirectional
self-attention) predicts the target embedding, only pooling at the very end. Their version uses
a 3D surface protein encoder (PRECISE/sPRECISE) we don't have; this adapts the same idea to our
sequence-based protein embeddings and BRICS fragment tokens instead of their fixed 3 CoNCISE
codes.

Deliberately a drop-in replacement for ConciseJEPAFragment's jepa_predictor: the DTI/pooling
backbone (ConciseFragment) is reused UNCHANGED, so attribution/AUPRC results stay comparable
across pooling methods. Only the "how do we predict the JEPA target" step changes. Output dict
shape matches ConciseJEPAFragment exactly, so LitFragment (including Solution 1's chem/group
heads and Solution 3's contrastive loss) works with this model with no changes at all.

Note: this result was descoped from the project's presentation deck for scope/simplicity, not
because it was wrong — see docs/PROJECT_HANDOFF.md §8 if picking this back up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fragment_encoder import ConciseFragment


class CrossAttnJEPAPredictor(nn.Module):
    """
    Cross-modal transformer over UNPOOLED drug-fragment tokens + protein residue tokens.

    Input:
      frag_embs:         [B, F, drug_dim]   unpooled, per-fragment FSQ embeddings
      frag_mask:         [B, F] bool         True = valid fragment
      protein_embedding: [B, R, residue_dim] raw protein residue tokens (Raygun, R=50)
    Output:
      jepa_pred: [B, target_dim]
    """

    def __init__(
        self,
        drug_dim: int,
        residue_dim: int,
        target_dim: int = 256,
        predictor_dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, predictor_dim)
        self.protein_proj = nn.Linear(residue_dim, predictor_dim)
        # modality embedding: 0 = protein token, 1 = drug-fragment token
        self.modality_embed = nn.Embedding(2, predictor_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — more stable to train than post-norm
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(predictor_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )

    def forward(
        self,
        frag_embs: torch.Tensor,
        frag_mask: torch.Tensor,
        protein_embedding: torch.Tensor,
    ) -> torch.Tensor:
        B, Fn, _ = frag_embs.shape
        R = protein_embedding.shape[1]
        device = frag_embs.device

        drug_tok = self.drug_proj(frag_embs) + self.modality_embed(
            torch.ones(B, Fn, dtype=torch.long, device=device)
        )
        prot_tok = self.protein_proj(protein_embedding) + self.modality_embed(
            torch.zeros(B, R, dtype=torch.long, device=device)
        )

        x = torch.cat([prot_tok, drug_tok], dim=1)  # [B, R+F, D]
        protein_valid = torch.ones(B, R, dtype=torch.bool, device=device)
        valid = torch.cat([protein_valid, frag_mask], dim=1)
        key_padding_mask = ~valid  # PyTorch convention: True = ignore

        x = self.blocks(x, src_key_padding_mask=key_padding_mask)
        x = self.out_norm(x)

        drug_x = x[:, R:, :]  # [B, F, D] — attended drug-fragment tokens
        mask_f = frag_mask.unsqueeze(-1).float()
        pooled = (drug_x * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)  # masked mean pool

        return self.output_proj(pooled)


class ConciseJEPAFragmentXAttn(nn.Module):
    """
    Drop-in alternative to ConciseJEPAFragment. Same output dict keys/shapes, so
    LitFragment (and Solutions 1/3) work unchanged. Only the JEPA prediction path
    differs: instead of an MLP on [pooled_d_emb, pooled_r_emb], a small cross-attention
    transformer attends over the unpooled fragment tokens + protein residue tokens.
    """

    def __init__(
        self,
        concise_fragment: ConciseFragment,
        smiles_target_dim: int = 256,
        predictor_dim: int = 256,
        predictor_depth: int = 3,
        predictor_heads: int = 8,
        predictor_mlp_ratio: float = 2.0,
        predictor_dropout: float = 0.1,
        clip_logit_scale_init: float = 14.0,
    ) -> None:
        super().__init__()
        self.concise = concise_fragment
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(clip_logit_scale_init))))

        residue_dim = self.concise.r_project.in_features
        self.jepa_predictor = CrossAttnJEPAPredictor(
            drug_dim=self.concise.latent_dim,
            residue_dim=residue_dim,
            target_dim=smiles_target_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_ratio=predictor_mlp_ratio,
            dropout=predictor_dropout,
        )

    def forward(
        self,
        protein_embedding: torch.Tensor,  # [B, 50, 1280]
        frag_fps: torch.Tensor,           # [B, F, 2048]
        frag_mask: torch.Tensor,          # [B, F] bool
    ) -> dict[str, torch.Tensor]:
        dti_out = self.concise(frag_fps, frag_mask, protein_embedding)

        jepa_pred = self.jepa_predictor(dti_out["frag_embs"], frag_mask, protein_embedding)

        drug_features = F.normalize(dti_out["d_emb"], dim=-1)
        protein_features = F.normalize(dti_out["r_emb"], dim=-1)
        similarity_cosines = dti_out["pairwise_binding"]
        similarity_logits = self.logit_scale.exp() * similarity_cosines

        return {
            "binding": dti_out["binding"],
            "frag_codes": dti_out["frag_codes"],
            "jepa_pred": jepa_pred,
            "drug_features": drug_features,
            "protein_features": protein_features,
            "similarity_cosines": similarity_cosines,
            "similarity_logits": similarity_logits,
            "pooled_drug_emb": dti_out["pooled_drug_emb"],
        }
