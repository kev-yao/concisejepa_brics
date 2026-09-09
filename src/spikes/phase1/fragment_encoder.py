"""fragment_encoder.py — the core fragment-based model: ConciseFragment + ConciseJEPAFragment.

This is the central file of the whole fragment-level project. It replaces the original
ConciseJEPA's whole-molecule Morgan fingerprint with BRICS fragment fingerprints: each
fragment gets its own FSQ code via a shared DrugEncoder, the per-fragment embeddings are
combined ("pooled" — see fragment_pooling.py for the pooling strategies compared throughout
this project) into one molecule embedding, and everything downstream of that (drug-target
binding scoring, JEPA prediction) runs unchanged from the original Concise architecture.

Two classes:
  - ConciseFragment       — the DTI (binding-prediction) model. Fragment encode -> pool ->
                            pairwise drug/protein attention -> binding score.
  - ConciseJEPAFragment   — wraps ConciseFragment and adds an MLP predictor head that maps
                            (drug, protein) context to a 256-d COATI embedding (the "JEPA"
                            reconstruction/generation target). This is the drop-in-replaceable
                            predictor: fragment_xattn.py's ConciseJEPAFragmentXAttn is an
                            alternate predictor with the same interface (see that file for the
                            cross-attention variant that turned out to reconstruct much better).

Trained via lit_fragment.py / fragment_train.py; loaded and evaluated by nearly every script
in scripts/ (jepa_reconstruct.py, jepa_denovo.py, the attribution scripts, the embedding-
retrieval scripts, etc.) — this is the one file almost everything else in the project imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from esm.model.esm2 import TransformerLayer

from concisejepa.models.drug_decoder import DrugEncoder
from concisejepa.models.concise import activation_choices

from .fragment_pooling import build_pooling, POOLING_NEEDS_CONTEXT


class ConciseFragment(nn.Module):
    """
    Drop-in replacement for Concise with BRICS fragment-level FSQ.

    Drug path (new):
        frag_fps [B, F, ligand_dim]
          → flatten [B*F, ligand_dim]
          → shared DrugEncoder (pre_transform + ResidualFSQ)
          → frag_embs [B, F, drug_dim]  +  frag_codes [B, F, n_factors]
          → pool([B, F, drug_dim], mask) → mol_emb [B, drug_dim]
          → unsqueeze [B, 1, drug_dim]
          → d_project [B, 1, proj_dim]
          → (same attention + pairwise scoring as Concise)

    Protein path: unchanged from Concise. With protein-conditioned pooling,
    each pair (i, j) pools drug i's shared fragment embeddings using protein j.
    Returned molecule embeddings and attribution weights remain aligned (i, i).
    """

    def __init__(
        self,
        drug_layers,
        pooling: str = "mean",
        ligand_dim: int = 2048,
        residue_dim: int = 1280,
        drug_dim: int = 128,
        proj_dim: int = 256,
        nheads: int = 32,
        activation: str = "tanh",
        pairwise_attention_chunk_size: int = 4096,
        use_pairwise_attention_checkpoint: bool = True,
        drug_quantizer=None,
    ):
        super().__init__()
        self.latent_dim = drug_dim
        self.pooling_name = pooling
        self.pairwise_attention_chunk_size = pairwise_attention_chunk_size
        self.use_pairwise_attention_checkpoint = use_pairwise_attention_checkpoint

        act_cls = activation_choices[activation]

        # DrugEncoder — weights shared across all fragments of a molecule.
        self.d_encoder = DrugEncoder(
            drug_layers,
            ligand_dim,
            drug_dim,
            activation=act_cls,
            quantizer=drug_quantizer,
        )
        self.num_tokens = 1  # after pooling: 1 drug token

        # Fragment pooling layer.
        self.pooling_layer = build_pooling(pooling, drug_dim, residue_dim)
        self.pooling_needs_context = pooling in POOLING_NEEDS_CONTEXT

        # Projection and attention layers — identical to Concise.
        self.r_project = nn.Linear(residue_dim, proj_dim)
        self.d_project = nn.Linear(drug_dim, proj_dim)

        self.d_to_d_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )
        self.r_to_r_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )
        self.r_to_d_attention = nn.MultiheadAttention(proj_dim, nheads, batch_first=True)

        # Final scorer (same shape as Concise with num_tokens=1).
        self.final = nn.Sequential(
            nn.Linear((self.num_tokens + 1) * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Fragment encoding helpers
    # ------------------------------------------------------------------

    def _encode_fragments(
        self,
        frag_fps: torch.Tensor,   # [B, F, ligand_dim]
        frag_mask: torch.Tensor,  # [B, F] bool — True = valid fragment
        r_emb_raw: torch.Tensor,  # [B, R, residue_dim] — used by context-conditioned poolers
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns aligned mol_emb [B, drug_dim], frag_codes [B, F, n_factors], and the
        UNPOOLED per-fragment embeddings frag_embs [B, F, drug_dim] (needed by
        multi-token predictors that attend over fragments directly instead of a
        single pooled vector — see ConciseJEPAFragmentXAttn)."""
        B, F, D = frag_fps.shape
        flat = frag_fps.reshape(B * F, D)
        d_outs = self.d_encoder(flat)

        frag_embs = d_outs["emb"].squeeze(1).reshape(B, F, self.latent_dim)  # [B, F, drug_dim]
        frag_codes = d_outs["codes"].reshape(B, F, -1)                        # [B, F, n_factors]

        if self.pooling_needs_context:
            mol_emb = self.pooling_layer(frag_embs, frag_mask, r_emb_raw)
        else:
            mol_emb = self.pooling_layer(frag_embs, frag_mask)

        return mol_emb, frag_codes, frag_embs

    # ------------------------------------------------------------------
    # Protein residue pooling (unchanged from Concise)
    # ------------------------------------------------------------------

    def _pool_residue_embeddings(self, r_emb: torch.Tensor) -> torch.Tensor:
        r_emb_wt = F.softmax(10 * r_emb, dim=1)
        return (r_emb * r_emb_wt).sum(dim=1)

    # ------------------------------------------------------------------
    # Pairwise scoring
    # ------------------------------------------------------------------

    def _project_drug_embeddings(self, mol_emb: torch.Tensor) -> torch.Tensor:
        d_emb = self.d_project(mol_emb.unsqueeze(1))
        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        return d_emb + rearrange(d_d_mixed, "n b k -> b n k")

    def _score_aligned_embeddings(self, d_pair: torch.Tensor, r_pair: torch.Tensor) -> torch.Tensor:
        d_pair = rearrange(d_pair, "b n k -> b (n k)")
        return self.final(torch.cat([d_pair, r_pair], dim=-1)).squeeze(-1)

    def _score_fragment_aligned_embeddings(self, d_pair, r_pair):
        """Opt-in primary readout hook; whole scoring keeps the original scorer."""
        return self._score_aligned_embeddings(d_pair, r_pair)

    def _attend_and_score_pair_chunk(self, d_pair, r_pair, r_pooled_pair, fragment_view=True):
        d_r_mixed, _ = self.r_to_d_attention(d_pair, r_pair, r_pair, need_weights=False)
        d_pair = d_pair + d_r_mixed
        scorer = self._score_fragment_aligned_embeddings if fragment_view else self._score_aligned_embeddings
        scores = scorer(d_pair, r_pooled_pair)
        return scores, d_pair

    def _condition_and_score_pair_chunk(
        self, frag_embs, frag_mask, r_emb_raw, r_emb, r_pooled, drug_indices, protein_indices
    ):
        # Gather inside the checkpoint so backward retains the shared tensors and
        # indices, not a separate expanded raw-protein/fragment tensor per chunk.
        # Attribution describes the aligned batch, not the last pair chunk.
        # Restore it even when non-reentrant checkpoint recomputation stops early.
        tracks_weights = hasattr(self.pooling_layer, "last_weights")
        aligned_weights = getattr(self.pooling_layer, "last_weights", None)
        try:
            mol_emb = self.pooling_layer(
                frag_embs.index_select(0, drug_indices),
                frag_mask.index_select(0, drug_indices),
                r_emb_raw.index_select(0, protein_indices),
            )
        finally:
            if tracks_weights:
                self.pooling_layer.last_weights = aligned_weights
        d_pair = self._project_drug_embeddings(mol_emb)
        return self._attend_and_score_pair_chunk(
            d_pair,
            r_emb.index_select(0, protein_indices),
            r_pooled.index_select(0, protein_indices),
        )

    def _pairwise_drug_to_protein_scores(
        self,
        d_emb: torch.Tensor,    # [B, 1, proj_dim]
        r_emb: torch.Tensor,    # [B, 50, proj_dim]
        r_pooled: torch.Tensor, # [B, proj_dim]
        *,
        frag_embs: torch.Tensor | None = None,
        frag_mask: torch.Tensor | None = None,
        r_emb_raw: torch.Tensor | None = None,
        condition_on_fragments: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Whole-molecule scoring bypasses fragment pooling, but shares the same
        # pairwise attention/scorer and the actual candidate protein for [i, j].
        conditioned = self.pooling_needs_context if condition_on_fragments is None else condition_on_fragments
        if conditioned and any(x is None for x in (frag_embs, frag_mask, r_emb_raw)):
            raise ValueError("Protein-conditioned pair scoring requires fragments, mask, and raw proteins")
        batch_size, drug_tokens, proj_dim = d_emb.shape
        total_pairs = batch_size * batch_size
        chunk_size = min(int(self.pairwise_attention_chunk_size), total_pairs)
        pairwise_binding = d_emb.new_empty(batch_size, batch_size)
        diagonal_d_emb = d_emb.new_empty(batch_size, drug_tokens, proj_dim)

        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            pair_indices = torch.arange(start, end, device=d_emb.device)
            drug_indices = torch.div(pair_indices, batch_size, rounding_mode="floor")
            protein_indices = pair_indices.remainder(batch_size)

            if conditioned:
                score_chunk = self._condition_and_score_pair_chunk
                pair_inputs = (
                    frag_embs, frag_mask, r_emb_raw, r_emb, r_pooled, drug_indices, protein_indices
                )
            else:
                score_chunk = self._attend_and_score_pair_chunk
                pair_inputs = (
                    d_emb.index_select(0, drug_indices),
                    r_emb.index_select(0, protein_indices),
                    r_pooled.index_select(0, protein_indices),
                    condition_on_fragments is not False,
                )

            if self.use_pairwise_attention_checkpoint and torch.is_grad_enabled():
                scores, d_pair = checkpoint(score_chunk, *pair_inputs, use_reentrant=False)
            else:
                scores, d_pair = score_chunk(*pair_inputs)

            pairwise_binding.reshape(-1)[start:end] = scores

            diagonal_mask = drug_indices == protein_indices
            if diagonal_mask.any():
                diagonal_d_emb.index_copy_(0, drug_indices[diagonal_mask], d_pair[diagonal_mask])

        return pairwise_binding, diagonal_d_emb

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frag_fps: torch.Tensor,   # [B, F, ligand_dim]
        frag_mask: torch.Tensor,  # [B, F] bool
        r_emb: torch.Tensor,      # [B, 50, residue_dim]
        whole_molecule_fingerprint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mol_emb, frag_codes, frag_embs = self._encode_fragments(frag_fps, frag_mask, r_emb)

        # Reusable drug path for protein-independent pooling. Context-conditioned
        # pair chunks below must instead pool/project using the scored protein.
        d_emb = self._project_drug_embeddings(mol_emb)

        # Protein path: project + self-attention
        r_emb_proj = self.r_project(r_emb)  # [B, 50, proj_dim]
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb_proj, "b n k -> n b k"))
        r_emb_proj = r_emb_proj + rearrange(r_r_mixed, "n b k -> b n k")

        r_pooled = self._pool_residue_embeddings(r_emb_proj)   # [B, proj_dim]
        pairwise_binding, d_emb = self._pairwise_drug_to_protein_scores(
            d_emb, r_emb_proj, r_pooled,
            frag_embs=frag_embs, frag_mask=frag_mask, r_emb_raw=r_emb,
        )
        binding = pairwise_binding.diagonal()

        d_emb_flat = rearrange(d_emb, "b n k -> b (n k)")  # [B, proj_dim]

        outputs = {
            "d_emb": d_emb_flat,          # [B, proj_dim]
            "r_emb": r_pooled,            # [B, proj_dim]
            "binding": binding,           # [B]
            "pairwise_binding": pairwise_binding,  # [B, B]
            "frag_codes": frag_codes,     # [B, F, n_factors]
            "pooled_drug_emb": mol_emb,   # [B, drug_dim] — post-FSQ, post-pooling; analog of
                                          # Concise's single-token "quantized" embedding, used
                                          # for chem/group auxiliary supervision.
            "frag_embs": frag_embs,       # [B, F, drug_dim] — UNPOOLED per-fragment embeddings,
                                          # for multi-token predictors (ConciseJEPAFragmentXAttn).
        }
        if whole_molecule_fingerprint is not None:
            whole = self.d_encoder(whole_molecule_fingerprint)
            whole_d = self._project_drug_embeddings(whole["emb"].squeeze(1))
            whole_scores, _ = self._pairwise_drug_to_protein_scores(
                whole_d, r_emb_proj, r_pooled, condition_on_fragments=False,
            )
            outputs.update({
                "whole_binding": whole_scores.diagonal(),
                "whole_pairwise_binding": whole_scores,
                "whole_codes": whole["codes"],
            })
        return outputs


class ConciseJEPAFragment(nn.Module):
    """
    ConciseJEPA with fragment-level FSQ drug encoder.

    Expected inputs:
      - protein_embedding: [B, 50, 1280]
      - frag_fps:          [B, F, 2048]
      - frag_mask:         [B, F] bool
    """

    def __init__(
        self,
        concise_fragment: ConciseFragment,
        smiles_target_dim: int = 256,
        jepa_hidden_dim: int = 512,
        clip_logit_scale_init: float = 14.0,
    ) -> None:
        super().__init__()
        self.concise = concise_fragment
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(clip_logit_scale_init))))

        proj_dim = self.concise.r_project.out_features
        drug_token_count = self.concise.num_tokens  # 1

        self.jepa_predictor = nn.Sequential(
            nn.Linear((drug_token_count + 1) * proj_dim, jepa_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(jepa_hidden_dim, jepa_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(jepa_hidden_dim, smiles_target_dim),
        )

    def forward(
        self,
        protein_embedding: torch.Tensor,  # [B, 50, 1280]
        frag_fps: torch.Tensor,           # [B, F, 2048]
        frag_mask: torch.Tensor,          # [B, F] bool
    ) -> dict[str, torch.Tensor]:
        dti_out = self.concise(frag_fps, frag_mask, protein_embedding)
        return self._jepa_outputs(dti_out)

    def _jepa_outputs(self, dti_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """JEPA and features depend exclusively on the aligned fragment path."""
        context = torch.cat([dti_out["d_emb"], dti_out["r_emb"]], dim=-1)  # [B, 2*proj_dim]
        jepa_pred = self.jepa_predictor(context)

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
