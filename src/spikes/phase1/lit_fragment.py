"""lit_fragment.py — the PyTorch Lightning training module for the fragment pipeline: computes
every loss term, logs every metric, and drives training/validation/test steps for whichever
model (ConciseJEPAFragment or ConciseJEPAFragmentXAttn) it's handed.

Losses (all summed into one training loss, individually logged):
  - loss_dti — in-batch contrastive DTI loss (drug/protein similarity vs. binding label).
  - loss_jepa — MSE (default) or contrastive (jepa_loss_type="contrastive", Solution 3 from the
    reconstruction investigation — see _contrastive_jepa_loss's docstring for the rationale, and
    docs/history/reconstruction-theories-and-solutions.md for the full theory).
  - loss_neg_diag — penalizes known negatives that score higher similarity than the row average.
  - loss_chem / loss_group — optional auxiliary chem-property / functional-group supervision on
    the pooled drug embedding (chem_supervision_weight / group_supervision_weight > 0 to enable).
    Ported from the original (non-fragmented) LitConciseJEPA after the reconstruction audit found
    reconstruction fidelity correlates with having these losses, independent of loss_jepa's own
    magnitude — see docs/history/reconstruction-audit.md. In the end this confound turned out to
    be real but small; max-pooling's own structural property mattered far more (see
    docs/PROJECT_HANDOFF.md §6).

Simplified from the original LitConciseJEPA — omits synthetic UMAP (added there post-QA, not
carried over here).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from concisejepa.models.chem_property_head import ChemPropertyHead, compute_property_bins
from concisejepa.models.functional_group_head import FunctionalGroupHead, compute_functional_group_bits


class LitFragment(pl.LightningModule):
    """
    Lightning module wrapping ConciseJEPAFragment.

    Batch format (from FragmentCollator):
        protein_emb:   [B, 50, 1280]
        frag_fps:      [B, F, 2048]
        frag_mask:     [B, F] bool
        smiles_emb:    [B, 256]
        label:         [B]
        smiles_list:   list[str]  (optional)
        sequences:     list[str]  (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        negative_diagonal_weight: float = 1.0,
        negative_diagonal_margin: float = 0.0,
        chem_supervision_weight: float = 0.0,
        group_supervision_weight: float = 0.0,
        jepa_loss_type: str = "mse",
        jepa_logit_scale_init: float = 14.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.negative_diagonal_weight = negative_diagonal_weight
        self.negative_diagonal_margin = negative_diagonal_margin

        self.jepa_loss_type = jepa_loss_type
        if jepa_loss_type not in ("mse", "contrastive"):
            raise ValueError(f"Unknown jepa_loss_type: {jepa_loss_type!r}")
        self.jepa_loss = nn.MSELoss()
        if jepa_loss_type == "contrastive":
            self.jepa_logit_scale = nn.Parameter(torch.log(torch.tensor(float(jepa_logit_scale_init))))
        for stage in ("train", "val", "test"):
            setattr(self, f"auprc_{stage}", BinaryAveragePrecision())
            setattr(self, f"auroc_{stage}", BinaryAUROC())

        # Auxiliary chem/group supervision on the pooled, post-FSQ drug embedding
        # (weight <= 0 disables — mirrors LitConciseJEPA's chem_supervision/group_supervision).
        latent_dim = int(model.concise.latent_dim)
        self.chem_supervision_weight = float(chem_supervision_weight)
        self.chem_property_head = ChemPropertyHead(in_dim=latent_dim) if self.chem_supervision_weight > 0 else None
        self._smiles_property_cache: dict[str, torch.Tensor] = {}

        self.group_supervision_weight = float(group_supervision_weight)
        self.group_head = FunctionalGroupHead(in_dim=latent_dim) if self.group_supervision_weight > 0 else None
        self._smiles_group_cache: dict[str, torch.Tensor] = {}

        self.save_hyperparameters({
            "lr": lr, "weight_decay": weight_decay,
            "chem_supervision_weight": chem_supervision_weight,
            "group_supervision_weight": group_supervision_weight,
            "jepa_loss_type": jepa_loss_type,
        })

    def _fill_property_cache(self, smiles_list: list[str]) -> None:
        for smi in smiles_list:
            if smi not in self._smiles_property_cache:
                self._smiles_property_cache[smi] = compute_property_bins(smi)

    def _fill_group_cache(self, smiles_list: list[str]) -> None:
        for smi in smiles_list:
            if smi not in self._smiles_group_cache:
                self._smiles_group_cache[smi] = compute_functional_group_bits(smi)

    # ------------------------------------------------------------------
    # Loss helpers (copied from LitConciseJEPA)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_identity_matrix(values: list[str], device: torch.device) -> torch.Tensor:
        mat = [[l == r for r in values] for l in values]
        return torch.tensor(mat, dtype=torch.bool, device=device)

    def _known_positive_mask(self, labels, smiles_list, sequence_list, B, device):
        pos_col_mask = (labels > 0.5).unsqueeze(0)
        smiles_match = (
            self._build_identity_matrix(smiles_list, device)
            if len(smiles_list) == B
            else torch.zeros(B, B, dtype=torch.bool, device=device)
        )
        seq_match = (
            self._build_identity_matrix(sequence_list, device)
            if len(sequence_list) == B
            else torch.zeros(B, B, dtype=torch.bool, device=device)
        )
        return (smiles_match | seq_match) & pos_col_mask

    def _contrastive_logits_targets(self, similarity_logits, labels, smiles_list, sequence_list):
        device = similarity_logits.device
        B = similarity_logits.shape[0]
        pos_mask = labels > 0.5
        pos_count = int(pos_mask.sum().item())
        if pos_count == 0:
            empty = similarity_logits.new_empty((0, B))
            return empty, torch.empty(0, dtype=torch.long, device=device), empty, torch.empty(0, dtype=torch.long, device=device), 0

        eye = torch.eye(B, dtype=torch.bool, device=device)
        kp_mask = self._known_positive_mask(labels, smiles_list, sequence_list, B, device)
        invalid_neg = kp_mask & ~eye

        row_logits = similarity_logits[pos_mask].masked_fill(invalid_neg[pos_mask], -1e9)
        row_targets = torch.arange(B, device=device)[pos_mask]
        col_logits = similarity_logits.T[pos_mask].masked_fill(invalid_neg.T[pos_mask], -1e9)
        col_targets = torch.arange(B, device=device)[pos_mask]
        return row_logits, row_targets, col_logits, col_targets, pos_count

    def _contrastive_dti_loss(self, similarity_logits, labels, smiles_list, sequence_list):
        row_l, row_t, col_l, col_t, n_pos = self._contrastive_logits_targets(
            similarity_logits, labels, smiles_list, sequence_list)
        if n_pos == 0:
            return similarity_logits.new_zeros(())
        return 0.5 * (F.cross_entropy(row_l, row_t) + F.cross_entropy(col_l, col_t))

    def _contrastive_jepa_loss(self, jepa_pred, smiles_emb, smiles_list):
        """
        Solution 3 (reconstruction-fidelity investigation): CLIP-style in-batch contrastive
        loss replacing the plain MSE loss_jepa. Rationale (see
        docs/history/reconstruction-theories-and-solutions.md, theory 4): pointwise MSE
        under residual ambiguity collapses toward a blurry conditional-mean embedding that
        decodes to nothing real. A contrastive loss only asks jepa_pred to be identifiable
        against other real molecules in the batch, rather than matching one point exactly —
        it can't be satisfied by an averaged, non-committal prediction.

        EMPIRICAL RESULT: the theory didn't pan out — this made reconstruction fidelity WORSE
        across every pooler tested (see docs/PROJECT_HANDOFF.md §6/§5.6). Kept in the codebase
        as a working, real alternative for anyone who wants to probe why, not because it's the
        recommended setting — the default (jepa_loss_type="mse") remains the better choice.

        jepa_pred:  [B, D] predicted COATI embeddings (never L2-normalized elsewhere upstream).
        smiles_emb: [B, D] true COATI embeddings for the batch (the "keys"/bank).
        smiles_list: SMILES strings, length B — used only to mask same-drug collisions (two
            rows sharing the same true drug must not be treated as false negatives of each
            other; unlike the DTI loss this only depends on SMILES identity, not sequence,
            since the JEPA target is the drug's own embedding regardless of protein).
        """
        device = jepa_pred.device
        B = jepa_pred.shape[0]
        pred_n = F.normalize(jepa_pred, dim=-1)
        true_n = F.normalize(smiles_emb, dim=-1)
        logits = self.jepa_logit_scale.exp() * (pred_n @ true_n.T)  # [B, B]

        eye = torch.eye(B, dtype=torch.bool, device=device)
        same_drug = (
            self._build_identity_matrix(smiles_list, device)
            if len(smiles_list) == B
            else eye
        )
        invalid_neg = same_drug & ~eye
        logits = logits.masked_fill(invalid_neg, -1e9)

        targets = torch.arange(B, device=device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    def _negative_diagonal_cosine_loss(self, similarity_cosines, labels, smiles_list, sequence_list):
        device = similarity_cosines.device
        B = similarity_cosines.shape[0]
        neg_mask = labels <= 0.5
        if int(neg_mask.sum()) == 0 or B <= 1:
            return similarity_cosines.new_zeros(())

        eye = torch.eye(B, dtype=torch.bool, device=device)
        kp_mask = self._known_positive_mask(labels, smiles_list, sequence_list, B, device)
        valid_ref = ~(eye | kp_mask)
        valid_neg_rows = neg_mask & (valid_ref.sum(1) > 0)
        if int(valid_neg_rows.sum()) == 0:
            return similarity_cosines.new_zeros(())

        row_sums = similarity_cosines.masked_fill(~valid_ref, 0.0).sum(1)
        row_counts = valid_ref.sum(1).clamp_min(1)
        row_means = row_sums / row_counts
        diag_scores = similarity_cosines.diagonal()
        penalties = F.relu(diag_scores - row_means + self.negative_diagonal_margin)
        return penalties[valid_neg_rows].mean()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _forward_step(self, batch, stage: str) -> tuple[torch.Tensor, dict]:
        protein_emb, frag_fps, frag_mask, smiles_emb, label = batch[:5]
        smiles_list = batch[5] if len(batch) > 5 else []
        seq_list = batch[6] if len(batch) > 6 else []

        outputs = self.model(protein_emb, frag_fps, frag_mask)

        loss_dti = self._contrastive_dti_loss(
            outputs["similarity_logits"], label, smiles_list, seq_list)
        loss_neg_diag = self._negative_diagonal_cosine_loss(
            outputs["similarity_cosines"], label, smiles_list, seq_list)
        if self.jepa_loss_type == "contrastive":
            loss_jepa = self._contrastive_jepa_loss(outputs["jepa_pred"], smiles_emb, smiles_list)
        else:
            loss_jepa = self.jepa_loss(outputs["jepa_pred"], smiles_emb)

        loss = loss_dti + loss_jepa + self.negative_diagonal_weight * loss_neg_diag

        loss_chem = outputs["binding"].new_zeros(())
        if self.chem_property_head is not None and smiles_list:
            self._fill_property_cache(smiles_list)
            loss_chem, _ = self.chem_property_head.loss(
                outputs["pooled_drug_emb"], smiles_list, self._smiles_property_cache)
            loss = loss + self.chem_supervision_weight * loss_chem

        loss_group = outputs["binding"].new_zeros(())
        if self.group_head is not None and smiles_list:
            self._fill_group_cache(smiles_list)
            loss_group, _ = self.group_head.loss(
                outputs["pooled_drug_emb"], smiles_list, self._smiles_group_cache)
            loss = loss + self.group_supervision_weight * loss_group

        B = int(label.shape[0])
        auprc = getattr(self, f"auprc_{stage}")(outputs["binding"], label.int())
        auroc = getattr(self, f"auroc_{stage}")(outputs["binding"], label.int())

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_dti", loss_dti, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_jepa", loss_jepa, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_neg_diag", loss_neg_diag, on_step=False, on_epoch=True, batch_size=B)
        if self.chem_property_head is not None:
            self.log(f"{stage}/loss_chem", loss_chem, on_step=False, on_epoch=True, batch_size=B)
        if self.group_head is not None:
            self.log(f"{stage}/loss_group", loss_group, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/dti_auprc", auprc, on_step=False, on_epoch=True, prog_bar=(stage != "train"), batch_size=B)
        self.log(f"{stage}/dti_auroc", auroc, on_step=False, on_epoch=True, prog_bar=(stage != "train"), batch_size=B)

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._forward_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._forward_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self._forward_step(batch, "test")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
