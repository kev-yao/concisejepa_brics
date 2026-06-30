# phase1_lit_fragment.py — Lightning module for fragment-level FSQ training.
#
# Simplified version of LitConciseJEPA: DTI (contrastive) + JEPA (MSE) losses,
# AUROC/AUPRC metrics. Omits chem/group supervision and synthetic UMAP (added post-QA).

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


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
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.negative_diagonal_weight = negative_diagonal_weight
        self.negative_diagonal_margin = negative_diagonal_margin

        self.jepa_loss = nn.MSELoss()
        for stage in ("train", "val", "test"):
            setattr(self, f"auprc_{stage}", BinaryAveragePrecision())
            setattr(self, f"auroc_{stage}", BinaryAUROC())

        self.save_hyperparameters({"lr": lr, "weight_decay": weight_decay})

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
        loss_jepa = self.jepa_loss(outputs["jepa_pred"], smiles_emb)

        loss = loss_dti + loss_jepa + self.negative_diagonal_weight * loss_neg_diag

        B = int(label.shape[0])
        auprc = getattr(self, f"auprc_{stage}")(outputs["binding"], label.int())
        auroc = getattr(self, f"auroc_{stage}")(outputs["binding"], label.int())

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_dti", loss_dti, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_jepa", loss_jepa, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{stage}/loss_neg_diag", loss_neg_diag, on_step=False, on_epoch=True, batch_size=B)
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
