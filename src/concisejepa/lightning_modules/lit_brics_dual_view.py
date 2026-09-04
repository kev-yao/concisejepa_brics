"""Training objective for the dual-view BRICS experiment."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class LitBricsDualViewJEPA(pl.LightningModule):
    def __init__(self, model: nn.Module, experiment_config: DictConfig) -> None:
        super().__init__()
        fingerprint_length = int(experiment_config.data.fingerprint_length)
        ligand_dim = int(experiment_config.model.ligand_dim)
        if fingerprint_length != ligand_dim:
            raise ValueError(
                "Fingerprint/model dimension mismatch: "
                f"data.fingerprint_length={fingerprint_length}, model.ligand_dim={ligand_dim}."
            )

        self.model = model
        self.lr = float(experiment_config.lr)
        self.weight_decay = float(experiment_config.weight_decay)
        loss_cfg = experiment_config.dual_view_loss
        self.dti_weight = float(loss_cfg.dti_weight)
        self.jepa_weight = float(loss_cfg.jepa_weight)
        self.alignment_weight = float(loss_cfg.alignment_weight)
        self.bce = nn.BCELoss()
        self.jepa_loss = nn.MSELoss()

        self.auprc_by_stage = nn.ModuleDict(
            {
                f"{stage}_metric": BinaryAveragePrecision()
                for stage in ("train", "val", "test")
            }
        )
        self.auroc_by_stage = nn.ModuleDict(
            {f"{stage}_metric": BinaryAUROC() for stage in ("train", "val", "test")}
        )
        self.save_hyperparameters(ignore=["model", "experiment_config"])

    def _forward_losses_metrics(self, batch, stage: str):
        (
            protein_embedding,
            fragment_fingerprints,
            fragment_mask,
            whole_molecule_fingerprint,
            smiles_target_embedding,
            label,
        ) = batch[:6]
        outputs = self.model(
            protein_embedding=protein_embedding,
            fragment_fingerprints=fragment_fingerprints,
            fragment_mask=fragment_mask,
            whole_molecule_fingerprint=whole_molecule_fingerprint,
        )

        label = label.float()
        loss_dti_whole = self.bce(outputs["whole_binding"], label)
        loss_dti_fragment = self.bce(outputs["fragment_binding"], label)
        loss_dti = 0.5 * (loss_dti_whole + loss_dti_fragment)
        loss_jepa = self.jepa_loss(outputs["jepa_pred"], smiles_target_embedding)
        loss_alignment = (1.0 - outputs["fragment_molecule_similarity"]).mean()
        loss = (
            self.dti_weight * loss_dti
            + self.jepa_weight * loss_jepa
            + self.alignment_weight * loss_alignment
        )

        labels_int = label.to(torch.int)
        metric_key = f"{stage}_metric"
        auprc = self.auprc_by_stage[metric_key](outputs["binding"], labels_int)
        auroc = self.auroc_by_stage[metric_key](outputs["binding"], labels_int)
        losses = {
            "loss": loss,
            "loss_dti": loss_dti,
            "loss_dti_whole": loss_dti_whole,
            "loss_dti_fragment": loss_dti_fragment,
            "loss_jepa": loss_jepa,
            "loss_alignment": loss_alignment,
        }
        return loss, losses, auprc, auroc, outputs

    def _step(self, batch, stage: str):
        loss, losses, auprc, auroc, outputs = self._forward_losses_metrics(batch, stage)
        batch_size = int(batch[5].shape[0])
        on_step = stage == "train"
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=on_step and name == "loss",
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch_size,
            )
        self.log(
            f"{stage}/dti_auprc",
            auprc,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "train",
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/dti_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "train",
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/binding_whole_mean",
            outputs["whole_binding"].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/binding_fragment_mean",
            outputs["fragment_binding"].mean(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


__all__ = ["LitBricsDualViewJEPA"]
