"""Optional aligned BCE and objective weighting for separate binding branches."""

import math

from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .lit_fragment import LitFragment


class LitSecondaryBinding(LitFragment):
    def __init__(self, model, experiment_config, secondary_loss_weight=0.25,
                 jepa_loss_type="mse", jepa_logit_scale_init=14.0,
                 binding_bce_weight=0.0, binding_contrastive_weight=1.0,
                 binding_negative_weight=1.0, jepa_loss_weight=1.0):
        weights = {
            "secondary_loss_weight": secondary_loss_weight,
            "binding_bce_weight": binding_bce_weight,
            "binding_contrastive_weight": binding_contrastive_weight,
            "binding_negative_weight": binding_negative_weight,
            "jepa_loss_weight": jepa_loss_weight,
        }
        for name, value in weights.items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            weights[name] = value
        super().__init__(model, experiment_config, jepa_loss_type, jepa_logit_scale_init)
        for name, value in weights.items():
            setattr(self, name, value)
        self.save_hyperparameters(weights)
        # Log Metric objects (not per-batch scalar calls): Lightning computes
        # across the epoch and resets states at each stage/epoch boundary.
        self.binding_metrics = nn.ModuleDict({
            f"{stage}_stage": nn.ModuleDict({
                f"{view}_pooled_{name}": cls()
                for view in ("fragment", "whole", "fused")
                for name, cls in (("ap", BinaryAveragePrecision), ("auroc", BinaryAUROC))
            }) for stage in ("train", "val", "test")
        })

    def _unpack_batch(self, batch):
        if len(batch) != 8:
            raise ValueError("Secondary binding requires the 8-item dual-view batch with whole fingerprint at index 3")
        protein, fragments, mask, whole, target, label, smiles, sequences = batch
        return (protein, fragments, mask, whole), target, label, smiles, sequences

    def _binding_losses(self, outputs, label, smiles_list, seq_list, stage):
        losses, components = {}, {}
        for view in ("fragment", "whole"):
            branch = {
                "similarity_logits": outputs[f"{view}_similarity_logits"],
                "similarity_cosines": outputs[f"{view}_similarity_cosines"],
            }
            contrastive, negative = super()._binding_losses(branch, label, smiles_list, seq_list, stage)
            # These are sigmoid probabilities, not raw binary logits. Only the
            # observed aligned labels supervise BCE; off-diagonals are unknown.
            bce = (F.binary_cross_entropy(outputs[f"{view}_binding"].float(), label.float())
                   if self.binding_bce_weight else contrastive.new_zeros(()))
            weighted_ce = contrastive if self.binding_contrastive_weight == 1 else self.binding_contrastive_weight * contrastive
            weighted_negative = negative if self.binding_negative_weight == 1 else self.binding_negative_weight * negative
            weighted_bce = self.binding_bce_weight * bce
            # Skip the inactive addition to retain exact legacy arithmetic.
            dti = weighted_ce + weighted_bce if self.binding_bce_weight else weighted_ce
            losses[view] = dti, weighted_negative
            components[view] = {
                "bce": weighted_bce,
                "contrastive": weighted_ce,
                "neg_diag": self.negative_diagonal_weight * weighted_negative,
            }
            # Retain the historical branch `dti` alias for raw contrastive CE.
            raw = {"dti": contrastive, "contrastive": contrastive, "neg_diag": negative, "bce": bce}
            for name, loss in raw.items():
                self.log(f"{stage}/loss_{view}_{name}", loss, on_step=False,
                         on_epoch=True, batch_size=len(label))
            for name, loss in components[view].items():
                self.log(f"{stage}/loss_{view}_{name}_weighted", loss, on_step=False,
                         on_epoch=True, batch_size=len(label))
        for name in components["fragment"]:
            combined = components["fragment"][name] + self.secondary_loss_weight * components["whole"][name]
            self.log(f"{stage}/loss_{name}_weighted", combined, on_step=False,
                     on_epoch=True, batch_size=len(label))
        return tuple(fragment + self.secondary_loss_weight * whole
                     for fragment, whole in zip(losses["fragment"], losses["whole"]))

    def _log_binding_metrics(self, outputs, label, stage, batch_size):
        for view in ("fragment", "whole", "fused"):
            predictions = outputs["binding" if view == "fused" else f"{view}_binding"].detach()
            for name in ("ap", "auroc"):
                key = f"{view}_pooled_{name}"
                metric = self.binding_metrics[f"{stage}_stage"][key]
                metric.update(predictions, label.int())
                self.log(f"{stage}/{key}", metric, on_step=False, on_epoch=True,
                         batch_size=batch_size, prog_bar=(view == "fused" and stage != "train"))


__all__ = ["LitSecondaryBinding"]
