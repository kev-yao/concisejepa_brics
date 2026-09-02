"""Standard-runner adapter for the BRICS fragment Lightning task."""

import torch.nn as nn
from omegaconf import DictConfig

from spikes.phase1.lit_fragment import LitFragment as _SpikeLitFragment


class LitFragment(_SpikeLitFragment):
    """Construct the reference BRICS objective from a composed experiment config."""

    def __init__(
        self,
        model: nn.Module,
        experiment_config: DictConfig,
        jepa_loss_type: str = "mse",
        jepa_logit_scale_init: float = 14.0,
    ) -> None:
        fingerprint_length = int(experiment_config.data.fingerprint_length)
        ligand_dim = int(experiment_config.model.concise_fragment.ligand_dim)
        if fingerprint_length != ligand_dim:
            raise ValueError(
                "Fragment fingerprint/model dimension mismatch: "
                f"data.fingerprint_length={fingerprint_length}, "
                f"model.concise_fragment.ligand_dim={ligand_dim}."
            )
        negative_cfg = experiment_config.negative_diagonal_loss
        chem_cfg = experiment_config.chem_supervision
        group_cfg = experiment_config.group_supervision
        super().__init__(
            model=model,
            lr=float(experiment_config.lr),
            weight_decay=float(experiment_config.weight_decay),
            negative_diagonal_weight=(
                float(negative_cfg.weight) if bool(negative_cfg.enabled) else 0.0
            ),
            negative_diagonal_margin=float(negative_cfg.margin),
            chem_supervision_weight=float(chem_cfg.weight) if bool(chem_cfg.enabled) else 0.0,
            group_supervision_weight=(
                float(group_cfg.weight) if bool(group_cfg.enabled) else 0.0
            ),
            jepa_loss_type=jepa_loss_type,
            jepa_logit_scale_init=jepa_logit_scale_init,
        )


__all__ = ["LitFragment"]
