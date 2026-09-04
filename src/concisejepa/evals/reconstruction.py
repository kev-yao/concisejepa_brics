"""Unique-molecule validation for the dual-view predictor."""

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence


class UniqueJEPAValidation(pl.Callback):
    """Weight every canonical validation molecule equally for checkpoint selection."""

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        dm = trainer.datamodule
        frame = dm.val_dataset.df
        key = "canonical_smiles" if "canonical_smiles" in frame else dm.smiles_col
        smiles = sorted(frame.drop_duplicates(key)[dm.smiles_col].tolist())
        squared_error = torch.zeros((), device=pl_module.device)
        count = 0
        for start in range(0, len(smiles), 128):
            chunk = smiles[start : start + 128]
            fragments = [dm.fragment_fps[s].float() for s in chunk]
            padded = pad_sequence(fragments, batch_first=True).to(pl_module.device)
            lengths = torch.tensor([len(f) for f in fragments], device=pl_module.device)
            mask = torch.arange(padded.shape[1], device=pl_module.device)[None, :] < lengths[:, None]
            pooled, _, _, _ = pl_module.model._encode_fragments(padded, mask)
            pred = pl_module.model.coati_predictor(pooled)
            target = torch.stack([dm.smiles_embeddings[s].reshape(-1).float() for s in chunk]).to(pl_module.device)
            squared_error += (pred - target).square().sum()
            count += target.numel()
        if not count:
            raise RuntimeError("No validation molecules available")
        pl_module.log("val/unique_jepa_mse", squared_error / count, on_epoch=True, batch_size=len(smiles))
