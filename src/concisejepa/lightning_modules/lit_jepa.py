import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from torchmetrics.classification import BinaryAveragePrecision


class LitConciseJEPA(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.model = instantiate(config.model)
        self.lr = config.lr
        self.weight_decay = config.weight_decay

        self.auprc_by_stage = {
            "train": BinaryAveragePrecision(),
            "val": BinaryAveragePrecision(),
            "test": BinaryAveragePrecision(),
        }
        self.jepa_loss = nn.MSELoss()
        self.dti_loss = nn.BCELoss()

        coati_cfg = getattr(config, "coati_validation", {})
        self.enable_val_smiles_decode = bool(coati_cfg.get("enabled", False))
        self.coati_device = coati_cfg.get("device", "cpu")
        self.coati_doc_url = coati_cfg.get("doc_url", "s3://terray-public/models/grande_closed.pkl")
        self.coati_noise_scale = float(coati_cfg.get("noise_scale", 0.0))
        self.max_decode_per_batch = int(coati_cfg.get("max_decode_per_batch", -1))
        self.table_max_rows = int(coati_cfg.get("table_max_rows", 64))

        self._coati_encoder = None
        self._coati_tokenizer = None
        self._reset_val_decode_state()

        self.save_hyperparameters(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }
        )

    def _reset_val_decode_state(self) -> None:
        self._val_tanimoto_sum = 0.0
        self._val_tanimoto_count = 0
        self._val_decode_count = 0
        self._val_total_count = 0
        self._val_table_rows = []

    def _ensure_coati_decoder_loaded(self) -> None:
        if self._coati_encoder is not None and self._coati_tokenizer is not None:
            return

        from coati.models.io import load_e3gnn_smiles_clip_e2e

        encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
            freeze=True,
            device=self.coati_device,
            doc_url=self.coati_doc_url,
        )
        encoder.eval()
        self._coati_encoder = encoder
        self._coati_tokenizer = tokenizer

    @staticmethod
    def _canonicalize_largest_fragment(smiles: str | None) -> str | None:
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None
        largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        largest_smiles = Chem.MolToSmiles(largest)
        try:
            return Chem.CanonSmiles(largest_smiles)
        except Exception:
            return None

    @staticmethod
    def _batch_tanimoto_similarity(
        target_smiles: list[str],
        predicted_smiles: list[str],
    ) -> list[float]:
        pred_fps = []
        for smi in predicted_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                pred_fps.append(None)
            else:
                pred_fps.append(RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=2048))

        grouped_indices = {}
        for i, smi in enumerate(target_smiles):
            grouped_indices.setdefault(smi, []).append(i)

        similarities = [0.0] * len(target_smiles)
        for tgt_smi, indices in grouped_indices.items():
            tgt_mol = Chem.MolFromSmiles(tgt_smi)
            if tgt_mol is None:
                continue
            tgt_fp = RDKFingerprint(tgt_mol, minPath=1, maxPath=7, fpSize=2048)

            valid_pred_fps = []
            valid_indices = []
            for idx in indices:
                fp = pred_fps[idx]
                if fp is not None:
                    valid_pred_fps.append(fp)
                    valid_indices.append(idx)

            if not valid_pred_fps:
                continue

            vals = BulkTanimotoSimilarity(tgt_fp, valid_pred_fps)
            for idx, score in zip(valid_indices, vals):
                similarities[idx] = float(score)

        return similarities

    def on_validation_epoch_start(self) -> None:
        self._reset_val_decode_state()

    def _collect_validation_decode_metrics(
        self,
        jepa_pred: torch.Tensor,
        target_smiles_list: list[str],
    ) -> None:
        if not self.enable_val_smiles_decode:
            return

        self._ensure_coati_decoder_loaded()

        num_items = jepa_pred.shape[0]
        if self.max_decode_per_batch > 0:
            num_items = min(num_items, self.max_decode_per_batch)
        if num_items <= 0:
            return

        with torch.no_grad():
            pred_vectors = jepa_pred[:num_items].detach().to(self.coati_device).float()
            generated_smiles = self._coati_encoder.hclip_to_2d_batch(
                h_clip=pred_vectors,
                tokenizer=self._coati_tokenizer,
                noise_scale=self.coati_noise_scale,
            )

        valid_target_smiles = []
        valid_pred_smiles = []
        for target_raw, generated_raw in zip(target_smiles_list[:num_items], generated_smiles):
            self._val_total_count += 1
            target_smiles = self._canonicalize_largest_fragment(target_raw)
            pred_smiles = self._canonicalize_largest_fragment(generated_raw)

            if pred_smiles is not None:
                self._val_decode_count += 1
            if target_smiles is None or pred_smiles is None:
                continue

            valid_target_smiles.append(target_smiles)
            valid_pred_smiles.append(pred_smiles)

        if not valid_target_smiles:
            return

        tanimotos = self._batch_tanimoto_similarity(valid_target_smiles, valid_pred_smiles)
        self._val_tanimoto_sum += float(sum(tanimotos))
        self._val_tanimoto_count += len(tanimotos)

        remaining = max(0, self.table_max_rows - len(self._val_table_rows))
        if remaining <= 0:
            return

        for target_smi, pred_smi, score in zip(
            valid_target_smiles[:remaining],
            valid_pred_smiles[:remaining],
            tanimotos[:remaining],
        ):
            self._val_table_rows.append(
                {
                    "target_smiles": target_smi,
                    "generated_smiles": pred_smi,
                    "tanimoto": float(score),
                }
            )

    def _log_wandb_table(self) -> None:
        if not self._val_table_rows:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        experiment = self.logger.experiment
        if experiment is None or not hasattr(experiment, "log"):
            return

        table = wandb.Table(columns=["target_smiles", "generated_smiles", "tanimoto"])
        for row in self._val_table_rows:
            table.add_data(row["target_smiles"], row["generated_smiles"], row["tanimoto"])

        experiment.log(
            {
                "val/jepa_smiles_table": table,
                "trainer/global_step": self.global_step,
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self.enable_val_smiles_decode:
            return

        mean_tanimoto = 0.0
        if self._val_tanimoto_count > 0:
            mean_tanimoto = self._val_tanimoto_sum / self._val_tanimoto_count

        decode_rate = 0.0
        valid_pair_rate = 0.0
        if self._val_total_count > 0:
            decode_rate = self._val_decode_count / self._val_total_count
            valid_pair_rate = self._val_tanimoto_count / self._val_total_count

        self.log("val/jepa_tanimoto", mean_tanimoto, prog_bar=True, on_epoch=True)
        self.log("val/jepa_decode_rate", decode_rate, on_epoch=True)
        self.log("val/jepa_valid_pair_rate", valid_pair_rate, on_epoch=True)
        self._log_wandb_table()

    def _forward_losses_metrics(
        self,
        batch,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        protein_embedding, morgan_fingerprint, smiles_target_embedding, label = batch[:4]

        outputs = self.model(
            protein_embedding=protein_embedding,
            morgan_fingerprint=morgan_fingerprint,
        )

        dti_probs = outputs["binding"]
        loss_dti = self.dti_loss(dti_probs, label)
        loss_jepa = self.jepa_loss(outputs["jepa_pred"], smiles_target_embedding)
        loss = loss_dti + loss_jepa

        labels = label.to(torch.int)
        auprc = self.auprc_by_stage[stage](dti_probs, labels)

        losses = {
            "loss": loss,
            "loss_dti": loss_dti,
            "loss_jepa": loss_jepa,
        }
        return loss, losses, auprc, outputs

    def _log_stage_metrics(
        self,
        stage: str,
        losses: dict[str, torch.Tensor],
        auprc: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.log(
            f"{stage}/loss",
            losses["loss"],
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_dti",
            losses["loss_dti"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss_jepa",
            losses["loss_jepa"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/dti_auprc",
            auprc,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=batch_size,
        )

    def _step(self, batch, stage: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, losses, auprc, outputs = self._forward_losses_metrics(batch, stage)
        batch_size = int(batch[3].shape[0])
        self._log_stage_metrics(stage, losses, auprc, batch_size)
        return loss, outputs

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _ = self._step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, outputs = self._step(batch, stage="val")
        if len(batch) > 4:
            self._collect_validation_decode_metrics(outputs["jepa_pred"], batch[4])
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _ = self._step(batch, stage="test")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
