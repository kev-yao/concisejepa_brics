import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDKFingerprint
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

        coati_cfg = getattr(config, "coati_validation", {})
        self.enable_val_smiles_decode = bool(coati_cfg.get("enabled", False))
        self.coati_device = coati_cfg.get("device", "cpu")
        self.coati_doc_url = coati_cfg.get("doc_url", "s3://terray-public/models/grande_closed.pkl")
        self.coati_noise_scale = float(coati_cfg.get("noise_scale", 0.0))
        self.max_decode_per_batch = int(coati_cfg.get("max_decode_per_batch", -1))
        self.table_max_rows = int(coati_cfg.get("table_max_rows", 64))
        self.max_decode_batches_per_epoch = int(coati_cfg.get("max_decode_batches_per_epoch", -1))

        synthetic_cfg = getattr(config, "synthetic_validation", {})
        self.enable_synthetic_validation = bool(synthetic_cfg.get("enabled", False))
        self.synthetic_seed = int(synthetic_cfg.get("seed", 17))
        self.synthetic_decode_batch_size = int(synthetic_cfg.get("decode_batch_size", 256))
        self.synthetic_morgan_radius = int(synthetic_cfg.get("morgan_radius", 2))
        self.synthetic_morgan_n_bits = int(synthetic_cfg.get("morgan_n_bits", 2048))
        self.synthetic_umap_neighbors = int(synthetic_cfg.get("umap_n_neighbors", 30))
        self.synthetic_umap_min_dist = float(synthetic_cfg.get("umap_min_dist", 0.15))
        self.synthetic_morgan_umap_metric = synthetic_cfg.get("morgan_umap_metric", "jaccard")
        self.synthetic_coati_umap_metric = synthetic_cfg.get("coati_umap_metric", "euclidean")
        self.synthetic_point_size = float(synthetic_cfg.get("point_size", 26.0))
        self.synthetic_point_alpha = float(synthetic_cfg.get("point_alpha", 0.9))
        self.synthetic_max_decode = int(synthetic_cfg.get("max_decode", -1))

        self._coati_encoder = None
        self._coati_tokenizer = None
        self._fixed_protein_embedding = None
        self._synthetic_code_indices = None
        self._synthetic_first_digit = None
        self._synthetic_second_digit = None
        self._synthetic_third_digit = None
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
        self._val_decode_batches = 0

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

    def _maybe_cache_fixed_protein_embedding(self, protein_embedding: torch.Tensor) -> None:
        if self._fixed_protein_embedding is not None:
            return
        if protein_embedding.ndim != 3 or protein_embedding.shape[0] == 0:
            return
        self._fixed_protein_embedding = protein_embedding[:1].detach().cpu().clone()

    def _collect_validation_decode_metrics(
        self,
        jepa_pred: torch.Tensor,
        target_smiles_list: list[str],
    ) -> None:
        if not self.enable_val_smiles_decode:
            return
        if self.max_decode_batches_per_epoch > 0 and self._val_decode_batches >= self.max_decode_batches_per_epoch:
            return

        self._ensure_coati_decoder_loaded()
        self._val_decode_batches += 1

        num_items = jepa_pred.shape[0]
        if self.max_decode_per_batch > 0:
            perm = torch.randperm(num_items, device=jepa_pred.device)
            num_items = min(num_items, self.max_decode_per_batch)
            perm = perm[:num_items]
        else:
            perm = None
        if num_items <= 0:
            return

        with torch.no_grad():
            if perm is not None:
                pred_vectors = jepa_pred[perm].detach().to(self.coati_device).float()
                target_smiles_list = [target_smiles_list[i] for i in perm.tolist()]
            else:
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

    def _get_primary_fsq(self):
        d_encoder = self.model.concise.d_encoder
        if len(d_encoder.residualfsqs) != 1:
            raise ValueError(
                "Synthetic validation currently expects exactly one ResidualFSQ layer. "
                f"Found {len(d_encoder.residualfsqs)}."
            )

        fsq = d_encoder.residualfsqs[0].fsq
        if int(fsq.codebook_dim) != 3:
            raise ValueError(
                "Synthetic validation currently expects a 3-digit FSQ codebook. "
                f"Found codebook_dim={fsq.codebook_dim}."
            )
        return fsq

    def _build_synthetic_code_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._synthetic_code_indices is not None and self._synthetic_first_digit is not None:
            return self._synthetic_code_indices, self._synthetic_first_digit

        fsq = self._get_primary_fsq()
        levels = fsq._levels.detach().cpu().to(torch.long)
        basis = fsq._basis.detach().cpu().to(torch.long)

        first_digit, second_digit = torch.meshgrid(
            torch.arange(int(levels[0]), dtype=torch.long),
            torch.arange(int(levels[1]), dtype=torch.long),
            indexing="ij",
        )
        first_digit = first_digit.reshape(-1)
        second_digit = second_digit.reshape(-1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.synthetic_seed)
        third_digit = torch.randint(
            low=0,
            high=int(levels[2]),
            size=(first_digit.shape[0],),
            generator=generator,
            dtype=torch.long,
        )

        digits = torch.stack([first_digit, second_digit, third_digit], dim=-1)
        code_indices = (digits * basis.unsqueeze(0)).sum(dim=-1).unsqueeze(1)

        self._synthetic_code_indices = code_indices
        self._synthetic_first_digit = first_digit
        self._synthetic_second_digit = second_digit
        self._synthetic_third_digit = third_digit
        return self._synthetic_code_indices, self._synthetic_first_digit

    def _decode_smiles_from_codes(
        self,
        code_indices: torch.Tensor,
        protein_embedding: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        self._ensure_coati_decoder_loaded()

        generated_smiles = []
        pred_vector_chunks = []
        model_device = self.device
        protein_embedding = protein_embedding.to(model_device)

        with torch.no_grad():
            for code_chunk in code_indices.split(self.synthetic_decode_batch_size):
                chunk_size = code_chunk.shape[0]
                protein_chunk = protein_embedding.expand(chunk_size, -1, -1)
                outputs = self.model.predict_from_codes(
                    protein_embedding=protein_chunk,
                    drug_codes=code_chunk.to(model_device),
                )
                pred_vectors = outputs["jepa_pred"].detach().to(self.coati_device).float()
                pred_vector_chunks.append(pred_vectors.cpu())
                generated_smiles.extend(
                    self._coati_encoder.hclip_to_2d_batch(
                        h_clip=pred_vectors,
                        tokenizer=self._coati_tokenizer,
                        noise_scale=self.coati_noise_scale,
                    )
                )

        return generated_smiles, torch.cat(pred_vector_chunks, dim=0)

    def _morgan_array_from_smiles(self, smiles: str):
        import numpy as np

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.synthetic_morgan_radius,
            nBits=self.synthetic_morgan_n_bits,
        )
        arr = np.zeros((self.synthetic_morgan_n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _plot_umap_figure(
        self,
        embedding_2d,
        color_values,
        title: str,
        summary_lines: list[str],
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        cmap = plt.get_cmap("gist_ncar", 32)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=np.asarray(color_values),
            cmap=cmap,
            vmin=1,
            vmax=32,
            s=self.synthetic_point_size,
            alpha=self.synthetic_point_alpha,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("First FSQ digit")
        colorbar.set_ticks([1, 8, 16, 24, 32])
        fig.tight_layout()
        return fig

    def _log_synthetic_umap(self) -> None:
        if not self.enable_synthetic_validation:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return
        if self._fixed_protein_embedding is None:
            return

        import matplotlib
        import numpy as np
        from umap import UMAP

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        code_indices, first_digit = self._build_synthetic_code_grid()
        if self.synthetic_max_decode > 0 and code_indices.shape[0] > self.synthetic_max_decode:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.synthetic_seed)
            perm = torch.randperm(code_indices.shape[0], generator=generator)[: self.synthetic_max_decode]
            code_indices = code_indices[perm]
            first_digit = first_digit[perm]
        generated_smiles, coati_latents = self._decode_smiles_from_codes(
            code_indices=code_indices,
            protein_embedding=self._fixed_protein_embedding,
        )

        valid_fingerprints = []
        valid_coati_latents = []
        valid_first_digit = []
        valid_smiles = []
        decoded_count = 0

        for raw_smiles, latent, raw_first_digit in zip(
            generated_smiles,
            coati_latents,
            first_digit.tolist(),
        ):
            canonical_smiles = self._canonicalize_largest_fragment(raw_smiles)
            if canonical_smiles is None:
                continue

            decoded_count += 1
            fp = self._morgan_array_from_smiles(canonical_smiles)
            if fp is None:
                continue

            valid_fingerprints.append(fp)
            valid_coati_latents.append(latent.cpu().numpy())
            valid_first_digit.append(raw_first_digit + 1)
            valid_smiles.append(canonical_smiles)

        total_count = len(generated_smiles)
        valid_count = len(valid_fingerprints)
        valid_rate = valid_count / total_count if total_count > 0 else 0.0

        self.log("val/synthetic_canonical_decode_count", float(decoded_count), on_epoch=True)
        self.log("val/synthetic_decode_valid_count", float(valid_count), on_epoch=True)
        self.log("val/synthetic_decode_valid_rate", valid_rate, on_epoch=True)
        self.log("val/synthetic_unique_smiles", float(len(set(valid_smiles))), on_epoch=True)
        self.log(
            "val/synthetic_unique_fingerprints",
            float(len({fp.tobytes() for fp in valid_fingerprints})),
            on_epoch=True,
        )

        if valid_count < 3:
            return

        fingerprints = np.asarray(valid_fingerprints, dtype=np.float32)
        coati_array = np.asarray(valid_coati_latents, dtype=np.float32)
        n_neighbors = max(2, min(self.synthetic_umap_neighbors, valid_count - 1))
        summary_lines = [
            f"valid={valid_count}/{total_count}",
            f"unique_smiles={len(set(valid_smiles))}",
            "fixed_protein=first validation sample",
            f"third_digit_seed={self.synthetic_seed}",
        ]

        morgan_umap_input = (
            fingerprints.astype(bool) if self.synthetic_morgan_umap_metric == "jaccard" else fingerprints
        )
        morgan_reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.synthetic_umap_min_dist,
            metric=self.synthetic_morgan_umap_metric,
            random_state=self.synthetic_seed,
        )
        morgan_embedding = morgan_reducer.fit_transform(morgan_umap_input)
        morgan_fig = self._plot_umap_figure(
            embedding_2d=morgan_embedding,
            color_values=valid_first_digit,
            title=f"Synthetic Morgan UMAP at Epoch {self.current_epoch}",
            summary_lines=summary_lines + [f"metric={self.synthetic_morgan_umap_metric}"],
        )

        coati_reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.synthetic_umap_min_dist,
            metric=self.synthetic_coati_umap_metric,
            random_state=self.synthetic_seed,
        )
        coati_embedding = coati_reducer.fit_transform(coati_array)
        coati_fig = self._plot_umap_figure(
            embedding_2d=coati_embedding,
            color_values=valid_first_digit,
            title=f"Synthetic COATI Latent UMAP at Epoch {self.current_epoch}",
            summary_lines=summary_lines + [f"metric={self.synthetic_coati_umap_metric}"],
        )

        if self.logger is not None and hasattr(self.logger, "experiment"):
            experiment = self.logger.experiment
            if experiment is not None and hasattr(experiment, "log"):
                experiment.log(
                    {
                        "val/synthetic_morgan_umap": wandb.Image(morgan_fig),
                        "val/synthetic_coati_umap": wandb.Image(coati_fig),
                        "trainer/global_step": self.global_step,
                    }
                )

        plt.close(morgan_fig)
        plt.close(coati_fig)

    def on_validation_epoch_end(self) -> None:
        if self.enable_val_smiles_decode:
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

        self._log_synthetic_umap()

    @staticmethod
    def _build_identity_matrix(values: list[str], device: torch.device) -> torch.Tensor:
        matrix = [[left == right for right in values] for left in values]
        return torch.tensor(matrix, dtype=torch.bool, device=device)

    def _contrastive_dti_loss(
        self,
        similarity_logits: torch.Tensor,
        labels: torch.Tensor,
        smiles_list: list[str],
        sequence_list: list[str],
    ) -> torch.Tensor:
        device = similarity_logits.device
        batch_size = similarity_logits.shape[0]
        positive_mask = labels > 0.5

        if positive_mask.sum() == 0:
            return similarity_logits.new_zeros(())

        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        smiles_match = self._build_identity_matrix(smiles_list, device)
        sequence_match = self._build_identity_matrix(sequence_list, device)
        positive_column_mask = positive_mask.unsqueeze(0)
        known_positive_mask = (smiles_match | sequence_match) & positive_column_mask
        invalid_negative_mask = known_positive_mask & ~eye_mask

        row_logits = similarity_logits[positive_mask].masked_fill(invalid_negative_mask[positive_mask], -1e9)
        row_targets = torch.arange(batch_size, device=device)[positive_mask]
        loss_drug_to_protein = F.cross_entropy(row_logits, row_targets)

        col_logits = similarity_logits.T[positive_mask].masked_fill(invalid_negative_mask.T[positive_mask], -1e9)
        col_targets = torch.arange(batch_size, device=device)[positive_mask]
        loss_protein_to_drug = F.cross_entropy(col_logits, col_targets)

        return 0.5 * (loss_drug_to_protein + loss_protein_to_drug)

    def _forward_losses_metrics(
        self,
        batch,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        protein_embedding, morgan_fingerprint, smiles_target_embedding, label = batch[:4]
        smiles_list = batch[4] if len(batch) > 4 else []
        sequence_list = batch[5] if len(batch) > 5 else []

        outputs = self.model(
            protein_embedding=protein_embedding,
            morgan_fingerprint=morgan_fingerprint,
        )

        dti_scores = outputs["binding"]
        loss_dti = self._contrastive_dti_loss(
            similarity_logits=outputs["similarity_logits"],
            labels=label,
            smiles_list=smiles_list,
            sequence_list=sequence_list,
        )
        loss_jepa = self.jepa_loss(outputs["jepa_pred"], smiles_target_embedding)
        loss = loss_dti + loss_jepa

        labels = label.to(torch.int)
        auprc = self.auprc_by_stage[stage](dti_scores, labels)

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
        self._maybe_cache_fixed_protein_embedding(batch[0])
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
