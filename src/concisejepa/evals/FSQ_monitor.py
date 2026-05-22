import torch
import pandas as pd
import numpy as np
from einops import rearrange
from collections import Counter
from pathlib import Path
import pytorch_lightning as pl


class FSQMonitor:
    def __init__(self, out_dir="fsq_logs", num_codes=32):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_codes = num_codes
        self.reset()

    def reset(self):
        self.step_values = {}  # {(layer_idx, step_name): list of all values across batches}
        self.code_batches = []

    @torch.no_grad()
    def summarize(self, name, tensor, epoch, batch_idx, layer_idx):
        x = tensor.detach().float().cpu().flatten()
        
        # Store raw values for aggregation
        key = (layer_idx, name)
        if key not in self.step_values:
            self.step_values[key] = []
        self.step_values[key].extend(x.numpy().tolist())


    @torch.no_grad()
    def inspect_residual_fsq(self, layer, x, epoch, batch_idx, layer_idx):
        self.summarize("input_pre_ln", x, epoch, batch_idx, layer_idx)

        x = layer.ln1(x)
        self.summarize("after_ln1", x, epoch, batch_idx, layer_idx)

        h = layer.in_proj[0](x)
        self.summarize("after_in_linear1", h, epoch, batch_idx, layer_idx)

        h = layer.in_proj[1](h)
        self.summarize("after_gelu", h, epoch, batch_idx, layer_idx)

        projected = layer.in_proj[2](h)
        self.summarize("after_in_linear2_projected", projected, epoch, batch_idx, layer_idx)

        fsq = layer.fsq
        z = rearrange(projected, "b n (c d) -> b n c d", c=fsq.num_codebooks)

        half_l = (fsq._levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(fsq._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()

        bounded = (z + shift).tanh() * half_l - offset
        self.summarize("after_bound", bounded, epoch, batch_idx, layer_idx)

        rounded = bounded.round()
        self.summarize("after_round", rounded, epoch, batch_idx, layer_idx)

        rounded_ste = z + (rounded - z).detach()
        self.summarize("after_round_ste", rounded_ste, epoch, batch_idx, layer_idx)

        half_width = fsq._levels // 2
        quantized_points = rounded_ste / half_width
        self.summarize("after_normalize_quantized_points", quantized_points, epoch, batch_idx, layer_idx)

        scaled_shifted = quantized_points * half_width + half_width
        self.summarize("after_scale_shift_for_index", scaled_shifted, epoch, batch_idx, layer_idx)

        # Track individual code dimensions (each 0-31) instead of product index
        # scaled_shifted shape: [batch, seq, num_codebooks=1, codebook_dim=3]
        for dim_idx in range(scaled_shifted.shape[-1]):
            dim_values = scaled_shifted[..., dim_idx]
            self.summarize(f"indices_dim_{dim_idx}", dim_values, epoch, batch_idx, layer_idx)
        
        codes = rearrange(quantized_points, "b n c d -> b n (c d)")
        self.summarize("final_quantized_points", codes, epoch, batch_idx, layer_idx)
        
        # Extract code values (each dimension 0-31)
        # scaled_shifted shape: [batch, seq, num_codebooks=1, codebook_dim=3]
        # Take mean across sequence to get representative codes per batch
        batch_codes = scaled_shifted.mean(dim=1).squeeze(1).to(torch.int32)  # [batch, 3]

        o = layer.out_proj[0](codes)
        self.summarize("after_out_linear1", o, epoch, batch_idx, layer_idx)

        o = layer.out_proj[1](o)
        self.summarize("after_out_gelu", o, epoch, batch_idx, layer_idx)

        o = layer.out_proj[2](o)
        self.summarize("after_out_linear2", o, epoch, batch_idx, layer_idx)

        o = layer.ln2(o)
        self.summarize("after_ln2", o, epoch, batch_idx, layer_idx)

        activated = layer.activation(o)
        self.summarize("after_tanh", activated, epoch, batch_idx, layer_idx)

        residual = x - activated
        self.summarize("residual", residual, epoch, batch_idx, layer_idx)

        return {
            "indices": batch_codes.unsqueeze(1),
            "quantized": activated,
            "points": codes,
            "residual": residual,
        }

    @torch.no_grad()
    def inspect_drug_encoder(self, drug_encoder, morgan, epoch, batch_idx):
        x = drug_encoder.pre_transform(morgan)

        codebookids = []
        residual = x
        
        num_layers = len(drug_encoder.residualfsqs)
        if batch_idx == 0 and epoch == 0:
            print(f"[FSQ Monitor] Processing {num_layers} ResidualFSQ layers")

        for i, layer in enumerate(drug_encoder.residualfsqs):
            res = self.inspect_residual_fsq(
                layer=layer,
                x=residual,
                epoch=epoch,
                batch_idx=batch_idx,
                layer_idx=i + 1,
            )
            codebookids.append(res["indices"])
            residual = res["residual"]

        codes = torch.cat(codebookids, dim=1).squeeze(-1)
        self.code_batches.append(codes.detach().cpu())

        return codes

    def save_epoch(self, epoch):
        # Aggregate step statistics across all batches
        step_stats = []
        for (layer_idx, step_name), values in self.step_values.items():
            values_array = np.array(values)
            step_stats.append({
                "epoch": epoch,
                "layer": layer_idx,
                "step": step_name,
                "min": float(np.min(values_array)),
                "p01": float(np.percentile(values_array, 1)),
                "p05": float(np.percentile(values_array, 5)),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "p50": float(np.percentile(values_array, 50)),
                "p95": float(np.percentile(values_array, 95)),
                "p99": float(np.percentile(values_array, 99)),
                "max": float(np.max(values_array)),
            })
        
        # Debug: print layers found
        layers_found = set(layer_idx for layer_idx, _ in self.step_values.keys())
        print(f"[FSQ Monitor] Epoch {epoch}: Found layers {sorted(layers_found)}, Total steps: {len(step_stats)}")
        
        stats_df = pd.DataFrame(step_stats)
        # Reorder columns: epoch, layer, step, min, max, mean, then rest
        stats_df = stats_df[["epoch", "layer", "step", "min", "max", "mean", "p01", "p05", "std", "p50", "p95", "p99"]]
        stats_path = self.out_dir / f"fsq_step_ranges_epoch_{epoch}.csv"
        stats_df.to_csv(stats_path, index=False)

        codes = torch.cat(self.code_batches, dim=0).numpy()
        
        # Handle multi-dimensional codes: [batch, num_layers, seq_len] -> [batch, num_layers]
        # Take the most common code per layer across sequence dimension
        if codes.ndim > 2:
            batch_size, num_layers = codes.shape[0], codes.shape[1]
            codes_2d = np.zeros((batch_size, num_layers), dtype=codes.dtype)
            for b in range(batch_size):
                for l in range(num_layers):
                    # Use the most common code for this batch/layer across sequence
                    seq_codes = codes[b, l, :].flatten().astype(int)
                    codes_2d[b, l] = np.bincount(seq_codes).argmax()
            codes = codes_2d
        
        np.save(self.out_dir / f"fsq_codes_epoch_{epoch}.npy", codes)

        dist_rows = []
        for layer_idx in range(codes.shape[1]):
            layer_codes = codes[:, layer_idx].flatten().astype(int)
            counts = np.bincount(layer_codes, minlength=self.num_codes)
            freqs = counts / counts.sum()

            for code_id in range(self.num_codes):
                dist_rows.append({
                    "epoch": epoch,
                    "layer": layer_idx + 1,
                    "code": code_id,
                    "count": int(counts[code_id]),
                    "frequency": float(freqs[code_id]),
                })

        dist_df = pd.DataFrame(dist_rows)
        dist_df.to_csv(self.out_dir / f"fsq_code_distribution_epoch_{epoch}.csv", index=False)

        joint_codes = [".".join(map(str, row)) for row in codes.astype(int)]
        joint_counts = Counter(joint_codes)

        joint_df = pd.DataFrame([
            {"epoch": epoch, "hierarchical_code": k, "count": v}
            for k, v in joint_counts.items()
        ]).sort_values("count", ascending=False)

        joint_df["frequency"] = joint_df["count"] / joint_df["count"].sum()
        joint_df.to_csv(self.out_dir / f"fsq_joint_distribution_epoch_{epoch}.csv", index=False)

        self.reset()


class FSQMonitorCallback(pl.Callback):
    """PyTorch Lightning callback that monitors FSQ layers during training."""

    def __init__(self, output_dir: str = "fsq_logs", num_codes: int = 32, monitor_train: bool = False):
        """
        Args:
            output_dir: Directory to save FSQ monitoring outputs
            num_codes: Number of codes in the codebook
            monitor_train: If True, monitor training batches; if False, only validation
        """
        super().__init__()
        self.fsq_monitor = FSQMonitor(out_dir=output_dir, num_codes=num_codes)
        self.monitor_train = monitor_train
        self.num_codes = num_codes

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.monitor_train:
            self.fsq_monitor.reset()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.monitor_train:
            self.fsq_monitor.reset()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.monitor_train:
            return

        self._process_batch(pl_module, batch, batch_idx, trainer.current_epoch)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self.monitor_train:
            return

        self._process_batch(pl_module, batch, batch_idx, trainer.current_epoch)

    def _process_batch(self, pl_module: pl.LightningModule, batch, batch_idx: int, epoch: int) -> None:
        """Extract morgan fingerprint and pass through FSQ monitor."""
        with torch.no_grad():
            morgan_fingerprint = batch[1]
            self.fsq_monitor.inspect_drug_encoder(
                drug_encoder=pl_module.model.concise.d_encoder,
                morgan=morgan_fingerprint,
                epoch=epoch,
                batch_idx=batch_idx,
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.monitor_train:
            epoch = trainer.current_epoch
            print(f"\n[FSQ Monitor] Saving epoch {epoch} statistics...")
            self.fsq_monitor.save_epoch(epoch)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.monitor_train:
            epoch = trainer.current_epoch
            print(f"\n[FSQ Monitor] Saving epoch {epoch} statistics...")
            self.fsq_monitor.save_epoch(epoch)