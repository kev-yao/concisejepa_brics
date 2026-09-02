import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class EpochMetricsWriter(Callback):
    """Persist resolved epoch and final Lightning metrics as JSON."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.epoch_metrics_jsonl_path = self.output_dir / "epoch_metrics.jsonl"
        self.epoch_metrics_json_path = self.output_dir / "epoch_metrics.json"
        self.final_metrics_path = self.output_dir / "final_metrics.json"
        self._epoch_records: list[dict[str, float | list[float] | str]] = []
        self._last_written_epoch: int | None = None

    @staticmethod
    def _serialize_metric_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, (int, float)):
            return float(value)
        return value

    def _serialized_callback_metrics(self, trainer: pl.Trainer) -> dict[str, float | list[float] | str]:
        serialized: dict[str, float | list[float] | str] = {}
        for key, value in trainer.callback_metrics.items():
            metric_value = self._serialize_metric_value(value)
            if isinstance(metric_value, (float, list, str)):
                serialized[str(key)] = metric_value
        return serialized

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if self._last_written_epoch == epoch:
            return
        self._last_written_epoch = epoch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"epoch": epoch}
        metrics.update(self._serialized_callback_metrics(trainer))
        self._epoch_records.append(metrics)
        with self.epoch_metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        with self.epoch_metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(self._epoch_records, f, indent=2, sort_keys=True)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_metrics = self._serialized_callback_metrics(trainer)
        final_metrics["epoch"] = int(trainer.current_epoch)
        with self.final_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, sort_keys=True)
