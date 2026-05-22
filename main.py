import json
import os
import uuid
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from concisejepa.datamodules import BindingDBDictDataModule
from concisejepa.lightning_modules import LitConciseJEPA
from concisejepa.evals.FSQ_monitor import FSQMonitorCallback


class EpochMetricsWriter(Callback):
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    pl.seed_everything(cfg.seed, workers=True)

    run_id = str(uuid.uuid4())[:8]
    run_name = f"{cfg.logging.run_name_prefix}-{run_id}"
    metrics_dir = os.path.join(cfg.logging.save_dir, run_name)
    forward_capture_cfg = getattr(cfg, "forward_capture", {})
    if bool(forward_capture_cfg.get("enabled", False)):
        with open_dict(cfg):
            cfg.forward_capture.output_path = os.path.join(
                metrics_dir,
                forward_capture_cfg.get("filename", "train_forward_logits_last.json"),
            )
    csv_logger = CSVLogger(
        save_dir=cfg.logging.save_dir,
        name=run_name,
    )

    checkpoint_dir = os.path.join(cfg.checkpoint.dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
    )
    progress_bar_callback = TQDMProgressBar()
    metrics_callback = EpochMetricsWriter(output_dir=metrics_dir)
    
    fsq_monitor_dir = os.path.join(metrics_dir, "fsq_logs")
    fsq_monitor_callback = FSQMonitorCallback(
        output_dir=fsq_monitor_dir,
        num_codes=32,
        monitor_train=False,
    )

    lit_module = LitConciseJEPA(cfg)
    data_module = BindingDBDictDataModule(cfg.datamodule)

    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=[checkpoint_callback, progress_bar_callback, metrics_callback, fsq_monitor_callback],
        **cfg.trainer,
    )

    trainer.fit(lit_module, datamodule=data_module)

    best_ckpt = checkpoint_callback.best_model_path
    last_ckpt = checkpoint_callback.last_model_path
    test_ckpt = best_ckpt or last_ckpt or None
    if test_ckpt:
        print(f"Testing with checkpoint: {test_ckpt}")
    else:
        print("No saved best/last checkpoint found; testing with current in-memory weights.")
    trainer.test(lit_module, datamodule=data_module, ckpt_path=test_ckpt)
    print(f"Epoch metrics JSONL: {os.path.join(metrics_dir, 'epoch_metrics.jsonl')}")
    print(f"Final metrics JSON: {os.path.join(metrics_dir, 'final_metrics.json')}")
    print(f"Lightning metrics CSV: {os.path.join(metrics_dir, 'version_0', 'metrics.csv')}")
    print(f"FSQ monitoring logs: {fsq_monitor_dir}")
    if bool(forward_capture_cfg.get("enabled", False)):
        print(f"Forward capture JSON: {cfg.forward_capture.output_path}")


if __name__ == "__main__":
    main()
