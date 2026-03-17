import os
import uuid

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from concisejepa.datamodules import BindingDBDictDataModule
from concisejepa.lightning_modules import LitConciseJEPA


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    pl.seed_everything(cfg.seed, workers=True)

    run_id = str(uuid.uuid4())[:8]
    run_name = f"{cfg.wandb.run_name_prefix}-{run_id}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        mode=cfg.wandb.mode,
        save_dir=cfg.wandb.save_dir,
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

    lit_module = LitConciseJEPA(cfg)
    data_module = BindingDBDictDataModule(cfg.datamodule)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **cfg.trainer,
    )

    trainer.fit(lit_module, datamodule=data_module)
    trainer.test(lit_module, datamodule=data_module, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    main()
