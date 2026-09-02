import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint


def _prepare_runtime_config(cfg: DictConfig) -> None:
    """Resolve a unique run directory before instantiating components."""
    run_id = str(uuid.uuid4())[:8]
    run_name = f"{cfg.run.name_prefix}-{run_id}"
    run_dir = Path(cfg.run.output_root).expanduser().resolve() / run_name

    with open_dict(cfg):
        cfg.runtime.run_id = run_id
        cfg.runtime.run_name = run_name
        cfg.runtime.run_dir = str(run_dir)
        cfg.runtime.checkpoint_dir = str(run_dir / "checkpoints")
        if bool(cfg.forward_capture.get("enabled", False)):
            cfg.forward_capture.output_path = str(run_dir / cfg.forward_capture.filename)


def _instantiate_callbacks(callback_cfg: DictConfig) -> tuple[list, dict[str, object]]:
    """Instantiate enabled callback configs and retain stable config names."""
    callbacks = []
    callbacks_by_name: dict[str, object] = {}
    for name, raw_cfg in callback_cfg.items():
        if raw_cfg is None or not bool(raw_cfg.get("enabled", True)):
            continue
        # Resolve interpolations while the callback is still attached to the
        # full config tree (where ``runtime.*`` lives), then detach it.
        component_cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
        with open_dict(component_cfg):
            component_cfg.pop("enabled", None)
        callback = instantiate(component_cfg)
        callbacks.append(callback)
        callbacks_by_name[str(name)] = callback
    return callbacks, callbacks_by_name


def _write_run_manifest(cfg: DictConfig) -> None:
    """Write the fully resolved experiment config beside the run artifacts."""
    run_dir = Path(cfg.runtime.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        git_commit = None
        git_dirty = None

    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "run_id": cfg.runtime.run_id,
                "run_name": cfg.runtime.run_name,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "config": resolved,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    pl.seed_everything(cfg.seed, workers=True)
    _prepare_runtime_config(cfg)
    _write_run_manifest(cfg)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    # ``config`` is the first positional parameter of hydra.instantiate itself,
    # so use an unambiguous task-constructor keyword for the global config.
    task = instantiate(cfg.task, experiment_config=cfg, model=model)
    callbacks, callbacks_by_name = _instantiate_callbacks(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(task, datamodule=datamodule)

    best_callback = callbacks_by_name.get("best_checkpoint")
    final_callback = callbacks_by_name.get("final_checkpoint")
    best_ckpt = best_callback.best_model_path if isinstance(best_callback, ModelCheckpoint) else ""
    final_ckpt = final_callback.best_model_path if isinstance(final_callback, ModelCheckpoint) else ""
    test_ckpt = final_ckpt or None
    if test_ckpt:
        print(f"Testing with FINAL-EPOCH checkpoint: {test_ckpt}")
        print(f"(best-by-monitor checkpoint, for reference only: {best_ckpt})")
    else:
        print("No saved final checkpoint found; testing with current in-memory weights.")
    trainer.test(task, datamodule=datamodule, ckpt_path=test_ckpt)
    print(f"Run directory: {cfg.runtime.run_dir}")
    print(f"Resolved config: {Path(cfg.runtime.run_dir) / 'resolved_config.yaml'}")


if __name__ == "__main__":
    main()
