#!/usr/bin/env python3
# phase1_fragment_train.py — Training entry point for fragment-level FSQ pooling ablation.
#
# Usage (env-var driven; set in sbatch script):
#   POOLING_STRATEGY=mean python fragment_train.py
#
# Or direct:
#   python fragment_train.py --pooling mean --output_dir /path/to/runs/mean

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# Ensure src/ is on PYTHONPATH
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent.parent  # src/
sys.path.insert(0, str(_SRC))

from concisejepa.models.drug_decoder import DrugEncoder

from .fragment_encoder import ConciseFragment, ConciseJEPAFragment
from .fragment_datamodule import FragmentDataModule
from .lit_fragment import LitFragment

# ---------------------------------------------------------------------------
# Defaults (match job.sh / config.yaml where possible)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    pooling="mean",
    drug_layers=[[32, 32, 32]],
    ligand_dim=2048,
    residue_dim=1280,
    drug_dim=128,
    proj_dim=256,
    nheads=32,
    activation="tanh",
    smiles_target_dim=256,
    jepa_hidden_dim=512,
    lr=1e-4,
    weight_decay=1e-2,
    max_epochs=30,
    batch_size=256,
    num_workers=0,
    seed=42,
    # Data paths — override via env vars or CLI flags
    train_csv="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/train.csv",
    val_csv="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/val.csv",
    test_csv="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/test.csv",
    protein_embeddings_path="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/raygun_embeddings.pt",
    fragment_fps_path="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/fragment_fps_r4_2048.pt",
    morgan_embeddings_path="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/morgan_embeddings.pt",
    smiles_embeddings_path="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/coati_embeddings.pt",
    output_dir="/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study",
)


def _env_override(d: dict) -> dict:
    """Override dict values with uppercase env vars if set."""
    result = dict(d)
    for key in result:
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            # coerce to same type
            orig = result[key]
            if isinstance(orig, int):
                result[key] = int(env_val)
            elif isinstance(orig, float):
                result[key] = float(env_val)
            else:
                result[key] = env_val
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Fragment-level FSQ pooling ablation training")
    for key, val in DEFAULTS.items():
        if isinstance(val, list):
            continue  # drug_layers is not a CLI flag
        p.add_argument(f"--{key}", type=type(val) if val is not None else str, default=None)
    return p.parse_args()


def build_cfg(args):
    """Merge DEFAULTS → env vars → CLI args into a SimpleNamespace."""
    cfg = _env_override(DEFAULTS)
    # CLI flags override env vars
    for key in DEFAULTS:
        if isinstance(DEFAULTS[key], list):
            continue
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    return SimpleNamespace(**cfg)


def build_model(cfg) -> ConciseJEPAFragment:
    backbone = ConciseFragment(
        drug_layers=cfg.drug_layers,
        pooling=cfg.pooling,
        ligand_dim=cfg.ligand_dim,
        residue_dim=cfg.residue_dim,
        drug_dim=cfg.drug_dim,
        proj_dim=cfg.proj_dim,
        nheads=cfg.nheads,
        activation=cfg.activation,
        drug_quantizer={"type": "fsq"},
    )
    return ConciseJEPAFragment(
        concise_fragment=backbone,
        smiles_target_dim=cfg.smiles_target_dim,
        jepa_hidden_dim=cfg.jepa_hidden_dim,
    )


def main():
    args = parse_args()
    cfg = build_cfg(args)

    pl.seed_everything(cfg.seed, workers=True)

    run_id = str(uuid.uuid4())[:8]
    run_name = f"fragment_{cfg.pooling}_{run_id}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Fragment pooling ablation: pooling={cfg.pooling} ===")
    print(f"Run dir: {run_dir}")

    model = build_model(cfg)
    lit = LitFragment(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    data_cfg = SimpleNamespace(
        train_csv=cfg.train_csv,
        val_csv=cfg.val_csv,
        test_csv=cfg.test_csv,
        protein_embeddings_path=cfg.protein_embeddings_path,
        fragment_fps_path=cfg.fragment_fps_path,
        morgan_embeddings_path=cfg.morgan_embeddings_path,
        smiles_embeddings_path=cfg.smiles_embeddings_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    dm = FragmentDataModule(data_cfg)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch:02d}-{val/dti_auprc:.4f}",
        monitor="val/dti_auprc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb, TQDMProgressBar()],
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm)

    best_ckpt = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    trainer.test(lit, datamodule=dm, ckpt_path=best_ckpt or None)

    # Write summary JSON for easy collection
    summary = {
        "pooling": cfg.pooling,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_checkpoint": best_ckpt,
    }
    for key, val in trainer.callback_metrics.items():
        if isinstance(val, torch.Tensor):
            summary[key] = float(val.item())
        elif isinstance(val, (int, float)):
            summary[key] = float(val)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
