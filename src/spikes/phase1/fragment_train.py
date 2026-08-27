#!/usr/bin/env python3
"""fragment_train.py — training entry point for the fragment pipeline (ConciseFragment /
ConciseJEPAFragment / ConciseJEPAFragmentXAttn).

Config is CLI-args-and-env-vars driven (env vars, if set, override DEFAULTS below; CLI flags
override both — see build_cfg()). Every sbatch script in this project sets the data-path env
vars explicitly, so DEFAULTS below is really just a fallback for ad-hoc/interactive runs.

Usage (env-var driven; typically set in an sbatch script):
    POOLING_STRATEGY=mean python fragment_train.py

Or direct, with explicit flags:
    python fragment_train.py --pooling max --fsq_levels 32,32,32 --predictor_type mlp \
        --output_dir /path/to/runs/max

Key flags (see docs/PROJECT_HANDOFF.md §4 for the full table): --pooling
{weighted_sum,f2r,max,whole_mol,...}, --fsq_levels (comma-separated FSQ factors — "32,32,32"
hierarchical vs. "32768"/"1500" flat), --jepa_loss_type {mse,contrastive}, --predictor_type
{mlp,xattn}, --chem_supervision_weight / --group_supervision_weight (auxiliary losses, see
lit_fragment.py).

Writes: run_dir/checkpoints/final.ckpt (the true-final-epoch checkpoint — see the comment above
final_ckpt_cb below for why this exists as a separate callback) + run_dir/summary.json.
"""

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

from .fragment_encoder import ConciseFragment, ConciseJEPAFragment
from .fragment_xattn import ConciseJEPAFragmentXAttn
from .fragment_datamodule import FragmentDataModule
from .lit_fragment import LitFragment

# ---------------------------------------------------------------------------
# Defaults (match job.sh / config.yaml where possible)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    pooling="mean",
    fsq_levels="32,32,32",   # comma-separated FSQ factor levels, e.g. "32,32,32" (hierarchical,
                              # 3 factors, 32768 codes) or "32768" (flat, 1 factor, same capacity)
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
    chem_supervision_weight=0.0,   # >0 enables auxiliary chem-property supervision (see LitFragment)
    group_supervision_weight=0.0,  # >0 enables auxiliary functional-group supervision (see LitFragment)
    jepa_loss_type="mse",           # "mse" (default) or "contrastive" (see LitFragment)
    predictor_type="mlp",           # "mlp" (default, ConciseJEPAFragment) or "xattn"
                                     # (ConciseJEPAFragmentXAttn — cross-attention over
                                     # unpooled fragment tokens, see fragment_xattn.py)
    predictor_dim=256,               # xattn predictor only
    predictor_depth=3,                # xattn predictor only
    predictor_heads=8,                 # xattn predictor only
    predictor_mlp_ratio=2.0,            # xattn predictor only
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
# NOTE: the *_path/*_csv defaults above point at an OLDER, now-stale data directory
# (count_combined_embeddings/, missing train/val/test.csv and raygun_embeddings.pt as of this
# writing). Every real training run in this project overrode these via env vars pointing at the
# current canonical directory instead: /hpc/group/singhlab/user/cy244/projects/peptides/
# BindingDB_embeddings/ (see docs/PROJECT_HANDOFF.md §3). Left as-is here rather than silently
# changed, since these are only fallbacks — but don't run this script bare without overriding
# them, or you'll train against the wrong (smaller, older) dataset.


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
    fsq_levels = [int(x) for x in str(cfg.fsq_levels).split(",")]
    backbone = ConciseFragment(
        drug_layers=[fsq_levels],
        pooling=cfg.pooling,
        ligand_dim=cfg.ligand_dim,
        residue_dim=cfg.residue_dim,
        drug_dim=cfg.drug_dim,
        proj_dim=cfg.proj_dim,
        nheads=cfg.nheads,
        activation=cfg.activation,
        drug_quantizer={"type": "fsq"},
    )
    if cfg.predictor_type == "xattn":
        return ConciseJEPAFragmentXAttn(
            concise_fragment=backbone,
            smiles_target_dim=cfg.smiles_target_dim,
            predictor_dim=cfg.predictor_dim,
            predictor_depth=cfg.predictor_depth,
            predictor_heads=cfg.predictor_heads,
            predictor_mlp_ratio=cfg.predictor_mlp_ratio,
        )
    if cfg.predictor_type != "mlp":
        raise ValueError(f"Unknown predictor_type: {cfg.predictor_type!r}")
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
    lit = LitFragment(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay,
        chem_supervision_weight=cfg.chem_supervision_weight,
        group_supervision_weight=cfg.group_supervision_weight,
        jepa_loss_type=cfg.jepa_loss_type,
    )

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
        use_whole_mol=(cfg.pooling == "whole_mol"),
    )
    dm = FragmentDataModule(data_cfg)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    # Best-by-AUPRC checkpoint, kept for reference only (not used for eval/downstream by
    # default anymore). auto_insert_metric_name=False avoids a real bug: Lightning's default
    # filename formatting prepends the raw metric NAME before its value, and since the metric
    # key itself is "val/dti_auprc" (contains a literal "/"), that slash landed in the
    # filename and silently created a spurious "epoch=NN-val/" subdirectory. This also
    # appears to have broken save_last's independence from the monitor (last.ckpt ended up
    # frozen at whatever epoch AUPRC last improved, never updating past that point) — rather
    # than rely on save_last at all, final-epoch weights are now guaranteed by a fully
    # separate, monitor-free callback below.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch={epoch:02d}-auprc={val/dti_auprc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/dti_auprc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    # Guaranteed true-final-epoch checkpoint, independent of any monitored metric. With
    # monitor=None every epoch is treated as "the one to keep" (save_top_k=1 just overwrites),
    # so whatever remains when training finishes is exactly the last epoch's weights — this is
    # now the canonical checkpoint used for testing and all downstream evaluation.
    final_ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="final",
        monitor=None,
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb, final_ckpt_cb, TQDMProgressBar()],
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm)

    best_ckpt = checkpoint_cb.best_model_path
    final_ckpt = final_ckpt_cb.best_model_path
    trainer.test(lit, datamodule=dm, ckpt_path=final_ckpt or None)

    # Write summary JSON for easy collection
    summary = {
        "pooling": cfg.pooling,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_checkpoint": final_ckpt,
        "best_auprc_checkpoint": best_ckpt,
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
