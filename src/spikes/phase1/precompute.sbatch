#!/bin/bash
# precompute.sbatch — Precompute BRICS fragment Morgan FPs for all SMILES in BindingDB split.
# CPU-only job; takes ~20-40 min.
#
# Submit: sbatch precompute.sbatch
#
#SBATCH --job-name=frag_precompute
#SBATCH --partition=singhlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/precompute_%j.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/precompute_%j.err

set -euo pipefail

mkdir -p /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study

echo "===== PRECOMPUTE METADATA ====="
echo "JOBID=$SLURM_JOB_ID"
hostname
date
echo "================================"

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PROJECT_ROOT=/hpc/home/cy244/projects/concisejepa
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

TRAIN_CSV="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/train.csv"
VAL_CSV="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/val.csv"
TEST_CSV="/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/test.csv"
OUTPUT_PATH="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/fragment_fps_r4_2048.pt"

echo "Train CSV:  $TRAIN_CSV"
echo "Output:     $OUTPUT_PATH"

python - <<'PYEOF'
import sys, os
import pandas as pd
from pathlib import Path
from concisejepa.datamodules.dataloader import _collect_unique_sequences_and_smiles
from spikes.phase1.fragment_datamodule import build_fragment_fp_cache

TRAIN_CSV = os.environ.get("TRAIN_CSV", "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/train.csv")
VAL_CSV   = os.environ.get("VAL_CSV",   "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/val.csv")
TEST_CSV  = os.environ.get("TEST_CSV",  "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/bindingdb/test.csv")
OUTPUT    = os.environ.get("OUTPUT_PATH", "/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/fragment_fps_r4_2048.pt")

csv_paths = [TRAIN_CSV, VAL_CSV, TEST_CSV]
_, smiles_list = _collect_unique_sequences_and_smiles(
    csv_paths=csv_paths,
    sequence_col="Target Sequence",
    smiles_col="SMILES",
)
print(f"Unique SMILES to process: {len(smiles_list)}")

build_fragment_fp_cache(
    smiles_values=smiles_list,
    output_path=Path(OUTPUT),
    fingerprint_kind="ecfp-count:4",
    fingerprint_length=2048,
)
print("Done.")
PYEOF

echo "===== PRECOMPUTE DONE ====="
date
ls -lh "$OUTPUT_PATH" 2>/dev/null || echo "Output file not found?"
