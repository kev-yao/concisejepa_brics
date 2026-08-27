#!/bin/bash
#SBATCH --job-name=e3_f2r_attr
#SBATCH --partition=scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/e3_f2r_attr_%j.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/e3_f2r_attr_%j.err

set -e
set -o pipefail
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PROJECT_ROOT=/hpc/home/cy244/projects/concisejepa
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TMPDIR="/tmp/${USER}/e3_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo "===== E3: F2R Attribution ====="
echo "JOBID=$SLURM_JOB_ID"
hostname
date

cd "$PROJECT_ROOT"

python scripts/e3_f2r_attribution.py \
  --checkpoint /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/fragment_f2r_02241b1f/checkpoints/epoch=75-val/dti_auprc=0.5547.ckpt \
  --pairs scripts/matched_pairs.json \
  --clean_pairs scripts/matched_pairs_clean.json \
  --protein_embs /hpc/group/singhlab/user/cy244/projects/peptides/peptide_embeddings/raygun_embeddings.pt \
  --output scripts/contrastive_results_f2r.json

echo "===== DONE ====="
date
