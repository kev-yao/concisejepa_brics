#!/bin/bash
#SBATCH --job-name=finetune_cliff
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/finetune_cliff_%j.log

set -eo pipefail
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PROJECT=/hpc/home/cy244/projects/concisejepa
export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"

CKPT=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/fragment_weighted_sum_60f394d3/checkpoints/epoch=66-val/dti_auprc=0.5732.ckpt
PAIRS=$PROJECT/scripts/matched_pairs.json
PROTEIN_EMB=/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/raygun_embeddings.pt
OUTPUT_DIR=/hpc/group/singhlab/user/cy244/projects/peptide_evals/finetune_cliff_ws/

echo "===== FINETUNE CLIFF ====="
date
echo "Checkpoint: $CKPT"
echo "Pairs: $PAIRS"

cd $PROJECT
python scripts/finetune_cliff.py \
    --checkpoint "$CKPT" \
    --pooling weighted_sum \
    --pairs "$PAIRS" \
    --protein_emb "$PROTEIN_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 30 \
    --lr 1e-4 \
    --margin 0.1 \
    --batch_size 16 \
    --val_frac 0.15

echo "===== DONE ====="
date
