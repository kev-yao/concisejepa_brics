#!/bin/bash
# train_long.sbatch — D1 diagnostic: 80-epoch runs for top models.
# Resolves whether fragment methods close the gap with whole_mol at longer training.
#
#   0: latent_query      (best fragment method overall)
#   1: multi_head_weighted_sum_k2  (highest AUROC)
#   2: whole_mol         (baseline that won AUPRC at 30 epochs)
#
# Submit: sbatch --array=0-2 train_long.sbatch
#
#SBATCH --job-name=frag_long
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/long_train_%A_%a.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/long_train_%A_%a.err

set -e
set -o pipefail

export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

mkdir -p /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study

POOLING_STRATEGIES=(latent_query multi_head_weighted_sum_k2 whole_mol)
POOLING_STRATEGY="${POOLING_STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

echo "===== D1 LONG TRAINING ====="
echo "JOBID=$SLURM_JOB_ID  ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "POOLING_STRATEGY=$POOLING_STRATEGY"
hostname
date
echo "============================"

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PROJECT_ROOT=/hpc/home/cy244/projects/concisejepa
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TORCH_HOME="/tmp/torch_${SLURM_JOB_ID}"
export TMPDIR="/tmp/${USER}/frag_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR"

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==============="

export POOLING_STRATEGY
export OUTPUT_DIR="/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study"
export MAX_EPOCHS=80
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
export SEED="${SEED:-42}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

echo "Max epochs: $MAX_EPOCHS  Batch size: $BATCH_SIZE  LR: $LR"

python -m spikes.phase1.fragment_train \
    --pooling "$POOLING_STRATEGY" \
    --output_dir "$OUTPUT_DIR" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS"

echo "===== DONE ====="
echo "Pooling: $POOLING_STRATEGY  Epochs: $MAX_EPOCHS"
date
