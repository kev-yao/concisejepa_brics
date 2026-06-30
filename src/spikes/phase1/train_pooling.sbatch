#!/bin/bash
# train_pooling.sbatch — Train one fragment pooling variant.
# Submit as a job array to run all 5 in parallel:
#   sbatch --array=0-4 train_pooling.sbatch
#
# Or individually:
#   POOLING_STRATEGY=mean sbatch train_pooling.sbatch
#
#SBATCH --job-name=frag_pool
#SBATCH --partition=singhlab-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/train_%A_%a.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/train_%A_%a.err

set -euo pipefail

mkdir -p /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study

# Map array index → pooling strategy (used when run as job array)
POOLING_STRATEGIES=(mean max weighted_sum cross_attention latent_query)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    POOLING_STRATEGY="${POOLING_STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
fi
# Fall back to env var or default
POOLING_STRATEGY="${POOLING_STRATEGY:-mean}"

echo "===== FRAGMENT POOL TRAINING ====="
echo "JOBID=$SLURM_JOB_ID  ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-none}"
echo "POOLING_STRATEGY=$POOLING_STRATEGY"
hostname
date
echo "=================================="

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

PROJECT_ROOT=/hpc/home/cy244/projects/concisejepa
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TORCH_HOME="/tmp/torch_${SLURM_JOB_ID:-manual}"
export TMPDIR="/tmp/${USER}/frag_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR"

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==============="

export POOLING_STRATEGY
export OUTPUT_DIR="/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study"
export MAX_EPOCHS="${MAX_EPOCHS:-30}"
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

echo "===== TRAINING DONE ====="
echo "Pooling: $POOLING_STRATEGY"
date
