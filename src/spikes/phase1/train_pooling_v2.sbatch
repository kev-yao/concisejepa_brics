#!/bin/bash
# train_pooling_v2.sbatch — Phase 2 pooling ablation (8 experiments).
# Builds on Phase 1 results (latent_query + weighted_sum were best).
#
# Experiments:
#   0: latent_query_q1          — n_queries=1 (single learned query / CLS-token style)
#   1: latent_query_q2          — n_queries=2
#   2: latent_query_q8          — n_queries=8
#   3: whole_mol                — whole-molecule FP baseline (no fragmentation)
#   4: pre_attn_latent_query    — fragment self-attention then latent query (n_queries=4)
#   5: mlp_weighted_sum         — 2-layer MLP scorer instead of linear
#   6: multi_head_weighted_sum_k2 — 2 independent weighted sums concatenated
#   7: multi_head_weighted_sum_k4 — 4 independent weighted sums concatenated
#
# Submit: sbatch --array=0-7 train_pooling_v2.sbatch
#
#SBATCH --job-name=frag_pool_v2
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/v2_train_%A_%a.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/v2_train_%A_%a.err

set -e
set -o pipefail

export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

mkdir -p /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study

POOLING_STRATEGIES=(
    latent_query_q1
    latent_query_q2
    latent_query_q8
    whole_mol
    pre_attn_latent_query
    mlp_weighted_sum
    multi_head_weighted_sum_k2
    multi_head_weighted_sum_k4
)
POOLING_STRATEGY="${POOLING_STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

echo "===== FRAGMENT POOL TRAINING v2 ====="
echo "JOBID=$SLURM_JOB_ID  ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "POOLING_STRATEGY=$POOLING_STRATEGY"
hostname
date
echo "======================================"

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
