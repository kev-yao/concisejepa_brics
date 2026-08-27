#!/bin/bash
# train_f2r.sbatch — E2: 80-epoch training for F2R (Fragment-to-Residue) pooler.
#
# F2R is protein-conditional: fragment pooling weights are computed by cross-attending
# each fragment against protein residues. Unlike WeightedSumPool, attribution is
# protein-specific — the same fragment gets different weights with different proteins.
#
# Submit: sbatch src/spikes/phase1/train_f2r.sbatch
#
#SBATCH --job-name=frag_f2r
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/f2r_train_%j.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/f2r_train_%j.err

set -e
set -o pipefail
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

mkdir -p /hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study

echo "===== F2R TRAINING ====="
echo "JOBID=$SLURM_JOB_ID"
hostname
date
echo "========================"

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

cd "$PROJECT_ROOT"

export OUTPUT_DIR="/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study"
export MAX_EPOCHS=80
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
export SEED="${SEED:-42}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

echo "Pooling: f2r  Max epochs: $MAX_EPOCHS  Batch size: $BATCH_SIZE  LR: $LR"

python -m spikes.phase1.fragment_train \
    --pooling f2r \
    --output_dir "$OUTPUT_DIR" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS"

echo "===== DONE ====="
echo "Pooling: f2r  Epochs: $MAX_EPOCHS"
date
