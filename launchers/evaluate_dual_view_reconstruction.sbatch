#!/bin/bash
# Post-hoc COATI reconstruction evaluation for a dual-view BRICS run.
#
# Usage:
#   RUN_DIR=/path/to/run CONCISEJEPA_ENV=/path/to/env \
#     sbatch launchers/evaluate_dual_view_reconstruction.sbatch

#SBATCH --job-name=brics-recon
#SBATCH --partition=singhlab-gpu
#SBATCH --account=singhlab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

PROJECT_ROOT="${CONCISEJEPA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed dual-view run directory}"
N_MOLECULES="${N_MOLECULES:-100}"
SEED="${SEED:-42}"

if [[ -n "${CONCISEJEPA_ENV:-}" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate "$CONCISEJEPA_ENV"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp/${USER}/concisejepa_${SLURM_JOB_ID:-manual}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ -z "${CHECKPOINT:-}" ]]; then
    shopt -s nullglob
    checkpoints=("$RUN_DIR"/checkpoints/epoch=*.ckpt)
    if [[ ${#checkpoints[@]} -ne 1 ]]; then
        echo "Expected exactly one best checkpoint; found ${#checkpoints[@]}. Set CHECKPOINT explicitly." >&2
        exit 1
    fi
    CHECKPOINT="${checkpoints[0]}"
fi
OUTPUT="${OUTPUT:-$RUN_DIR/reconstruction_best_n${N_MOLECULES}_seed${SEED}.json}"

cd "$PROJECT_ROOT"
python scripts/evaluate_dual_view_reconstruction.py \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --n-molecules "$N_MOLECULES" \
    --seed "$SEED" \
    --device cuda:0 \
    --output "$OUTPUT"
