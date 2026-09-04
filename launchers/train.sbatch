#!/bin/bash
# Generic collaborator-safe launcher.
#
# Examples:
#   CONCISEJEPA_ENV=/path/to/env sbatch launchers/train.sbatch
#   EXPERIMENT=diveq CONCISEJEPA_ENV=/path/to/env sbatch launchers/train.sbatch
#   sbatch --partition=h200-hp --account=singhlab_h200_r \
#     --qos=singhlab_h200 --gres=gpu:h200:1 launchers/train.sbatch
#
# Dataset and output locations are supplied through CONCISEJEPA_* environment
# variables; no collaborator-owned path is baked into this launcher.

#SBATCH --job-name=concisejepa
#SBATCH --partition=singhlab-gpu
#SBATCH --account=singhlab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONCISEJEPA_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$CONCISEJEPA_PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
EXPERIMENT="${EXPERIMENT:-baseline}"
RUN_ROOT="${CONCISEJEPA_RUN_ROOT:-$PROJECT_ROOT/scratch/runs}"

if [[ -n "${CONCISEJEPA_ENV:-}" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate "$CONCISEJEPA_ENV"
fi

# Some compute nodes prefer the system libstdc++ unless the activated
# environment is placed first explicitly. SciPy and PyTorch in the project
# environment require the newer CXXABI provided there.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if ! python -c "import hydra, torch, pytorch_lightning"; then
    echo "ERROR: activate the project environment or set CONCISEJEPA_ENV." >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CONCISEJEPA_RUN_ROOT="$RUN_ROOT"
export TMPDIR="${TMPDIR:-/tmp/${USER}/concisejepa_${SLURM_JOB_ID:-manual}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$RUN_ROOT" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "$PROJECT_ROOT"

echo "experiment=$EXPERIMENT"
echo "project_root=$PROJECT_ROOT"
echo "run_root=$RUN_ROOT"
echo "git_commit=$(git rev-parse HEAD)"

python main.py \
    experiment="$EXPERIMENT" \
    hydra.run.dir="$RUN_ROOT/hydra/${SLURM_JOB_ID:-manual}"
