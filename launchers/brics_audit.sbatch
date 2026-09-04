#!/bin/bash
# Requires AUDIT_ROOT, CONCISEJEPA_AUDIT_DATA_ROOT and CONCISEJEPA_ENV.
# Each array element runs one preregistered arm over three matched seeds.
#SBATCH --job-name=brics-audit
#SBATCH --partition=singhlab-gpu
#SBATCH --account=singhlab
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -eo pipefail
PROJECT_ROOT="${CONCISEJEPA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
eval "$(micromamba shell hook --shell=bash)"
micromamba activate "${CONCISEJEPA_ENV:?Set CONCISEJEPA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:$PROJECT_ROOT:${PYTHONPATH:-}"
export TMPDIR="/tmp/${USER}/brics_audit_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/cache"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$PROJECT_ROOT"
arms=(fsq_joint continuous_joint fsq_jepa continuous_jepa fsq_dti continuous_dti)
python scripts/brics_audit_suite.py --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --output "${AUDIT_ROOT:?Set AUDIT_ROOT}" "$@"
