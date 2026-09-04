#!/bin/bash
# Submit with --dependency=afterany:<training-array-id>.
#SBATCH --job-name=brics-audit-report
#SBATCH --partition=common
#SBATCH --account=singhlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -eo pipefail
PROJECT_ROOT="${CONCISEJEPA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
eval "$(micromamba shell hook --shell=bash)"
micromamba activate "${CONCISEJEPA_ENV:?Set CONCISEJEPA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/${USER}/brics_report_${SLURM_JOB_ID}/matplotlib"
export XDG_CACHE_HOME="/tmp/${USER}/brics_report_${SLURM_JOB_ID}/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$PROJECT_ROOT"
python scripts/summarize_brics_audit.py "${AUDIT_ROOT:?Set AUDIT_ROOT}"
python scripts/brics_conditional_metrics.py "$AUDIT_ROOT"
python scripts/plot_brics_audit.py "$AUDIT_ROOT"
