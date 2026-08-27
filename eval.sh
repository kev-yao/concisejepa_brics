#!/bin/bash
#SBATCH --job-name=concisejepa_train
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/group/singhlab/user/cy244/projects/peptide_evals/concisejepa_train_%j.log
#SBATCH --error=/hpc/group/singhlab/user/cy244/projects/peptide_evals/concisejepa_train_%j.err



# Usage:
#   CKPT_PATH=/path/to/model.ckpt ./validation/run_cross_dataset_eval.sh
#
# Optional overrides:
#   EVAL_DATASET=peptide|bindingdb
#   TRAIN_DATASET=peptide|bindingdb
#   BATCH_SIZE=64
#   NUM_WORKERS=0

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu

PROJECT_ROOT="/hpc/home/cy244/projects/concisejepa"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="/tmp/${USER}/concisejepa_eval_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

PEPTIDE_DIR="/hpc/group/singhlab/user/cy244/projects/peptides/peptide_embeddings"
BINDINGDB_DIR="/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings"

dataset_dir() {
  case "$1" in
    peptide) echo "$PEPTIDE_DIR" ;;
    bindingdb) echo "$BINDINGDB_DIR" ;;
    *)
      echo "ERROR: Unknown dataset '$1'. Use peptide or bindingdb." >&2
      exit 1
      ;;
  esac
}

CKPT_PATH="/hpc/group/singhlab/user/cy244/projects/peptide_evals/runs/concisejepa_47040572/checkpoints/concisejepa-a75c1c13/last.ckpt"
if [[ -z "$CKPT_PATH" ]]; then
  echo "ERROR: set CKPT_PATH to a checkpoint file before running." >&2
  exit 1
fi

TRAIN_DATASET="${TRAIN_DATASET:-bindingdb}"
EVAL_DATASET="${EVAL_DATASET:-peptide}"

TRAIN_DIR="$(dataset_dir "$TRAIN_DATASET")"
EVAL_DIR="$(dataset_dir "$EVAL_DATASET")"

TRAIN_CSV="${TRAIN_CSV:-$TRAIN_DIR/train.csv}"
VAL_CSV="${VAL_CSV:-$EVAL_DIR/val.csv}"
TEST_CSV="${TEST_CSV:-$EVAL_DIR/test.csv}"

PROTEIN_EMB_PATH="${PROTEIN_EMB_PATH:-$EVAL_DIR/raygun_embeddings.pt}"
MORGAN_EMB_PATH="${MORGAN_EMB_PATH:-$EVAL_DIR/morgan_embeddings.pt}"
SMILES_EMB_PATH="${SMILES_EMB_PATH:-$EVAL_DIR/coati_embeddings.pt}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-/hpc/group/singhlab/user/cy244/projects/peptide_evals/validation}"

echo "CKPT_PATH=$CKPT_PATH"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "EVAL_DATASET=$EVAL_DATASET"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "TEST_CSV=$TEST_CSV"
echo "PROTEIN_EMB_PATH=$PROTEIN_EMB_PATH"
echo "MORGAN_EMB_PATH=$MORGAN_EMB_PATH"
echo "SMILES_EMB_PATH=$SMILES_EMB_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python "$PROJECT_ROOT/validation/cross_dataset_eval.py" \
  --config "$PROJECT_ROOT/configs/config.yaml" \
  --ckpt-path "$CKPT_PATH" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --test-csv "$TEST_CSV" \
  --protein-embeddings-path "$PROTEIN_EMB_PATH" \
  --morgan-embeddings-path "$MORGAN_EMB_PATH" \
  --smiles-embeddings-path "$SMILES_EMB_PATH" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --accelerator "$ACCELERATOR" \
  --devices "$DEVICES" \
  --precision "$PRECISION" \
  --output-dir "$OUTPUT_DIR"
