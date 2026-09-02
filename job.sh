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

set -e
set -o pipefail

export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
set +u

echo "===== JOB METADATA ====="
echo "JOBID=$SLURM_JOB_ID"
echo "SUBMIT_DIR=$SLURM_SUBMIT_DIR"
pwd
hostname
date
echo "========================"

module purge
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT=/hpc/home/cy244/projects/concisejepa
OUTPUT_ROOT=/hpc/group/singhlab/user/cy244/projects/peptide_evals
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export TORCH_HOME="/tmp/torch_${SLURM_JOB_ID:-manual}"
export TMPDIR="/tmp/${USER}/concisejepa_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

echo "===== ENV ====="
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
which python
python --version
echo "==============="

echo "===== C++ / sqlite sanity ====="
python -c "import sqlite3; print('sqlite3 ok')"
echo "==============================="

echo "===== GPU ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi
GPU_COMPATIBLE=$(python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    try:
        torch.cuda.init()
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        print("device 0:", name)
        print("device capability:", f"{major}.{minor}")
        # This environment's Torch build supports sm_70+; P100 is sm_60.
        if major >= 7:
            print("GPU_COMPATIBLE=1")
        else:
            print("GPU_COMPATIBLE=0")
    except Exception as exc:
        print("cuda init failed:", exc)
        print("GPU_COMPATIBLE=0")
else:
    print("GPU_COMPATIBLE=0")
PY
)
echo "$GPU_COMPATIBLE"
if echo "$GPU_COMPATIBLE" | grep -q "GPU_COMPATIBLE=1"; then
  TRAINER_ACCELERATOR="gpu"
  TRAINER_DEVICES=1
else
  echo "ERROR: GPU is unavailable/incompatible for this Torch build (requires compute capability >= 7.0)." >&2
  echo "ERROR: This run is configured as GPU-only and will stop here." >&2
  exit 1
fi
echo "=============="

echo "===== RUN TRAINING ====="
RUN_ROOT="$OUTPUT_ROOT/runs"
RUN_DIR="$RUN_ROOT/concisejepa_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"

CONFIG_PATH="$PROJECT_ROOT/configs/config.yaml"

#Dataset toggles (peptide/bindingdb) are disabled.
# TRAIN_DATASET="${TRAIN_DATASET:-peptide}"
# FINAL_EVAL_DATASET="${FINAL_EVAL_DATASET:-peptide}"
# PEPTIDE_DIR="/hpc/group/singhlab/user/cy244/projects/peptides/peptide_embeddings"
# BINDINGDB_DIR="/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings"
# dataset_dir() {
#   case "$1" in
#     peptide) echo "$PEPTIDE_DIR" ;;
#     bindingdb) echo "$BINDINGDB_DIR" ;;
#     *)
#       echo "ERROR: Unknown dataset '$1'. Use peptide or bindingdb." >&2
#       exit 1
#       ;;
#   esac
# }
# TRAIN_DIR="$(dataset_dir "$TRAIN_DATASET")"
# EVAL_DIR="$(dataset_dir "$FINAL_EVAL_DATASET")"
# TRAIN_CSV="${TRAIN_CSV:-$TRAIN_DIR/train.csv}"
# VAL_CSV="${VAL_CSV:-$EVAL_DIR/val.csv}"
# TEST_CSV="${TEST_CSV:-$EVAL_DIR/test.csv}"
# PROTEIN_EMB_PATH="${PROTEIN_EMB_PATH:-$TRAIN_DIR/raygun_embeddings.pt}"
# MORGAN_EMB_PATH="${MORGAN_EMB_PATH:-$TRAIN_DIR/morgan_embeddings.pt}"
# SMILES_EMB_PATH="${SMILES_EMB_PATH:-$TRAIN_DIR/coati_embeddings.pt}"

 # Combined embeddings setup (disabled)
 COMBINED_DIR="/hpc/group/singhlab/user/cy244/projects/peptides/combined_embeddings"
 TRAIN_CSV="/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings/train.csv"
 VAL_CSV="/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings/val.csv"
 TEST_CSV="/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings/test.csv"
 PROTEIN_EMB_PATH="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/raygun_embeddings.pt"
 MORGAN_EMB_PATH="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/morgan_embeddings.pt"
 SMILES_EMB_PATH="/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/coati_embeddings.pt"

#  TRAIN_CSV= "/hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/moodeng/train.csv"
#   VAL_CSV= "/hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/val.csv"
#   TEST_CSV= "/hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/test_remaining.csv"
#   PROTEIN_EMB_PATH= "/hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/raygun_embeddings.pt"
#   MORGAN_EMB_PATH= "/hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/morgan_embeddings.pt"
#   SMILES_EMB_PATH= "/hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/coati_embeddings.pt"

SEED="${SEED:-42}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ENABLE_COATI_VALIDATION="${ENABLE_COATI_VALIDATION:-true}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"

DRUG_QUANTIZER_TYPE="fsq"

echo "CONFIG_PATH=$CONFIG_PATH"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "TEST_CSV=$TEST_CSV"
echo "PROTEIN_EMB_PATH=$PROTEIN_EMB_PATH"
echo "MORGAN_EMB_PATH=$MORGAN_EMB_PATH"
echo "SMILES_EMB_PATH=$SMILES_EMB_PATH"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "PERSISTENT_WORKERS=$PERSISTENT_WORKERS"
echo "TRAINER_ACCELERATOR=$TRAINER_ACCELERATOR"
echo "TRAINER_DEVICES=$TRAINER_DEVICES"
echo "DRUG_QUANTIZER_TYPE=$DRUG_QUANTIZER_TYPE"

python "$PROJECT_ROOT/main.py" \
  --config-path "$PROJECT_ROOT/configs" \
  --config-name config \
  seed="$SEED" \
  lr="$LR" \
  weight_decay="$WEIGHT_DECAY" \
  model.concise_backbone.drug_quantizer.type="$DRUG_QUANTIZER_TYPE" \
  trainer.max_epochs="$MAX_EPOCHS" \
  trainer.accelerator="$TRAINER_ACCELERATOR" \
  trainer.devices="$TRAINER_DEVICES" \
  data.batch_size="$BATCH_SIZE" \
  data.train_csv="$TRAIN_CSV" \
  data.val_csv="$VAL_CSV" \
  data.test_csv="$TEST_CSV" \
  data.protein_embeddings_path="$PROTEIN_EMB_PATH" \
  data.morgan_embeddings_path="$MORGAN_EMB_PATH" \
  data.smiles_embeddings_path="$SMILES_EMB_PATH" \
  data.num_workers="$NUM_WORKERS" \
  data.persistent_workers="$PERSISTENT_WORKERS" \
  run.output_root="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR/hydra" \
  hydra.sweep.dir="$RUN_DIR/multirun" \
  coati_validation.enabled="$ENABLE_COATI_VALIDATION"

echo "===== TRAINING COMPLETE ====="
echo "Run directory: $RUN_DIR"
EXPERIMENT_DIR=$(find "$RUN_DIR" -maxdepth 1 -mindepth 1 -type d -name 'concisejepa-*' | sort | tail -n 1 || true)
METRICS_DIR="$EXPERIMENT_DIR"
if [[ -n "$METRICS_DIR" ]]; then
  echo "Metrics directory: $METRICS_DIR"
  find "$METRICS_DIR" -maxdepth 3 -type f \( -name 'metrics.csv' -o -name 'epoch_metrics.jsonl' -o -name 'final_metrics.json' -o -name 'resolved_config.yaml' \) | sort || true
  find "$METRICS_DIR/checkpoints" -maxdepth 1 -type f | sort || true
fi

FINAL_METRICS_JSON=""
if [[ -n "$METRICS_DIR" ]]; then
  FINAL_METRICS_JSON="$METRICS_DIR/final_metrics.json"
  echo "Final metrics JSON: $FINAL_METRICS_JSON"
  [[ -f "$FINAL_METRICS_JSON" ]] && cat "$FINAL_METRICS_JSON" || true
fi

date
