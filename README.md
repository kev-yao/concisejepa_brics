## Setup

For architecture, experiment, migration, and contributor guidance, start with the
[collaborator documentation](docs/README.md).

COATI is required for JEPA training because the target regression embedding is a COATI SMILES latent.

Install project dependencies (including COATI from GitHub):

```bash
pip install -e .
```

If your cluster has a problematic shared temp dir, force pip to use a private tmp location:

```bash
mkdir -p "$HOME/tmp"
TMPDIR="$HOME/tmp" pip install -e .
```

## Generate COATI embeddings

Generate and save SMILES embeddings before training:

```bash
python scripts/gen_coati_embeddings.py
```

By default this writes `coati_embeddings.pt` next to `train_csv`. You can override paths and device:

```bash
python scripts/gen_coati_embeddings.py \
  train_csv=/path/train.csv \
  val_csv=/path/val.csv \
  test_csv=/path/test.csv \
  smiles_embeddings_path=/path/coati_embeddings.pt \
  embedding_device=cuda:0
```

## Train

Experiments are composed from Hydra config groups under `configs/`:

```text
model + data + task + callbacks + trainer + logger + experiment overrides
```

Run the default whole-molecule FSQ baseline:

```bash
python main.py experiment=baseline
```

Swap the model and its compatible callbacks together:

```bash
python main.py experiment=diveq
```

Use the short debug trainer while developing:

```bash
python main.py experiment=debug
```

Run the promoted BRICS fragment pipeline with its reference mean pooling and
training defaults:

```bash
export CONCISEJEPA_BRICS_DATA_ROOT=/path/to/compatible/brics/csvs-and-caches
python main.py experiment=brics_mean
```

The BRICS model accepts the original pooling names as config overrides, including
`max`, `weighted_sum`, `latent_query`, `f2r`, and the multi-head variants:

```bash
python main.py experiment=brics_mean model.concise_fragment.pooling=f2r
```

Use the unpooled fragment/protein cross-attention JEPA predictor with:

```bash
python main.py experiment=brics_xattn
```

`experiment=brics_debug` limits train, validation, and test to two batches. BRICS
experiments intentionally preserve the reference spike defaults: 30 epochs, batch
size 256, MSE JEPA loss, and chemical/group auxiliary supervision disabled.

Every run writes a resolved config, run manifest, metrics, and checkpoints to a unique
directory below `CONCISEJEPA_RUN_ROOT` (default: `outputs/`). The manifest records the Git
commit, dirty-tree state, Slurm job ID, and launch arguments.

### Collaborator-specific paths

Do not edit shared experiment configs just to change storage locations. Set:

```bash
export CONCISEJEPA_CSV_ROOT=/path/to/bindingdb/csvs
export CONCISEJEPA_EMBEDDING_ROOT=/path/to/precomputed/embeddings
export CONCISEJEPA_RUN_ROOT=/path/to/experiment/outputs
```

BRICS experiments use one additional root containing mutually compatible CSVs,
protein embeddings, COATI embeddings, Morgan fallback embeddings, and the fragment
fingerprint cache:

```bash
export CONCISEJEPA_BRICS_DATA_ROOT=/path/to/BindingDB_embeddings
```

The generic Slurm launcher additionally accepts `CONCISEJEPA_ENV` and `EXPERIMENT`:

```bash
CONCISEJEPA_ENV=/path/to/micromamba/env \
EXPERIMENT=baseline \
sbatch launchers/train.sbatch
```

### Adding an experiment

Add reusable components to their config groups, then create a small file such as
`configs/experiment/my_architecture.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: my_architecture
  - override /data: bindingdb
  - override /task: dti_jepa
  - override /callbacks: wholemol
  - override /trainer: default

run:
  name_prefix: my-architecture
```

Launch it with `python main.py experiment=my_architecture`. Models used with the existing
`dti_jepa` task must accept `protein_embedding` and `morgan_fingerprint` and return the output
keys consumed by `LitConciseJEPA`. A model with a different batch or output contract should
provide its own task config instead of adding architecture-specific branches to `main.py`.

To override only the COATI cache for an ad-hoc run:

```bash
python main.py data.smiles_embeddings_path=/path/coati_embeddings.pt
```
