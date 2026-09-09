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

## Training bugs fixed (2026-09-08)

Two correctness bugs were identified while auditing fragment-codebook concentration on
BindingDB. Both are fixed in the current source. Neither has been established as the cause
of collapse in a particular training run.

### 1. Wrong protein context for off-diagonal pair scores

**Affected:** original BRICS models using `f2r` or `cross_attention` **pooling**.
Protein-independent poolers are unaffected. The `brics_xattn` experiment defaults to mean
pooling; its cross-attention *predictor* does not by itself trigger this bug.

The model pooled drug `i` using its aligned protein `i`, then reused that vector when scoring
all candidate proteins `j`:

```text
Before: score[i,j] = score(pool(fragments[i], protein[i]), protein[j])
After:  score[i,j] = score(pool(fragments[i], protein[j]), protein[j])
```

Off-diagonal training scores therefore depended on an unrelated anchor protein and differed
from evaluating the same drug–protein pair independently. This also created an opportunity
for the contrastive objective to exploit agreement with the conditioning protein rather than
binding compatibility.

**Fix:** encode fragments once, but perform protein-conditioned pooling and drug projection
inside each pair-scoring chunk using the actual candidate protein. Keep checkpointing and
preserve aligned predictions, pooled molecule embeddings, and F2R attribution weights through
backward recomputation. Conditioned pooling now requires additional per-pair computation.

- Implementation: [fragment_encoder.py](src/spikes/phase1/fragment_encoder.py).
- Regression tests: [test_fragment_pair_conditioning.py](tests/test_fragment_pair_conditioning.py).
  Tests compare independent/batched scores and gradients, checkpointed optimizer updates,
  protein-order/chunk-size/padding invariance, attribution, and aligned JEPA outputs.

### 2. Positive mask used column labels for both identity cases

**Affected:** the original BRICS contrastive task and the duplicated whole-molecule
contrastive task. The mask is used both to exclude known positives from contrastive negatives
and to exclude them from negative-diagonal reference means.

For entry `[i,j] = (drug_i, protein_j)`, matching drugs must inherit **label `j`**, whereas
matching proteins must inherit **label `i`**:

```text
Before: known_positive = (same_drug OR same_protein) AND positive_column
After:  known_positive = (same_drug AND positive_column)
                     OR (same_protein AND positive_row)
```

For a batch containing `(A, P, positive)` and `(B, P, negative)`, the old mask treated A's
duplicate positive as a negative and excluded B's actual negative from discrimination. With
deterministic scores, the binding objective became `log(2)/2`, with **zero discrimination
gradients**, regardless of the ordering of A and B. This does not imply zero JEPA gradients.

**Fix:** apply the label direction appropriate to each identity case. The corrected mask
restores gradients that raise the positive ligand's score and lower the negative ligand's
score while preserving the same-drug logic. It remains direct identity-based inference, not
a complete lookup of all known interactions; conflicting labels still require data handling.

- Implementations: [BRICS task](src/spikes/phase1/lit_fragment.py) and
  [whole-molecule task](src/concisejepa/lightning_modules/lit_jepa.py).
- Regression tests: [test_positive_pair_mask.py](tests/test_positive_pair_mask.py).
  Tests cover shared receptors/drugs, negative references, missing identity metadata,
  reordered and degenerate batches, and gradients through real training steps.

### Verification and implications for existing runs

The 12 targeted tests and all 33 tests in the recommended validation set passed. H200 checks
also verified finite gradients and restored discrimination; the conditioning fix was tested
with real BindingDB inputs. These checks validate correctness, not comparative retraining performance.

Run the targeted regressions with:

```bash
PYTHONPATH=src:. python -m unittest -v \
  tests.test_fragment_pair_conditioning \
  tests.test_positive_pair_mask
```

Existing checkpoints still load: parameter names, shapes, and checkpoint formats are unchanged.
However, affected losses/off-diagonal scores and subsequent training trajectories change.
Loading an old checkpoint does not undo learning under the buggy objectives; use new training
runs to measure the fixes' impact. FSQ initialization and other representation limitations
remain separate research questions, not resolved by these corrections.

See [architecture contracts](docs/architecture.md) and
[the full validation command](docs/development.md#required-local-checks) for details.

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

Matched quantizer controls for the BRICS pipeline:

```bash
# Same 3-D bottleneck, projections, bounds and scale; remove rounding only.
python main.py experiment=brics_mean model.concise_fragment.pooling=f2r \
  model.concise_fragment.drug_quantizer.type=continuous_fsq

# Five factors, same 32,768-code / 15-bit capacity as [32,32,32].
python main.py experiment=brics_mean model.concise_fragment.pooling=f2r \
  'model.concise_fragment.drug_layers=[[8,8,8,8,8]]'
```

`continuous_fsq` emits **-1 sentinels**, not discrete codes; codebook occupancy,
code enumeration, and embedding from integer codes are not meaningful for this control.
Keep the resolved quantizer type with checkpoints: its parameter keys/shapes intentionally
match discrete FSQ, so weights alone cannot identify the mode. These controls do not change
the default quantizer or establish which representation is best; compare task and identity
metrics across seeds rather than selecting by occupancy alone.

For a **secondary whole-molecule binding branch**, use `experiment=brics_f2r_whole`
(or `brics_f2r_whole_debug`). The whole input is scored separately and fused only
at the binding-score level; it never enters F2R pooling or JEPA. This opt-in
bundle requires a verified count-Morgan cache and logs pooled fragment/whole/fused
AP/AUROC. See [secondary binding](docs/secondary-binding.md) for inputs, losses,
cache requirements, and commands. Existing experiments remain unchanged.

Use the unpooled fragment/protein cross-attention JEPA predictor with:

```bash
python main.py experiment=brics_xattn
```

Run the shared-codebook dual-view experiment, in which a Set Transformer predicts
the molecule COATI representation from fragment codes while whole-molecule and
fragment-set vectors receive cosine-probability DTI supervision:

```bash
python main.py experiment=brics_dual_view
python main.py experiment=brics_dual_view_debug
```

See [the dual-view experiment specification](docs/dual-view-brics.md) for its
tensor contract, losses, and planned ablations.

`experiment=brics_debug` limits train, validation, and test to two batches. BRICS
experiments intentionally preserve the reference spike defaults: 30 epochs, batch
size 256, MSE JEPA loss, and chemical/group auxiliary supervision disabled.

Every run writes a resolved config, run manifest, metrics, and checkpoints to a unique
directory below `CONCISEJEPA_RUN_ROOT` (default: `outputs/`). The manifest records the Git
commit, dirty-tree state, Slurm job ID, and launch arguments.

### End-to-end Set Transformer binding readout

`experiment=brics_f2r_set_e2e` trains CONCISE and the primary Set Transformer
jointly while retaining **discrete FSQ with straight-through gradients**. Unlike
the earlier frozen-feature fit, binding gradients reach the fingerprint encoder,
F2R, and protein layers. The whole-molecule branch stays separate; fragment JEPA
remains enabled. Candidate fits skip test evaluation by default.

See [Set Transformer readout](docs/set-transformer-readout.md#end-to-end-training-with-discrete-fsq)
for the starting objective, debug configuration, and optional warm start using
`initialization.backbone_checkpoint`. No historical run artifacts are modified.

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
