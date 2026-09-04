# ConciseJEPA collaborator guide

This directory is the canonical technical guide for collaborators and coding agents. Start here
before changing training code or adding an experiment.

## Reading order

1. [Architecture](architecture.md) — runtime flow, tensor contracts, model outputs, and directory ownership.
2. [Experiments](experiments.md) — Hydra composition, available experiments, launch commands, and extension patterns.
3. [Development](development.md) — invariants, tests, storage rules, and the change checklist.
4. [Migration notes](migration-notes.md) — what changed in the Hydra/BRICS refactor and what remains transitional.
5. [Dual-view BRICS](dual-view-brics.md) — the shared-codebook Set Transformer experiment and losses.
6. [BRICS audit protocol](brics-audit-experiments.md) — cold-molecule splits, controlled ablations, numerical cache checks, and evaluation rules.
7. [BRICS audit results](brics-audit-results.md) — completed comparisons, interpretation limits, and the matched-input follow-up status.

## Quick reference

The generic runner composes six independently configured components:

```text
model + data + task + callbacks + trainer + logger
                         │
                         └── selected together by an experiment config
```

Supported named experiments:

| Experiment | Purpose |
| --- | --- |
| `baseline` | Whole-molecule Morgan input, FSQ, MLP JEPA predictor |
| `diveq` | Whole-molecule DiVeQ architecture and compatible callbacks |
| `debug` | Whole-molecule FSQ with two train/validation/test batches |
| `brics_mean` | BRICS fragment FSQ with mean pooling and reference spike defaults |
| `brics_xattn` | BRICS fragment FSQ with the unpooled cross-attention JEPA predictor |
| `brics_debug` | BRICS mean model with two train/validation/test batches |
| `brics_dual_view` | Shared fragment/whole codebook, Set Transformer, cosine-probability DTI |
| `brics_dual_view_debug` | Two-batch dual-view BRICS smoke run |
| `audit_{fsq,continuous}_{joint,jepa,dti}` | Six controlled conditions on the cleaned cold-molecule split, with legacy whole inputs |
| `matched_{fsq,continuous}_{joint,dti}` | Binding-trained conditions with verified count fingerprints for both molecular views |

Audit warning: the inherited BindingDB whole-molecule cache contains binary values
despite count metadata. New matched-count experiments use a separate verified cache;
do not relabel or overwrite the shared source. See the audit protocol before choosing
an experiment for new research.

Typical launches:

```bash
python main.py experiment=baseline
python main.py experiment=brics_mean
python main.py experiment=brics_mean model.concise_fragment.pooling=f2r
python main.py experiment=brics_debug
python main.py experiment=brics_dual_view_debug
```

The Slurm launcher uses `EXPERIMENT=baseline` when no experiment is supplied:

```bash
EXPERIMENT=brics_mean sbatch launchers/train.sbatch
```

## Rules that should survive future refactors

- `main.py` is architecture-agnostic. Do not add model-name conditionals there.
- A changed batch or output contract gets a compatible data/task config, not a special case in the runner.
- Experiment YAML files select compatible components; reusable implementation belongs in component groups.
- Paths and collaborator-specific resources come from environment variables or CLI overrides.
- Generated datasets, embeddings, checkpoints, run outputs, and scratch files are never committed.
- Training changes require a trajectory or gradient test. A config that merely instantiates is insufficient.
- Keep legacy/reference construction available until exact parity tests pass.

## Current source-of-truth boundary

Whole-molecule production code lives under `src/concisejepa/`.

BRICS is now supported through `concisejepa.models.fragment`,
`concisejepa.datamodules.fragment`, and `concisejepa.lightning_modules.lit_fragment`. During the
first migration stage, the numerical fragment implementation remains single-sourced under
`src/spikes/phase1/`; the supported modules adapt or re-export it. New experiments should target
the `concisejepa.*` paths. Do not create a second copy of the fragment math.

## Before handing work to another collaborator or agent

Record:

- the named experiment and all CLI overrides;
- the resolved config from the run directory;
- the Git commit and whether the tree was dirty;
- cache metadata and data-root choice;
- which parity and smoke tests passed;
- whether a real GPU/debug job was run.

The runner writes most of this automatically to `resolved_config.yaml` and `run_manifest.json`.
