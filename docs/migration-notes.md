# Hydra and BRICS migration notes

## Motivation

The original runner used one monolithic config and constructed training components directly. That
made it difficult for several collaborators to compare architecture variants without editing
shared files or copying launch scripts. BRICS research also lived in a separate CLI/env-driven
spike pipeline with stale fallback paths.

The current refactor introduces a stable experiment composition layer while preserving existing
training behavior.

## Hydra refactor

Configuration is now split into:

```text
configs/model/
configs/data/
configs/task/
configs/callbacks/
configs/trainer/
configs/logger/
configs/experiment/
```

`main.py` independently instantiates each component and injects the model into the task. It also
creates unique run directories and writes `resolved_config.yaml` plus `run_manifest.json`.

The generic Slurm launcher reads experiment, environment, data, and output locations from
environment variables. Collaborator-owned absolute paths are no longer part of the launcher.

The legacy whole-molecule constructor remains supported: `LitConciseJEPA(config)` still
instantiates its model internally, while the new runner uses
`LitConciseJEPA(experiment_config=cfg, model=model)`.

## Whole-molecule behavior preservation

The refactor initially exposed a default embedding-path discrepancy. The active data config was
corrected to match the pre-refactor defaults exactly.

`tests/test_training_refactor_parity.py` compares legacy construction with composed construction
through two deterministic AdamW steps. It verifies exact equality for initial state, loss
components, forward outputs, gradients, and updated state.

`tests/test_experiment_configs.py` additionally locks the baseline optimizer, architecture, data,
trainer, and auxiliary-loss defaults.

No neural-network layer or loss formula was changed by the Hydra refactor.

## BRICS promotion

The supported BRICS surface now consists of:

```text
concisejepa.models.fragment
concisejepa.datamodules.fragment
concisejepa.lightning_modules.lit_fragment
configs/model/brics_*.yaml
configs/data/brics_bindingdb.yaml
configs/task/brics_jepa.yaml
configs/experiment/brics_*.yaml
```

The fragment data adapter exposes explicit Hydra keyword arguments and validates cache metadata.
The task adapter maps global optimizer and loss configuration onto the reference `LitFragment`
constructor and validates fingerprint/model dimensions.

The promoted experiments intentionally retain the spike defaults:

- FSQ levels `[32, 32, 32]`;
- mean pooling for `brics_mean`;
- MSE JEPA loss;
- learning rate `1e-4` and weight decay `1e-2`;
- batch size 256;
- 30 epochs;
- chemical and group auxiliary losses disabled.

`tests/test_brics_hydra_integration.py` verifies:

- compatible Hydra bundles for mean, cross-attention, and debug experiments;
- fragment cache and batch contracts;
- model/data fingerprint dimension rejection;
- forward contracts for every existing pooler and the cross-attention predictor;
- exact spike-vs-Hydra initialization, outputs, gradients, and two-step optimizer trajectory.

## Transitional decisions

The numerical fragment encoder, poolers, cross-attention predictor, collator, and Lightning task
remain under `src/spikes/phase1/` for now. The supported modules re-export or adapt that single
implementation. This avoids a copied implementation drifting during the behavioral migration.

Consequences:

- new configs and production launches should use `concisejepa.*` targets;
- old scripts and checkpoints can continue importing `spikes.phase1.*`;
- changes to shared fragment math affect both paths and must run the parity suite;
- `src/spikes/phase1` cannot yet be deleted.

The root `configs/datamodule.yaml` and old spike Slurm scripts are legacy artifacts and are not
selected by the new Hydra defaults. `job.sh` remains a compatibility launcher; new shared jobs
should use `launchers/train.sbatch`.

## Recommended next stages

1. Run `brics_debug` against the canonical shared caches on GPU.
2. Add fragment-aware FSQ/codebook monitoring that respects masks and `[B, F, factors]` codes.
3. Decide which pooling variants are supported experiments versus historical ablations.
4. Move numerical fragment implementations into `src/concisejepa/`, leaving compatibility
   re-exports at the old spike paths.
5. Search downstream scripts for direct spike imports and migrate them incrementally.
6. Add checkpoint-loading tests before removing any legacy import path.
7. Introduce a generic configured-head interface if multiple new auxiliary outputs are planned.

## Validation status at the time of migration

The relevant CPU suite completed with 21 passing tests. Ruff, Python compilation, shell syntax,
resolved-config checks, and `git diff --check` also passed. This validates deterministic CPU
behavior and configuration wiring; it does not replace a real CUDA/data-loader smoke job.
