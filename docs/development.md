# Development and validation rules

## Core invariants

Treat these as architectural constraints:

1. `main.py` remains a generic composition root.
2. Model/data/task compatibility is explicit in experiment YAML.
3. Existing experiment defaults do not change silently during refactors.
4. A cache must be validated before its tensors reach the model.
5. Refactors require numerical parity, not only import or shape tests.
6. Experimental code becomes supported through stable `concisejepa.*` paths.
7. Generated and collaborator-specific files stay outside Git.

## Recommended change workflow

For a new architecture or research module:

1. Write down its batch and output contracts.
2. Decide whether the existing task can consume it.
3. Add the smallest compatible component set.
4. Add a named debug experiment.
5. Compare old and new construction under fixed seeds.
6. Run a short real-data GPU job.
7. Inspect the resolved config, metrics, and checkpoint paths.
8. Only then launch the full experiment.

If there is an existing spike, keep it as a reference until the new path matches initialization,
forward results, loss, gradients, and optimizer updates.

## Required local checks

The tests are written with `unittest`, so they do not require `pytest`:

```bash
PYTHONPATH=src:. python -m unittest -q \
  tests.test_brics_hydra_integration \
  tests.test_fragment_pair_conditioning \
  tests.test_positive_pair_mask \
  tests.test_continuous_fsq \
  tests.test_training_refactor_parity \
  tests.test_experiment_configs \
  tests.test_fingerprint_embeddings \
  tests.test_morgan_sparsity_analysis
```

Static checks:

```bash
ruff check main.py src tests
python -m compileall -q main.py src tests
git diff --check
bash -n launchers/train.sbatch job.sh
```

Resolve configs without loading data:

```bash
python main.py experiment=baseline --cfg job --resolve >/dev/null
python main.py experiment=brics_mean --cfg job --resolve >/dev/null
python main.py experiment=brics_xattn --cfg job --resolve >/dev/null
```

Some cluster environments emit harmless warnings when Lightning detects Slurm outside `srun`,
when a test calls `self.log()` without a Trainer, or when PyTorch disables nested-tensor
optimization for pre-norm transformers. Warnings should be reviewed, but these three do not
indicate numerical test failure.

## What parity tests must compare

A refactor parity test should use fixed initialization and dropout seeds and compare with zero
tolerance when both paths execute the same CPU kernels:

- initial state dict;
- total and component losses;
- model output tensors;
- every populated parameter gradient;
- state dict after at least two optimizer updates.

For a new module rather than a refactor, require shape, masking, finite-gradient, and optimizer
tests. For masked fragment models, explicitly prove that padded fragment values do not affect the
pooled output.

## Real-data smoke tests

CPU synthetic parity does not validate CUDA kernels, cache contents, worker behavior, or cluster
paths. Before a full run, submit:

```bash
EXPERIMENT=debug sbatch launchers/train.sbatch
EXPERIMENT=brics_debug sbatch launchers/train.sbatch
```

For each job verify:

- train, validation, and test all execute;
- losses are finite and on the expected scale;
- the final checkpoint exists and is used for test;
- `resolved_config.yaml` points at the intended caches;
- `run_manifest.json` records the intended Git commit;
- the datamodule retains a nonzero number of rows after cache-key filtering.

## Data and cache safety

Do not move, delete, or regenerate shared caches as part of a code refactor. Cache generation is a
separate, explicit operation.

Whole-molecule Morgan caches require matching fingerprint metadata. BRICS fragment caches require:

```json
{
  "type": "fragment_fps",
  "fingerprint_kind": "ecfp-count:4",
  "fingerprint_length": 2048,
  "max_frags": 16
}
```

The BRICS data root must be internally consistent. Mixing CSVs, protein embeddings, fragment
fingerprints, and COATI embeddings from different dataset generations can silently discard rows,
even when tensor dimensions match.

## Storage conventions

`scratch` is ignored and may be a symlink to cluster scratch storage. Use it for run outputs,
temporary caches, downloaded weights, and large derived artifacts. Never commit the symlink target
or assume another collaborator has the same absolute path.

The generic launcher defaults run storage to `scratch/runs` when the symlink exists. Override it
with `CONCISEJEPA_RUN_ROOT` when needed.

Do not use repository `tmp/`, `/tmp`, or a collaborator's home path as a shared scientific data
contract. Node-local temporary directories are appropriate only for disposable process caches.

## Code organization going forward

- Stable reusable code belongs under `src/concisejepa/`.
- Exploratory spikes may live under `src/spikes/`, but must have an owner, question, and exit plan.
- Promote a successful spike through a standard namespace and config bundle before new production runs.
- Keep temporary compatibility imports while old checkpoints/scripts still rely on spike paths.
- Remove a spike only after downstream import search, checkpoint loading, and parity tests succeed.
- Prefer explicit constructor arguments over passing an untyped `cfg` object into data/model classes.
- Prefer shared interface keys over callbacks that inspect a specific concrete model class.

## Pull-request or handoff checklist

- [ ] New and changed configs resolve.
- [ ] Default values were compared against their reference path.
- [ ] Model, data, task, and callback compatibility is documented.
- [ ] Synthetic behavioral tests pass.
- [ ] Existing whole-molecule tests still pass.
- [ ] A real debug job was run, or the absence of one is stated.
- [ ] No caches, checkpoints, outputs, or personal paths were added.
- [ ] Documentation and example commands were updated.
