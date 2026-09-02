# Experiment configuration guide

## Composition model

The base config selects one item from each reusable group:

```yaml
defaults:
  - model: wholemol_fsq
  - data: bindingdb
  - task: dti_jepa
  - callbacks: wholemol_fsq
  - trainer: default
  - logger: csv
  - _self_
  - experiment: null
```

A named experiment is a small global override that selects a compatible bundle. It should not
repeat entire model or data configs.

Effective precedence is:

```text
component defaults → base config → named experiment → CLI overrides
```

Environment-variable resolvers provide collaborator-specific default paths. A direct CLI value
can still override the resolved field.

`python main.py` uses the base whole-molecule setup with run prefix `concisejepa`. The generic
Slurm launcher explicitly defaults to `experiment=baseline`, which has the same training setup
and prefix `baseline-fsq`.

## Available bundles

| Experiment | Model | Data | Task | Trainer |
| --- | --- | --- | --- | --- |
| `baseline` | `wholemol_fsq` | `bindingdb` | `dti_jepa` | `default` |
| `diveq` | `wholemol_diveq` | `bindingdb` | `dti_jepa` | `default` |
| `debug` | `wholemol_fsq` | `bindingdb` | `dti_jepa` | `debug` |
| `brics_mean` | `brics_fsq` | `brics_bindingdb` | `brics_jepa` | `fragment_default` |
| `brics_xattn` | `brics_xattn` | `brics_bindingdb` | `brics_jepa` | `fragment_default` |
| `brics_debug` | `brics_fsq` | `brics_bindingdb` | `brics_jepa` | `debug` |

## Paths and launching

Whole-molecule experiments read:

```bash
export CONCISEJEPA_CSV_ROOT=/path/to/csvs
export CONCISEJEPA_EMBEDDING_ROOT=/path/to/embeddings
```

BRICS requires one mutually compatible root containing `train.csv`, `val.csv`, `test.csv`,
`raygun_embeddings.pt`, `fragment_fps_r4_2048.pt`, `morgan_embeddings.pt`, and
`coati_embeddings.pt`:

```bash
export CONCISEJEPA_BRICS_DATA_ROOT=/path/to/BindingDB_embeddings
```

All experiments can use:

```bash
export CONCISEJEPA_RUN_ROOT=/path/to/run/output
python main.py experiment=brics_mean
```

For Slurm:

```bash
CONCISEJEPA_ENV=/path/to/environment \
EXPERIMENT=brics_mean \
sbatch launchers/train.sbatch
```

Never edit a shared YAML solely to point at a personal directory.

## Adding a compatible model variant

If the batch and output contracts are unchanged:

1. add the model class under `src/concisejepa/models/`;
2. add a model config under `configs/model/` using `_target_`;
3. add a small experiment that overrides `/model` and any model-specific callbacks;
4. add initialization, shape, gradient, and optimizer-trajectory tests.

For a BRICS pooling study, a new config file is often unnecessary:

```bash
python main.py experiment=brics_mean model.concise_fragment.pooling=weighted_sum
```

Create a named experiment when the setting will be shared, compared repeatedly, or published.

## Adding a new batch contract

If a model needs new inputs, treat model, data, and task as one compatible bundle:

1. define the collated batch explicitly;
2. implement a data module with explicit Hydra keyword arguments;
3. implement a model with a documented forward signature;
4. implement a task that unpacks that batch and consumes the model outputs;
5. select all three in an experiment config.

Do not teach `main.py` how to recognize the model name. Its generic construction calls must stay
unchanged.

## Adding an output head

Today, chemical-property and functional-group heads are owned by the Lightning tasks. They are
not yet a Hydra component group. For a new head now:

1. put the reusable `nn.Module` in `src/concisejepa/models/`;
2. document which model output representation it consumes;
3. add labels to the data contract only if they cannot be derived from existing batch fields;
4. construct and score the head in the compatible task;
5. put `enabled`, `weight`, and head hyperparameters in configuration;
6. test disabled-mode parity and enabled-mode gradient flow independently.

The intended future design is a `configs/heads/` group instantiated into a task-owned
`nn.ModuleDict`, with each head declaring an `input_key` and loss weight. Do not implement a
one-off head registry in `main.py`; introduce the generic head contract first.

## Callback compatibility

Callbacks are named dictionaries with an optional `enabled` flag. The runner removes that flag
before instantiation and retains stable callback names for checkpoint selection.

Whole-molecule FSQ monitoring assumes whole-molecule inputs and output key `codes`. BRICS returns
`frag_codes`, so it currently uses `brics_fsq` callbacks without the old FSQ monitor or chemical
codebook callback. A future fragment monitor must understand fragment masks and code tensors of
shape `[B, F, factors]`; do not attach the whole-molecule callback unchanged.

## Configuration review checklist

- Does every `_target_` import successfully?
- Does the experiment select a compatible model/data/task/callback set?
- Are dimensions validated between caches and model inputs?
- Are reference loss weights, batch size, epochs, and optimizer defaults preserved?
- Is the path configurable without editing YAML?
- Does `python main.py experiment=<name> --cfg job --resolve` succeed?
- Does a debug experiment exist before launching a full run?
