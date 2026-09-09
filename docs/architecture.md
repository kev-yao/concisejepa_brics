# Architecture and runtime contracts

## End-to-end runtime

`main.py` is a composition root. It owns orchestration, not model behavior:

```text
Hydra config composition
        │
        ├── cfg.model ───────────────► neural network
        ├── cfg.data ────────────────► LightningDataModule
        ├── cfg.task + model ────────► LightningModule
        ├── cfg.callbacks ───────────► callback list
        ├── cfg.logger ──────────────► logger
        └── cfg.trainer ─────────────► Trainer
                                           │
                                           ├── fit(task, datamodule)
                                           └── test(task, datamodule)
```

Before component construction, the runner:

1. seeds PyTorch and Lightning;
2. creates a unique run ID and directory;
3. resolves runtime paths;
4. writes the resolved configuration and run manifest.

After fitting, the final-epoch checkpoint is used for testing. The best monitored checkpoint is
retained for reference.

## Component responsibilities

| Component | Owns | Must not own |
| --- | --- | --- |
| Model | tensors-to-tensors neural computation | file paths, Trainer setup, run directories |
| Data module | CSV filtering, cache loading, collation, loaders | loss definitions, model selection |
| Task | losses, metrics, optimizer, train/validation/test steps | collaborator paths, Slurm setup |
| Callback | orthogonal monitoring or artifact generation | primary forward/loss behavior |
| Trainer config | execution limits and hardware settings | architecture hyperparameters |
| Experiment config | a compatible component bundle and focused overrides | duplicated implementations |

## Whole-molecule contract

The BindingDB collator returns:

```text
protein_embedding       [B, R, 1280]
morgan_fingerprint      [B, 2048]
smiles_target_embedding [B, 256]
label                   [B]
smiles_list             list[str]
sequence_list           list[str]
```

The default model call is:

```python
model(
    protein_embedding=protein_embedding,
    morgan_fingerprint=morgan_fingerprint,
)
```

The task consumes these required output keys:

| Key | Meaning |
| --- | --- |
| `binding` | aligned drug-target binding prediction, shape `[B]` |
| `similarity_logits` | scaled in-batch drug/protein similarity matrix, shape `[B, B]` |
| `similarity_cosines` | unscaled similarity matrix, shape `[B, B]` |
| `jepa_pred` | predicted COATI target embedding, shape `[B, 256]` by default |
| `codes` | whole-molecule discrete code factors |
| `pre_quantized` | continuous representation before quantization |
| `quantized` | representation after quantization |

The default objective combines DTI contrastive loss, JEPA MSE, negative-diagonal loss, chemical
property supervision, and functional-group supervision. The two auxiliary terms each have weight
`0.1` in the whole-molecule baseline.

## BRICS fragment contract

The fragment collator returns:

```text
protein_embedding       [B, R, 1280]
fragment_fingerprints   [B, F, 2048]
fragment_mask           [B, F] boolean; true means a real fragment
smiles_target_embedding [B, 256]
label                   [B]
smiles_list             list[str]
sequence_list           list[str]
```

`F` varies by molecule and is padded to the largest fragment count in the batch. Preprocessing
caps each molecule at 16 fragments and records this contract in cache metadata.

The fragment model call is:

```python
model(protein_embedding, fragment_fingerprints, fragment_mask)
```

Its shared task contract includes:

| Key | Meaning |
| --- | --- |
| `binding` | aligned binding prediction |
| `similarity_logits` | scaled in-batch similarities |
| `similarity_cosines` | unscaled in-batch similarities |
| `jepa_pred` | predicted COATI target |
| `frag_codes` | code factors for every padded fragment position |
| `pooled_drug_emb` | post-FSQ molecule representation after fragment pooling |

The reference BRICS objective uses DTI contrastive loss, JEPA MSE, and negative-diagonal loss.
Chemical and functional-group auxiliary losses are disabled in `brics_mean` and `brics_xattn` to
preserve the spike training defaults.

## BRICS model flow

```text
fragment fingerprints [B, F, 2048]
        │
        ▼ shared DrugEncoder + FSQ
fragment embeddings/codes [B, F, 128]
        │
        ▼ configured pooling strategy + fragment mask
molecule embedding [B, 128]
        │
        ├── protein/drug attention ──► binding + pairwise similarities
        │
        └── MLP or cross-attention predictor ──► COATI embedding prediction
```

Supported pooling names are `mean`, `max`, `whole_mol`, `weighted_sum`, `cross_attention`,
`latent_query`, `latent_query_q1`, `latent_query_q2`, `latent_query_q8`,
`pre_attn_latent_query`, `mlp_weighted_sum`, `multi_head_weighted_sum_k2`,
`multi_head_weighted_sum_k4`, and `f2r`.

### Protein-conditioned pair scoring

Fragment fingerprints are encoded once per batch. Protein-independent poolers produce a
reusable molecule vector. For `f2r` and `cross_attention`, however, score matrix entry `[i, j]`
pools drug `i`'s fragment embeddings using **protein `j`**, then projects and scores that pair:

```text
score[i, j] = score(pool(fragment_embeddings[i], protein[j]), protein[j])
```

Conditioned pooling and drug projection run inside the existing pair chunks. With checkpointing
enabled, pair-specific tensor gathers are also recomputed during backward rather than retaining
expanded raw-protein tensors for every pair. These poolers require more computation than
protein-independent pooling; `pairwise_attention_chunk_size` controls the per-chunk working set.

`binding`, `d_emb`, `pooled_drug_emb`, JEPA outputs, and F2R `last_weights` still describe the
aligned `(i, i)` pairs. Pair-chunk forwards and checkpoint recomputation do not overwrite the
aligned attribution weights. In evaluation mode, every matrix entry must agree with evaluating
that pair independently, including after independently permuting the protein batch.

Earlier code reused `pool(fragments[i], protein[i])` against all candidate proteins, introducing
an unrelated anchor-protein dependency into off-diagonal scores. The correction changes those
scores and subsequent training for context-conditioned poolers. Parameter names/shapes and the
checkpoint format are unchanged; historical context-conditioned training trajectories are not
numerically reproducible under the corrected scoring. Protein-independent scoring is unchanged.

## Contrastive positive masking

The whole-molecule and original BRICS contrastive tasks interpret score entry `[i, j]` as
`(drug_i, protein_j)`. Their identity-based mask uses two different label directions:

```text
known_positive[i, j] = (same_drug[i, j] AND positive_label[j])
                   OR (same_protein[i, j] AND positive_label[i])
```

The same mask excludes known positives from contrastive negatives and from negative-diagonal
reference means. For `(A, P, positive)` and `(B, P, negative)`, both A-P entries are known
positives; neither B-P entry is. This preserves discrimination between ligands sharing a
receptor. Diagonal targets remain available in the contrastive loss.

Earlier code incorrectly applied column labels to both identity cases. Correcting this changes
losses and training on affected repeated-receptor batches but does not change parameters or
checkpoint formats. This is direct identity-based inference, not a complete dataset-wide lookup
of known interactions; conflicting labels and missing identity metadata remain data concerns.

## Run artifacts

Every run receives a unique directory below `run.output_root`. It contains, as applicable:

```text
resolved_config.yaml
run_manifest.json
checkpoints/
logs/
epoch_metrics.jsonl
epoch_metrics.json
final_metrics.json
```

The manifest records the command line, Git commit, dirty-tree state, run ID/name, Slurm job ID,
and fully resolved config. These artifacts are the provenance record; do not reconstruct a run
from memory or an unexpanded experiment YAML alone.

## Directory ownership

```text
configs/                    composable runtime configuration
launchers/                  generic environment/Slurm entry points
src/concisejepa/models/     supported neural modules
src/concisejepa/datamodules supported data and batch contracts
src/concisejepa/lightning_modules/ supported objectives and optimization
src/concisejepa/callbacks/  supported orthogonal monitoring
src/spikes/                 transitional/reference research implementations
tests/                      behavioral and integration regression tests
qa/                         older exploratory QA tests
scripts/                    preprocessing and downstream analysis tools
scratch                     ignored, collaborator-specific external-storage symlink
```
