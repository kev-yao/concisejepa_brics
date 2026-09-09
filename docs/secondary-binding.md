# Secondary whole-molecule binding (corrected F2R)

`brics_f2r_whole` is an opt-in extension of the corrected fragment F2R/MLP-JEPA
model. It is **not** the older Set-Transformer/BCE/alignment dual-view experiment.
Existing fragment-only and whole-molecule experiments are unchanged.

```text
BRICS fingerprints → shared DrugEncoder/quantizer → F2R → fragment binding score
                                                     └→ JEPA → COATI target
Whole fingerprint  → same DrugEncoder/quantizer ────────→ whole binding score

final binding = (1 − alpha) × fragment score + alpha × whole score
```

The whole input bypasses fragment pooling entirely. Drug projection, attention,
final sigmoid scorer, and encoded protein tensors are shared. Each pair matrix
entry `[i,j]` uses drug `i` and protein `j`; F2R conditions on that candidate
protein, not drug `i`'s unrelated aligned protein. Whole binding has its own
pairwise drug/protein attention but no fragment pooling. Both paths use existing
chunking/checkpointing. JEPA, pooled fragment embeddings, fragment codes/features,
and F2R attribution weights remain fragment-only.

Sharing parameters still couples **training** between the two views. Separating
forward paths does not imply independent learned representations or causal
fragment explanations. Legacy unfragmentable/fallback cases may already supply
a whole-molecule fingerprint as their sole fragment token; this experiment does
not change those cases or append another whole token to the fragment list.

## Objective and fusion

For each branch `v`, retain the corrected contrastive DTI and negative-diagonal
objectives (including the same known-positive identity mask):

```text
L_binding(v) = L_contrastive(v) + negative_diagonal_weight × L_negative_diagonal(v)
L_total      = L_binding(fragment) + beta × L_binding(whole) + L_JEPA(fragment)
```

By default, no fused-score training loss, BCE loss, alignment loss, extra
normalization, or whole-to-COATI predictor is added. Chemical/group auxiliaries
stay disabled in the named configurations. `model.whole_binding_weight` (`alpha`) and
`task.secondary_loss_weight` (`beta`) both start at **0.25**, not a tuned result.
Alpha controls prediction only; beta controls secondary supervision independently.
Alpha must be finite in `[0,1]`; beta must be finite and nonnegative. Alpha endpoints
permit reporting either branch alone, but do not disable secondary supervision;
set beta to zero as well for a no-secondary-loss ablation. Select any fusion weight
using validation only and retain it in the resolved config/checkpoint provenance.

### Opt-in supervised objective screen

The secondary task exposes finite, nonnegative weights, preserving the above
objective exactly at defaults:

| Task setting | Default | Applies to |
| --- | ---: | --- |
| `binding_bce_weight` | 0 | Mean probability BCE on observed aligned labels only |
| `binding_contrastive_weight` | 1 | Existing contrastive CE |
| `binding_negative_weight` | 1 | Existing relative negative-diagonal penalty |
| `jepa_loss_weight` | 1 | Fragment-only JEPA loss |

For each branch, `D_v = w_B * BCE(p_v,y) + w_C * C_v`, and
`N_v = w_N * N_original_v`. The total is
`D_fragment + beta*D_whole + negative_diagonal_weight*(N_fragment + beta*N_whole)
+ w_J*JEPA`. Fusion alpha is still independent of every loss coefficient.
Unknown off-diagonal pairs are **not** assigned negative labels for BCE. The
existing `similarity_logits` contain scaled probabilities and must not be passed
to BCEWithLogitsLoss. BCE is computed in float32 from the branch probabilities;
when disabled it is not computed and its logs contain zero.

Per-branch `loss_{fragment,whole}_{bce,contrastive,neg_diag}` are raw components;
legacy `loss_{fragment,whole}_dti` remains an alias for raw contrastive CE.
Corresponding `_weighted` logs apply objective weights (including the global
negative-diagonal coefficient for negative terms) but not beta. Aggregate
`loss_{bce,contrastive,neg_diag}_weighted` also includes the beta-weighted whole
contribution. `loss_dti` aggregates weighted BCE and CE; `loss_jepa` is raw and
`loss_jepa_weighted` is the actual JEPA contribution. Weights are saved in task
hyperparameters and the resolved configuration; model structure is unchanged.

Candidate fits must reserve test for a separately selected winner:

```bash
python main.py experiment=brics_f2r_whole \
  data.morgan_embeddings_path=/path/to/verified_count_morgan.pt \
  task.binding_bce_weight=1 task.binding_contrastive_weight=0.1 \
  task.binding_negative_weight=0 task.jepa_loss_weight=1 \
  trainer.max_epochs=60 evaluation.run_test=false
```

`evaluation.run_test` defaults to true (also for older configs lacking the key),
retaining automatic **final-checkpoint** testing. False means the runner never
calls `Trainer.test` or sets up the test dataloader. It does not change training,
validation-best checkpoint monitoring or final checkpoint saving. Loading and
testing a locked validation-selected winner is a separate explicit operation.

## Data/output contracts

The model requires:

```python
model(protein_embedding, fragment_fingerprints, fragment_mask,
      whole_molecule_fingerprint)
```

The reused `DualViewFragmentDataModule` returns an eight-item batch:
`(protein, fragments, mask, whole_fingerprint, COATI_target, label, smiles, sequences)`.
The whole fingerprint must have shape `[B, fingerprint_dim]`; omission is an error.

Both caches must use matching count-Morgan fingerprints (default ECFP radius 4,
2048 dimensions). The new bundle enables metadata validation and
`require_count_values: true`, which also rejects binary whole caches mislabeled
as counts. Override the whole-cache path with a verified count cache rather than
regenerating or overwriting any shared input. Presence of metadata alone is not
proof of count values. No caches are created by this feature.

Model outputs:

| Keys | Meaning |
| --- | --- |
| `binding`, `similarity_cosines`, `similarity_logits` | Fused aligned sigmoid score, fused pair matrix, and its scaled counterpart |
| `fragment_binding`, `fragment_similarity_cosines`, `fragment_similarity_logits` | Independent fragment scores |
| `whole_binding`, `whole_similarity_cosines`, `whole_similarity_logits` | Independent whole-molecule scores |
| `jepa_pred`, `pooled_drug_emb`, `frag_codes`, `drug_features`, `protein_features` | Unchanged fragment-path outputs |
| `whole_codes` | Whole input codes; invalid `-1` sentinels under continuous FSQ |

The historical `similarity_cosines` name is retained for compatibility; these are
**sigmoid scorer outputs**, not newly introduced cosine similarities.

Train/validation/test log `{stage}/{fragment,whole,fused}_pooled_{ap,auroc}`.
These are epoch-pooled metrics via TorchMetrics objects, with Lightning-managed
stage/epoch reset—not averages of per-batch AP/AUROC. Existing metrics callbacks
save all three views to epoch/final/test JSON. Best-checkpoint monitoring uses
`val/fused_pooled_ap`; the generic runner still tests the **final-epoch** checkpoint.
Per-branch DTI/negative-diagonal losses are also logged. Unqualified `loss_dti` and
`loss_neg_diag` include the beta-weighted secondary contributions.

## Usage

Resolve without loading data:

```bash
python main.py experiment=brics_f2r_whole --cfg job --resolve
python main.py experiment=brics_f2r_whole_debug --cfg job --resolve
```

Two-batch debug (requires an independently verified count cache):

```bash
python main.py experiment=brics_f2r_whole_debug \
  data.morgan_embeddings_path=/path/to/verified_count_morgan.pt
```

Use `experiment=brics_f2r_whole` for a separately approved full run. The no-rounding
control remains available through
`model.concise_fragment.drug_quantizer.type=continuous_fsq`; FSQ is the default.
No new training job is automatically submitted by these changes.

Focused regression tests:

```bash
PYTHONPATH=src:. python -m unittest -q tests.test_secondary_binding
```
