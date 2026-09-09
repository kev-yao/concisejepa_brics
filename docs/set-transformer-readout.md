# Set Transformer binding readout

`concisejepa.models.set_binding_head.SetTransformerBindingHead` is a standalone
neural primary binding head originally used for the frozen-backbone comparison.
The opt-in `brics_f2r_set_e2e` experiment now trains it jointly with CONCISE (see
below). In both cases it replaces the binding readout, **not F2R fragment pooling**.

## Inputs and scope

The historical head-only experiment kept the trained FSQ3/F2R backbone and its
fragment-only JEPA predictor frozen. For each drug–protein pair, form four
256-dimensional tokens:

| Type ID | Feature token |
|---|---|
| 0 | normalized fragment-path drug feature `d` |
| 1 | normalized protein feature `p` |
| 2 | elementwise product `d * p` |
| 3 | absolute difference `abs(d - p)` |

These are **four derived feature vectors**, not individual BRICS fragments, integer
FSQ code IDs, raw fingerprints, or whole-molecule tokens. The Set Transformer has
no backbone, whole-molecule input, data loading, or classifier-library dependency.
It does not itself compute `d`, `p`, or their interactions.

The caller retains the separate whole-molecule binding score and performs the
planned late mixture `0.95 * primary_probability + 0.05 * whole_probability`.
Nothing from the whole path enters this head or JEPA. The head is fully neural,
but training it on frozen features is **not end-to-end backbone fine-tuning**.

## Architecture and API

```text
Type-indexed input normalization
    → shared input projection + learned token-type embeddings
    → configurable self-attention blocks (SAB)
    → learned single-seed pooling by multihead attention (PMA)
    → small feed-forward logit readout
```

Attention blocks use residual multihead attention, row-wise feed-forward layers,
and LayerNorm. There are no positional encodings. Jointly permuting tokens, type
IDs and masks preserves evaluation outputs; reassigning type IDs changes semantic
roles, not just positions. Training dropout is stochastic.

```python
head = SetTransformerBindingHead(
    feature_dim=256, model_dim=64, num_heads=4,
    num_blocks=2, dropout=0.1, ff_multiplier=2,
)
head.set_input_stats(train_mean, train_scale)  # both [4, 256]
out = head(tokens, token_types=types, mask=valid)
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    out["binding_logits"], labels.float()
)
```

- `tokens`: floating-point `[B,F,feature_dim]`.
- `token_types`: integer IDs 0–3, `[F]` or `[B,F]`, on the token device. Omitted
  types default to `[0,1,2,3]` only when `F=4`; other set lengths require IDs.
- `mask`: optional boolean `[B,F]` on the same device; true means valid.
  Every example must contain at least one valid token. Empty batches/sets are
  rejected. Padded positions still require in-range type IDs.
- Outputs: `binding_logits [B]` (signed, suitable for BCEWithLogits),
  `binding [B]` (sigmoid probability), and `pooled [B,model_dim]`.

Padded values are zeroed before normalization/projection and excluded as keys in
all attention blocks, including PMA. NaN/Inf padding is ignored, while nonfinite
**valid** tokens are rejected.

`input_mean` and `input_scale` are registered buffers `[4,feature_dim]`, initially
zero/one. The caller must compute them from **training features only** and supply
finite, strictly positive scales (for a zero-variance coordinate, the caller may
use scale1). `set_input_stats` copies detached values; it fits nothing. Statistics
are gathered by type rather than set position and are included in the state dict.
Keep constructor settings alongside the state dict to reconstruct the head.

Focused tests: `PYTHONPATH=src:.:tests python -m unittest -q tests.test_set_binding_head`.
This module and its tests do not train the experimental candidates, select
checkpoints, access a test split, or claim an achieved AP.

## End-to-end training with discrete FSQ

Use `experiment=brics_f2r_set_e2e` (or `brics_f2r_set_e2e_debug` for a two-batch
smoke). `ConciseFragmentSetReadout` computes the four feature roles **live** after
candidate-protein-conditioned F2R and drug/protein attention. The Set head scores
all candidate pairs, including the aligned pairs used for observed-label BCE.
There are no detached/cached CONCISE features and no frozen model parameters.

```text
BRICS fingerprints → DrugEncoder → discrete FSQ (STE backward) → F2R
                                                             ↓
protein embeddings → CONCISE protein layers → pair attention → [d,p,d*p,|d-p|]
                                                             ↓
                                                      Set Transformer → primary

whole fingerprint → shared encoder/FSQ → separate attention/scorer → whole
fragment context → JEPA predictor → fixed COATI target
final = 0.95 × primary + 0.05 × whole
```

- FSQ rounding and discrete factor IDs are retained; backward uses the existing
  straight-through estimator. Integer IDs are metadata, not embedding lookup
  inputs to the Set head. The new model rejects `continuous_fsq`.
- CONCISE's fingerprint encoder, FSQ input/output projections, F2R, protein
  projections/attention, and Set head receive primary binding gradients.
  The existing optimizer covers the entire task, including JEPA and the whole
  scorer. JEPA's predictor is trained by its own objective, not binding BCE.
- Input Morgan/BRICS processing and cached protein embeddings/COATI targets
  remain preprocessing. This does **not** fine-tune the external protein or
  COATI embedding generators.
- Whole fingerprints never enter F2R, the Set head, or fragment JEPA. The whole
  scorer's architecture is unchanged, but its predictions are **not frozen**:
  both branches update shared encoder/projection/attention parameters.
- Fresh heads use identity input-stat buffers (zero mean/unit scale), with the
  existing L2-normalized `d,p` and internal LayerNorms. No historical feature
  cache or validation-fitted normalization is loaded. The old head's selected
  weights and statistics are not automatically reused.

### Starting recipe (not a measured/tuned result)

The named config uses a fresh dim128/two-SAB/four-head/dropout0.2 Set head,
AdamW at the inherited `lr=1e-4`, weight decay0.01, batch32, 30 epochs, float32,
and gradient clipping1.0. It retains the original branch contrastive and
negative-diagonal objectives and fragment JEPA, **adding observed-pair BCE**:

```text
L = [BCE_fragment + CE_fragment + N_fragment]
  + 0.25 × [BCE_whole + CE_whole + N_whole]
  + MSE_JEPA
```

`N` includes the existing negative-diagonal coefficient and known-positive mask.
BCE uses the branch sigmoid probabilities in float32 via the existing secondary
training task; `similarity_logits` remain scaled probabilities, **not binary
logits**. Unknown off-diagonal pairs do not receive BCE labels. The 0.05 fusion
weight is independent of the 0.25 whole supervision weight. This objective and
live batch size differ from the old frozen-head fit: this is not a matched
one-variable comparison or an AP claim.

Validation-best checkpoints monitor `val/fused_pooled_ap`. Candidate configs
set `evaluation.run_test=false`; evaluate a locked winner separately, never via
candidate fitting. The benchmark has already been inspected and is not an
untouched holdout.

### Warm-start the original FSQ3/F2R backbone

The standard runner accepts `initialization.backbone_checkpoint` for this model.
It loads all original model tensors from a **trusted legacy secondary-binding
Lightning checkpoint**, allowing only the newly introduced Set head to be absent.
Unexpected/missing backbone keys and shape mismatches fail. No optimizer state
is resumed and nothing is frozen. The runner records the source checkpoint path,
SHA256 and loaded/fresh tensor counts in `initialization.json`; normal full
checkpoints subsequently contain both backbone and Set head. Reload those full
checkpoints normally, not through the legacy-backbone initializer.

Use matching FSQ levels and architecture from the source resolved config;
quantizer modes/levels are not fully encoded in state dicts. Omitting the path
trains from scratch. For the original FSQ3/F2R run, from this repository in the
compatible environment and a GPU allocation:

```bash
BACKBONE=/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics_training/20260908-2120-f2r-secondary-binding/training/brics-f2r-secondary-fsq3-b6476734/checkpoints/final.ckpt
COUNTS=/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/morgan_embeddings.pt
PYTHONPATH=src:. python main.py experiment=brics_f2r_set_e2e_debug \
  initialization.backbone_checkpoint="$BACKBONE" \
  data.morgan_embeddings_path="$COUNTS" trainer.accelerator=gpu
```

Use a new output root for any full fit; do not modify the historical run bundle.
The normal data module still verifies count-cache metadata. Changing the debug
experiment to `brics_f2r_set_e2e` enables the full recipe, but no SLURM job is
submitted automatically.

Regression coverage in `tests/test_set_binding_e2e.py` isolates primary BCE and
checks finite nonzero gradients **and optimizer updates before FSQ**, discrete
forward-grid/code consistency, branch isolation, candidate-pair correctness,
checkpoint recomputation with dropout, strict warm starts, full-model checkpoint
roundtrip, and a real synthetic Lightning fit without a test stage.

## Completed frozen-backbone experiment

The four-candidate seed42 screen selected model dim128, two SABs, four heads,
PMA with one seed, dropout0.2, and epoch14. Fixed95/5 primary/whole fusion reached
**validation AP0.643760 and test AP0.659632** on the unchanged benchmark.
Primary-only test AP was0.645528. This clears the0.65 fused-score target but does
not match the boosted-head reference0.696998. Selection used validation only;
one selected Set Transformer checkpoint was evaluated on test.

[Results, provenance, checkpoints and verification](../../concisejepa_brics_training/20260909-0145-set-transformer-readout/RESULTS.md)
are retained in the run bundle. Load `checkpoint["model_kwargs"]` into this class,
then strictly load `checkpoint["state_dict"]`; this restores the TRAIN-only
normalization buffers too. Neither this head checkpoint nor the old backbone
checkpoint alone implements the full95/5 prediction pipeline. This remains a
one-seed result on an already monitored, non-cold benchmark.
