# Dual-view BRICS experiment

## Research question

Can one shared encoder and FSQ vocabulary learn both how BRICS fragments compose into a molecule
and how that molecule interacts with a protein?

The initial experiment is intentionally simple. It avoids drug-to-protein cross-attention and
uses independently encoded molecule and receptor vectors.

## Inputs

Each row contains a protein, molecule, and binary interaction label. The data module retrieves:

```text
protein_embedding            [B, R, 1280]
fragment_fingerprints        [B, F, 2048]
fragment_mask                [B, F]
whole_molecule_fingerprint   [B, 2048]
COATI target                 [B, 256]
interaction label            [B]
```

Both molecular inputs use count-based ECFP radius 4 fingerprints. The existing canonical BRICS
cache contains every required tensor; this experiment changes collation, not preprocessing.

## Architecture

```text
fragment fingerprints ─┐
                       ├── shared DrugEncoder + FSQ
whole-molecule FP ─────┘

fragment quantized vectors [B,F,D]
        │
        ▼ position-free masked Set Transformer
fragment set vector [B,D]
        ├──► COATI predictor
        └──► shared drug DTI projection + ReLU

whole quantized vector [B,D]
        └──► same drug DTI projection + ReLU

protein residues
        └──► protein projection + self-attention + pooling + ReLU
```

The Set Transformer prepends a learned summary token, applies no positional embeddings, and masks
padded fragments. Its summary is therefore invariant to fragment order.

There is exactly one `DrugEncoder`/FSQ instance. Fragment and whole-molecule calls share all of its
parameters, and all losses backpropagate through it.

The protein representation is independent of both molecule views. There is no drug-to-protein or
protein-to-drug cross-attention.

Both named experiments require a CUDA accelerator. This prevents a scheduled GPU run from silently
falling back to CPU when an allocated device is unhealthy.

## DTI probabilities

All three DTI-space vectors pass through ReLU before normalization. Their coordinates are
nonnegative, so pairwise cosine similarities lie in `[0,1]` and are interpreted directly as
probabilities:

```text
p_whole   = cosine(whole molecule, receptor)
p_fragment = cosine(fragment set, receptor)
p_final    = 0.5 × (p_whole + p_fragment)
```

Training uses `BCELoss`, not `BCEWithLogitsLoss`, and the model applies no sigmoid to these scores.
The final probability is used for AUPRC/AUROC; whole and fragment means are logged separately.

## Objective

```text
L_whole_DTI = BCE(p_whole, label)
L_frag_DTI  = BCE(p_fragment, label)
L_DTI       = 0.5 × (L_whole_DTI + L_frag_DTI)

L_JEPA      = MSE(COATI_predictor(fragment_set), COATI_target)
L_align     = mean(1 - cosine(fragment_set_DTI, whole_molecule_DTI))

L_total     = 1.0 × L_DTI + 1.0 × L_JEPA + 1.0 × L_align
```

`L_align` applies to positive and negative protein pairs because it describes two views of the
same molecule, not the interaction label.

## Launching

Two-batch smoke run:

```bash
EXPERIMENT=brics_dual_view_debug sbatch launchers/train.sbatch
```

Thirty-epoch experiment:

```bash
EXPERIMENT=brics_dual_view sbatch launchers/train.sbatch
```

Override the canonical data root when necessary:

```bash
CONCISEJEPA_BRICS_DATA_ROOT=/path/to/BindingDB_embeddings \
EXPERIMENT=brics_dual_view \
sbatch launchers/train.sbatch
```

## Interpretation limits and planned ablations

The fragment DTI branch sends binding gradients through the Set Transformer, but it does not make
attention weights causal explanations. The first attribution method should be leave-one-fragment-
out change in the fragment DTI probability.

Because fragment pooling is protein-independent, it can learn generally binding-relevant molecular
features but cannot explicitly select different fragments for different receptors. A later
late-interaction experiment may score contextualized fragment vectors against the independently
encoded receptor and aggregate those scores without modifying the receptor representation.

Minimum ablation matrix:

```text
A  whole-molecule DTI only
B  whole + fragment DTI
C  B + fragment/whole alignment
D  B + fragment-to-COATI JEPA
E  B + alignment + JEPA (this initial experiment)
```

These comparisons are required before concluding that the JEPA or alignment objectives improve
DTI generalization.
