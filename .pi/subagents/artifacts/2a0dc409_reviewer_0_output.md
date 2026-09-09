## Review

Read-only audit of the original fragment pipeline. No files changed or jobs submitted. The dual-view audit results are **not evidence about this architecture**: its BCE/alignment objective and protein-independent JEPA path differ.

### Ranked findings

**1. Blocker — P1: Positive masking removes real ligand discrimination within a receptor.**

`src/spikes/phase1/lit_fragment.py:113-125` constructs:
```python
(smiles_match | seq_match) & positive_column
```
But matrix entry `[i,j]` represents drug **i** against protein **j**. The identity-based positive rule needs different label directions:
```python
(smiles_match & positive_column) | (seq_match & positive_row)
```

Concrete counterexample: batch `[(A,P,positive), (B,P,negative)]`.

- Current known-positive mask is `[[True,False],[True,False]]`.
- Row contrastive loss incorrectly treats the duplicate positive `(A,P)` as a negative.
- Column contrastive loss excludes the actual negative `(B,P)`.
- The negative-diagonal loss excludes its only reference.

Evidence: masking application at `lit_fragment.py:138-142`; reference exclusion at `lit_fragment.py:202-205`.

**Established consequence:** in deterministic evaluation mode, scores have form `[[a,a],[b,b]]`. Current DTI loss is exactly `ln(2)/2`, independent of `a,b`, and negative-diagonal loss is zero. Thus this mixed-label, same-receptor batch supplies **no binding-discrimination gradient**, whatever its predictions.

**Unproven:** how frequently this pattern materially contributes to the reported collapse.

---

**2. Blocker — P1, conditional on pooler: Off-diagonal scoring uses the wrong protein-conditioned drug representation.**

`src/spikes/phase1/fragment_encoder.py:138-139` pools drug `i` using its aligned protein `i` **once**. The pairwise loop then reuses that representation against every protein `j` at `fragment_encoder.py:185-186`.

Consequently:
```text
S[i,j] = scorer(pool(fragments_i, protein_i), protein_j)
```
rather than:
```text
S[i,j] = scorer(pool(fragments_i, protein_j), protein_j)
```

This affects `cross_attention` and `f2r`, registered at `src/spikes/phase1/fragment_pooling.py:270`. Off-diagonal negatives have an extra anchor-protein dependency absent from independently evaluated pairs. Training can potentially exploit this conditioning mismatch rather than molecular discrimination.

**Established:** inconsistent pair semantics.  
**Unproven:** exploitation by a particular checkpoint or causation of collapse.

**Unaffected:** protein-independent poolers. Both named `brics_mean` and `brics_xattn` default to mean pooling (`configs/model/brics_fsq.yaml:6`, `configs/model/brics_xattn.yaml:6`). The latter names the **predictor**, not a context-conditioned pooler.

---

**3. Note — P2: Protein information can support JEPA despite impoverished fragment codes.**

For the MLP predictor:

- Drug features already incorporate protein attention: `fragment_encoder.py:161-164`.
- JEPA additionally receives the pooled protein explicitly: `fragment_encoder.py:292-293`.

For the cross-attention predictor:

- Raw protein tokens enter directly: `src/spikes/phase1/fragment_xattn.py:85-97`.
- JEPA consumes **unpooled** fragment embeddings, bypassing the configured pooling layer: `fragment_xattn.py:150`.

**Established:** JEPA is not a drug-only reconstruction constraint. The cross-attention JEPA branch also supplies no direct gradient to pooling-layer parameters.

**Collapse hypothesis:** the predictor could reduce MSE through receptor-associated molecular statistics while preserving little fragment identity. A constant predictor can also learn the target mean. Neither mechanism proves that a trained model uses this shortcut.

Both named experiments disable chemical/group auxiliaries (`configs/experiment/brics_mean.yaml:14-18`, `configs/experiment/brics_xattn.yaml:12-16`). The active objective therefore has no direct code-diversity term (`lit_fragment.py:224-248`).

**Important qualification:** positive contrastive loss still opposes complete collapse in ordinary informative batches; collapse is not generally a zero-loss solution.

---

**4. Note — P2: Negative supervision tolerates ties; “cosines” are actually sigmoid scores.**

The scorer ends in sigmoid (`fragment_encoder.py:110-115`). Those scores—not normalized-feature cosine similarities—become `similarity_cosines` and scaled contrastive logits (`fragment_encoder.py:295-298`; `fragment_xattn.py:152-155`).

The negative penalty is:
```text
ReLU(negative_diagonal − reference_row_mean + margin)
```
at `lit_fragment.py:207-212`; configured margin is zero (`configs/config.yaml:26-29`).

**Established:**

- Any constant score matrix has zero negative-diagonal penalty.
- An all-negative batch has zero contrastive DTI loss (`lit_fragment.py:149-150`).
- Therefore a constant matrix on an all-negative batch receives no DTI correction, irrespective of its absolute binding probability.
- Sigmoid saturation attenuates score-head gradients. The learned contrastive scale does not provide an explicit saturation safeguard.

**Unproven:** whether saturation or this weak negative constraint explains observed collapse. This is a ranking objective, not calibrated binary-label supervision.

---

**5. Note — P2: F2R reduces residues to their mean before determining fragment importance.**

At `src/spikes/phase1/fragment_pooling.py:227-233`, projected fragment–residue dot products are averaged over residues and heads **before** fragment softmax.

By linearity, this is equivalent to scoring fragments against the **mean projected protein vector**. It cannot distinguish two residue tensors with identical means, regardless of their different residue-level patterns.

This limits the interpretation of “fragment-to-residue attention.” It does not independently establish collapse.

### Correct / existing safeguards

- Supported models re-export the same spike implementation, and the supported task inherits its objective: `src/concisejepa/models/fragment.py:9-23`, `src/concisejepa/lightning_modules/lit_fragment.py:9-42`. These findings apply to both construction paths.
- Cross-attention JEPA correctly negates the valid-fragment mask for attention and masks final pooling: `fragment_xattn.py:93-102`.
- The molecular JEPA path consumes post-encoder fragment embeddings; I found no raw-fingerprint bypass around quantization in these files.
- Integration tests compare initialization, gradients, and optimizer trajectories, but their batch uses distinct receptors (`tests/test_brics_hydra_integration.py:63-80,249-295`). Pooler coverage primarily checks shapes (`:200-247`), so these tests do not establish masking or pair-conditioning correctness.

### Falsifiable CPU probes — proposed, not executed

| Probe | Expected result from current code |
|---|---|
| Two distinct drugs, identical protein tensors, labels `[1,0]`; evaluation mode with gradients enabled; backpropagate DTI plus negative-diagonal loss only | Loss `ln(2)/2`; binding-discrimination gradients zero, up to floating-point error. |
| Keep drugs fixed, independently permute protein batch by `p`; compare new matrix with original `S[:,p]` | Mean pooling agrees within tolerance. Context-conditioned pooling generally disagrees when its context affects pooling. |
| Constant score matrix with all-negative labels | Both DTI components zero, including for uniformly high binding scores. |
| Replace fragment embeddings with one constant vector; retain proteins and masks | Measures remaining protein/count-driven predictive capacity. Compare against protein replacement and target-mean controls; predictive degradation is checkpoint-dependent. |
| Add nonzero, residue-wise zero-mean perturbations to F2R context | Pooling weights/output unchanged within floating-point tolerance. |
| Backpropagate cross-attention JEPA loss alone | No pooling-layer gradient; fragment encoder and JEPA predictor remain connected. |

Recommended existing regression command for the supervisor:
```bash
PYTHONPATH=src:. python -m unittest -q tests.test_brics_hydra_integration
```