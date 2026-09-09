# Code Context

## Scope and findings
Read-only CPU diagnosis, using TRAIN labels for fitting and VALIDATION labels only for evaluation. No test CSV, test labels, test metrics files, or checkpoint inference were loaded. No jobs submitted. Existing filtering and duplicate weighting preserved. Model validation AP 0.5483 is supplied context, not remeasured.

**High — objective deserves priority over further representation complexity.** A train-only additive receptor+drug identity logistic regression (one-hot, unseen identities ignored, C=1, liblinear, max_iter=300) reaches validation pooled AP **0.528984**, AUROC **0.888588**. This is only 0.0193 AP below supplied fused model performance, despite no molecular or protein features. It does not prove representation failure or establish that 0.65 is attainable; it shows that marginal identity effects explain much of current pooled performance. Do not promote an identity baseline as evidence of biochemical generalization.

**High — substantial prevalence shift:** training is 50.22% positive versus validation 14.12%. Constant-score validation AP is 0.141239. A scalar intercept/prior correction or temperature cannot improve AP because it preserves ranking. Balanced training by itself is not evidence of a bug; focus on global ranking and explicit labeled-negative supervision rather than calibration-only fixes.

**Medium — repeated/conflicting measurements and overlap:** 17.14% of validation rows have an exact TRAIN receptor–SMILES pair. Overlap is a benchmark property, not evidence of accidental code leakage. Conflicts preclude perfect prediction for identical input pairs, but these counts alone do not imply a hard AP ceiling. Do not deduplicate, reconcile labels, or change the eligible benchmark.

## Split distribution (same eligibility as dual-view datamodule)
Inputs: `/hpc/group/singhlab/user/cy244/projects/peptides/BindingDB_embeddings/{train,val}.csv`.

| Quantity | TRAIN | VALIDATION |
|---|---:|---:|
| Raw rows | 12,667 | 6,644 |
| Raw positive prevalence | 0.500039 | 0.139524 |
| Eligible rows | 12,545 | 6,535 |
| Dropped missing-cache rows | 122 | 109 |
| Positive / negative eligible rows | 6,300 / 6,245 | 923 / 5,612 |
| Eligible prevalence | 0.502192 | 0.141239 |
| Receptor keys / SMILES keys | 1,010 / 3,779 | 800 / 1,750 |
| Unique exact receptor–SMILES pairs | 10,370 | 6,114 |
| Extra repeated-pair rows | 2,175 | 421 |
| Pairs repeated within split | 1,185 | 333 |
| Pairs with both labels | 268 | 65 |
| Rows belonging to conflicting pairs | 975 | 163 |

Keys use `astype(str).str.strip()` exactly as dataset code; no chemical canonicalization. “Duplicates” here means duplicate model-input pair, not necessarily a fully identical CSV record.

Cache key counts: Raygun 1,238; BRICS 7,142; COATI 7,130; whole COUNT cache 10,228. Whole cache path: `/hpc/group/singhlab/user/cy244/projects/peptides/count_combined_embeddings/morgan_embeddings.pt`. Eligible rows require receptor in Raygun AND SMILES in all three drug caches. The whole cache contains more keys than the supplied 7,107 eligible-smiles population; cache size is not an eligible-population count.

| Key | Shared TRAIN/VAL unique keys | VAL rows seen / unseen in TRAIN | Seen / unseen VAL prevalence |
|---|---:|---:|---:|
| Receptor | 711 | 6,419 / 116 | 0.140676 / 0.172414 |
| SMILES | 556 | 5,263 / 1,272 | 0.125214 / 0.207547 |
| Receptor–SMILES pair | 899 | 1,120 / 5,415 | 0.309821 / 0.106371 |

Among 908 validation rows whose exact training pair has a single consistent training label, **206 (22.69%) disagree** with that label. Training receptor count median/p90/max: 7/24/398; drug count median/p90/max: 1/2/364. Thus many drugs are singletons, while evaluation rows heavily emphasize a minority of shared drugs.

## TRAIN-only frequency baselines, scored only on VALIDATION
Rate predictor: `(training_positive_count + alpha * training_prevalence)/(training_count + alpha)`, unseen keys fall back to training prevalence. Alpha values below are a disclosed diagnostic grid, not test-based selection. Counts and rates preserve every original eligible training row.

| Predictor | alpha=0 AP | alpha=1 AP | alpha=10 AP | alpha=100 AP |
|---|---:|---:|---:|---:|
| Receptor positive rate | 0.443305 | 0.466372 | 0.440531 | 0.389465 |
| Drug positive rate | 0.392927 | 0.399518 | 0.321572 | 0.288379 |
| Exact pair positive rate | 0.291206 | 0.322019 | 0.317261 | 0.316045 |

Raw training occurrence-count AP: receptor 0.288327; drug 0.154436; pair 0.262320. Additive receptor+drug logistic regression AP **0.528984** (single fixed C=1). Rate alpha=1 AUROC: receptor 0.829261, drug 0.771433, pair 0.653787. Pair lookup alone is weak despite overlap; neither marginal priors nor exact-pair memorization demonstrate sufficient performance for the goal.

## Files Retrieved
Paths below are relative to `/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics` unless absolute.
1. `src/concisejepa/datamodules/dataloader.py` lines 282–360 — string normalization and exact eligible-row filtering.
2. `src/spikes/phase1/fragment_datamodule.py` lines 240–334 — base cache intersections, lookup, fit/test setup separation.
3. `src/concisejepa/datamodules/fragment.py` lines 143–212 — dual-view adds whole-cache intersection, requires count fingerprints.
4. `src/concisejepa/lightning_modules/lit_fragment.py` lines 1–45 — standard runner delegates to spike objective.
5. `src/spikes/phase1/lit_fragment.py` lines 131–155 and 197–251 — positive-only contrastive targets, relative negative-diagonal penalty, additive JEPA loss.
6. `src/concisejepa/lightning_modules/lit_secondary_binding.py` lines 12–54 (targeted grep) — fragment plus 0.25 whole branch loss and pooled branch metrics.
7. `../concisejepa_brics_training/20260908-2120-f2r-secondary-binding/training/brics-f2r-secondary-fsq3-b6476734/resolved_config.yaml` lines 1–161 — actual cache paths, batch size 256, MSE JEPA, negative weight 1.0, whole loss weight 0.25, 30 epochs.
8. Absolute train/val CSVs above — all rows, loaded by pandas; embedding `.pt` dictionaries loaded on CPU for keys only. Test split untouched.

## Key Code and Architecture
`BindingDBDataset` strips strings and filters membership, without deduplication. Dual-view uses `fragment ∩ COATI ∩ whole` SMILES and Raygun sequences. Our probe replicated this directly without constructing a datamodule, avoiding any accidental test setup or cache generation.

Actual training objective is NOT direct supervised binary cross-entropy on observed pair labels:
- `pos_mask = labels > 0.5`; contrastive row/column cross-entropy is computed only for positive anchor rows.
- Negative rows get `relu(diag_scores - row_means + margin)` against in-batch reference similarities (margin 0 in this run).
- `loss = loss_dti + loss_jepa + negative_diagonal_weight * loss_neg_diag`.
- Secondary binding adds whole-branch losses at weight 0.25.

This trains relative in-batch retrieval and auxiliary representation prediction, whereas pooled AP compares observed positive/negative pairs globally. In-batch unmatched pairs need not be verified biological negatives. This is a plausible objective mismatch, not a measured causal diagnosis.

## Start Here / recommended validation-only diagnostics
Start with `src/spikes/phase1/lit_fragment.py:131–251`, not further fingerprint engineering.

1. **Objective-controlled ablation:** freeze the existing representation and fit a small supervised pair scorer using all eligible TRAIN labeled pairs (BCEWithLogits); compare with the existing score on the unchanged full VALIDATION set. Then, if useful, test a narrowly scoped BCE auxiliary/replacement loss with matched seed, training budget and same rows. Frozen readout first distinguishes scoring/objective limitations from missing representational signal. Calibration-only transformations cannot improve AP.
2. **Honest representation probe:** fixed-budget train-only regularized classifier on pooled Raygun plus whole COUNT/COATI pair features, including controlled low-rank interactions; compare against the 0.528984 identity baseline. Keep hyperparameter budget small and declared. This tests whether rich existing caches contain learnable interaction signal without yet changing FSQ/fragment machinery. No current probe establishes their achievable AP.
3. **Validation error decomposition:** report unchanged pooled AP plus receptor/drug/pair seen-vs-unseen strata, per-target AP where both labels exist, and receptor-cluster bootstrap uncertainty for baseline-versus-model differences. Retain duplicates/conflicts in primary metrics; report their error contribution diagnostically. Fit any baseline+model residual combination using training or training-only out-of-fold predictions, never validation labels as training inputs.

## Residual risks
No test distribution was audited by design; no test-based selection is authorized. Exact-string keys do not detect chemically equivalent alternate SMILES or homologous receptors. No model predictions were available in this probe, so complementarity, statistical significance, and whether 0.65 is reachable remain unknown. Duplicate/conflict effects are descriptive, not a license to alter filtering. Repeated validation tuning can overfit this split; preregister a small comparison set and reserve test for the final locked evaluation.