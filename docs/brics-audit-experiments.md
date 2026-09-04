# BRICS audit and ablation protocol

The original run `brics-dual-view-171419e2` remains an exploratory reference. Its DTI
metrics averaged batch AUPRC/AUROC, and its source CSVs contain overlapping and
contradictory drug–protein pairs. Neither its headline scores nor its checkpoint
selection establish generalization. Recomputing pooled scores cannot retroactively
recover discarded epoch checkpoints.

## Dataset

`scripts/prepare_brics_audit_data.py` filters for available embeddings, canonicalizes
full molecules with stereochemistry and all components, and groups pairs by canonical
SMILES plus protein sequence. It excludes every pair with conflicting binary labels,
deduplicates consistent pairs, then assigns canonical molecules to train/validation/test
with seed 17 and 70/15/15 molecule fractions. A stable raw SMILES representative keeps
cache lookup compatible. It never changes the source CSVs or embedding caches.

Dataset: `scratch/datasets/cold_molecule_v1`. Counts: train 16,969 pairs / 4,904
molecules; validation 3,248 / 1,050; test 4,155 / 1,052. Excluded: 804 contradictory
pairs (3,316 rows), 4,492 repeated consistent rows, and 419 missing-embedding rows.
Its manifest contains input/output hashes and counts. Excluded conflicts are saved
for provenance, not silently resolved into positive or negative examples.

This tests unseen molecules, **not unseen scaffolds**. Proteins can overlap; metrics
also separate seen and unseen proteins. The source was already preprocessed, so its
class prevalence is not an estimate of natural binding prevalence. Existing cached
protein embedding provenance is inherited and is not independently established here.

## Fixed comparison

Each condition uses seeds 42, 43, 44, 30 epochs, batch size 256, AdamW, learning rate
1e-4, weight decay 1e-2, and full-precision training. No early stopping or parameter
tuning on the test set. All conditions are fixed before viewing their results.

| Arm suffix | DTI weight | JEPA weight | Alignment weight |
|---|---:|---:|---:|
| joint | 1 | 1 | 1 |
| jepa | 0 | 1 | 0 |
| dti | 1 | 0 | 1 |

Each objective is crossed with `fsq` and `continuous` representations, making six
conditions and 18 runs. The `dti` conditions retain alignment to isolate JEPA's
incremental contribution versus `joint`. Their untrained JEPA heads are not interpreted.
Likewise, JEPA-only conditions do not train a binding head; their DTI scores are controls,
not candidate binding models.

`continuous` bypasses the entire FSQ block and applies tanh to the shared 128-D encoder
output. It removes both quantization and the 3-D bottleneck. Thus this experiment cannot
attribute a difference specifically to rounding; a matched 3-D continuous control would
be a follow-up if warranted. FSQ retains the original forward path and parameters.

## Measurements

- Pooled split-wide AUPRC/AUROC, BCE, Brier score, and positive prevalence for mean,
  fragment-only, and whole-only binding predictions. Row predictions are saved.
- Best DTI checkpoint selected on pooled validation AUPRC; best reconstruction
  checkpoint selected on MSE giving each unique validation molecule equal weight.
  Both get validation and test DTI evaluation with separate labels in the output.
- Full unique-validation MSE before training; zero and unique-training-mean baselines.
  Baseline probes preserve RNG state and training mode is explicitly restored.
- For reconstruction-trained conditions, 100 common validation molecules, sample seed
  42, greedy COATI decoding (`k=1`, seed 123), valid and exact counts, ECFP radius-2
  2048-bit and RDK Tanimoto, rotated-target control, and true-latent decoder control.
  Invalid decodes score zero in all-sample means; valid-only means remain separate.
  Molecular comparison currently uses the largest component and is explicitly limited
  to that component. ECFP uses the existing default without chirality.
- Latent nearest-neighbor retrieval against the 1,050 validation targets distinguishes
  embedding identity from generative decoding; it is a finite-catalog diagnostic.
- Initial and trained loss-gradient norms/cosines on the last shared encoder projection,
  on a fixed sample of training pairs, with dropout disabled for the diagnostic.
  These describe that parameter block, not all gradient interactions.
- Leave-one-fragment-out sensitivity on up to 20 validation molecules having both
  positive and negative receptor examples. This measures receptor-dependent model
  sensitivity, not causal binding contributions. Cached token indices cannot be assumed
  to map to a newly recomputed BRICS ordering.

The original decoder harness used a fixed molecule-selection seed but unseeded
stochastic decoding. The updated standalone harness exposes `--decode-seed`, `--top-k`,
and `--split`; its default greedy decoding changes the protocol and should not be
compared as if it were the same trial.

## Running and interpreting

```bash
export CONCISEJEPA_AUDIT_DATA_ROOT="$PWD/scratch/datasets/cold_molecule_v1"
export AUDIT_ROOT="$PWD/scratch/audits/cold_molecule_v1"
export CONCISEJEPA_ENV=/path/to/environment
sbatch launchers/brics_audit.sbatch
```

The array runs one objective/representation combination per task, three seeds per
task, with at most two Singh Lab GPUs concurrently. Existing completed runs are skipped;
incomplete run directories cause an error so retries cannot silently overwrite evidence.
Each run saves its resolved Hydra configuration, git manifest, initial metrics,
checkpoints, epoch metrics, row predictions, reconstruction examples, fragment sensitivity,
and an `audit_summary.json`. JSON epoch output now records metrics after training-epoch
aggregation; test metrics from the standard runner have a separate `test_metrics.json`.

Primary questions are paired-seed differences: does continuous versus FSQ improve
reconstruction under JEPA-only training; does adding DTI/alignment worsen reconstruction;
and does adding JEPA improve binding under the same representation and alignment?
Use validation to choose follow-ups. Test results assess the fixed comparisons and must
not become a repeated hyperparameter-selection loop. Three seeds and one molecular split
are preliminary evidence; uncertainty across dataset splits is not measured.

## Additional diagnostic controls

`brics_shortcut_baselines.py` fits a training-only per-protein positive-rate prior
with a fixed pseudocount of 10, and reports both held-out splits. It also identifies
distinct canonical molecules with exactly identical multisets of cached fragment
fingerprints. These controls do not change the training conditions.
It saves `fragment_input_groups.csv` so binding scores can be stratified by whether
the fragment multiset appeared in training, alongside receptor-prior scores on those
same subsets. Canonical-molecule disjointness alone does not imply novel model inputs.
With `--whole-cache`, it additionally checks whether any fragment token has the exact
whole-molecule fingerprint, and saves cached token counts. The conditional report
stratifies reconstruction into single/multiple-fragment and whole-token-present/absent
subsets; successful prediction from an intact-molecule token is not evidence of assembly.

`brics_conditional_metrics.py` computes validation ranking metrics within proteins
and within molecules, restricted to groups with at least five rows and both labels.
Macro AUROC weights groups equally; pair-weighted AUROC weights each within-group
positive/negative comparison equally. AP minus prevalence is only a descriptive
comparison with a constant predictor, not a permutation significance test.
The companion `probability_and_novelty_validation_metrics.json` reports the novelty
subsets and exact-zero/near-zero predictions on positive examples. Nonnegative cosine
is mathematically in [0,1], but that alone does not establish probability calibration;
Brier score and boundary counts check separate failure modes without changing the head.

`summarize_brics_audit.py` writes CSV, JSON, and a Markdown report, including paired
seed contrasts. `plot_brics_audit.py` exports comparison plots and the first six
random reconstruction examples from each seed-42 sample, without cherry-picking.
`launchers/brics_audit_report.sbatch` can run these after the training array finishes.

Validation: 33 relevant tests pass, including pooled-metric/reset checks, canonical
split integrity, continuous gradient isolation, unique-molecule validation weighting,
and existing training-refactor parity checks. The older `test_quantizers.py` suite
has six failures and two errors concerning historical residual-branch expectations;
the same failures were reproduced in an untouched checkout of the preceding commit.
One broad-suite temporary-directory cleanup error disappeared with local `/tmp`.

## Matched-fingerprint follow-up

During the audit, numerical inspection found that the legacy whole-molecule cache
is binary despite metadata labeling it `ecfp-count:4`. Fragment inputs are actual
counts. `prepare_brics_count_fingerprints.py` generated a separate count cache from
the same raw SMILES, using the existing count generator. For all 7,006 molecules,
binarizing the new count fingerprint exactly reproduces the legacy fingerprint.
Every regenerated molecule has at least one count above one. The new cache and its
hash audit are under `scratch/datasets/cold_molecule_v1`; no source cache or metadata
was changed. The legacy-input suite remains useful but is explicitly conditional
on this mismatch.

Follow-up: `matched_{fsq,continuous}_{joint,dti}` repeats the four binding-trained
conditions with the same splits, three seeds, 30 epochs, objectives, and checkpoint
rules. Only the whole-molecule cache changes. The matched data config enables a
value-level count check; legacy defaults are preserved. This is an adaptive diagnostic
follow-up prompted by a cache audit, not part of the initial fixed six-arm comparison.

The six JEPA-only runs are reused: whole inputs do not affect their active objective.
A test checks bit-exact three-step AdamW parameter/loss trajectories under binary vs
count whole inputs for both representation choices. Their unused DTI diagnostics are
not interpreted. A `study_manifest.json` records reuse explicitly; there are 12 new
training runs, not 18 independent reruns. Total study: 30 trained runs.

```bash
# After the legacy-input suite, retaining a two-GPU concurrency limit:
export AUDIT_ROOT="$PWD/scratch/audits/cold_molecule_matched_count_v1"
sbatch --array=0-1,4-5%2 --time=00:25:00 \
  --dependency=afterok:LEGACY_ARRAY_ID \
  launchers/brics_audit.sbatch --config-prefix matched
```

Reconstruction subset diagnostics use the regenerated count cache as their structural
whole-token reference for both suites. This does not change what either model sees.
