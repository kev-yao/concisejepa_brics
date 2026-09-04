# BRICS experiment audit: first results

Status, September 4, 2026: **18 initial runs completed successfully; 12 matched-input
follow-up runs are underway.** This document reports the completed comparison, not the
outcome of the unfinished follow-up. The [protocol](brics-audit-experiments.md) describes
configuration, checkpoint selection, controls, and reproduction.

The main finding is that fragments contain useful molecular identity information,
but the original joint FSQ setup preserves much less of it than the continuous
JEPA-only control. Reliable molecular generation and fragment-level binding
explanations have not been demonstrated. A numerical fingerprint-cache mismatch
requires a matched-input follow-up before selecting a research architecture.

## Completed comparison

All conditions use the same cleaned canonical-molecule split, seeds 42/43/44,
30 epochs, batch size 256, optimizer, and learning rate. Values below are means ±
sample standard deviations across training seeds, not confidence intervals.

**Important:** these initial runs inherit binary whole-molecule fingerprints and
count fragment fingerprints. Both cache metadata files claimed counts. We discovered
the mismatch by inspecting values, not by inferring it from model performance.

| Representation and objective | Validation AP | Test AP | Unique validation JEPA MSE | ECFP Tanimoto, all 100 samples |
|---|---:|---:|---:|---:|
| FSQ, joint | 0.7432 ± 0.0088 | 0.6404 ± 0.0248 | 0.1390 ± 0.0094 | 0.1093 ± 0.0087 |
| Continuous, joint | 0.7626 ± 0.0031 | 0.6802 ± 0.0053 | 0.0754 ± 0.0030 | 0.1733 ± 0.0109 |
| FSQ, JEPA only | untrained binding head | untrained binding head | 0.0906 ± 0.0021 | 0.1571 ± 0.0051 |
| Continuous, JEPA only | untrained binding head | untrained binding head | 0.0473 ± 0.0002 | 0.2360 ± 0.0174 |
| FSQ, DTI + alignment | 0.7365 ± 0.0052 | 0.6144 ± 0.0381 | untrained predictor | — |
| Continuous, DTI + alignment | 0.7573 ± 0.0086 | 0.6638 ± 0.0084 | untrained predictor | — |

Binding columns use the checkpoint selected by pooled validation AP. Reconstruction
columns use the checkpoint selected by unique-molecule validation MSE. They need
not be the same checkpoint; the CSV also records binding AP at the reconstruction
checkpoint. Test scores did not select checkpoints or tune these conditions.

The unique-training-mean target baseline has validation MSE **0.1741**; the zero
baseline is **0.2246**. Each run saves its actual pre-training MSE separately.

## What the experiments establish

### 1. The fragment predictor learns identity, but decoding remains weak

| Objective | FSQ top-10 retrieval | Continuous top-10 retrieval |
|---|---:|---:|
| Joint | 15.0% | 80.3% |
| JEPA only | 67.3% | 96.0% |

Retrieval uses the same 100 validation queries and all 1,050 validation target
embeddings as candidates. Random ranks give approximately 0.95% top-10 accuracy.
Continuous JEPA-only top-1 retrieval is 76.3%. This is a finite-catalog identity
diagnostic, not de novo molecular generation.

Of the 100 queries, 91 have multiple cached fragments. Continuous JEPA-only top-10
retrieval on those 91 is 95.6%, so the result is not explained solely by intact
molecules appearing as single-fragment inputs.

Continuous JEPA-only decoding is valid for 69/63/69 molecules across seeds and gives
only 2/2/3 exact largest-component matches. Its all-sample ECFP mean is 0.2360 and
valid-only mean 0.3521. Feeding the *true* COATI targets to the same greedy decoder
produces 79 valid molecules, 35 exact matches, and all-sample ECFP 0.5343. Thus both
prediction error and the target/decoder pipeline limit reconstruction. The true-latent
control is a reference, not a guaranteed upper bound for every individual example.

All decodes use the same selection seed, greedy `k=1`, and decode seed 123. Invalid
decodes contribute zero to all-sample Tanimoto. These are not directly comparable
to the earlier unseeded stochastic 25-test-molecule trial.

### 2. The limited representation and joint objectives both matter

With JEPA-only training, continuous reduces MSE relative to FSQ by 0.0433 on average;
the improvement occurs in all three seeds. Adding DTI **and alignment together**
increases MSE by 0.0484 for FSQ and 0.0281 for continuous, also in every seed.
This supports objective competition in this setup, but does not isolate alignment
from DTI as its cause.

The continuous control bypasses the entire FSQ block, including the 3-D bottleneck,
and exposes a bounded 128-D representation. It is **not** a rounding-only ablation.
Do not conclude that discretization itself is the problem. A matched 3-D continuous
control is needed to distinguish dimensionality from quantization.

Gradient diagnostics on one shared encoder projection also show that equal scalar
loss weights do not imply equal gradient magnitudes. Directions vary across seeds
and checkpoints, so one gradient cosine is not a global proof of interference.

### 3. JEPA's incremental binding benefit is modest in this comparison

Adding JEPA to DTI + alignment raises validation AP by 0.0067 on average for FSQ
(paired differences +0.0029/+0.0074/+0.0098). For continuous, the mean increase is
0.0053, with paired differences −0.0075/+0.0124/+0.0111. This is preliminary evidence,
not a robust large binding gain. The matched-count follow-up will repeat this test.

A training-only receptor-frequency baseline, with fixed smoothing pseudocount 10,
already reaches validation AP **0.7186** and test AP **0.6505** without molecular
features. Global AP alone therefore does not establish ligand-specific binding.

Within the 128 validation receptors having at least five rows and both labels:

| Model | Macro within-receptor AUROC | Pair-weighted within-receptor AUROC |
|---|---:|---:|
| FSQ, joint | 0.5766 | 0.7133 |
| Continuous, joint | 0.6683 | 0.7927 |
| FSQ, DTI + alignment | 0.5771 | 0.7221 |
| Continuous, DTI + alignment | 0.6434 | 0.7827 |

These evaluate ligand ranking for the same receptor, where a receptor-only constant
score has AUROC 0.5. The subset covers 1,181/3,248 validation rows; it is not the
whole validation population. Macro and pair-weighted metrics answer different
questions and should remain separately labeled.

### 4. Fragment importance is not yet a validated output

Leave-one-fragment-out tests use the same 20 validation molecules with a positive
and negative receptor example. FSQ joint models change their top-ranked fragment
between receptors in 0/0/1 cases across seeds; continuous joint models do so in
4/2/3 cases. Effects can differ in magnitude even when the top index stays the same.

This measures model sensitivity, not causal binding contribution. Masking a cached
fragment is not necessarily a chemically valid perturbation. Do not present these
rankings as experimentally established binding fragments or infer fragment identities
from a newly recomputed BRICS ordering.

### 5. Cosine probabilities have valid bounds, but problematic boundaries

The nonnegative-cosine argument is mathematically correct: the implemented score
lies in [0,1]. However, at the binding-selected checkpoints, the averaged continuous
joint score assigns **exactly zero** to 19/20/17 of 764 positive validation pairs.
The FSQ joint counts are 1/1/19. Disjoint ReLU supports can have zero local gradients
as well as zero probability; shared updates from other examples can still move them.

Brier scores are reported separately from ranking metrics. The mean validation
Brier is 0.1070 for continuous joint, 0.1097 for FSQ joint, and 0.1117 for the receptor
prior. Neither valid bounds nor good ranking establishes probability calibration.
The head was not changed during this study.

## Data and measurement caveats

- The cleaned split has 16,969/3,248/4,155 training/validation/test pairs and
  4,904/1,050/1,052 canonical molecules. Molecules are disjoint; scaffolds and
  proteins are not. Only 57 validation and 56 test rows involve unseen proteins.
- We excluded 804 conflicting binary-label pairs (3,316 rows), removed 4,492
  consistent duplicates, and filtered 419 missing-embedding rows. Conflicts can
  reflect assay context; exclusion does not establish which original label was true.
- 816 of 7,006 canonical molecules share exactly identical cached fragment inputs
  with another molecule before FSQ. Examples include stereoisomers and differently
  assembled molecules. The cache loses information the predictor cannot recover
  uniquely from its inputs.
- 103/1,050 validation molecules and 91/1,052 test molecules share fragment inputs
  with training molecules. Novel-input subset metrics are saved. On the 2,861
  validation rows with novel fragment multisets, joint validation AP remains 0.7489
  for FSQ and 0.7635 for continuous, versus a same-subset receptor prior of 0.7178.
- Pair sampling weights molecules unevenly. The ten most frequent molecules account
  for 2,370/16,969 training rows and 1,827/3,248 validation rows. Unique-molecule JEPA
  validation fixes evaluation weighting; training sampling was deliberately unchanged.
- One molecular split and three training seeds do not measure uncertainty over
  datasets or scaffolds. The original source is preprocessed; its label prevalence
  is not natural binding prevalence. Protein-cache provenance is inherited.

## The matched-count follow-up

The new whole-molecule cache contains actual counts (maximum 72). For **all 7,006
molecules**, its nonzero pattern exactly matches the legacy binary fingerprint.
This makes the follow-up a clean input-value comparison on the same molecular rows.
The source caches and their metadata were left untouched.

Four binding-trained conditions × three seeds run as Slurm array **54380032**,
with at most two Singh Lab GPUs concurrently. The first tasks started at 17:40–17:41
EDT after slots became available earlier than the scheduler's estimate. The six JEPA-only
controls are reused because their active objective is independent of whole inputs;
bit-exact three-step optimizer parity was tested for both representations. This is
**12 new training runs**, making 30 planned trained runs overall, not 36 independent
replications.

Report job **54380099** runs after the follow-up array ends and writes metrics,
paired-seed comparisons, conditional diagnostics, and figures. It reports the number
of completed runs explicitly even if an array task fails. No matched-count result is
claimed in the table above.

For new research, use the verified matched-count data configuration. Retain the
legacy configuration for reproducibility; do not silently reinterpret or overwrite
its caches. Architecture selection should wait for the matched-input results.

After that comparison, the most informative next questions are a matched 3-D
continuous bottleneck, an alignment-only ablation, and receptor-specific ranking
evaluation. Increasing model size or decoding more examples would not resolve those
confounders by itself. These additional experiments have **not** been launched.

## Artifacts and checks

Cluster artifacts are under the repository's `scratch/` symlink and are not in GitHub:

- [Initial suite report](../scratch/audits/cold_molecule_v1/REPORT.md),
  [comparison figure](../scratch/audits/cold_molecule_v1/comparison.png), and
  [per-run CSV](../scratch/audits/cold_molecule_v1/suite_summary.csv).
- [Continuous JEPA-only random examples](../scratch/audits/cold_molecule_v1/continuous_jepa/seed_42/random_reconstructions.svg):
  the first six sampled molecules, not selected successes.
- [Matched-input report](../scratch/audits/cold_molecule_matched_count_v1/REPORT.md)
  and [study/reuse manifest](../scratch/audits/cold_molecule_matched_count_v1/study_manifest.json).
- [Count-cache audit](../scratch/datasets/cold_molecule_v1/count_fingerprint_audit.json)
  and [data split manifest](../scratch/datasets/cold_molecule_v1/manifest.json).
- Each run contains resolved configuration, git provenance, initial metrics,
  epoch logs, checkpoints, row predictions, and reconstruction/sensitivity diagnostics.

Pooled re-evaluation of the original `epoch=24-auprc=0.6001.ckpt` gives validation AP
0.5915 and test AP 0.5817 (test AUROC 0.8876), rather than the old batch-averaged
test AP of approximately 0.5874. Its overlapping source splits and different label
prevalence make it incomparable to the cleaned-split table. Recomputing metrics also
cannot recover discarded checkpoints for a retrospective pooled-AP selection.

Validation: 33 relevant tests pass, including existing refactor parity tests, pooled
metric/reset checks, molecule-split integrity, continuous gradient isolation, numerical
count-cache rejection, and JEPA-only whole-input optimizer parity. A full one-epoch
GPU audit smoke run completed before the suite. All six initial array tasks exited
successfully. Historical residual-quantizer test failures were reproduced unchanged
on the preceding source version; see the protocol for details.
