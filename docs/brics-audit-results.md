# BRICS experiment audit: completed results

Status, September 4, 2026: **all 30 training runs completed successfully**: 18 initial
runs and 12 matched-input follow-up runs. Six JEPA-only controls are reused across
the two comparisons, not counted as independent reruns. The [protocol](brics-audit-experiments.md) describes
configuration, checkpoint selection, controls, and reproduction.

The main finding is that fragments contain useful molecular identity information,
but the original joint FSQ setup preserves much less of it than the continuous
JEPA-only control. Reliable molecular generation and fragment-level binding
explanations have not been demonstrated. Correcting a numerical fingerprint-cache
mismatch did not remove the representation gap. JEPA gives modest global binding AP
gains, but no convincing improvement in ranking ligands within the same receptor.

## Primary result: matched count fingerprints

All conditions use the same cleaned canonical-molecule split, seeds 42/43/44,
30 epochs, batch size 256, optimizer, and learning rate. Values below are means ±
sample standard deviations across training seeds, not confidence intervals.

In the binding-trained conditions, both molecular views now use verified count
fingerprints. The JEPA-only rows reuse controls whose active objective is independent
of whole inputs. The initial suite inherited
binary whole-molecule fingerprints and count fragment fingerprints, although both
metadata files claimed counts. We discovered this by inspecting values. The
matched-input follow-up changed only the whole-molecule cache, not the split or losses.

| Representation and objective | Validation AP | Test AP | Unique validation JEPA MSE | ECFP Tanimoto, all 100 samples |
|---|---:|---:|---:|---:|
| FSQ, joint | 0.7436 ± 0.0029 | 0.6490 ± 0.0401 | 0.1506 ± 0.0082 | 0.1057 ± 0.0137 |
| Continuous, joint | 0.7615 ± 0.0077 | 0.6764 ± 0.0235 | 0.0739 ± 0.0005 | 0.1894 ± 0.0094 |
| FSQ, JEPA only | untrained binding head | untrained binding head | 0.0906 ± 0.0021 | 0.1571 ± 0.0051 |
| Continuous, JEPA only | untrained binding head | untrained binding head | 0.0473 ± 0.0002 | 0.2360 ± 0.0174 |
| FSQ, DTI + alignment | 0.7305 ± 0.0088 | 0.6453 ± 0.0100 | untrained predictor | — |
| Continuous, DTI + alignment | 0.7528 ± 0.0160 | 0.6663 ± 0.0234 | untrained predictor | — |

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
| Joint | 9.3% | 84.7% |
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
increases MSE by 0.0600 for FSQ and 0.0266 for continuous, also in every seed.
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

Adding JEPA to DTI + alignment raises validation AP by 0.0130 on average for FSQ
(paired differences +0.0238/+0.0076/+0.0076). For continuous, the mean increase is
0.0088, with paired differences +0.0083/+0.0260/−0.0080. This is preliminary evidence,
not a robust large binding gain. Test AP increases by only 0.0037 for FSQ and 0.0101
for continuous on average, with substantial seed variation.

A training-only receptor-frequency baseline, with fixed smoothing pseudocount 10,
already reaches validation AP **0.7186** and test AP **0.6505** without molecular
features. Global AP alone therefore does not establish ligand-specific binding.

Within the 128 validation receptors having at least five rows and both labels:

| Model | Macro within-receptor AUROC | Pair-weighted within-receptor AUROC |
|---|---:|---:|
| FSQ, joint | 0.5695 | 0.7132 |
| Continuous, joint | 0.6652 | 0.7985 |
| FSQ, DTI + alignment | 0.5919 | 0.7155 |
| Continuous, DTI + alignment | 0.6634 | 0.7964 |

These evaluate ligand ranking for the same receptor, where a receptor-only constant
score has AUROC 0.5. The subset covers 1,181/3,248 validation rows; it is not the
whole validation population. Macro and pair-weighted metrics answer different
questions and should remain separately labeled. Within-receptor ranking barely
changes for continuous when JEPA is added, and macro AUROC worsens for FSQ. Thus
the global AP gain should not be described as established binding-specific benefit.

### 4. Fragment importance is not yet a validated output

Leave-one-fragment-out tests use the same 20 validation molecules with a positive
and negative receptor example. FSQ joint models change their top-ranked fragment
between receptors in 0/0/0 cases across seeds; continuous joint models do so in
3/4/1 cases. Effects can differ in magnitude even when the top index stays the same.

This measures model sensitivity, not causal binding contribution. Masking a cached
fragment is not necessarily a chemically valid perturbation. Do not present these
rankings as experimentally established binding fragments or infer fragment identities
from a newly recomputed BRICS ordering.

### 5. Cosine probabilities have valid bounds, but problematic boundaries

The nonnegative-cosine argument is mathematically correct: the implemented score
lies in [0,1]. However, at the binding-selected checkpoints, the averaged continuous
joint score assigns **exactly zero** to 12/28/17 of 764 positive validation pairs.
The FSQ joint counts are 5/1/4. Disjoint ReLU supports can have zero local gradients
as well as zero probability; shared updates from other examples can still move them.

Brier scores are reported separately from ranking metrics. The mean validation
Brier is 0.1128 for continuous joint, 0.1095 for FSQ joint, and 0.1117 for the receptor
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
  validation rows with novel fragment multisets, joint validation AP is 0.7463
  for FSQ and 0.7587 for continuous, versus a same-subset receptor prior of 0.7178.
- Pair sampling weights molecules unevenly. The ten most frequent molecules account
  for 2,370/16,969 training rows and 1,827/3,248 validation rows. Unique-molecule JEPA
  validation fixes evaluation weighting; training sampling was deliberately unchanged.
- One molecular split and three training seeds do not measure uncertainty over
  datasets or scaffolds. The original source is preprocessed; its label prevalence
  is not natural binding prevalence. Protein-cache provenance is inherited.

## What the input-cache correction changed

The new whole-molecule cache contains actual counts (maximum 72). For **all 7,006
molecules**, its nonzero pattern exactly matches the legacy binary fingerprint.
This makes the follow-up a clean input-value comparison on the same molecular rows.
The source caches and their metadata were left untouched.

The four binding-trained conditions × three seeds completed as Slurm array
**54380032**, using at most two Singh Lab GPUs concurrently. The six JEPA-only
controls were reused because their active objective is independent of whole inputs;
bit-exact three-step optimizer parity was tested for both representations. This is
**12 new training runs**, making 30 completed trained runs overall, not 36 independent
replications.

Report job **54380099** completed and wrote metrics, paired-seed comparisons,
conditional diagnostics, and figures. All four follow-up array tasks exited with
code zero; all selected checkpoint scores agree with the saved epoch records.

For reference, the initial mixed-input suite was:

| Binding-trained condition | Validation AP | Test AP | Unique JEPA MSE |
|---|---:|---:|---:|
| FSQ, joint | 0.7432 | 0.6404 | 0.1390 |
| Continuous, joint | 0.7626 | 0.6802 | 0.0754 |
| FSQ, DTI + alignment | 0.7365 | 0.6144 | untrained |
| Continuous, DTI + alignment | 0.7573 | 0.6638 | untrained |

The count correction changed joint validation AP by +0.0003 for FSQ and −0.0011
for continuous on average; individual seeds moved both ways. FSQ joint reconstruction
MSE **increased** by 0.0116 in the paired comparison, while continuous changed by
−0.0016. Continuous decoded ECFP improved by 0.0161. Input consistency is important
for interpreting the experiment, but is not a universal performance fix.

For new research, use the verified matched-count data configuration. Retain the
legacy configuration for reproducibility; do not silently reinterpret or overwrite
its caches. The continuous JEPA-only model is the strongest chemical-representation
reference in this study; it is not a trained DTI model and has no discrete codebook.
For quantized research, retain FSQ as the reference and investigate the bottleneck
and competing objectives rather than treating the continuous ablation as a final replacement.

The most informative next experiments are a matched 3-D continuous bottleneck and
an ablation that removes alignment while retaining DTI + JEPA, evaluated with the
same receptor-specific ranking metrics. Increasing model size or decoding more examples would not resolve those
confounders by itself. These additional experiments have **not** been launched.

## Artifacts and checks

Cluster artifacts are under the repository's `scratch/` symlink and are not in GitHub:

- [Initial suite report](../scratch/audits/cold_molecule_v1/REPORT.md),
  [comparison figure](../scratch/audits/cold_molecule_v1/comparison.png), and
  [per-run CSV](../scratch/audits/cold_molecule_v1/suite_summary.csv).
- [Continuous JEPA-only random examples](../scratch/audits/cold_molecule_v1/continuous_jepa/seed_42/random_reconstructions.svg):
  the first six sampled molecules, not selected successes.
- [Matched-input report](../scratch/audits/cold_molecule_matched_count_v1/REPORT.md)
  with its [comparison figure](../scratch/audits/cold_molecule_matched_count_v1/comparison.png)
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
successfully, as did all four follow-up tasks. Across all 30 distinct runs, every
epoch metric is finite, all 30 epoch records are present, checkpoint selections
match independently evaluated scores, and clean-split CSV hashes are unchanged.
Historical residual-quantizer test failures were reproduced unchanged
on the preceding source version; see the protocol for details.
