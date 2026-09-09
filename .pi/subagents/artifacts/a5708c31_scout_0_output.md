# Code Context

## Files Retrieved
1. `scripts/analyze_codebook_utilization.py:22-134` — checkpoint/cache selection, encoding, statistics.
2. `codebook_utilization.json:1-28` — reported occupancy and fragment counts.
3. `src/spikes/phase1/fragment_datamodule.py:58-153,176-224,263-330` — preprocessing, validity, padding, sampling.
4. `src/concisejepa/datamodules/fragment.py:21-117,200-215` — metadata validation and adapter.
5. `src/concisejepa/datamodules/dataloader.py:282-329` — CSV-row dataset.
6. `src/concisejepa/evals/FSQ_monitor.py:31-40,109-120,148-173,375-418,494-503` — factor aggregation and callback input.
7. `src/spikes/phase1/fragment_train.py:53-99,170-249,257-270` — reference defaults and provenance limitations.
8. `configs/data/brics_bindingdb.yaml:1-28`; `configs/callbacks/brics_fsq.yaml:1-28`; `docs/experiments.md:49-79,129-136` — promoted training contract.
9. `tests/test_brics_hydra_integration.py:121-249`; `qa/test_fragment_pooling.py:94-123` — cheap synthetic seams.
10. `job.sh:25-36,137-177`; `launchers/train.sbatch:28-79` — legacy versus configurable launcher.

## Key Findings

### High: utilization artifact does not establish collapse on the actual BindingDB training distribution
`scripts/analyze_codebook_utilization.py:26-29,61-76` encodes **every entry of the older `count_combined_embeddings` fragment cache**, without CSV split selection or the training data module’s protein/COATI availability filters.

`src/spikes/phase1/fragment_train.py:91-99` explicitly warns that these defaults are stale and says real runs overrode them to `BindingDB_embeddings`. That comment is not independently verified run configuration, but makes the mismatch a concrete concern.

The artifact (`codebook_utilization.json:2-8`) establishes, conditional on correct code extraction:
- 1,073 observed codes out of 32,768;
- entropy 6.925 bits, equivalent to approximately **121.5 equally likely codes**;
- 45,654 fragment occurrences across **7,148 cache SMILES**, mean 6.387 fragments/molecule.

It does **not** establish one-code collapse, temporal deterioration, training-set occupancy, or collapse caused by pooling. It lacks initialization/control measurements and unique-input counts. The denominator is theoretical capacity, not the number of distinct fragment inputs available.

### High: old monitor destroys factor tuples when writing code distributions
`FSQ_monitor.py:120` produces factor vectors; `:148-173` aggregates them as `[batch, layers, factors]`. Then `:375-386` mistakes the factor axis for sequence and takes its mode:

```python
seq_codes = codes[b, l, :].flatten().astype(int)
codes_2d[b, l] = np.bincount(seq_codes).argmax()
```

Consequently `:388-418` writes scalar-mode distributions, not joint FSQ triplet occupancy. Distinct tuples can map to the same scalar. This is separate from `codebook_utilization.json`, whose script counts tuples directly.

The callback also reads `batch[1]` without a fragment mask (`:494-503`); fragment batches supply `[B,F,D]`, not its assumed whole-molecule input. **Do not attach unchanged.** Current BRICS callbacks exclude it (`configs/callbacks/brics_fsq.yaml:1-28`), consistent with `docs/experiments.md:133-136`; therefore this defect is not evidence that it affected the identified run.

### Medium: failure fingerprints are valid tokens, despite “filtered downstream” comment
`fragment_datamodule.py:109,116` returns one zero fingerprint on certain failures. Cache construction retains it (`:135-139`); valid-SMILES filtering checks keys only (`:282`); collation marks every cached row valid (`:218-221`).

Thus repeated failure vectors can contribute a shared code and losses as real fragments. Actual incidence is unknown. Ordinary padding is masked false, and **the utilization script does not add padding** (`analyze_codebook_utilization.py:75-76`), so normal batch padding cannot explain that JSON unless already present in the cache.

### Medium: effective sample population differs between training and analysis
- BRICS decomposition returns a set, then retains at most 16 largest fragments (`fragment_datamodule.py:76,95-96`): within-molecule multiplicity is removed, but common fragments can recur across molecules.
- Cache keys are raw SMILES, not demonstrated canonical molecular identities.
- Analysis counts fragment occurrences, not unique fragment fingerprints.
- Training samples CSV interaction rows (`dataloader.py:297-328`), with shuffle but no molecule balancing or deduplication (`fragment_datamodule.py:312-324`). Drugs with many interactions therefore contribute repeatedly.
- Fragment counts affect analysis weighting: molecules with more fragments contribute more codes.

Unique-FP count, duplicate rate, zero rate, and per-factor occupancy are necessary before interpreting 3.27% as abnormal. Per-factor statistics are printed but **not persisted** by the script (`:104-108,121-130`).

### Medium: insufficient source/config provenance
The JSON contains no checkpoint/cache path, hashes, source commit, resolved model config, timestamp, split, or code histograms (`analyze_codebook_utilization.py:121-130`). Encoder architecture is hard-coded (`:42-48`) rather than reconstructed from checkpoint configuration.

Fragment metadata validates declared type/kind/length/cap (`src/concisejepa/datamodules/fragment.py:21-55`), but not content, source CSVs, chemistry-library versions, duplicate counts, or actual count-valued fingerprints. Spike loading bypasses this validation (`fragment_datamodule.py:263-282`).

## Exact Run Located

Only the specifically referenced external run directory was inspected:

`/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study/fragment_latent_query_719fa566`

Its `summary.json:2-11` confirms:
- pooling `latent_query`;
- checkpoint `checkpoints/epoch=29-val/dti_auprc=0.5587.ckpt`;
- test AUPRC `0.5821300148963928`, AUROC `0.8852358460426331`.

`logs/version_0/hparams.yaml:1-2` records only `lr: 0.0001`, `weight_decay: 0.01`. No resolved config was found in the bounded run listing. “q4” is the analysis script’s label, not verified saved configuration. Checkpoint files exist but were not loaded.

The exact old cache exists, approximately **377 MB**; its `.meta.json:1-7` declares count ECFP4, 2048 dimensions, cap 16, and 7,148 SMILES—matching the artifact’s molecule count. This supports correspondence, not cryptographic identity.

## Architecture

CSV interaction rows → embedding-key filtering → cached variable-length fragment tensors → padded fragment batch plus mask → fragment encoder/pooler. The promoted Hydra data module wraps the spike data module and adds metadata checks.

The offline utilization script bypasses that entire sampling/filtering path and loads only the checkpoint’s drug encoder. The old monitor is another independent path, manually mirroring encoder internals.

## Start Here

Open `scripts/analyze_codebook_utilization.py:22-90` first: establish the intended dataset and exact encoder code shape before treating the artifact as model-collapse evidence.

Cheap future checks, without training:
- Synthetic tuple histogram regression: distinct factor tuples must remain distinct.
- Synthetic collator test covering failed zero fingerprints versus padded rows.
- Metadata missing/mismatch tests using temporary small caches.
- Existing `tests/test_brics_hydra_integration.py:154-198` exercises a tiny cache/batch contract.
- Existing `qa/test_fragment_pooling.py:97-123` checks padding invariance for mean/weighted-sum pooling.
- Recompute statistics on a deliberately bounded sample with unique-FP and per-factor counts, preserving masks and provenance. No such encoding was performed here.

Environment: requested group path resolves physically to `/hpc/home/me196/projects/obsidian/runs/concisejepa_brics`; source HEAD `63b6f22a40b34f427884d77ed951be1dc94a7423`, origin `https://github.com/kev-yao/concisejepa_brics.git`. Available default Python is `/hpc/group/singhlab/tools/conda/miniconda3/bin/python`, version 3.11.4. Legacy launcher activates `/hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu`; dependency availability was not tested.