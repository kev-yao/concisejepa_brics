# Code Context

## Files Retrieved
All repository paths below are relative to `/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics`. Personally inspected; only this requested report was written. No source/config edits, tests or jobs.

1. `src/concisejepa/models/secondary_binding.py` (1–45): independent whole scoring, 75/25 score fusion.
2. `src/concisejepa/lightning_modules/lit_secondary_binding.py` (1–61): branch objectives, secondary weight, pooled metrics.
3. `src/spikes/phase1/fragment_encoder.py` (1–372): actual fragment implementation, sigmoid scorer, candidate-conditioned pooling, separate whole path, fragment-only JEPA.
4. `src/spikes/phase1/lit_fragment.py` (1–293): contrastive targets, negative-diagonal penalty, loss assembly, AdamW.
5. `src/concisejepa/lightning_modules/lit_fragment.py` (1–47), `src/concisejepa/models/fragment.py` (1–41): adapter/re-export, not independent implementations.
6. `src/concisejepa/models/drug_decoder.py` (1–211), `src/concisejepa/models/fsq.py` (290–332): matched continuous-FSQ mode and limitations.
7. `configs/model/brics_f2r_whole.yaml` (1–9), `configs/task/brics_f2r_whole.yaml` (1–5), `configs/experiment/brics_f2r_whole.yaml` (1–24): configuration seams.
8. `tests/test_secondary_binding.py` (1–302): forward separation, gradients, pairwise parity, baseline loss parity, metrics and reload coverage.
9. `main.py` (65–121): automatic final-checkpoint test evaluation.
10. Artifact directory `/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics_training/20260908-2120-f2r-secondary-binding/training/brics-f2r-secondary-fsq3-b6476734`: `resolved_config.yaml` (1–136), `epoch_metrics.jsonl` (1–30; all records parsed), `final_metrics.json` (1–32), `test_metrics.json` (1–15).

## Key Code / severity-tagged findings

**High — candidate-selection protocol:** `main.py:103–115` computes best checkpoint but sets `test_ckpt = final_ckpt or None` and tests automatically. Add a narrowly scoped evaluation gate before future candidate runs: candidates must skip test; explicitly evaluate only the frozen validation-selected winner using its validation-best checkpoint. Existing test AP .570362 is final-epoch, not validation-best performance.

**High — objective mismatch, a research hypothesis rather than correctness bug:** `src/spikes/phase1/lit_fragment.py:131–155,197–216,228–250` applies contrastive CE only to positive aligned pairs. Unknown off-diagonal pairs compete as negatives except identity-derived known-positive masks. Labeled negative diagonals merely need to fall below an unlabelled row mean (margin0), rather than receiving explicit binary supervision. Local row/column ranking may fail to produce globally useful positive/negative ranking. Unrecognized off-diagonal true positives remain possible. Never label all off-diagonals zero for BCE.

**High — misleading output names:** `src/spikes/phase1/fragment_encoder.py:112–118,353–372` ends the shared scorer in Sigmoid. `similarity_cosines` are sigmoid scores, not cosines; `similarity_logits` are temperature-scaled probabilities, not signed binary logits. **Do not use existing similarity_logits with BCEWithLogitsLoss.** The minimum change is float32 BCE on existing aligned branch probabilities. Stable raw-logit exposure is possible later, but not necessary architecture churn for this round.

**Medium — bottleneck not established:** `drug_layers=[[32,32,32]]` means one three-factor quantization code with 32^3 possible combinations, not three residual layers. `continuous_fsq` removes rounding but retains the three-coordinate projection/bounds and parameter shapes. Improvement implicates rounding; no improvement does NOT rule out dimensional compression. No evidence inspected here establishes actual code collapse. Do not jump to DiVeQ or expanded factors this round.

**Invariant verified in source:** `secondary_binding.py:25–42`, `fragment_encoder.py:257–308,353–372`: fragments are encoded and candidate-protein pooled for fragment scoring and JEPA context. Whole fingerprint is encoded separately, sharing projection/attention/scorer but setting `condition_on_fragments=False`; only scores are fused. Whole input must not change fragment/JEPA forward outputs at fixed weights. Shared-weight learning from whole supervision is allowed. The existing different dual-view model/pooler/objective is excluded as a non-equivalent control.

## Architecture

Hydra constructs the secondary model/task; standard model imports re-export the spike implementation. Secondary task inherits the standard adapter and spike loss assembly. Current loss:

`L = C_fragment + N_fragment + MSE_JEPA + .25*(C_whole + N_whole)`

Prediction: `.75*p_fragment + .25*p_whole`. Supervision and prediction weights are independent. Keep both coefficients .25; retain F2R, fragment-only JEPA, MSE targets, disabled chemistry/group auxiliaries, exact split/cache paths, count fingerprints, max_fragments16, batch256, seed42, AdamW lr1e-4 and weight_decay.01.

## Current trajectory

Every epoch was parsed. Zero-based epochs:

| epoch | train fused AP | val fragment AP | val whole AP | val fused AP |
|---|---:|---:|---:|---:|
| 0 | .7355 | .4288 | .4686 | .4519 |
| 11 | .8746 | .5216 | .4998 | .5310 |
| 15 | .8916 | .5310 | .4868 | .5357 |
| 22 | .9033 | .5371 | .5086 | .5457 |
| 27 | .9123 | .5390 | .5054 | **.5489** |
| 29 | .9179 | .5322 | .5111 | .5483 |

**Best validation fused AP is 0.548917651 at epoch27**, not final 0.548266113. `final_metrics.json` records completed-run epoch30 while the last training record is epoch29. Existing final test fused/fragment/whole AP are .570362/.555950/.524557. These are descriptive only, not candidate-selection evidence.

Weighted contrastive loss falls 6.825→3.378 train and 6.585→3.792 validation. Final JEPA MSE is .0964/.0970; weighted negative penalty .0836/.0956. Small scalar MSE does not establish small encoder gradients; measure gradients before diagnosing JEPA dominance. Late best-checkpoint improvements support testing a longer budget, but AP improves only about .018 between epochs11 and27. Large train/validation AP gap warrants caution, not a numerical overfitting estimate: online train metrics and possible prevalence/distribution differences confound that comparison. Whole standalone is weaker late but contributes to fixed-mixture validation AP, so increasing whole dominance is unsupported. No trajectory extrapolation credibly promises >.65 test AP.

## Small controlled round: maximum four candidates, two concurrent GPU jobs

Pre-register four fresh seed42 runs, each **max_epochs=60**, best checkpoint by `val/fused_pooled_ap`, fixed fusion .25. Same duration avoids objective-versus-budget confounding. Existing 30-epoch result is only the same-task historical reference. Do not selectively warm-start candidates. No scheduler, class weighting, resampling, fusion search, auxiliary heads or data changes.

Let `B_v = mean BCE(p_v,y)` on observed aligned labeled pairs only. Define:

`L_new = B_fragment + .25*B_whole + beta*(C_fragment+.25*C_whole) + gamma*(N_fragment+.25*N_whole) + lambda*MSE_JEPA`.

| candidate | objective | beta | gamma | lambda | quantizer | contrast |
|---|---|---:|---:|---:|---|---|
| A | exact legacy, no BCE | 1 | 1 | 1 | fsq | length-matched control |
| B | new | .10 | 0 | 1 | fsq | aligned BCE plus weak contrastive regularization |
| C | new | .10 | 0 | .10 | fsq | JEPA balance versus B |
| D | new | .10 | 0 | 1 | continuous_fsq | rounding versus B |

Schedule A/B then C/D, never more than two simultaneous one-GPU jobs. No jobs were submitted. B intentionally changes an objective package (BCE addition, CE reduction, negative-relative penalty removal), so it is not a pure causal test of BCE alone. C and D provide single-factor contrasts to B. Pure BCE, factor expansion and more fusion weights are deferred, not hidden extra candidates. If fixed60 is unaffordable, predeclare a common early-stop rule before launch, e.g. minimum30 epochs and patience15 on validation AP, rather than terminating selectively.

## Exact minimal implementation seam (future work only)

1. In `configs/task/brics_f2r_whole.yaml` and `LitSecondaryBinding.__init__`, introduce `binding_bce_weight=0`, `binding_contrastive_weight=1`, `binding_negative_weight=1`, `jepa_loss_weight=1`. Validate finite nonnegative values and save hyperparameters. These are **proposed**, not currently supported overrides. B uses 1/.1/0/1; C changes last value to .1. Existing `secondary_loss_weight=.25` remains unchanged.
2. Extend `LitSecondaryBinding._binding_losses` to calculate `binary_cross_entropy(outputs[f'{view}_binding'].float(), label.float())` for each view, retaining independent branch supervision. Return weighted BCE+CE as first tuple component and weighted N as second; inherited negative weight remains1. Log explicit branch BCE and weighted CE/BCE/N because aggregate `loss_dti` will no longer mean pure CE. Avoid computing inactive BCE in legacy mode to preserve default arithmetic exactly.
3. Narrow shared assembly change in `src/spikes/phase1/lit_fragment.py:241–250`: multiply JEPA loss by `self.jepa_loss_weight`, default1 initialized in the base constructor and set from secondary task configuration. Log raw and weighted JEPA. Avoid copying the whole forward loop or losing auxiliary behavior. No model forward changes required.
4. D uses existing `model.concise_fragment.drug_quantizer.type=continuous_fsq` only. Save resolved configuration with checkpoint; state-dict shapes alone cannot distinguish mode. Its -1 code sentinels are not discrete utilization observations; discrete enumeration/embedding operations are unavailable.
5. Gate `main.py:103–115` with a proposed `evaluation.run_test=false` for candidate fits; after selection, explicitly load the selected best checkpoint for a single final test evaluation. Preserve unrelated default workflows if needed. Record winner ID/checkpoint/config and selection rule before test access.

## Regression and success criteria

Before GPU approval, extend `tests/test_secondary_binding.py`; run existing secondary-binding, pair-conditioning, positive-mask and continuous-FSQ suites. Tests were inspected but not executed here.

- Legacy defaults reproduce losses, outputs and gradients; preserve zero-secondary baseline parity, optimizer behavior and checkpoint roundtrip.
- Hand-calculated BCE arithmetic for mixed/all-positive/all-negative batches and B=1; finite float32 endpoint behavior and gradients. Negative-label BCE must give a learning signal even where the relative penalty is zero.
- Whole-input perturbation leaves fragment prediction, pooled embedding and JEPA unchanged in eval mode. JEPA/fragment gradients to whole input absent; whole-only BCE updates shared encoder/scorer but not F2R pool parameters; fragment BCE updates pool parameters.
- Pairwise matrices equal single-pair scoring; preserve protein permutation, chunk-size and checkpoint-gradient parity, padding/max-fragment behavior and input metadata.
- Epoch-pooled AP/AUROC match concatenated-prediction metrics and reset each epoch; never average batch AP. Confirm candidate fits never test and final evaluation loads validation-best rather than final.
- Log branch/fused AP/AUROC, raw/weighted losses, score distributions by label, occasional loss-specific shared-encoder gradient norms on fixed training batches. Do not turn diagnostics into unregistered hyperparameter tuning.

Rank by maximum validation fused pooled AP only, tie within .001 resolved by earlier checkpoint then simpler FSQ candidate. Predeclare **meaningful screen improvement as ≥.02 absolute validation AP over A**, with fragment AP at the selected checkpoint no more than .01 below A's selected checkpoint. Report all candidates, selected epochs, and improvement versus historical best .5489. A gain solely accompanied by substantial primary-fragment deterioration is not the requested improvement. If none clears the bar, report inconclusive/negative results rather than expanding search via test feedback. Freeze the validation-selected winner before one held-out test evaluation; >.65 is an outcome criterion, never the selection criterion.

## Start Here

Open `src/concisejepa/lightning_modules/lit_secondary_binding.py`: branch probabilities and independent weighting already exist, making this an objective-only extension. Gate automatic testing in `main.py` before future training. Do not migrate to the non-equivalent dual-view architecture or insert whole features into pooling/JEPA.

## Scientific caveats / residual risks

Single-seed, four-candidate best-of-epoch validation selection has winner's bias; later seed confirmation needs separately approved budget. AP is ranking-based, so BCE calibration gains need not improve AP. False negatives, label noise, distribution shift and dimensional compression may dominate. Historical different-split AP is not comparable. Freeze split/cache hashes and effective sample counts/prevalence; no task/data changes. Existing dirty working tree predates this review, so reconcile source with run manifest before attributing results to the new objective. The already-observed test makes it less pristine; never tune to its branch metrics, repeatedly evaluate winners, or keep searching until test passes. Continuous-FSQ is diagnostic and cannot meet a requirement for discrete enumeration. Shared-weight updates couple branch learning despite correct forward separation. No implementation or prospective improvement is attested.