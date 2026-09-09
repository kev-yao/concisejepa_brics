# Objective-weight and evaluation-gate implementation

## Implemented

Added backward-compatible optional weights to `LitSecondaryBinding`:

- `binding_bce_weight=0`: mean float32 probability BCE, observed aligned branch labels only. Inactive BCE is not evaluated.
- `binding_contrastive_weight=1`: existing corrected contrastive loss.
- `binding_negative_weight=1`: existing corrected negative-diagonal loss, still multiplied by the existing global negative-diagonal coefficient.
- `jepa_loss_weight=1`: fragment-only JEPA through a minimal shared base-task assembly seam.

All weights, including existing secondary supervision, are finite/nonnegative validated and saved as task hyperparameters. Fragment and whole branches remain independently supervised; existing secondary_loss_weight combines them. No model/scorer/encoder/pooling/forward changes. Probability BCE does NOT consume the existing temperature-scaled `similarity_logits` and does not assign labels to unknown off-diagonal pairs.

Raw branch BCE/contrastive/negative logs and objective-weighted per-branch/combined logs are explicit. Historical branch `loss_*_dti` remains the raw-CE alias; aggregate `loss_dti` includes weighted BCE+CE. Negative weighted logs include the global negative-diagonal weight; combined weighted logs additionally include whole supervision beta. Raw and weighted JEPA are logged separately. Disabled BCE logs zero because it is not measured. Default arithmetic skips inactive additions/unit multiplications for exact legacy parity.

Added `evaluation.run_test=true` to the root config and a runner gate. False skips `Trainer.test` entirely. Missing settings in old configs retain true behavior. True/default still tests FINAL checkpoint (not best); no alternate checkpoint-selection machinery was introduced. Parent collection owns explicit validation-best winner testing.

## Changed files for this slice

1. `src/concisejepa/lightning_modules/lit_secondary_binding.py`
2. `src/spikes/phase1/lit_fragment.py`
3. `configs/task/brics_f2r_whole.yaml`
4. `configs/config.yaml`
5. `main.py`
6. `docs/secondary-binding.md`
7. `tests/test_secondary_binding.py` — existing task-swap fixture removes the four newly explicit secondary-only config keys.
8. `tests/test_secondary_objectives.py` — six new objective tests.
9. `tests/test_runner_evaluation.py` — three new runner/config/integration tests.

Other dirty changes predate this slice and were preserved. No named supervised experiment was necessary: documented existing-experiment overrides select all approved candidates. No staging, commits, scheduler submissions, real-data training, or cache changes.

## Validation

Environment: `/hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu`; preload its `lib/libstdc++.so.6`, prepend its lib directory, OMP/MKL threads4, `PYTHONPATH=src:.:tests`, `TMPDIR=/tmp/me196-objective-worker`.

- Tests first failed before implementation because weight/gate config seams did not exist: `objective-before.log`.
- **69 compatible tests passed** (the prior60 plus9 new): `objective-compatible-tests.log`, 16.401 seconds.
- Focused new/existing secondary tests: **18 passed**, `objective-focused.log`.
- `objective-frozen-parity.py` invokes independent current/frozen-source processes. **Bit-exact CPU FSQ and continuous-FSQ default initialization, outputs, losses, all populated parameter gradients, and two AdamW updates** versus immutable `20260908-2120-f2r-secondary-binding/source/`: `objective-frozen-parity.log` and paired trajectory artifacts.
- BCE arithmetic tested for independent objective/global-negative/secondary coefficients; mixed, all-positive, all-negative and singleton batches; finite float32 loss/gradients at/near endpoints; negative-label gradient when old hinge has zero penalty; disabled BCE is not called.
- Whole BCE reaches shared encoder but not F2R; fragment BCE reaches F2R; weighted JEPA has no gradient to separate whole input. Existing forward/pair/chunk/checkpoint/parity tests pass unchanged except the config-swapping fixture adaptation.
- A real one-batch CPU Lightning runner fit with nondefault weights and `run_test=false` calls only datamodule fit setup, creates final metrics/checkpoint, creates no test metrics, and reloads all configured weights from checkpoint. Mocked runner cases verify false never invokes test/setup and true/default/old-config invoke test with FINAL rather than validation-best checkpoint.
- Ruff on all changed Python files, compileall and `git diff --check` passed. Candidate/legacy configs resolve. Candidate config asserts weights `[1,.1,0,.1]`, max60 and boolean false test gate; baseline retains true. `objective-static.log`, `objective-candidate-config.yaml`, `objective-baseline-config.yaml`.
- Verified every `src/concisejepa/models/*.py` and `src/spikes/phase1/fragment_encoder.py` hash remains identical to the immutable secondary snapshot.
- `git diff --cached --name-only` is empty.

The known unrelated `tests/test_quantizers` module (17 tests with6 failures/2 errors in preexisting residual-quantizer expectations) was intentionally excluded and not repaired/reinvestigated. Prior frozen-source reproduction is retained in the earlier validation bundle. Do not claim the full repository suite globally green.

## Risks / next step

Parent should run its CUDA smoke and fresh independent review, then freeze source/configs for candidate jobs. No real-data/GPU training or test-set evaluation was performed by this worker. Objective effectiveness and testAP>0.65 remain unknown. Candidate runs must explicitly set `evaluation.run_test=false`; the legacy default intentionally remains true. Full pairwise forward computation remains unchanged even when contrastive/negative weights are zero—no performance/architecture refactor was included. Preserve config with checkpoints as before.