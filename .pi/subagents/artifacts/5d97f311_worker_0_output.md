# Implementation: secondary whole-molecule binding

## Implemented

Added opt-in `brics_f2r_whole` / `brics_f2r_whole_debug` bundles. The corrected fragment F2R/MLP-JEPA path stays primary. A separate whole fingerprint is encoded through the same DrugEncoder/FSQ (or continuous FSQ), projected and scored using shared protein tensors/attention/scorer, **without calling fragment pooling**. Whole fingerprints are never appended to fragment tokens and never feed JEPA. No additional trainable parameters were introduced.

Late sigmoid-score fusion is `(1-alpha)*fragment + alpha*whole`, starting at alpha 0.25. The unchanged fragment contrastive/negative-diagonal objective receives an additional beta 0.25 times the identical whole-branch objective; JEPA remains fragment-only. Alpha and beta are independently configured/validated; alpha endpoints 0/1 work. New outputs explicitly expose each branch's binding/pair scores/scaled logits; the existing unprefixed binding/similarity keys represent fusion. Existing fragment features/codes/JEPA keys remain fragment-only. Missing or incorrectly shaped whole inputs fail clearly.

The new task reuses the original loss helpers/identity masks through small batch/loss/metric seams, not copied training logic. Pooled epoch AP/AUROC are logged for fragment, whole, fused on train/val/test using registered TorchMetrics objects with Lightning resets. Existing metric callbacks persist them; best monitoring uses `val/fused_pooled_ap`, while the runner retains final-checkpoint test evaluation.

Reused the eight-item DualViewFragmentDataModule/collator and enabled `require_count_values: true`. A verified matching count-Morgan cache must be supplied; legacy binary caches mislabeled by metadata are rejected. No cache or existing run bundle was modified. Legacy unfragmentable/fallback whole-as-single-fragment behavior remains explicitly documented.

## Changed files (this task only)

- `src/spikes/phase1/fragment_encoder.py`: optional separate whole scoring, explicit bypass of conditioned fragment pooling, shared JEPA-output helper; old output contract unchanged without whole input.
- `src/spikes/phase1/lit_fragment.py`: narrow batch/loss/metric hooks, preserving old objective behavior.
- `src/concisejepa/models/secondary_binding.py`: opt-in model and score fusion.
- `src/concisejepa/lightning_modules/lit_secondary_binding.py`: secondary supervision and pooled branch metrics.
- `configs/model/brics_f2r_whole.yaml`
- `configs/data/brics_f2r_whole.yaml`
- `configs/task/brics_f2r_whole.yaml`
- `configs/experiment/brics_f2r_whole.yaml`
- `configs/experiment/brics_f2r_whole_debug.yaml`
- `tests/test_secondary_binding.py`: nine regression/integration tests.
- `docs/secondary-binding.md`: contracts, objective, interpretation limits, usage.
- `README.md`: brief feature link/usage entry.

Other dirty files/untracked tests were preexisting and left untouched. Nothing staged or committed.

## Validation

Environment used: `ENV=/hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu`, `LD_PRELOAD=$ENV/lib/libstdc++.so.6`, `LD_LIBRARY_PATH=$ENV/lib`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Normal `PYTHONPATH=src:.`; the final broad test command additionally uses `tests` because existing `test_brics_audit.py` imports a sibling as a top-level module. Node-local TMPDIR avoids unrelated network-filesystem TemporaryDirectory cleanup errors.

1. Tests first: five initial new tests failed as expected before missing configs/model existed (`secondary-before.log`).
2. Final focused/broader suite: **60 tests passed**, including nine new tests plus corrected pair/mask/continuous FSQ tests, baseline training parity, config/data tests, existing dual-view and audit tests. See `final-tests-passed.log`.
3. `frozen_parity.py`: compared against immutable corrected source from `20260908-1700-f2r-quantizer-ablation/source/`; **exact zero-tolerance parity** for CPU and CUDA, both mean and F2R, initial state dicts, model outputs, losses, gradients, and two AdamW updates. See `frozen-parity.log`.
4. Nine new tests cover whole/fragment input perturbation isolation (FSQ and continuous FSQ), JEPA-to-whole gradient absence, whole DTI gradient into shared encoder but not pooling, fragment DTI gradient into pooling, independent pair matrices, protein permutation, chunk sizes, checkpoint gradients/attribution preservation, independent-pair whole gradients, alpha endpoints/invalid weights/missing input, exact objective arithmetic, zero-secondary full training parity, compatible collation/count guard, optimizer/state roundtrip, actual Lightning checkpoint roundtrip, two-epoch metric artifact persistence, epoch-pooled metric correctness and reset across differently sized repeated validations.
5. Changed-Python-file Ruff, compileall, git diff whitespace checks passed (`static-checks.log`). Both new configs and baseline/brics_mean/brics_xattn resolve successfully; resolved YAML artifacts retained here.
6. Broad discovery is **not globally green**: preexisting `test_quantizers` expects residual-code/blend/norm outputs no longer supplied by the preexisting encoder. Six failures and two errors were reproduced against frozen source, unchanged (`frozen-quantizer-tests.log`). The initial discovery also had two network-TMPDIR cleanup errors; both passed using node-local TMPDIR (`tempdir-retest.log`). An explicit-module broad run hit the existing audit sibling-import issue before adding `tests` to PYTHONPATH; the corrected final run passed. No unrelated code/test repairs were made.

## Risks / next step

- Parent still needs independent review and the planned real-data GPU smoke with a verified count cache. No new training run or scheduler job was launched by this worker.
- Alpha/beta 0.25 are starting values, not selected hyperparameters or evidence of improved performance.
- Shared encoder/scorer weights couple learning even though forward inputs and JEPA are separated.
- Preserve resolved model/config when reloading; fusion/quantizer modes are configuration, not uniquely identifiable from parameter keys.
- Existing frozen-run external prediction-capture callbacks assume the old batch layout; they were intentionally not modified or attached. New standard metrics callbacks were exercised successfully.