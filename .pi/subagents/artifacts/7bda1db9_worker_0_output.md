# Matched continuous-FSQ implementation

## Implemented

Added explicit `drug_quantizer.type=continuous_fsq`. The encoder builds the identical learnable architecture as `fsq`, consuming exactly the same initialization RNG. `ResidualFSQ(discretize=False)` replaces only rounding with `fsq.bound(projected) / (fsq._levels // 2)`; bounds, scaling, projections, normalization, activation and default FSQ behavior remain unchanged.

Continuous outputs contain integer `-1` factor-code sentinels. `embed` and `all_code_indices` explicitly reject continuous mode; `get_levels` exposes its bounding levels. Inline documentation warns that the mode must be retained in checkpoint configuration because state-dict keys/shapes deliberately match FSQ.

Changed source/test paths only:
- `src/concisejepa/models/drug_decoder.py`
- `src/concisejepa/models/fsq.py`
- `tests/test_continuous_fsq.py` (new, six tests)

All pre-existing uncommitted correctness fixes were preserved. No staging, commits, training launches, or run-bundle writes.

## Validation

TDD: first formula/seed/sentinel test failed with unknown `continuous_fsq`, then passed after implementation. The embedding/enumeration rejection test failed before its guards were added, then passed.

Six new tests cover independent bounded/no-round formula, same-seed state-dict and RNG equality, sentinel/unsupported-API contracts, finite/nonzero CPU and CUDA gradients in train/eval, unchanged default FSQ forward/gradient composition, five-factor 32,768-code capacity and roundtrip, and a tiny corrected checkpointed F2R optimizer step plus serialized model/optimizer reload.

**24 tests passed**: six new + six conditioning + six positive-mask + one refactor-parity + five Hydra-integration tests. Only expected existing warnings about direct Lightning calls without a Trainer and Transformer nested tensors appeared.

Exact suite command (run from repository root):
```bash
E=/hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu
mkdir -p /tmp/concisejepa-ablation-tests
LD_PRELOAD=$E/lib/libstdc++.so.6 PYTHONPATH=$PWD/src TMPDIR=/tmp/concisejepa-ablation-tests OMP_NUM_THREADS=4 "$E/bin/python" - <<'PY'
import unittest
patterns = ['test_continuous_fsq.py', 'test_fragment_pair_conditioning.py', 'test_positive_pair_mask.py', 'test_training_refactor_parity.py', 'test_brics_hydra_integration.py']
suite = unittest.TestSuite(unittest.defaultTestLoader.discover('tests', pattern=p) for p in patterns)
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

Result: `Ran 24 tests in 3.821s`, `OK`, exit 0.

Static checks:
```bash
/hpc/group/singhlab/user/cy244/projects/micromamba/envs/concise311-gpu/bin/ruff check src/concisejepa/models/drug_decoder.py src/concisejepa/models/fsq.py tests/test_continuous_fsq.py
git diff --check
git diff --cached --quiet
```
All passed; no staged files.

Additional independent baseline probe imported the original `fsq.py` and `drug_decoder.py` from immutable snapshot `/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics_training/20260908-155827-f2r-fixed/source/src/concisejepa/models`, setting that private reference DrugEncoder module's ResidualFSQ symbol to its original snapshot class. Compared old/new default FSQ encoders at seed42, dim16, latent8, levels[32,32,32] across two AdamW steps at lr1e-4. **All returned tensors, all parameter gradients and post-step state tensors were bit-exact on both CPU and CUDA** (rtol=atol=0).

## Boundaries and remaining risks

- Parent must adapt monitoring to continuous mode; -1 sentinels are not a codebook, and occupancy/entropy must not be reported for them.
- Checkpoint reconstruction must use the recorded `continuous_fsq` config; weights alone intentionally cannot identify the mode.
- Full-size training smoke, launch, and experiment-level evaluation remain parent-owned. Tiny F2R coverage includes the real model, corrected loss, checkpointed pair scoring and optimizer/state serialization, not a Lightning Trainer-managed checkpoint.
- Pre-existing stale hybrid-residual expectations in `tests/test_quantizers.py` were not changed or run wholesale.

Recommended next step: independent review, snapshot these three changed paths, then full-batch smoke and the approved paired launches.