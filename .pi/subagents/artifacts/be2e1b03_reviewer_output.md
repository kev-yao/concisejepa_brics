## Review

**No blockers for a single collector invocation.** Paths below are relative to the artifact directory.

- **Fixed:** Import provenance concern resolved by snapshot pinning, imported-module path checks, and optimization rejection (`train_head.py:18–40`).
- **Correct:** Collection requires completed trials, selects validation AP with deterministic ties, saves selection before fresh validation, strictly reloads weights, and verifies TRAIN-derived normalization (`collect_and_test.py:40–78`). Preflight returns before test setup (`:138–142`); inherited datamodule `setup("fit")` creates only train/validation datasets.
- **Note — medium:** The one-test guard is not concurrency-safe: existence check at `collect_and_test.py:32` and replacing write at `:141` allow two simultaneous collectors to both test. Use exclusive marker creation immediately before test setup, or enforce exactly one collector operationally.
- **Limits:** Static review only; inspected the passing preflight record, but independently ran nothing and accessed no test data. Final winner validation/test outcomes remain unverified.