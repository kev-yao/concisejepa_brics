## Review

**No new modeling or leakage blocker found.** This is a **boosted-tree primary head on a frozen neural backbone—not end-to-end neural retraining**.

Paths below are relative to `../concisejepa_brics_training/20260909-0055-ap65-goal/`.

- **Correct — train-only fitting:** `boosted_readout_probe.py:15–33` fits HGB exclusively on cached training features/labels, with `early_stopping=False`. Validation data enters only prediction and metrics. No test-reading code appears in this probe.
- **Correct — selection:** `hgb-fusion-validation-scan.json:5–145` records all 27 positive-alpha combinations; HGB31/alpha0.05 is the maximum at **0.6710058808**, matching `hgb-selection.json`. Inference applies **95% primary + 5% unchanged whole score**, without feature mixing (`readout_candidate.py:60–74`).
- **Correct — provenance improvement:** `readout_candidate.py:22–56` now checks checkpoint, readout, configuration, manifest and manifest-listed source hashes, pins immutable imports, and explicitly rejects project modules imported from another location. This addresses the previous ambient-import concern. `readout-validation.json` records successful linear reproduction, whole-perturbation separation and immutable module paths.

### Caveats before test

- **Note — medium, HGB-specific preflight missing from reviewed artifacts:** `validate_readout_candidate.py:12–20` remains hardcoded to the **linear** selection, scaler/logistic attributes and 75/25 fusion. Its successful result does not validate the locked HGB artifact. Before test, verify HGB parameters/classes and reproduce its cached versus fresh-validation probabilities and **0.6710058808** AP through the wrapper. Tree thresholds make checking actual HGB outputs preferable to inferring equivalence from linear scores.
- **Note — launcher remains linear-specific:** `evaluate_locked_readout.py:14–20` uses linear result/selection filenames. Preserve those artifacts; use a candidate-specific HGB launcher/result namespace and bind its validation evidence to the HGB selection hash.
- **Note — low, assertion-dependent integrity:** Hash checks use `assert` (`readout_candidate.py:24–32`); run without `python -O`/`PYTHONOPTIMIZE`.
- **Note — interpretation/provenance:** The saved probe documents the reported inline algorithm, but static review cannot independently establish which command produced the serialized estimator. The unblended HGB31 validation AP (**0.6733008**) exceeds the selected fused AP; retaining 5% whole satisfies the separate-secondary-path constraint but does **not** demonstrate a validation benefit from that branch. Repeated benchmark checks remain monitored stopping evidence, not untouched-holdout validation.

No files changed, commands executed, or test data/results read.