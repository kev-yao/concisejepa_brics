## Review

**No modeling or leakage blockers found by static inspection. Proceed only after the planned validation-reproduction and separation checks pass.**

Paths below are relative to `../concisejepa_brics_training/`; **A** = `20260909-0055-ap65-goal`, **S** = `20260908-2120-f2r-secondary-binding/source`.

- **Correct — training/selection separation:** `A/frozen_readout_probe.py:21–43` loads only train/validation datasets, fits both scaler and logistic regression on **train only**, and uses validation labels solely for evaluation. `S/src/spikes/phase1/fragment_datamodule.py:305–310` confirms `setup("fit")` does not instantiate the test dataset. Existing eligibility filtering is retained.
- **Correct — inference reproduction:** Feature order `[d,p,d*p,abs(d-p)]`, saved preprocessing/classifier, positive-class probability, and fixed 75/25 fusion agree between `A/frozen_readout_probe.py:28–45` and `A/readout_candidate.py:30–48`. C=0.1 has the highest recorded fused validation AP, **0.6136697303**, matching the locked selection. The wrapper performs no fitting.
- **Correct — legitimate neural-feature head:** Inputs are normalized learned fragment/protein representations, not identity indicators or labels (`S/src/spikes/phase1/fragment_encoder.py:353–373`). F2R, fragment FSQ encoding, and fragment-only JEPA remain active. Whole fingerprints use the shared encoder/attention/scorer through a separate forward path without fragment pooling (`:284–307`). The new primary logistic head and unchanged whole final scorer are **distinct heads**, although upstream neural parameters remain shared. Describe this as a frozen neural backbone with a supervised primary readout—not an unchanged shared-final-scorer architecture.

### Caveats / preflight requirements

- **Note — medium, import provenance:** `A/readout_candidate.py:25–29` uses ambient Hydra imports; `readout-selection.json:11` records `source_root` but the wrapper never enforces it. Checkpoint/readout hashes do not pin source or resolved configuration. Before test, use a clean process pinned to immutable `S/src`, record actual `concisejepa` and `spikes` module locations/config digest, and reproduce validation predictions/AP. Strict state loading alone cannot detect same-shaped forward-code changes.
- **Note — validation pending:** Whole-input perturbation should leave the **new** primary probability and JEPA prediction unchanged; fragment perturbation should leave whole probability unchanged. Frozen synthetic coverage exists at `S/tests/test_secondary_binding.py:46–64`, but I did not execute it or the candidate checks.
- **Note — scientific scope:** `A/PLAN.md:33–37` originally specifies the A–D promotion policy and describes this readout as diagnostic; `A/readout-selection.json:7` explicitly records its subsequent promotion. Present that honestly as a validation-driven plan amendment. Prior test inspection is disclosed in `A/PLAN.md:7`; another check is not untouched-holdout evidence, and validation AP does not establish test AP >0.65.

No source/artifact edits, shell commands, dataset/label access, or inference were performed.