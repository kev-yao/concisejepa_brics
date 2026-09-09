**Both runs completed successfully:** exit code **0**, **30/30 epochs** (final index 29). `comparison.state="completed"`; `missing=[]`.

- Continuous3 finished **19:03:02 EDT**.
- FSQ5 finished **19:07:37 EDT**, September 8, 2026.

Final-checkpoint metrics; paired values are **validation / test**:

| Metric | Continuous3 | FSQ5 |
|---|---:|---:|
| Pooled AP | 0.5461 / 0.5429 | 0.5238 / 0.5477 |
| Validation within-receptor macro AP | 0.7232 | 0.7222 |
| Molecule-balanced JEPA MSE | 0.1226 / 0.1257 | 0.1232 / 0.1265 |
| Molecule-balanced retrieval R@1 | 0.0306 / 0.0184 | 0.0356 / 0.0176 |
| Molecule-balanced retrieval R@5 | 0.1121 / 0.0648 | 0.1128 / 0.0743 |
| Molecule-balanced retrieval MRR | 0.0830 / 0.0533 | 0.0895 / 0.0559 |

**FSQ5 codes:** 696 active, 436.7 effective, capacity 32,768 (2,407-molecule panel). FSQ3 reference: 1,075 active / 793.1 effective. Continuous3: not applicable.

**Artifacts**, under:
`/hpc/group/singhlab/user/me196/projects/obsidian/runs/concisejepa_brics_training/20260908-1700-f2r-quantizer-ablation/`
- `continuous3_training_status.json`
- `fsq5_training_status.json`
- `COMPARISON.md`
- `comparison.json`

Single seed (**42**) and existing **non-cold splits** limit generalization; these results do not establish a universal winner. All requested artifacts were present, so no SLURM fallback query was needed. Nothing modified or restarted.