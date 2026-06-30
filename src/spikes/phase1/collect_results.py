#!/usr/bin/env python3
"""
phase1_collect_results.py — Collect and print comparison table from completed training runs.

Usage:
    python -m spikes.phase1.collect_results --study_dir /hpc/group/.../fragment_pool_study
"""
import argparse
import json
from pathlib import Path


def collect(study_dir: Path) -> list[dict]:
    rows = []
    for summary_file in sorted(study_dir.rglob("summary.json")):
        try:
            data = json.loads(summary_file.read_text())
            rows.append(data)
        except Exception as e:
            print(f"[warn] Could not read {summary_file}: {e}")
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return

    keys = ["pooling", "val/dti_auroc", "val/dti_auprc", "val/loss_jepa", "test/dti_auroc", "test/dti_auprc"]
    header = " | ".join(f"{k:25s}" for k in keys)
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda r: r.get("val/dti_auprc", 0.0), reverse=True):
        cells = []
        for k in keys:
            val = row.get(k, "—")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        print(" | ".join(f"{c:25s}" for c in cells))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study_dir", default="/hpc/group/singhlab/user/cy244/projects/peptide_evals/fragment_pool_study")
    args = p.parse_args()
    rows = collect(Path(args.study_dir))
    print(f"\nFound {len(rows)} completed runs in {args.study_dir}\n")
    print_table(rows)
