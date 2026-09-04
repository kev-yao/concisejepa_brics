#!/usr/bin/env python3
"""Rank ligands within receptor groups, and receptors within ligand groups."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def conditional_metrics(frame, group_column, score_column="mean", min_rows=5):
    rows = []
    for key, group in frame.groupby(group_column, sort=True):
        if len(group) < min_rows or group.label.nunique() < 2:
            continue
        y = group.label
        pred = group[score_column]
        positives = int(y.sum())
        negatives = len(y) - positives
        rows.append(
            {
                "group": key,
                "n": len(y),
                "positive_rate": float(y.mean()),
                "auprc": float(average_precision_score(y, pred)),
                "auroc": float(roc_auc_score(y, pred)),
                "comparisons": positives * negatives,
            }
        )
    if not rows:
        return {"groups": 0, "rows": 0}
    comparisons = sum(r["comparisons"] for r in rows)
    return {
        "groups": len(rows),
        "rows": sum(r["n"] for r in rows),
        "minimum_group_rows": min_rows,
        "macro_auroc": sum(r["auroc"] for r in rows) / len(rows),
        "macro_auprc": sum(r["auprc"] for r in rows) / len(rows),
        "macro_prevalence": sum(r["positive_rate"] for r in rows) / len(rows),
        "macro_ap_lift_over_constant": sum(r["auprc"] - r["positive_rate"] for r in rows) / len(rows),
        "within_group_pair_weighted_auroc": sum(r["auroc"] * r["comparisons"] for r in rows) / comparisons,
    }


def run(root):
    result = {}
    for path in sorted(root.glob("*/seed_*/best_dti_val_predictions.csv")):
        frame = pd.read_csv(path)
        values = {}
        for group, col in (("within_protein", "protein_sha256"), ("within_molecule", "SMILES")):
            for view in ("mean", "whole", "fragment"):
                values[f"{group}_{view}"] = conditional_metrics(frame, col, view)
        result[str(path.parent.relative_to(root))] = values
    (root / "conditional_validation_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
